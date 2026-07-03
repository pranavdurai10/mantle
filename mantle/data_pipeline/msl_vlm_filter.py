#!/usr/bin/env python
# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : VLM-based boulder density filter for raw MSL imagery
#               – Uses Qwen3.5-35B-A3B (sparse MoE) to classify boulder density
#               – Sorts images into dense / mediocre / sparse / rejected buckets
#               – Supports resumable runs via checkpoint + per-image audit CSV
# =============================================================================

"""
Model notes (Qwen3.5-35B-A3B)
------------------------------
- Sparse MoE: 35B total params, only 3B activated — memory footprint is modest.

- Package requirements:

    # Requires transformers from the main branch: 
    pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
    pip install accelerate pillow torch torchvision
    pip install causal-conv1d
    pip install git+https://github.com/fla-org/flash-linear-attention

Usage
-----
python msl_vlm_filter.py \
    --input_dir  /path/to/raw_images \
    --output_dir /path/to/sorted \
    --model      Qwen/Qwen3.5-35B-A3B \
    --device     cuda \
    --extensions jpg jpeg png tif \
    --log_csv    results.csv

Supports automatic resume from interruption — re-run with the same
arguments to continue processing. Use --force-restart to start fresh.

Directory layout produced
-------------------------
<output_dir>/
    dense/
    mediocre/
    sparse/
    rejected/          # no boulders detected
    results.csv        # per-image audit trail
"""

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("boulder_filter")


# Prompt templates
STAGE1_SYSTEM = (
    "You are a planetary geology expert specialising in Mars surface imagery "
    "from the MSL Curiosity rover. Your task is to analyse images and determine "
    "whether they contain rocks or boulders. Answer ONLY with a single word: "
    "'yes' or 'no'."
)

STAGE1_USER = (
    "Does this Mars rover image contain any rocks or boulders? "
    "Rocks or boulders are defined as discrete solid fragments larger than "
    "approximately 2 cm. Respond with exactly one word: yes or no."
)

STAGE2_SYSTEM = (
    "You are a planetary geology expert specialising in Mars surface imagery "
    "from the MSL Curiosity rover. Your task is to estimate the spatial density "
    "of boulders / rocks in an image. Answer ONLY with a single word from this "
    "fixed vocabulary: dense, mediocre, or sparse."
)

STAGE2_USER = (
    "Estimate the spatial density of rocks and boulders in this Mars rover image.\n"
    "Use these definitions:\n"
    "  dense    – rocks or boulders cover more than ~40 % of the visible ground.\n"
    "  mediocre – rocks or boulders cover roughly 10–40 % of the visible ground.\n"
    "  sparse   – rocks or boulders cover less than ~10 % of the visible ground.\n"
    "Respond with exactly one word: dense, mediocre, or sparse."
)

VALID_DENSITY = {"dense", "mediocre", "sparse"}
OUTPUT_DIRS = ["dense", "mediocre", "sparse", "rejected"]


# Checkpoint management functions
def load_checkpoint(checkpoint_path: Path) -> dict:
    """
    Load checkpoint state from JSON file.

    Returns empty checkpoint structure if file doesn't exist or is corrupted.
    Backs up corrupted checkpoints to .checkpoint.json.backup.
    """
    if not checkpoint_path.exists():
        return {
            "version": "1.0",
            "config": {},
            "processed": {},
            "stats": {}
        }

    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning("Corrupted checkpoint file: %s. Starting fresh.", e)
        # Backup corrupted checkpoint
        backup_path = checkpoint_path.with_suffix('.json.backup')
        shutil.copy2(checkpoint_path, backup_path)
        log.info("Corrupted checkpoint backed up to %s", backup_path)
        return {
            "version": "1.0",
            "config": {},
            "processed": {},
            "stats": {}
        }


def save_checkpoint(checkpoint_path: Path, checkpoint_data: dict) -> None:
    """
    Save checkpoint state to JSON file with atomic write.

    Uses atomic write pattern (write to temp file, then rename) to prevent
    corruption if interrupted during write.
    """
    checkpoint_data["stats"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Atomic write: write to temp file, then rename
    temp_path = checkpoint_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    # Atomic rename (overwrites existing file)
    temp_path.replace(checkpoint_path)


def validate_checkpoint_config(checkpoint: dict, args: argparse.Namespace) -> bool:
    """
    Validate that checkpoint config matches current run arguments.

    Returns True if config matches or checkpoint is empty.
    Returns False if config mismatch detected.
    """
    if not checkpoint["config"]:
        return True  # Empty checkpoint, first run

    config = checkpoint["config"]
    mismatches = []

    if str(Path(config.get("input_dir", ""))) != str(Path(args.input_dir)):
        mismatches.append(f"input_dir: {config.get('input_dir')} != {args.input_dir}")
    if str(Path(config.get("output_dir", ""))) != str(Path(args.output_dir)):
        mismatches.append(f"output_dir: {config.get('output_dir')} != {args.output_dir}")
    if config.get("model") != args.model:
        mismatches.append(f"model: {config.get('model')} != {args.model}")

    if mismatches:
        log.warning("Checkpoint config mismatch detected:")
        for mismatch in mismatches:
            log.warning("  - %s", mismatch)
        return False

    return True


# Model definition wrapper
class QwenVLMFilter:
    """
    Thin wrapper around Qwen3.5-35B-A3B (or any compatible Qwen VLM checkpoint).

    Qwen3.5 behaviour:
      - Thinking mode disabled via chat_template_kwargs={"enable_thinking": False}
        so the model skips <think>...</think> and returns a direct single-word answer.
        with headroom in case a special token or newline is emitted before the answer.
      - Greedy decoding with max_new_tokens=16 — more than enough for one word,
    """
    def __init__(self, model_name: str, device: str, dtype=None):
        log.info("Loading processor from %s …", model_name)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        log.info("Loading model from %s …", model_name)
        load_kwargs = dict(trust_remote_code=True)
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        elif device == "cuda":
            load_kwargs["dtype"] = torch.bfloat16

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, **load_kwargs
        ).to(device)
        self.model.eval()
        self.device = device
        log.info("Model loaded on %s.", device)

    @torch.inference_mode()
    def _infer(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        """
        Run a single image-text inference and return the decoded answer.

        Thinking mode is disabled via chat_template_kwargs so the model skips
        the <think>...</think> block and answers directly with a single word.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        # Decode only newly generated tokens
        answer = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return answer.strip().lower()

    def has_boulders(self, image: Image.Image) -> bool:
        """
        Stage 1: returns True if boulders / rocks are detected.
        """
        answer = self._infer(image, STAGE1_SYSTEM, STAGE1_USER)
        # Normalise: accept any response that starts with 'yes'
        return answer.startswith("yes")

    def density(self, image: Image.Image) -> str:
        """
        Stage 2: returns one of 'dense', 'mediocre', 'sparse'.
        Falls back to 'sparse' if the model returns an unexpected token.
        """
        answer = self._infer(image, STAGE2_SYSTEM, STAGE2_USER)
        for label in VALID_DENSITY:
            if answer.startswith(label):
                return label
        log.warning("Unexpected density answer '%s', defaulting to 'sparse'.", answer)
        return "sparse"


# Load images from input directory
def collect_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(input_dir.rglob(f"*.{ext}"))
        paths.extend(input_dir.rglob(f"*.{ext.upper()}"))
    return sorted(set(paths))

# Create output directory structure
def setup_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {}
    for name in OUTPUT_DIRS:
        d = output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs

# Pipeline for filtering mechanism
def run_pipeline(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_csv    = output_dir / args.log_csv
    checkpoint_path = output_dir / ".checkpoint.json"

    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    images = collect_images(input_dir, args.extensions)
    if not images:
        log.error("No images found in %s with extensions %s", input_dir, args.extensions)
        sys.exit(1)

    log.info("Found %d image(s) to process.", len(images))
    dirs = setup_output_dirs(output_dir)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)

    # Handle --force-restart flag
    if args.force_restart:
        log.info("Force restart requested. Discarding checkpoint.")
        checkpoint = {
            "version": "1.0",
            "config": {},
            "processed": {},
            "stats": {}
        }
        if checkpoint_path.exists():
            backup_path = checkpoint_path.with_suffix('.json.old')
            shutil.move(checkpoint_path, backup_path)
            log.info("Old checkpoint moved to %s", backup_path)
    else:
        # Validate checkpoint config
        if not validate_checkpoint_config(checkpoint, args):
            log.error("Checkpoint config mismatch. Use --force-restart to start fresh.")
            sys.exit(1)

    # Update checkpoint config
    checkpoint["config"] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "extensions": args.extensions
    }
    checkpoint["stats"]["total_found"] = len(images)

    # Filter already processed images
    processed_names = set(checkpoint["processed"].keys())
    images_to_process = [img for img in images if img.name not in processed_names]

    if processed_names:
        log.info("Resuming from checkpoint: %d already processed, %d remaining.",
                 len(processed_names), len(images_to_process))

    if not images_to_process:
        log.info("All images already processed. Nothing to do.")
        log.info("Use --force-restart to reprocess all images.")
        return

    # Load model
    log.info("Loading model (this may take 1-2 minutes)...")
    vlm = QwenVLMFilter(
        model_name=args.model,
        device=args.device,
    )

    # CSV audit trail
    csv_rows = []
    counters  = {k: 0 for k in OUTPUT_DIRS}

    # Count already processed images by category
    for img_name, img_data in checkpoint["processed"].items():
        dest = img_data.get("dest", "—")
        for bucket in OUTPUT_DIRS:
            if f"/{bucket}/" in dest:
                counters[bucket] += 1
                break

    t0 = time.time()
    total_images = len(images)
    already_processed = len(processed_names)

    for idx, img_path in enumerate(images_to_process, start=already_processed + 1):
        log.info("[%d/%d] Processing: %s", idx, total_images, img_path.name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            log.warning("Cannot open %s: %s — skipping.", img_path.name, exc)
            checkpoint["processed"][img_path.name] = {
                "stage1": "error",
                "stage2": "—",
                "dest": "—",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            csv_rows.append(
                {"file": str(img_path), "stage1": "error", "stage2": "—", "dest": "—"}
            )
            save_checkpoint(checkpoint_path, checkpoint)
            continue

        # Stage 1 — presence
        try:
            present = vlm.has_boulders(image)
        except Exception as exc:
            log.warning("Stage-1 inference failed for %s: %s", img_path.name, exc)
            present = False

        if not present:
            dest_dir  = dirs["rejected"]
            stage2_out = "—"
            bucket     = "rejected"
        else:
            # Stage 2 — density
            try:
                stage2_out = vlm.density(image)
            except Exception as exc:
                log.warning("Stage-2 inference failed for %s: %s", img_path.name, exc)
                stage2_out = "sparse"   # conservative fallback
            bucket    = stage2_out
            dest_dir  = dirs[bucket]

        dest_path = dest_dir / img_path.name

        # Handle filename collisions
        if dest_path.exists():
            stem   = img_path.stem
            suffix = img_path.suffix
            dest_path = dest_dir / f"{stem}_{idx}{suffix}"

        # Copy file
        try:
            shutil.copy2(img_path, dest_path)
        except Exception as exc:
            log.error("Failed to copy %s to %s: %s", img_path.name, dest_path, exc)
            continue

        counters[bucket] += 1

        # Record in checkpoint
        checkpoint["processed"][img_path.name] = {
            "stage1": "yes" if present else "no",
            "stage2": stage2_out,
            "dest": str(dest_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        csv_rows.append(
            {
                "file":   str(img_path),
                "stage1": "yes" if present else "no",
                "stage2": stage2_out,
                "dest":   str(dest_path),
            }
        )

        # Save checkpoint after each image
        checkpoint["stats"]["total_processed"] = len(checkpoint["processed"])
        save_checkpoint(checkpoint_path, checkpoint)

        # Progress logging
        elapsed = time.time() - t0
        processed_this_run = idx - already_processed
        rate = processed_this_run / elapsed if elapsed > 0 else 0
        remaining = len(images_to_process) - processed_this_run
        eta = remaining / rate if rate > 0 else 0
        log.info(
            "  → %s | stage1=%s | stage2=%s | ETA %.0fs",
            bucket,
            "yes" if present else "no",
            stage2_out,
            eta,
        )

    # Append new results to CSV
    csv_mode = 'a' if log_csv.exists() else 'w'
    write_header = not log_csv.exists()

    with open(log_csv, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "stage1", "stage2", "dest"])
        if write_header:
            writer.writeheader()
        writer.writerows(csv_rows)

    # Log summary
    total_elapsed = time.time() - t0
    log.info("─" * 60)
    log.info("Done in %.1f s (%.2f img/s)", total_elapsed,
             len(images_to_process) / total_elapsed if total_elapsed > 0 else 0)
    log.info("Session processed: %d images", len(images_to_process))
    log.info("Total processed: %d / %d images", len(checkpoint["processed"]), total_images)
    log.info("  dense     : %d", counters["dense"])
    log.info("  mediocre  : %d", counters["mediocre"])
    log.info("  sparse    : %d", counters["sparse"])
    log.info("  rejected  : %d", counters["rejected"])
    log.info("  CSV log   : %s", log_csv)
    log.info("  Checkpoint: %s", checkpoint_path)



# Parser for CLI arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage VLM boulder filter for MSL Mastcam / Navcam images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root directory containing raw rover images (searched recursively).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination root; dense/, mediocre/, sparse/, rejected/ will be created.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-35B-A3B",
        help="HuggingFace model identifier for the Qwen VLM checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device: 'cuda', 'cpu', or 'mps'.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["jpg", "jpeg", "png"],
        help="Image file extensions to process.",
    )
    parser.add_argument(
        "--log_csv",
        default="results.csv",
        help="Filename (within output_dir) for the per-image audit CSV.",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing checkpoint and reprocess all images. "
             "Previous checkpoint will be backed up to .checkpoint.json.old",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
