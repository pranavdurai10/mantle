# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Headless SAM2 batch mask generator for HPC clusters
#               – Streams image → SAM2 → RLE JSON → disk, zero RAM accumulation
#               – SAM2AutomaticMaskGenerator instantiated once per run
# =============================================================================

"""
JSON output format (one file per image, same stem as source):
{
    "source":          "NLB_12345.png",
    "image_shape":     [H, W],
    "clahe_used":      true,
    "params":          { ... },
    "masks": [
        {
            "id":              0,
            "area":            4821,
            "predicted_iou":   0.91,
            "stability_score": 0.88,
            "segmentation":    {"counts": "...", "size": [H, W]},   ← COCO RLE
            "visible":         true
        },
        ...
    ]
}

Usage
-----
    python auto_SAM2_mask_generator.py --input_dir /path/to/images --json_dir /path/to/json_out

SLURM example
-------------
    #!/bin/bash
    #SBATCH --job-name=mantle_sam2
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=32G
    #SBATCH --time=12:00:00
    #SBATCH --output=logs/sam2_%j.log

    source activate torch_env
    python auto_SAM2_mask_generator.py \\
        --input_dir /scratch/data/msl_filtered \\
        --json_dir  /scratch/data/msl_json
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from huggingface_hub import hf_hub_download


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAM2 PARAMETERS — edit these before submitting to HPC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POINTS_PER_SIDE         = 32
PRED_IOU_THRESH         = 0.30
STABILITY_SCORE_THRESH  = 0.85
MIN_MASK_REGION_AREA    = 0.0    # absolute pixel area
MAX_MASK_AREA_PCT       = 0.30   # fraction of total image area (0.0 – 1.0)
ENABLE_CLAHE            = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def encode_mask(binary_mask: np.ndarray) -> dict:
    """Encode boolean/uint8 HxW mask to COCO RLE (JSON-serialisable)."""
    m = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def load_sam2(device: str) -> object:
    log.info("Loading SAM2 checkpoint (sam2_hiera_large) …")
    try:
        ckpt = hf_hub_download(
            repo_id="facebook/sam2-hiera-large",
            filename="sam2_hiera_large.pt",
        )
    except Exception:
        ckpt = "sam2_hiera_large.pt"
    model = build_sam2("sam2_hiera_l.yaml", ckpt, device=device)
    log.info("SAM2 ready on device: %s", device)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Headless SAM2 batch inference for Mantle HPC pipeline."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing pre-processed PNG images.",
    )
    parser.add_argument(
        "--json_dir", required=True,
        help="Directory to write per-image JSON annotation files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    json_dir  = Path(args.json_dir).resolve()

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    json_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True       # faster conv on fixed-size inputs
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        log.warning("No GPU detected — inference will be very slow on CPU.")
    log.info("Device: %s", device)

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    total = len(images)
    if total == 0:
        log.error("No images found in %s", input_dir)
        sys.exit(1)
    log.info("Found %d images.", total)

    params = {
        "points_per_side":        POINTS_PER_SIDE,
        "pred_iou_thresh":        PRED_IOU_THRESH,
        "stability_score_thresh": STABILITY_SCORE_THRESH,
        "min_mask_region_area":   MIN_MASK_REGION_AREA,
        "max_mask_area_pct":      MAX_MASK_AREA_PCT,
        "clahe_enabled":          ENABLE_CLAHE,
    }
    log.info("SAM2 params: %s", json.dumps(params, indent=2))

    set_seed(42)
    model = load_sam2(device)

    if device == "cuda" and hasattr(torch, "compile"):
        log.info("Applying torch.compile to SAM2 image encoder …")
        model.image_encoder = torch.compile(
            model.image_encoder,
            mode="reduce-overhead",   # optimised for repeated same-shape inputs
            fullgraph=False,           # SAM2 has control flow; partial compile is safe
        )
        log.info("torch.compile applied.")

    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        min_mask_region_area=MIN_MASK_REGION_AREA,
    )

    processed = skipped = 0

    for fname in tqdm(images, unit="img", dynamic_ncols=True, desc="SAM2 Inference"):
        img_path = input_dir / fname
        out_path = json_dir / (Path(fname).stem + ".json")

        if out_path.exists():
            processed += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            log.warning("Could not read image, skipping: %s", fname)
            skipped += 1
            continue

        h, w  = bgr.shape[:2]
        max_area = MAX_MASK_AREA_PCT * h * w

        rgb = apply_clahe(bgr) if ENABLE_CLAHE else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        try:
            with torch.inference_mode():
                raw_masks = generator.generate(rgb)
        except Exception as e:
            log.warning("Inference failed for %s: %s", fname, e)
            skipped += 1
            continue

        annotations = []
        for m in raw_masks:
            if m['area'] <= max_area:
                annotations.append({
                    "id":               len(annotations),
                    "area":             int(m['area']),
                    "predicted_iou":    float(m['predicted_iou']),
                    "stability_score":  float(m['stability_score']),
                    "segmentation":     encode_mask(m['segmentation']),
                    "visible":          True,
                })

        record = {
            "source":       fname,
            "image_shape":  [h, w],
            "clahe_used":   ENABLE_CLAHE,
            "params":       params,
            "masks":        annotations,
        }

        with open(out_path, 'w') as f:
            json.dump(record, f, separators=(',', ':'))

        processed += 1

    log.info("=" * 60)
    log.info("DONE")
    log.info("  Total images : %d", total)
    log.info("  Processed    : %d", processed)
    log.info("  Skipped      : %d", skipped)
    log.info("  JSON output  : %s", json_dir)


if __name__ == "__main__":
    main()
