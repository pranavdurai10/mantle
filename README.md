# MANTLE

## Developers

**Pranav Durai**
Stanford Center for Innovation in In Vivo Imaging, Stanford University School of Medicine, Stanford, CA 94305

**Dr. Gary B. Doran**
Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA 91109

## Abstract
Planetary surface exploration missions rely increasingly on autonomous robotic platforms capable of interpreting complex terrain to ensure safe navigation, enable targeted science, and enhance operational efficiency, as demonstrated across past Mars missions from Viking through Perseverance. Among the key perception capabilities, landform classification provides contextual information for landing site selection and scientific analysis, while boulder segmentation supports hazard assessment and path planning. This paper introduces MANTLE, a unified transformer architecture for Martian landform understanding. The model uses a shared DINOv2 backbone for high-level feature extraction with task-specific heads: a classification head for large-scale landform classification, and a segmentation head for pixel-wise boulder localization, each trained on newly curated datasets built respectively from HiRISE orbital imagery and MSL surface-level imagery. The classification head achieved an accuracy of 95.51\% across seven Martian terrain classes, while the segmentation head achieved a validation IoU of 0.753 and showed strong cross-SOL generalization on a held-out test set from previously unseen rover traverses. A key advantage of Mantle is its modular, extensible design, formalized here as the Modular Uplink Principle: only a shared, frozen backbone needs to remain onboard, while new perception capabilities are trained on Earth as lightweight task-specific heads and uplinked without retraining the full model. This work demonstrates two such high-impact capabilities, terrain classification and boulder segmentation, as an initial realization of a framework built to support many more over a mission's lifetime. With this foundation, future explorers need not arrive on Mars fully formed, but can continue to learn, adapt, and grow more capable with every uplink.

## Publication Status

This work is currently under review for publication. Datasets and model weights would be made available here shortly. 

## Installation
```bash
pip install -r requirements.txt
```

## Project Layout
```
mantle/                             # repo root — run everything from here
├── README.md
├── requirements.txt
└── mantle/                         # the importable package
    ├── __init__.py
    ├── main.py                         # single entry point — see below
    ├── configs.py
    ├── model.py
    ├── feature_extractor.py
    ├── train_weighted.py
    ├── train_classification.py
    ├── inference_lightweight.py
    ├── inference_classification.py
    ├── utils.py
    └── data_pipeline/                  # standalone data-prep tools (run directly, not via main.py)
        ├── extract_hirise_cutouts.py   # full-res HiRISE cutout generator
        ├── parse_unique_hirise_files.py
        ├── download_mastcam.py
        ├── fetch_pds.py
        ├── msl_vlm_filter.py
        ├── auto_SAM2_mask_generator.py
        ├── boulder_mask_explorator.py
        └── drivers/                    # annotation/trace data consumed by the scripts above
```

## Dataset Structure

### High-Resolution MSL Boulder Dataset

```
msl_boulder_dataset/
├── train/
│   └── boulders/
│       ├── images/
│       │   ├── image_001.PNG
│       │   └── ...
│       └── masks/
│           ├── image_001.PNG
│           └── ...
└── val/
    └── boulders/
        ├── images/
        └── masks/
```

- **Image/mask format**: `.PNG`/`.png` (grayscale binary masks, 0 = background, 255 = boulder)
- **Naming convention**: image and mask share the same filename
- Images are resized to 784×784 (56×56 DINOv2 patch grid) — see `configs.py`

### High-Resolution HiRISE Landform Dataset

```
terrain-classification-dataset/
├── train/
│   ├── crater/*.jpg
│   ├── dark_dune/*.jpg
│   ├── slope_streak/*.jpg
│   ├── bright_dune/*.jpg
│   ├── impact_ejecta/*.jpg
│   ├── swiss_cheese/*.jpg
│   └── spider/*.jpg
└── test/
    └── (same 7 class subfolders)
```

`train-classification` automatically splits `train/` into 80% train / 20% validation.

## Quick Start

All five capabilities run as a module from the **repo root** (`mantle/`), via `python -m mantle.main --mode <mode>`:

A. Boulder Segmentation Capability
```bash
# Extract and cache features for boulder segmentation
python -m mantle.main --mode extract --data-dir msl_boulder_dataset

# Train the segmentation head on cached features
python -m mantle.main --mode train --head-type convolutional --epochs 50

# Run segmentation inference + visualization
python -m mantle.main --mode inference --split val --visualize
```

B. Terrain Classification Capability
```bash
# Train the terrain classification head
python -m mantle.main --mode train-classification --classification-epochs 100

# Run terrain classification inference
python -m mantle.main --mode infer-classification

```

NOTE: Run with `-m` from the repo root, not `python mantle/main.py` as the package uses relative imports internally, which only resolve correctly when Python loads it as `mantle.main` rather than as a standalone script.

## Command Line Interface

```bash
python -m mantle.main --mode {extract,train,inference,train-classification,infer-classification} [OPTIONS]
```

### Shared / Extraction & Training Args

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data-dir` | str | `msl_boulder_dataset` | Root boulder dataset directory |
| `--features-dir` | str | `cached_features_vitb_784` | Cached DINOv2 feature directory |
| `--head-type` | str | `convolutional` | Segmentation head: `convolutional` |
| `--dino-model` | str | `dinov2_vits14` (configs.py) | DINOv2 backbone variant |
| `--epochs` | int | 50 (configs.py) | Segmentation training epochs |
| `--batch-size` | int | 16 (configs.py) | Segmentation training batch size |
| `--learning-rate` | float | 1e-4 (configs.py) | Segmentation training learning rate |
| `--pos-weight` | float | 1.2 | BCE pos_weight for boulder loss |
| `--extraction-batch-size` | int | 32 | Batch size used during feature extraction |
| `--splits` | list | `train val` | Which splits to extract features for |
| `--no-h5` | flag | False | Use pickle instead of HDF5 for cached features |

### Inference: Args for Segmentation (`--mode inference`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | auto-detect | Path to segmentation checkpoint |
| `--split` | str | `val` | Which split to evaluate (`train` or `val`) |
| `--threshold` | float | 0.5 | Binarization threshold |
| `--inference-batch-size` | int | 32 | Inference batch size |
| `--visualize` | flag | False | Generate visualization grids |

### INference: Args for Classification (`--mode train-classification` / `infer-classification`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--classification-data-dir` | str | `terrain-classification-dataset` | Root dir with `train/`+`test/` class subfolders |
| `--classification-test-dir` | str | `terrain-classification-dataset/test` | Test set directory (inference only) |
| `--classification-checkpoint` | str | `checkpoints/best_terrain_classification_model.pth` | Checkpoint path (inference only) |
| `--classification-batch-size` | int | 16 | Batch size |
| `--classification-epochs` | int | 100 | Training epochs |
| `--classification-lr` | float | 1e-6 | Learning rate |
| `--classification-image-size` | int | 224 | Input resolution |
| `--class-names` | list | the 7 MSL terrain classes | Override class names |

## Configurations

Default hyperparameters for boulder segmentation live in `configs.py`:

```python
class Config:
    DINOV2_MODEL = "dinov2_vitb14"   # dinov2_vits14 / vitb14 / vitl14 / vitg14
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 50
    LEARNING_RATE = 1e-4
    IMAGE_SIZE   = (784, 784)        # 56x56 DINOv2 patch grid
    LOSS_FUNCTION = "bce_dice"
    POS_WEIGHT   = 1.2
    OPTIMIZER    = "adamw"
    SCHEDULER    = "cosine"
    EARLY_STOPPING_PATIENCE = 10
```

## Output Structure

```
checkpoints/
├── best_model.pth                          # Best boulder segmentation model (by val IoU)
└── best_terrain_classification_model.pth   # Best terrain classification model (by val accuracy)

inference_results/
├── inference_grid_threshold_0.50.png       # Segmentation comparison grid
├── <sample>_boxes.png                      # Per-sample bounding-box overlays
└── classification_results.txt              # Per-image terrain classification predictions

mantle.log                              # Pipeline log
```

## Performance Metrics

**Boulder segmentation** (`--mode train` / `inference`):
- IoU, Accuracy, Precision, Recall, F1 — pixel-wise
- Instance-level TP/FP/FN via connected-component matching

**Terrain classification** (`--mode train-classification` / `infer-classification`):
- Overall and per-class Accuracy, Precision, Recall, F1
- Confusion matrix and most-confused class pairs
