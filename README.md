# MANTLE
<p align="center">
  <a href="https://arxiv.org/abs/2608.28724">
    <img src="https://img.shields.io/badge/arXiv-2608.28724-b31b1b.svg"/>
  </a>
  <a href="https://pranavdurai10.github.io/mantle-paper/">
    <img src="https://img.shields.io/badge/Project%20Page-MANTLE-blue.svg"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/DOI-coming%20soon-lightgrey"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg"/>
  <img src="https://img.shields.io/badge/PyTorch-2.8-orange.svg"/>
  <a href="https://github.com/pranavdurai10/mantle">
    <img src="https://hits.sh/github.com/pranavdurai10/mantle.svg?label=Views&color=brightgreen"/>
  </a>
</p>

---

## Developers
**Pranav Durai** - Stanford Center for Innovation in In Vivo Imaging, Stanford University School of Medicine, Stanford, CA 94305

**Dr. Gary Doran** - Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA 91109

---

## Publication Status
This work is available as a preprint on arXiv and is currently under peer review.

**arXiv**: https://arxiv.org/abs/2608.28724

### Datasets

| Dataset | DOI |
|---------|-----|
| HiRISE Landform Classification Dataset | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21300384.svg)](https://doi.org/10.5281/zenodo.21300384) |
| MSL Boulder Segmentation Dataset | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21313774.svg)](https://doi.org/10.5281/zenodo.21313774) |

> Model weights will be made available upon acceptance.

## Installation

```bash
# Clone the repository
git clone https://github.com/pranavdurai10/mantle.git
cd mantle

# Install dependencies
pip install -r requirements.txt
```

> A pip-installable PyPI package is coming soon.

## Quick Start

All capabilities run as a module from the **repo root** (`mantle/`), via `python -m mantle.main --mode <mode>`:

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

> Run with `-m` from the repo root, not `python mantle/main.py` as the package uses relative imports internally, which only resolve correctly when Python loads it as `mantle.main` rather than as a standalone script.

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

### Inference: Args for Classification (`--mode train-classification` / `infer-classification`)

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

---

## License and Copyright

This repository is licensed under the [MIT License](LICENSE).

Copyright © 2026 Pranav Durai and Gary Doran.

Raw HiRISE imagery courtesy of NASA/JPL-Caltech/University of Arizona, available through the University of Arizona HiRISE platform and the NASA Planetary Data System (PDS) under NASA's open data policy. MSL Mastcam imagery courtesy of NASA/JPL-Caltech/MSSS.
