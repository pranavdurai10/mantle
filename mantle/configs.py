# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Configuration file for Mantle pipeline
#               – Centralized hyperparameters and training settings
#               – Dataset paths and model configurations
#               – DINOv2-based semantic segmentation parameters
# =============================================================================
import torch
from pathlib import Path


class Config:
    """
    Configuration class for training parameters.
    """
    # Dataset paths — msl_boulder_dataset structure
    DATA_ROOT    = Path("msl_boulder_dataset")
    TRAIN_IMAGES = DATA_ROOT / "train" / "boulders" / "images"
    TRAIN_MASKS  = DATA_ROOT / "train" / "boulders" / "masks"
    VAL_IMAGES   = DATA_ROOT / "val"   / "boulders" / "images"
    VAL_MASKS    = DATA_ROOT / "val"   / "boulders" / "masks"

    # Model settings
    DINOV2_MODEL = "dinov2_vits14"              # ViT-B/14 — 768-dim features (upgraded from vits14)
    FEATURE_DIM  = 768                          # Feature dimension for vitb14
    NUM_CLASSES  = 1                            # Binary segmentation
    HEAD_TYPE    = "convolutional"              # Options: convolutional, bn

    # Training hyperparameters
    BATCH_SIZE      = 16
    NUM_EPOCHS      = 50
    LEARNING_RATE   = 1e-4
    WEIGHT_DECAY    = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image settings
    IMAGE_SIZE  = (784, 784)                    # 784 = 14 × 56 → 56×56 patch grid (upgraded from 518)
    PATCH_SIZE  = 14                            # DINOv2 patch size

    # Augmentation settings
    AUGMENTATION_PROB    = 0.5
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB   = 0.0

    # Loss settings
    # pos_weight = 1.2 — dataset is near-balanced (45% boulder / 55% background)
    # computed via compute_pos_weight.py on 6,232 SAM2-annotated Mastcam masks
    POS_WEIGHT    = 1.2
    LOSS_FUNCTION = "bce_dice"                  # Options: "bce", "dice", "bce_dice"
    BCE_WEIGHT    = 0.5                         # Weight for BCE in combined loss

    # Optimizer and scheduler
    OPTIMIZER = "adamw"                         # Options: "adam", "adamw", "sgd"
    SCHEDULER = "cosine"                        # Options: "cosine", "step", "none"

    # Logging and checkpoints
    LOG_INTERVAL   = 1
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    INFERENCE_RESULTS_DIR = Path("inference_results")
    INFERENCE_RESULTS_DIR.mkdir(exist_ok=True)

    # Validation settings
    VAL_INTERVAL            = 1                 # Validate every N epochs
    SAVE_BEST_ONLY          = True
    EARLY_STOPPING_PATIENCE = 10

    # Inference settings
    INFERENCE_BATCH_SIZE = 16
    OVERLAY_ALPHA        = 0.5                  # Transparency for mask overlay
    MASK_COLOR           = (255, 0, 0)          # Red color for mask overlay


config = Config()


def print_banner():
    banner = r""" 
     __  __   _   _  _ _____ _    ___ 
    |  \/  | /_\ | \| |_   _| |  | __|
    | |\/| |/ _ \| .` | | | | |__| _| 
    |_|  |_/_/ \_\_|\_| |_| |____|___|

    - Multi-task Adaptive Network for Terrain and Landform Extraction
    - Package version: v1.0.0                                                     
    """
    print(banner)
