# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Main entry point for the Mantle dual-task pipeline
#               – Feature extraction from msl_boulder_dataset
#               – Boulder segmentation training (BCEDice loss) and inference
#               – Terrain classification training and inference
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path

import torch

from .configs import config, print_banner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mantle.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_environment() -> None:
    """
    Logs training system and PyTorch enviroment information. 

    Returns:
        None
    """
    logger.info("=" * 55)
    logger.info("System Environment")
    logger.info("=" * 55)
    logger.info(f"PyTorch version : {torch.__version__}")
    logger.info(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version    : {torch.version.cuda}")
        logger.info(f"GPU             : {torch.cuda.get_device_name(0)}")
    logger.info(f"Device          : {config.DEVICE}")


def check_data_directories(data_dir: str = None) -> None:
    """
    Validates the dataset directory structure.
    Expects - msl_boulder_dataset/[train|val]/boulders/[images|masks]/

    Args:
        data_dir (str, optional): Root dataset directory path. 

    Returns:
        None

    Raises:
        FileNotFoundError: If any required dataset directory is missing.
    """
    if data_dir:
        root = Path(data_dir)
        config.DATA_ROOT    = root
        config.TRAIN_IMAGES = root / "train" / "boulders" / "images"
        config.TRAIN_MASKS  = root / "train" / "boulders" / "masks"
        config.VAL_IMAGES   = root / "val"   / "boulders" / "images"
        config.VAL_MASKS    = root / "val"   / "boulders" / "masks"

    dirs_to_check = [
        config.TRAIN_IMAGES,
        config.TRAIN_MASKS,
        config.VAL_IMAGES,
        config.VAL_MASKS,
    ]

    missing = [d for d in dirs_to_check if not d.exists()]
    if missing:
        logger.error("Missing dataset directories:")
        for d in missing:
            logger.error(f"  Missing: {d}")
        raise FileNotFoundError(
            f"Dataset structure not found under {config.DATA_ROOT}.\n"
            "Expected: msl_boulder_dataset/[train|val]/boulders/[images|masks]/"
        )

    logger.info(f"Dataset root    : {config.DATA_ROOT}")
    logger.info(f"Train images    : {config.TRAIN_IMAGES}")
    logger.info(f"Train masks     : {config.TRAIN_MASKS}")
    logger.info(f"Val images      : {config.VAL_IMAGES}")
    logger.info(f"Val masks       : {config.VAL_MASKS}")


def run_feature_extraction(args) -> None:
    """
    Runs DINOv2 feature extraction for specified dataset splits. 

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            dataset paths, model configuration, output directory, and batch size
            settings.

    Returns:    
        None
    """
    from .feature_extractor import DINOv2FeatureExtractor

    logger.info("=" * 55)
    logger.info("Mantle — Feature Extraction")
    logger.info("=" * 55)

    check_data_directories(args.data_dir)

    extractor = DINOv2FeatureExtractor(
        model_name=args.dino_model or config.DINOV2_MODEL,
        device=config.DEVICE
    )

    splits     = args.splits or ["train", "val"]
    output_dir = Path(args.features_dir)

    logger.info(f"Splits      : {splits}")
    logger.info(f"Batch size  : {args.extraction_batch_size}")
    logger.info(f"Output dir  : {output_dir}")

    for split in splits:
        logger.info(f"\nProcessing '{split}' split...")
        extractor.extract_and_save_features(
            data_split=split,
            batch_size=args.extraction_batch_size,
            output_dir=output_dir,
            use_h5=not args.no_h5,
        )

    logger.info("=" * 55)
    logger.info(f"Feature extraction complete → {output_dir}")
    logger.info("=" * 55)


def run_training(args) -> None:
    """
    Runs training using cached DINOv2 features and segmentation masks.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing 
            dataset paths, training configuration, model settings, and feature
            cache locations.

    Returns:
        None
    """
    from .train_weighted import Trainer
    from .feature_extractor import CachedFeaturesDataset
    from torch.utils.data import DataLoader
    import platform

    logger.info("=" * 55)
    logger.info("Mantle — Training")
    logger.info("=" * 55)

    check_data_directories(args.data_dir)

    # Override config from CLI args
    if args.epochs:        config.NUM_EPOCHS     = args.epochs
    if args.batch_size:    config.BATCH_SIZE     = args.batch_size
    if args.learning_rate: config.LEARNING_RATE  = args.learning_rate

    logger.info(f"Dataset     : {args.data_dir}")
    logger.info(f"Head type   : {args.head_type}")
    logger.info(f"Epochs      : {config.NUM_EPOCHS}")
    logger.info(f"Batch size  : {config.BATCH_SIZE}")
    logger.info(f"LR          : {config.LEARNING_RATE}")
    logger.info(f"pos_weight  : {args.pos_weight}")

    # Check for cached features
    features_dir   = Path(args.features_dir)
    train_features = features_dir / "train_features.h5"
    val_features   = features_dir / "val_features.h5"

    if not train_features.exists():
        train_features = features_dir / "train_features.pkl"
        val_features   = features_dir / "val_features.pkl"

    if not train_features.exists():
        logger.error(f"Cached features not found at {features_dir}")
        logger.error("Run feature extraction first:")
        logger.error(f"  python -m mantle.main --mode extract --data-dir {args.data_dir}")
        sys.exit(1)

    # Datasets
    logger.info("Loading cached features...")
    train_dataset = CachedFeaturesDataset(
        features_file=train_features,
        masks_dir=config.TRAIN_MASKS,
        augment=True
    )
    val_dataset = CachedFeaturesDataset(
        features_file=val_features,
        masks_dir=config.VAL_MASKS,
        augment=False
    )

    num_workers  = 0 if platform.system() == 'Windows' else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Build model
    if args.head_type == 'convolutional':
        from .feature_extractor import CachedFeaturesSegModel

        logger.info(
            "Model: ConvolutionalHead"
            "(SE+CBAM+dilated+double-conv+deep-supervision)"
        )
        model = CachedFeaturesSegModel(
            feature_dim=config.FEATURE_DIM,
            num_classes=1,
        ).to(config.DEVICE)

    elif args.head_type == 'aspp':
        from .feature_extractor import ASPPSegModel

        logger.info("Building model: ASPPHead (multi-scale atrous pyramid)")
        model = ASPPSegModel(
            feature_dim=config.FEATURE_DIM,
            num_classes=1
        ).to(config.DEVICE)

    else:
        from .feature_extractor import LightweightSegmentationModel
        logger.info("Building model: BNHead (lightweight)")
        model = LightweightSegmentationModel(
            feature_dim=config.FEATURE_DIM,
            num_classes=1,
            head_type='bn'
        ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        pos_weight=args.pos_weight,
        learning_rate=config.LEARNING_RATE,
        num_epochs=config.NUM_EPOCHS,
    )
    trainer.train()

    logger.info("Training complete.")
    logger.info("Best model saved to: checkpoints/best_model.pth")


def run_inference(args) -> None:
    """
    Runs boulder segmentation inference using cached DINOv2 features.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            checkpoint, features directory, head type, split, threshold,
            and visualization settings.

    Returns:
        None
    """
    from .inference_lightweight import run_lightweight_inference

    logger.info("=" * 55)
    logger.info("Mantle — Segmentation Inference")
    logger.info("=" * 55)

    metrics, counts = run_lightweight_inference(
        model_path=Path(args.checkpoint) if args.checkpoint else None,
        feature_path=Path(args.features_dir),
        visualize=args.visualize,
        optimal_threshold=args.threshold,
        batch_size=args.inference_batch_size,
        split=args.split,
        head_type=args.head_type,
    )

    if not metrics:
        logger.error("Inference failed — see errors above.")
        sys.exit(1)

    logger.info("=" * 55)
    logger.info(f"IoU: {metrics['iou']:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    if counts:
        logger.info(f"Instance counts — TP: {counts['tp']} | FP: {counts['fp']} | FN: {counts['fn']}")
    logger.info("=" * 55)


def run_train_classification(args) -> None:
    """
    Runs terrain classification training.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            classification dataset path, batch size, epochs, learning rate,
            and image size.

    Returns:
        None
    """
    from .train_classification import train_classification_model

    logger.info("=" * 55)
    logger.info("Mantle — Terrain Classification Training")
    logger.info("=" * 55)

    train_classification_model(
        data_dir=args.classification_data_dir,
        class_names=args.class_names,
        batch_size=args.classification_batch_size,
        num_epochs=args.classification_epochs,
        learning_rate=args.classification_lr,
        image_size=args.classification_image_size,
    )


def run_infer_classification(args) -> None:
    """
    Runs terrain classification inference.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            checkpoint path, test dataset path, batch size, and image size.

    Returns:
        None
    """
    from .inference_classification import run_classification_inference

    logger.info("=" * 55)
    logger.info("Mantle — Terrain Classification Inference")
    logger.info("=" * 55)

    run_classification_inference(
        model_path=args.classification_checkpoint,
        test_data_dir=args.classification_test_dir,
        class_names=args.class_names,
        batch_size=args.classification_batch_size,
        image_size=args.classification_image_size,
    )


def main():
    """
    Parses command-line arguments and runs the selected pipeline mode.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Mantle — Dual-Task Segmentation & Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        End-to-end Segmentation and Classification Pipeline
        for HiRISE and MSL Imagery.
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['extract', 'train', 'inference', 'train-classification', 'infer-classification'],
        required=True,
        help=(
            'Pipeline mode: extract (DINOv2 features), train (segmentation), '
            'inference (segmentation), train-classification, or infer-classification'
        )
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='msl_boulder_dataset',
        help='Root dataset directory (default: msl_boulder_dataset)'
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default='cached_features_vitb_784',
        help='Directory for cached DINOv2 features (default: cached_features_vitb_784 for ViT-B 784px)'
    )
    parser.add_argument(
        '--head-type',
        type=str,
        choices=['convolutional', 'bn', 'aspp'],
        default='convolutional',
        help='Segmentation head type: convolutional, bn, or aspp (default: convolutional)'
    )
    parser.add_argument(
        '--dino-model',
        type=str,
        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
        default=None,
        help='DINOv2 model variant (default: dinov2_vits14 from configs.py)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: 50 from configs.py)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size (default: 16 from configs.py)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: 1e-4 from configs.py)'
    )
    parser.add_argument(
        '--pos-weight',
        type=float,
        default=1.2,
        help='BCE pos_weight (default: 1.2 — calibrated for MSL boulder dataset)'
    )
    parser.add_argument(
        '--extraction-batch-size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 32)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['train', 'val'],
        default=None,
        help='Which splits to extract features for (default: train val)'
    )
    parser.add_argument(
        '--no-h5',
        action='store_true',
        help='Use pickle format instead of HDF5 for cached features'
    )

    # --- Segmentation inference (--mode inference) ---
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to segmentation checkpoint (default: auto-detect in checkpoints/)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val'],
        default='val',
        help='Which split to run segmentation inference on (default: val)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Binarization threshold for segmentation inference (default: 0.5)'
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=32,
        help='Batch size for segmentation inference (default: 32)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization grids during inference'
    )

    # --- Terrain classification (--mode train-classification / infer-classification) ---
    parser.add_argument(
        '--classification-data-dir',
        type=str,
        default='terrain-classification-dataset',
        help='Root dir with train/ and test/ class subfolders (default: terrain-classification-dataset)'
    )
    parser.add_argument(
        '--classification-test-dir',
        type=str,
        default='terrain-classification-dataset/test',
        help='Test dir with one subfolder of .jpg images per class (default: terrain-classification-dataset/test)'
    )
    parser.add_argument(
        '--classification-checkpoint',
        type=str,
        default='checkpoints/best_terrain_classification_model.pth',
        help='Path to classification checkpoint for inference'
    )
    parser.add_argument(
        '--classification-batch-size',
        type=int,
        default=16,
        help='Batch size for terrain classification (default: 16)'
    )
    parser.add_argument(
        '--classification-epochs',
        type=int,
        default=100,
        help='Number of terrain classification training epochs (default: 100)'
    )
    parser.add_argument(
        '--classification-lr',
        type=float,
        default=1e-6,
        help='Learning rate for terrain classification training (default: 1e-6)'
    )
    parser.add_argument(
        '--classification-image-size',
        type=int,
        default=224,
        help='Input image resolution for terrain classification (default: 224)'
    )
    parser.add_argument(
        '--class-names',
        nargs='+',
        default=None,
        help='Terrain class names (default: the 7 MSL terrain classes)'
    )

    args = parser.parse_args()

    try:
        check_environment()

        if args.mode == 'extract':
            run_feature_extraction(args)
        elif args.mode == 'train':
            run_training(args)
        elif args.mode == 'inference':
            run_inference(args)
        elif args.mode == 'train-classification':
            run_train_classification(args)
        elif args.mode == 'infer-classification':
            run_infer_classification(args)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    print_banner()
    main()
