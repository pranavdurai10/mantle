# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Inference pipeline for lightweight cached features models
#               – Batch processing and evaluation on test datasets
#               – Visualization grid generation (GT vs predictions)
#               – Export segmentation masks with overlay capabilities
#               – Supports optimal threshold application
# =============================================================================

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .configs import config
from .feature_extractor import (
    CachedFeaturesDataset,
    CachedFeaturesSegModel,
    ASPPSegModel,
    LightweightSegmentationModel,
)
from .utils import (
    calculate_metrics, overlay_mask_on_image, instance_metrics,
    draw_boxes_on_image,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightInferencePipeline:
    """
    Inference pipeline for lightweight segmentation models using cached features.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_path: Optional[Path] = None,
        optimal_threshold: float = 0.5
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Lightweight model to use for inference
            device: Device to run inference on
            checkpoint_path: Path to model checkpoint
            optimal_threshold: Optimal threshold for binarization (default 0.5, but 0.9 is better for this task)
        """
        self.model = model
        self.device = device
        self.optimal_threshold = optimal_threshold
        
        # Load checkpoint if provided
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.checkpoint_metrics = checkpoint.get('metrics', {})
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            if self.checkpoint_metrics:
                logger.info(f"Checkpoint metrics: IoU={self.checkpoint_metrics.get('iou', 0):.4f}")
        
        self.model.eval()
    
    def process_batch(
        self,
        data_loader: DataLoader,
        save_results: bool = True,
        visualize: bool = False,
        split: str = "val"
    ) -> Dict[str, float]:
        """
        Process a batch of images using cached features.
        
        Args:
            data_loader: Data loader with test features
            save_results: Whether to save results to disk
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary of average metrics
        """
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'iou': []
        }
        all_counts = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
        }
        
        results = []
        logger.info(f"Starting batch inference with threshold={self.optimal_threshold:.2f}...")
        
        with torch.no_grad():
            for batch_idx, (features, masks) in enumerate(tqdm(data_loader, desc="Inference")):
                features = features.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                preds = torch.sigmoid(outputs)
                
                # Apply optimal threshold
                preds_binary = (preds > self.optimal_threshold).float()
                
                # Calculate metrics with optimal threshold
                batch_counts = instance_metrics(preds_binary, masks, threshold=0.5)
                for key in all_counts:
                    if key in batch_counts:
                        all_counts[key] += batch_counts[key]

                batch_metrics = calculate_metrics(preds_binary, masks, threshold=0.5)
                for key in all_metrics:
                    if key in batch_metrics:
                        all_metrics[key].append(batch_metrics[key])
                
                # Store results for visualization
                if save_results or visualize:
                    for i in range(features.shape[0]):
                        results.append({
                            'features': features[i].cpu(),
                            'mask': masks[i].cpu(),
                            'pred_raw': preds[i].cpu(),
                            'pred_binary': preds_binary[i].cpu()
                        })

        
        # Calculate average metrics
        avg_metrics = {
            key: np.mean(values) for key, values in all_metrics.items()
        }
        
        logger.info(
            f"Inference completed - IoU: {avg_metrics['iou']:.4f}, "
            f"Accuracy: {avg_metrics['accuracy']:.4f}, "
            f"Precision: {avg_metrics['precision']:.4f}, "
            f"Recall: {avg_metrics['recall']:.4f}"
        )
        
        if save_results or visualize:
            self._save_results(data_loader.dataset, results, visualize, split=split)
            self._save_individual_results(data_loader.dataset, results, split=split, expand=1.0)
        
        return avg_metrics, all_counts

    def _create_heatmap(self, pred_mask: np.ndarray, max_distance: float = 150) -> np.ndarray:
        """
        Create a global heatmap showing boulder danger zones across entire image.
        
        Args:
            pred_mask: Binary prediction mask
            max_distance: Maximum distance for heat spreading (in pixels)
            
        Returns:
            Heatmap array with values 0-1 covering entire image
        """
        # Ensure binary mask
        binary_mask = (pred_mask > 0.5).astype(np.float32)
        
        if np.sum(binary_mask) == 0:
            # No boulders detected, return cold map with minimal base heat
            return np.ones_like(binary_mask) * 0.1  # Slight blue tint for "safe"
        
        # Create distance transform from boulder pixels
        distance_from_boulder = distance_transform_edt(1 - binary_mask)

        # Create global heatmap that covers entire image
        heatmap = np.zeros_like(distance_from_boulder)
        
        # Define heat zones with smoother transitions
        # Zone 1: Boulder pixels (intensity = 1.0)
        boulder_mask = binary_mask > 0
        heatmap[boulder_mask] = 1.0
        
        # Zone 2: Very near field (0-10% of max_distance) - Critical danger
        critical_mask = (distance_from_boulder > 0) & (distance_from_boulder <= max_distance * 0.1)
        heatmap[critical_mask] = 0.95 - (distance_from_boulder[critical_mask] / (max_distance * 0.1)) * 0.05
        
        # Zone 3: Near field (10-30% of max_distance) - High danger
        near_mask = (distance_from_boulder > max_distance * 0.1) & (distance_from_boulder <= max_distance * 0.3)
        heatmap[near_mask] = 0.75 - ((distance_from_boulder[near_mask] - max_distance * 0.1) / 
                                      (max_distance * 0.2)) * 0.15
        
        # Zone 4: Mid field (30-60% of max_distance) - Medium danger  
        mid_mask = (distance_from_boulder > max_distance * 0.3) & (distance_from_boulder <= max_distance * 0.6)
        heatmap[mid_mask] = 0.45 - ((distance_from_boulder[mid_mask] - max_distance * 0.3) / 
                                     (max_distance * 0.3)) * 0.15
        
        # Zone 5: Far field (60-100% of max_distance) - Low danger
        far_mask = (distance_from_boulder > max_distance * 0.6) & (distance_from_boulder <= max_distance)
        heatmap[far_mask] = 0.25 - ((distance_from_boulder[far_mask] - max_distance * 0.6) / 
                                    (max_distance * 0.4)) * 0.10
        
        # Zone 6: Beyond max_distance - Very low danger (but not zero)
        beyond_mask = distance_from_boulder > max_distance
        # Use exponential decay for areas beyond max_distance
        decay_rate = 0.01  # Slow decay rate
        heatmap[beyond_mask] = 0.15 * np.exp(-decay_rate * (distance_from_boulder[beyond_mask] - max_distance))
        
        # Ensure minimum heat level across entire image (no completely cold areas)
        heatmap = np.maximum(heatmap, 0.05)
        
        # Apply stronger Gaussian smoothing for global continuity
        heatmap = gaussian_filter(heatmap, sigma=5)
        
        # Add subtle noise for visual interest in safe areas
        noise = np.random.normal(0, 0.01, heatmap.shape)
        heatmap = heatmap + noise
        
        # Clip values to [0, 1]
        heatmap = np.clip(heatmap, 0, 1)
        
        return heatmap
    
    def _apply_heatmap_overlay(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Apply global heatmap overlay on entire image.
        
        Args:
            image: Original image
            heatmap: Heatmap array (0-1)
            alpha: Transparency factor (reduced for global coverage)
            
        Returns:
            Image with heatmap overlay covering entire area
        """
        # Create custom colormap with more gradual transitions
        # Deep blue -> Cyan -> Green -> Yellow -> Orange -> Red
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000']
        n_bins = 256  # More bins for smoother gradients
        cmap = LinearSegmentedColormap.from_list('boulder_heat_global', colors, N=n_bins)
        
        # Apply colormap to entire heatmap
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Apply overlay to entire image (no masking)
        # Use variable alpha based on heat intensity for better visibility
        # Lower heat areas get lower alpha for subtler effect
        alpha_map = alpha * (0.3 + 0.7 * heatmap)  # Scale alpha from 0.3*alpha to 1.0*alpha
        alpha_map = np.expand_dims(alpha_map, axis=2)  # Add channel dimension
        
        # Blend with original image
        result = (alpha_map * heatmap_colored + (1 - alpha_map) * image).astype(np.uint8)
        
        return result
    
    # Export heatmap visualization results 
    def _save_results(
        self,
        dataset,
        results: list,
        visualize: bool = True,
        num_samples: int = 8,
        split: str = "val"
    ) -> None:
        """
        Save inference results with visualizations.
        
        Args:
            results: List of result dictionaries
            visualize: Whether to create visualization plots
            num_samples: Number of samples to visualize
            split: Which split we're working with
        """
        # Create output directory
        output_dir = config.INFERENCE_RESULTS_DIR
        output_dir.mkdir(exist_ok=True)
        
        if not visualize:
            logger.info("Skipping visualization as requested")
            return
        
        # Select samples to visualize
        num_samples = min(num_samples, len(results))
        if num_samples == 0:
            logger.warning("No results to visualize")
            return
            
        sample_indices = np.linspace(0, len(results) - 1, num_samples, dtype=int)
        
        # Setup paths for images based on split
        images_dir = config.TRAIN_IMAGES if split == "train" else config.VAL_IMAGES
        
        sample_names = dataset.sample_names
        
        # Create visualization grid with bounding boxes and heatmap
        fig, axes = plt.subplots(
            num_samples, 6,  # 6 columns: original, GT, pred@0.5, pred@optimal, bboxes, heatmap
            figsize=(24, 4 * num_samples)
        )
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        successful_plots = 0
        
        for idx, sample_idx in enumerate(sample_indices):
            try:
                result = results[sample_idx]
                
                # Get masks
                gt_mask = result['mask'].squeeze().numpy()
                pred_raw = result['pred_raw'].squeeze().numpy()
                pred_binary = result['pred_binary'].squeeze().numpy()
                pred_standard = (pred_raw > 0.5).astype(float)
                
                # Try to load the actual image
                image_np = None
                if sample_idx < len(sample_names) and images_dir.exists():
                    sample_name = sample_names[sample_idx]
                    image_path = images_dir / f"{sample_name}.png"
                    
                    if image_path.exists():
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize((518, 518))
                        image_np = np.array(image)
                        logger.debug(f"Loaded image: {image_path}")
                    else:
                        logger.warning(f"Image not found: {image_path}")

                # If image not loaded, fallback to synthetic approach
                if image_np is None:
                    # Create a better synthetic image based on the mask
                    image_np = np.ones((518, 518, 3), dtype=np.uint8) * 220  # Light gray background
                    # Add some texture / noise
                    noise = np.random.randint(-20, 20, (518, 518, 3))
                    image_np = np.clip(image_np.astype(int) + noise, 0, 255).astype(np.uint8)
                    sample_name = f"Sample_{sample_idx}"
                
                # Create overlays
                overlay_standard = overlay_mask_on_image(
                    image_np.copy(), pred_standard,
                    alpha=0.5, color=(255, 0, 0)  # Red
                )
                
                overlay_optimal = overlay_mask_on_image(
                    image_np.copy(), pred_binary,
                    alpha=0.5, color=(0, 255, 0)  # Green
                )
                
                # Create bounding box visualization
                bbox_image = image_np.copy()
                
                # Find connected components
                labeled_mask = measure.label(pred_binary > 0.5, connectivity=2)
                regions = measure.regionprops(labeled_mask)
                
                # Create heatmap visualization
                heatmap = self._create_heatmap(pred_binary, max_distance=150)
                heatmap_overlay = self._apply_heatmap_overlay(image_np.copy(), heatmap, alpha=0.6)
                
                # Calculate metrics
                intersection = np.logical_and(gt_mask > 0.5, pred_binary > 0.5)
                union = np.logical_or(gt_mask > 0.5, pred_binary > 0.5)
                iou = np.sum(intersection) / (np.sum(union) + 1e-7)
                
                # Plot results
                axes[idx, 0].imshow(image_np)
                axes[idx, 0].set_title(f'Original Image\n{sample_name[:20]}...')
                axes[idx, 0].axis('off')
                
                axes[idx, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
                axes[idx, 1].set_title(f'Ground Truth\n{np.sum(gt_mask>0.5):.0f} pixels')
                axes[idx, 1].axis('off')
                
                axes[idx, 2].imshow(overlay_standard)
                axes[idx, 2].set_title(f'Pred @ 0.5\n{np.sum(pred_standard):.0f} pixels')
                axes[idx, 2].axis('off')
                
                axes[idx, 3].imshow(overlay_optimal)
                axes[idx, 3].set_title(f'Pred @ {self.optimal_threshold:.1f}\n{np.sum(pred_binary):.0f} px, IoU={iou:.3f}')
                axes[idx, 3].axis('off')
                
                # Plot bounding boxes
                axes[idx, 4].imshow(bbox_image)
                ax_bbox = axes[idx, 4]
                
                num_bboxes = 0
                for region in regions:
                    minr, minc, maxr, maxc = region.bbox
                    height = maxr - minr
                    width = maxc - minc
                    
                    # Expand by 50%
                    expand_h = height * 0.50
                    expand_w = width * 0.50
                    
                    minr_exp = max(0, minr - expand_h/2)
                    minc_exp = max(0, minc - expand_w/2)
                    maxr_exp = min(518, maxr + expand_h/2)
                    maxc_exp = min(518, maxc + expand_w/2)
                    
                    rect = patches.Rectangle(
                        (minc_exp, minr_exp),
                        maxc_exp - minc_exp,
                        maxr_exp - minr_exp,
                        linewidth=1,
                        edgecolor='lime',
                        facecolor='none'
                    )
                    ax_bbox.add_patch(rect)
                    num_bboxes += 1
                
                ax_bbox.set_title(f'Boulders\n{num_bboxes} detections')
                ax_bbox.axis('off')
                
                # Plot heatmap
                axes[idx, 5].imshow(heatmap_overlay)
                axes[idx, 5].set_title('Hazardous Zones\nRed=High Risk, Blue=Safe')
                axes[idx, 5].axis('off')
                
                # Add colorbar for heatmap
                if idx == 0:  # Only add colorbar for first row
                    # Create a small axes for colorbar
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(axes[idx, 5])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    
                    # Create colorbar
                    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
                    cmap = LinearSegmentedColormap.from_list('boulder_heat', colors, N=100)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, cax=cax)
                    cbar.set_label('Risk Level', rotation=270, labelpad=15)
                    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                    cbar.set_ticklabels(['Safe', 'Low', 'Medium', 'High', 'Critical'])
                
                successful_plots += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_idx}: {e}")
                # Clear the row if there was an error
                for col in range(6):
                    axes[idx, col].axis('off')
                continue
        
        if successful_plots > 0:
            plt.suptitle(
                f'Lightweight Model Inference Results with Hazardous Zones\n'
                f'Optimal Threshold: {self.optimal_threshold:.2f}, '
                f'Checkpoint IoU: {self.checkpoint_metrics.get("iou", 0):.4f}',
                fontsize=16
            )
            plt.tight_layout()
            
            # Save grid
            grid_path = output_dir / f"inference_grid_threshold_{self.optimal_threshold:.2f}.png"
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Successfully plotted {successful_plots}/{num_samples} samples")
            logger.info(f"Results saved to {output_dir}")
            logger.info(f"Visualization saved to {grid_path}")
        else:
            logger.error("No samples were successfully plotted!")
            plt.close()


    # Export heatmap visualization results 
    def _save_individual_results(
        self,
        dataset,
        results: list,
        split: str = "val",
        expand: float = 1.0,
    ) -> None:
        """
        Save inference results with visualizations.
        
        Args:
            results: List of result dictionaries
            visualize: Whether to create visualization plots
            num_samples: Number of samples to visualize
            split: Which split we're working with
        """
        # Create output directory
        output_dir = config.INFERENCE_RESULTS_DIR
        output_dir.mkdir(exist_ok=True)
        
        # Select samples to visualize
        num_samples = len(results)
        if num_samples == 0:
            logger.warning("No results to visualize")
            return
            
        sample_indices = list(range(num_samples))
        
        # Setup paths for images based on split
        images_dir = config.TRAIN_IMAGES if split == "train" else config.VAL_IMAGES
        
        sample_names = dataset.sample_names

        for idx, sample_idx in tqdm(list(enumerate(sample_indices))):
            try:
                result = results[sample_idx]
                
                # Get masks
                targ_binary = result['mask'].squeeze().numpy()
                pred_binary = result['pred_binary'].squeeze().numpy()
                
                # Try to load the actual image
                image_np = None
                if sample_idx < len(sample_names) and images_dir.exists():
                    sample_name = sample_names[sample_idx]
                    image_path = images_dir / f"{sample_name}.png"
                    
                    if image_path.exists():
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize((518, 518))
                        image_np = np.array(image)
                        logger.debug(f"Loaded image: {image_path}")
                    else:
                        logger.warning(f"Image not found: {image_path}")

                # If image not loaded, fallback to synthetic approach
                if image_np is None:
                    # Create a better synthetic image based on the mask
                    image_np = np.ones((518, 518, 3), dtype=np.uint8) * 220  # Light gray background
                    # Add some texture / noise
                    noise = np.random.randint(-20, 20, (518, 518, 3))
                    image_np = np.clip(image_np.astype(int) + noise, 0, 255).astype(np.uint8)
                    sample_name = f"Sample_{sample_idx}"

                image_np = draw_boxes_on_image(
                    image_np, targ_binary,
                    color=(255, 0, 0),
                )

                image_np = draw_boxes_on_image(
                    image_np, pred_binary,
                    color=(0, 255, 0),
                )

                outputfile = output_dir / f"{sample_name}_boxes.png"
                Image.fromarray(image_np).save(outputfile)

            except Exception as e:
                logger.error(f"Error processing sample {sample_idx}: {e}")
                continue


def run_lightweight_inference(
    model_path: Optional[Path] = None,
    feature_path: Optional[Path] = None,
    visualize: bool = True,
    optimal_threshold: float = 0.5,
    batch_size: int = 32,
    split: str = "val",
    head_type: str = "convolutional"
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Run inference on the val/train split using cached DINOv2 features.

    Args:
        model_path: Path to model checkpoint
        feature_path: Path to cached features directory
        visualize: Whether to visualize results
        optimal_threshold: Threshold for predictions (0.5 recommended)
        batch_size: Batch size for inference
        split: Which split to run inference on
        head_type: Segmentation head type — must match the head used to
            train the checkpoint ("convolutional", "aspp", or "bn")

    Returns:
        Tuple of (metrics_dict, counts_dict)
    """
    # Build model matching the trained head type
    if head_type == "convolutional":
        model = CachedFeaturesSegModel(
            feature_dim=config.FEATURE_DIM,
            num_classes=config.NUM_CLASSES,
        )
    elif head_type == "aspp":
        model = ASPPSegModel(
            feature_dim=config.FEATURE_DIM,
            num_classes=config.NUM_CLASSES,
        )
    else:
        model = LightweightSegmentationModel(
            feature_dim=config.FEATURE_DIM,
            num_classes=config.NUM_CLASSES,
            head_type="bn"
        )
    model = model.to(config.DEVICE)

    # Use default checkpoint if not provided
    possible_paths = [
        Path("checkpoints/best_model_weighted.pth"),
        Path("checkpoints/best_model_fast.pth"),
        config.CHECKPOINT_DIR / "best_model.pth",
    ]

    if model_path is None:
        model_path = next((p for p in possible_paths if p.exists()), None)
        if model_path:
            logger.info(f"Found checkpoint at: {model_path}")
    else:
        model_path = Path(model_path)

    if model_path is None or not model_path.exists():
        logger.error("Model checkpoint not found!")
        logger.error("Tried locations:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        return {}, {}
    
    # Create inference pipeline
    pipeline = LightweightInferencePipeline(
        model=model,
        device=config.DEVICE,
        checkpoint_path=model_path,
        optimal_threshold=optimal_threshold
    )
    
    # Determine features directory
    if feature_path is None:
        features_dir = Path("cached_features")
    else:
        features_dir = Path(feature_path)
    
    # Create data loader for cached features
    features_file = features_dir / f"{split}_features.h5"
    
    if not features_file.exists():
        # Try pickle format
        features_file = features_dir / f"{split}_features.pkl"
        if not features_file.exists():
            logger.error(f"Features file not found: {features_file}")
            logger.error("Please run feature extraction first:")
            logger.error("python -m mantle.main --mode extract")
            return {}, {}  # Return empty dicts
    
    # Determine masks directory
    masks_dir = config.TRAIN_MASKS if split == "train" else config.VAL_MASKS
    
    # Create dataset
    dataset = CachedFeaturesDataset(
        features_file=features_file,
        masks_dir=masks_dir,
        augment=False  # No augmentation during inference
    )
    
    # Create data loader
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    logger.info(f"Running inference on {split} split with {len(dataset)} samples")
    logger.info(f"Using optimal threshold: {optimal_threshold:.2f}")
    
    # Run inference
    start_time = time.time()
    metrics, counts = pipeline.process_batch(
        data_loader, 
        save_results=True, 
        visualize=visualize, 
        split=split
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"Inference completed in {elapsed_time:.1f} seconds")
    
    return metrics, counts


def main():
    """
    Main function for standalone inference.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightweight Model Inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--featurefile", type=str, default=None,
                       help="Path to saved features directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Optimal threshold for predictions")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--split", type=str, default="val",
                       choices=["train", "val"],
                       help="Which split to run inference on")
    parser.add_argument("--head-type", type=str, default="convolutional",
                       choices=["convolutional", "aspp", "bn"],
                       help="Segmentation head type — must match the trained checkpoint")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization plots")

    args = parser.parse_args()

    # Run inference
    metrics, counts = run_lightweight_inference(
        model_path=Path(args.checkpoint) if args.checkpoint else None,
        feature_path=Path(args.featurefile) if args.featurefile else None,
        visualize=args.visualize,
        optimal_threshold=args.threshold,
        batch_size=args.batch_size,
        split=args.split,
        head_type=args.head_type
    )
    
    # Print results - check if metrics exist first
    if not metrics:
        logger.error("Inference failed - no metrics to report")
        return
    
    print("\n" + "="*50)
    print("PIXEL-WISE SEGMENTATION RESULTS")
    print("="*50)
    print(f"Threshold: {args.threshold:.2f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print("="*50)

    if counts:
        print("\n" + "="*50)
        print("INSTANCE-LEVEL DETECTION RESULTS")
        print("="*50)
        print(f"True Positives (Correct detections):  {counts['tp']}")
        print(f"False Positives (False alarms):       {counts['fp']}")
        print(f"False Negatives (Missed boulders):    {counts['fn']}")
        
        # Calculate instance-level metrics
        precision_inst = counts['tp'] / (counts['tp'] + counts['fp'] + 1e-7)
        recall_inst = counts['tp'] / (counts['tp'] + counts['fn'] + 1e-7)
        f1_inst = 2 * (precision_inst * recall_inst) / (precision_inst + recall_inst + 1e-7)
        
        print(f"\nInstance Precision: {precision_inst:.4f}")
        print(f"Instance Recall:    {recall_inst:.4f}")
        print(f"Instance F1:        {f1_inst:.4f}")
        print("="*50)


if __name__ == "__main__":
    main()