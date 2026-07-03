# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Inference pipeline for Martian terrain classification
#               – Performs inference on test dataset
#               – Generates confusion matrix visualization
#               – Provides detailed analytics on classification performance
# =============================================================================

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .model import create_classification_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('terrain_classification_inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TerrainInferenceDataset(Dataset):
    """
    Dataset for terrain classification inference.
    """
    
    def __init__(
        self,
        root_dir: str,
        transform=None,
        class_names: List[str] = None
    ):
        """
        Initialize inference dataset.
        
        Args:
            root_dir: Path to terrain-classification-dataset-test directory
            transform: Transformations to apply
            class_names: List of terrain class names
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Default class names
        if class_names is None:
            class_names = [
                'crater', 'dark_dune', 'slope_streak',
                'bright_dune', 'impact_ejecta', 'swiss_cheese', 'spider'
            ]
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in class_names:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))
                for img_path in images:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                logger.info(f"Found {len(images)} images for class '{class_name}'")
            else:
                logger.warning(f"Class directory not found: {class_dir}")
        
        logger.info(f"Total samples loaded: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)


class TerrainClassificationInference:
    """
    Inference pipeline for terrain classification.
    """
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        class_names: List[str]
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            class_names: List of class names
        """
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        self.model = create_classification_model(
            model_name="dinov2_vits14",
            num_classes=self.num_classes,
            freeze_backbone=True,
            class_names=class_names,
            device=device
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            logger.info(f"Model validation accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    
    def predict_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict on a batch of images.
        
        Args:
            images: Batch of images
        
        Returns:
            Predictions and probabilities
        """
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
        
        return predictions, probabilities
    
    def run_inference(self, data_loader: DataLoader) -> Dict:
        """
        Run inference on entire dataset.
        
        Args:
            data_loader: DataLoader for inference
        
        Returns:
            Dictionary with predictions and metrics
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_paths = []
        
        logger.info("Running inference...")
        
        for images, labels, paths in tqdm(data_loader, desc="Inference"):
            images = images.to(self.device)
            
            # Get predictions
            predictions, probabilities = self.predict_batch(images)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'paths': all_paths
        }
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Calculate comprehensive metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
        
        Returns:
            Dictionary with metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str = "confusion_matrix.png"
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the figure
        """
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix for percentage
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix - Martian Terrain Classification', fontsize=16, pad=20)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add percentage annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                percentage = cm_normalized[i, j] * 100
                if percentage > 0:
                    plt.text(
                        j + 0.5, i + 0.7,
                        f'{percentage:.1f}%',
                        ha='center', va='center',
                        fontsize=8, color='gray'
                    )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def print_detailed_analytics(self, metrics: Dict):
        """
        Print detailed analytics including TP, FP, TN, FN.
        
        Args:
            metrics: Dictionary with calculated metrics
        """
        cm = metrics['confusion_matrix']
        
        logger.info("\n" + "=" * 80)
        logger.info("DETAILED CLASSIFICATION ANALYTICS")
        logger.info("=" * 80)
        
        # Overall metrics
        logger.info(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
        logger.info(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
        logger.info(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        # Per-class detailed metrics
        logger.info("\n" + "-" * 80)
        logger.info("PER-CLASS METRICS:")
        logger.info("-" * 80)
        
        for i, class_name in enumerate(self.class_names):
            # Calculate TP, FP, TN, FN
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            
            logger.info(f"\nClass: {class_name}")
            logger.info(f"  Support: {metrics['support'][i]}")
            logger.info(f"  Precision: {metrics['precision'][i]:.4f}")
            logger.info(f"  Recall: {metrics['recall'][i]:.4f}")
            logger.info(f"  F1-Score: {metrics['f1'][i]:.4f}")
            logger.info(f"  True Positives (TP): {TP}")
            logger.info(f"  False Positives (FP): {FP}")
            logger.info(f"  True Negatives (TN): {TN}")
            logger.info(f"  False Negatives (FN): {FN}")
            
            # Additional metrics
            if TP + FP > 0:
                precision = TP / (TP + FP)
                logger.info(f"  Calculated Precision: {precision:.4f}")
            
            if TP + FN > 0:
                sensitivity = TP / (TP + FN)
                logger.info(f"  Sensitivity (Recall): {sensitivity:.4f}")
            
            if TN + FP > 0:
                specificity = TN / (TN + FP)
                logger.info(f"  Specificity: {specificity:.4f}")
        
        # Classification report
        logger.info("\n" + "-" * 80)
        logger.info("CLASSIFICATION REPORT:")
        logger.info("-" * 80)
        report = classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=self.class_names,
            digits=4
        )
        logger.info("\n" + report)
        
        # Most confused pairs
        logger.info("\n" + "-" * 80)
        logger.info("MOST CONFUSED PAIRS:")
        logger.info("-" * 80)
        
        # Find top confused pairs
        confusion_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((cm[i, j], self.class_names[i], self.class_names[j]))
        
        confusion_pairs.sort(reverse=True)
        for count, true_class, pred_class in confusion_pairs[:10]:
            logger.info(f"  {true_class} → {pred_class}: {count} samples")


DEFAULT_CLASS_NAMES = [
    'crater', 'dark_dune', 'slope_streak',
    'bright_dune', 'impact_ejecta', 'swiss_cheese', 'spider'
]


def run_classification_inference(
    model_path: str = "checkpoints/best_terrain_classification_model.pth",
    test_data_dir: str = "terrain-classification-dataset/test",
    class_names: List[str] = None,
    batch_size: int = 16,
    image_size: int = 224,
) -> Dict:
    """
    Run terrain classification inference and report metrics.

    Args:
        model_path: Path to trained classification checkpoint
        test_data_dir: Directory with one subfolder of .jpg images per class
        class_names: List of terrain class names (defaults to the 7 MSL classes)
        batch_size: Inference batch size
        image_size: Input image resolution (must match training)

    Returns:
        Dictionary of computed metrics
    """
    MODEL_PATH = model_path
    TEST_DATA_DIR = test_data_dir
    CLASS_NAMES = class_names or DEFAULT_CLASS_NAMES
    BATCH_SIZE = batch_size
    IMAGE_SIZE = image_size

    # Device selection with MPS support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Determine pin_memory based on device
    use_pin_memory = (device.type == 'cuda')  # Only use for CUDA, not MPS or CPU
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model checkpoint not found at {MODEL_PATH}")
        logger.error("Please train the model first using train_classification.py")
        sys.exit(1)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    logger.info(f"Loading test dataset from {TEST_DATA_DIR}")
    test_dataset = TerrainInferenceDataset(
        root_dir=TEST_DATA_DIR,
        transform=transform,
        class_names=CLASS_NAMES
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory
    )
    
    # Initialize inference pipeline
    inference = TerrainClassificationInference(
        model_path=MODEL_PATH,
        device=device,
        class_names=CLASS_NAMES
    )
    
    # Run inference
    results = inference.run_inference(test_loader)

    # Calculate metrics
    metrics = inference.calculate_metrics(
        results['predictions'],
        results['labels']
    )
    metrics['labels'] = results['labels']
    metrics['predictions'] = results['predictions']
    
    # Plot confusion matrix
    inference.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path="confusion_matrix_terrain_classification.png"
    )
    
    # Print detailed analytics
    inference.print_detailed_analytics(metrics)
    
    # Save results
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save predictions to file
    results_file = output_dir / "classification_results.txt"
    with open(results_file, 'w') as f:
        f.write("Image Path,True Class,Predicted Class,Confidence\n")
        for path, true_label, pred_label, probs in zip(
            results['paths'],
            results['labels'],
            results['predictions'],
            results['probabilities']
        ):
            true_class = CLASS_NAMES[true_label]
            pred_class = CLASS_NAMES[pred_label]
            confidence = probs[pred_label]
            f.write(f"{path},{true_class},{pred_class},{confidence:.4f}\n")
    
    logger.info(f"\nResults saved to {results_file}")
    logger.info("\nInference complete!")

    return metrics


if __name__ == "__main__":
    run_classification_inference()
