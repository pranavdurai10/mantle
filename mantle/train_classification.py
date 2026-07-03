# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Training pipeline for Martian terrain classification
#               – Classifies between 7 terrain types using DINOv2 features
#               – Implements training with CrossEntropy loss
#               – Tracks comprehensive metrics for multi-class classification
# =============================================================================

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .model import create_classification_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('terrain_classification_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TerrainClassificationDataset(Dataset):
    """
    Dataset for Martian terrain classification.
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform=None,
        class_names: List[str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Path to terrain-classification-dataset directory
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
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
            else:
                logger.warning(f"Class directory not found: {class_dir}")
        
        logger.info(f"Loaded {len(self.samples)} samples from {root_dir}")
        logger.info("Class distribution:")
        for class_name in class_names:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            logger.info(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TerrainClassificationTrainer:
    """
    Trainer for Martian terrain classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        class_names: List[str],
        learning_rate: float = 1e-6
    ):
        """
        Initialize trainer.
        
        Args:
            model: Classification model
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            class_names: List of class names
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Loss function - CrossEntropy
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer - Adam with very low learning rate
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': []}
        self.valid_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': []}
        
        logger.info(f"Trainer initialized with LR={learning_rate}")
        logger.info(f"Training on device: {device}")
    
    def calculate_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            labels: Ground truth labels
        
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy
        if len(predictions.shape) > 1:
            preds = predictions.argmax(dim=1).cpu().numpy()
        else:
            preds = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(labels_np, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch} - Training",
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            all_preds.append(outputs.detach())
            all_labels.append(labels.detach())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'batch': f"{batch_idx+1}/{len(self.train_loader)}"
            })
        
        # Calculate epoch metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self.calculate_metrics(all_preds, all_labels)
        avg_loss = running_loss / len(self.train_loader)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch} Training - Loss: {avg_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, "
            f"Prec: {metrics['precision']:.4f}, "
            f"Rec: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        
        # Per-class accuracy
        preds_np = all_preds.argmax(dim=1).cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        for i, class_name in enumerate(self.class_names):
            class_mask = labels_np == i
            if class_mask.sum() > 0:
                class_acc = (preds_np[class_mask] == i).mean()
                logger.info(f"  {class_name}: {class_acc:.4f}")
        
        metrics['loss'] = avg_loss
        return metrics
    
    def validate(self, epoch: int, loader: DataLoader, split_name: str = "Validation") -> Dict[str, float]:
        """
        Validate/test the model.
        
        Args:
            epoch: Current epoch number
            loader: Data loader for validation/test
            split_name: Name of the split (for logging)
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                loader, 
                desc=f"Epoch {epoch} - {split_name}",
                leave=False
            )
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                running_loss += loss.item()
                all_preds.append(outputs)
                all_labels.append(labels)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self.calculate_metrics(all_preds, all_labels)
        avg_loss = running_loss / len(loader)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch} {split_name} - Loss: {avg_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, "
            f"Prec: {metrics['precision']:.4f}, "
            f"Rec: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        
        # Per-class accuracy
        preds_np = all_preds.argmax(dim=1).cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        for i, class_name in enumerate(self.class_names):
            class_mask = labels_np == i
            if class_mask.sum() > 0:
                class_acc = (preds_np[class_mask] == i).mean()
                logger.info(f"  {class_name}: {class_acc:.4f}")
        
        # Confusion matrix for validation
        if split_name == "Validation":
            cm = confusion_matrix(labels_np, preds_np)
            logger.info("Confusion Matrix (rows=true, cols=pred):")
            for i, class_name in enumerate(self.class_names):
                row = cm[i]
                logger.info(f"  {class_name}: {row}")
        
        metrics['loss'] = avg_loss
        return metrics
    
    def test(self) -> Dict[str, float]:
        """
        Test the model on test set.
        
        Returns:
            Dictionary with test metrics
        """
        logger.info("=" * 60)
        logger.info("Testing Best Model on Test Set")
        logger.info("=" * 60)
        
        # Load best model
        checkpoint_path = Path("checkpoints") / "best_terrain_classification_model.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # Test
        metrics = self.validate(epoch=0, loader=self.test_loader, split_name="Test")
        
        # Detailed confusion matrix
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        logger.info("\nDetailed Confusion Matrix:")
        logger.info("Rows = True Class, Columns = Predicted Class")
        logger.info(f"Classes: {', '.join(self.class_names)}")
        for i, class_name in enumerate(self.class_names):
            row_str = ' '.join([f"{val:4d}" for val in cm[i]])
            logger.info(f"{class_name:15s}: {row_str}")
        
        # Per-class precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        logger.info("\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            logger.info(
                f"{class_name:15s}: Prec={precision[i]:.3f}, "
                f"Rec={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}"
            )
        
        return metrics
    
    def train(self, num_epochs: int):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info("=" * 60)
        logger.info("Starting Terrain Classification Training")
        logger.info(f"Training for {num_epochs} epochs")
        logger.info(f"Classes: {', '.join(self.class_names)}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.train_history['precision'].append(train_metrics['precision'])
            self.train_history['recall'].append(train_metrics['recall'])
            
            # Validate
            valid_metrics = self.validate(epoch, self.valid_loader, "Validation")
            self.valid_history['loss'].append(valid_metrics['loss'])
            self.valid_history['accuracy'].append(valid_metrics['accuracy'])
            self.valid_history['precision'].append(valid_metrics['precision'])
            self.valid_history['recall'].append(valid_metrics['recall'])
            
            # Save best model
            if valid_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = valid_metrics['accuracy']
                self.best_epoch = epoch
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': valid_metrics,
                    'class_names': self.class_names
                }
                
                checkpoint_path = Path("checkpoints") / "best_terrain_classification_model.pth"
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                
                logger.info(f"New best model! Validation Accuracy: {self.best_val_acc:.4f}")
        
        elapsed = (time.time() - start_time) / 60
        logger.info(f"\nTraining complete in {elapsed:.1f} minutes")
        logger.info(f"Best Validation Accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        # Test on test set
        test_metrics = self.test()
        logger.info(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Save training history
        history = {
            'train': self.train_history,
            'valid': self.valid_history,
            'test': test_metrics
        }
        torch.save(history, Path("checkpoints") / "training_history.pth")


DEFAULT_CLASS_NAMES = [
    'crater', 'dark_dune', 'slope_streak',
    'bright_dune', 'impact_ejecta', 'swiss_cheese', 'spider'
]


def train_classification_model(
    data_dir: str = "terrain-classification-dataset",
    class_names: List[str] = None,
    batch_size: int = 16,
    num_epochs: int = 100,
    learning_rate: float = 1e-6,
    image_size: int = 224,
) -> nn.Module:
    """
    Train the DINOv2 terrain classification model.

    Args:
        data_dir: Root directory containing train/ and test/ class subfolders
        class_names: List of terrain class names (defaults to the 7 MSL classes)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Optimizer learning rate
        image_size: Input image resolution

    Returns:
        Trained classification model
    """
    DATA_DIR = data_dir
    CLASS_NAMES = class_names or DEFAULT_CLASS_NAMES
    BATCH_SIZE = batch_size
    NUM_EPOCHS = num_epochs
    LEARNING_RATE = learning_rate
    IMAGE_SIZE = image_size

    # Device selection with MPS support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    
    # Transforms - training with augmentation, validation/test without
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Define directories for train and test datasets
    TRAIN_DATA_DIR = Path(DATA_DIR) / "train"
    TEST_DATA_DIR  = Path(DATA_DIR) / "test"

    # Create datasets with different transforms
    logger.info(f"Loading training dataset from {TRAIN_DATA_DIR}")
    logger.info(f"Loading test dataset from {TEST_DATA_DIR}")

    # Load the training dataset
    train_val_dataset = TerrainClassificationDataset(
        root_dir=TRAIN_DATA_DIR,
        transform=None,  # No transform yet
        class_names=CLASS_NAMES
    )
    
    # Load the test dataset separately
    test_dataset_full = TerrainClassificationDataset(
        root_dir=TEST_DATA_DIR,
        transform=val_test_transform,  # Apply transform directly
        class_names=CLASS_NAMES
    )
    
    # Split train_val_dataset into 80% train and 20% validation
    train_val_samples = train_val_dataset.samples
    total_train_val_size = len(train_val_samples)
    train_size = int(0.8 * total_train_val_size)
    valid_size = total_train_val_size - train_size
    
    # Create indices for splitting train/validation
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)
    
    shuffled_indices = torch.randperm(total_train_val_size).tolist()
    
    train_indices = shuffled_indices[:train_size]
    valid_indices = shuffled_indices[train_size:]
    
    # Create separate datasets with appropriate transforms
    class SubsetDataset(Dataset):
        def __init__(self, samples, indices, transform, class_names):
            self.samples = [samples[i] for i in indices]
            self.transform = transform
            self.class_names = class_names
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    train_dataset = SubsetDataset(train_val_samples, train_indices, train_transform, CLASS_NAMES)
    valid_dataset = SubsetDataset(train_val_samples, valid_indices, val_test_transform, CLASS_NAMES)
    
    # Test dataset is already loaded
    test_size = len(test_dataset_full)
    
    logger.info(f"Dataset split - Train: {train_size} (80%), Valid: {valid_size} (20%), Test: {test_size} (separate)")
    logger.info(f"Training samples per class (approx): {train_size // len(CLASS_NAMES)}")
    logger.info(f"Validation samples per class (approx): {valid_size // len(CLASS_NAMES)}")
    logger.info(f"Test samples per class (approx): {test_size // len(CLASS_NAMES)}")

    # Determine pin_memory based on device
    use_pin_memory = (device.type == 'cuda')  # Only use for CUDA, not MPS or CPU
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset_full,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )

    # Create model
    logger.info("Creating DINOv2 classification model...")
    model = create_classification_model(
        model_name="dinov2_vits14",
        num_classes=len(CLASS_NAMES),
        freeze_backbone=True,
        class_names=CLASS_NAMES,
        device=device
    )
    
    # Create trainer
    trainer = TerrainClassificationTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        class_names=CLASS_NAMES,
        learning_rate=LEARNING_RATE
    )
    
    # Train
    trainer.train(num_epochs=NUM_EPOCHS)

    logger.info("Training pipeline complete!")

    return model


if __name__ == "__main__":
    train_classification_model()