# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Training pipeline for Mantle
#               – Uses standard BCEDiceLoss (dataset is near-balanced at 45/55)
#               – pos_weight = 1.2 (computed from 6,232 SAM2-annotated masks)
#               – Trainer for BNHead, ConvolutionalHead, and ASPPHead
#               – Supports cached DINOv2 features for fast head-only training
# =============================================================================

import logging
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configs import config
from .utils import calculate_metrics, BCEDiceLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoundaryLoss(nn.Module):
    """
    Boundary-aware binary cross-entropy loss for segmentation.

    Args:
        boundary_weight (float, optional): Weight applied to boundary
            pixels in the loss map. Defaults to 5.0.

    Returns:
        None
    """
    def __init__(self, boundary_weight: float = 5.0):
        """
        Initializes the boundary-aware loss module.

        Args:
            boundary_weight (float, optional): Multiplicative weight for
                boundary pixels. Defaults to 5.0.

        Returns:
            None
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        # Laplacian kernel for edge detection
        kernel = torch.tensor(
            [[0., 1., 0.],
             [1.,-4., 1.],
             [0., 1., 0.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts boundary pixels using a Laplacian filter.

        Args:
            mask (torch.Tensor): Binary segmentation mask of shape
                (B, 1, H, W).

        Returns:
            torch.Tensor: Boundary mask tensor of shape
                (B, 1, H, W).
        """
        # mask: (B, 1, H, W) float [0,1]
        boundary = F.conv2d(mask, self.kernel.to(mask.device), padding=1).abs()
        return (boundary > 0.1).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted BCE loss using boundary emphasis.

        Args:
            pred (torch.Tensor): Predicted segmentation logits.
            target (torch.Tensor): Ground-truth segmentation mask.

        Returns:
            torch.Tensor: Boundary-aware BCE loss value.
        """
        boundary    = self._get_boundary(target)
        weight_map  = 1.0 + (self.boundary_weight - 1.0) * boundary
        bce_loss    = F.binary_cross_entropy_with_logits(
            pred, target, weight=weight_map, reduction='mean'
        )
        return bce_loss


class DeepSupervisionLoss(nn.Module):
    """
    Loss function with auxiliary deep supervision support.

    Loss = main_loss + aux_weight × (aux1_loss + aux2_loss)

    Args:
        main_criterion (nn.Module): Primary segmentation loss function.
        boundary_criterion (nn.Module): Boundary-aware auxiliary loss.
        aux_weight (float, optional): Initial weighting factor for
            auxiliary losses. Defaults to 0.3.
        total_epochs (int, optional): Total number of training epochs used
            for auxiliary loss annealing. Defaults to 100.

    Returns:
        None
    """
    def __init__(
        self,
        main_criterion: nn.Module,
        boundary_criterion: nn.Module,
        aux_weight: float = 0.3,
        total_epochs: int = 100,
    ):
        """
        Initializes the deep supervision loss module.

        Args:
            main_criterion (nn.Module): Main segmentation loss function.
            boundary_criterion (nn.Module): Boundary-aware loss function.
            aux_weight (float, optional): Weight applied to auxiliary
                supervision losses. Defaults to 0.3.
            total_epochs (int, optional): Total training epochs for
                auxiliary weight annealing. Defaults to 100.

        Returns:
            None
        """
        super().__init__()
        self.main_crit     = main_criterion
        self.boundary_crit = boundary_criterion
        self.aux_weight    = aux_weight
        self.total_epochs  = total_epochs

    def forward(
        self,
        outputs,           # tuple (main, aux1, aux2) or just main tensor
        targets: torch.Tensor,
        epoch: int = 1,
    ) -> torch.Tensor:
        """
        Computes segmentation loss with optional auxiliary supervision.

        Args:
            outputs (Union[torch.Tensor, Tuple[torch.Tensor,
                torch.Tensor, torch.Tensor]]): Main prediction tensor or
                tuple containing main and auxiliary predictions.
            targets (torch.Tensor): Ground-truth segmentation masks.
            epoch (int, optional): Current training epoch used for
                auxiliary weight annealing. Defaults to 1.

        Returns:
            torch.Tensor: Combined segmentation loss value.
        """
        if isinstance(outputs, tuple):
            main, aux1, aux2 = outputs
        else:
            return self.main_crit(outputs, targets) + self.boundary_crit(outputs, targets)

        main_loss = self.main_crit(main, targets) + self.boundary_crit(main, targets)

        # Anneal aux weight: full weight for first half, linear decay to 0
        progress   = min(epoch / (self.total_epochs * 0.5), 1.0)
        cur_weight = self.aux_weight * (1.0 - progress)

        if cur_weight > 0:
            aux_loss = (self.main_crit(aux1, targets) +
                        self.main_crit(aux2, targets)) * 0.5
            return main_loss + cur_weight * aux_loss

        return main_loss


class Trainer:
    """
    Trainer for Mantle boulder segmentation models.

    Args:
        model (nn.Module): Segmentation model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device used for training and inference.
        pos_weight (float, optional): Positive class weight used in
            BCE-based loss computation. Defaults to 1.2.
        learning_rate (float, optional): Initial learning rate for the
            optimizer. Defaults to 1e-4.
        num_epochs (int, optional): Number of training epochs.
            Defaults to 50.

    Returns:
        None
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        pos_weight: float = 1.2,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
    ):
        """
        Initializes the training pipeline and optimization setup.

        Args:
            model (nn.Module): Segmentation model to train.
            train_loader (DataLoader): Training dataset loader.
            val_loader (DataLoader): Validation dataset loader.
            device (torch.device): Device for computation.
            pos_weight (float, optional): Positive class weighting factor.
                Defaults to 1.2.
            learning_rate (float, optional): Optimizer learning rate.
                Defaults to 1e-4.
            num_epochs (int, optional): Total training epochs.
                Defaults to 50.

        Returns:
            None
        """
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.num_epochs   = num_epochs

        # Main loss: BCEDice — appropriate for balanced dataset
        main_loss     = BCEDiceLoss(bce_weight=config.BCE_WEIGHT, pos_weight=pos_weight)
        boundary_loss = BoundaryLoss(boundary_weight=5.0)

        # Wrap with deep supervision handler
        self.criterion = DeepSupervisionLoss(
            main_criterion=main_loss,
            boundary_criterion=boundary_loss,
            aux_weight=0.3,
            total_epochs=num_epochs,
        )

        # AdamW with cosine annealing
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )

        self.best_val_iou = 0.0
        self.best_epoch   = 0

        logger.info(
            f"Trainer initialized | pos_weight={pos_weight} | "
            f"LR={learning_rate} | epochs={num_epochs}"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs one full training epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            Dict[str, float]: Dictionary containing averaged training
                metrics such as loss, IoU, recall, precision, F1-score,
                and accuracy.
        """
        self.model.train()
        running_loss = 0.0
        all_metrics  = {'accuracy': 0, 'precision': 0, 'recall': 0, 'iou': 0, 'f1': 0}

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch:>3} [train]")

        for features, masks in progress_bar:
            features = features.to(self.device)
            masks    = masks.to(self.device)

            outputs = self.model(features, return_aux=True)                 if hasattr(self.model, 'head') and                    hasattr(self.model.head, 'aux_head1') else                 self.model(features)
            loss = self.criterion(outputs, masks, epoch=epoch)

            if torch.isnan(loss):
                logger.warning("NaN loss detected — skipping batch")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                main_out      = outputs[0] if isinstance(outputs, tuple) else outputs
                preds         = torch.sigmoid(main_out)
                batch_metrics = calculate_metrics(preds, masks)
                for key in all_metrics:
                    if key in batch_metrics:
                        all_metrics[key] += batch_metrics[key]

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'iou':  f"{batch_metrics['iou']:.4f}",
                })

        n = len(self.train_loader)
        for key in all_metrics:
            all_metrics[key] /= n
        all_metrics['loss'] = running_loss / n

        logger.info(
            f"Epoch {epoch:>3} [train] | loss={all_metrics['loss']:.4f} | "
            f"iou={all_metrics['iou']:.4f} | recall={all_metrics['recall']:.4f} | "
            f"precision={all_metrics['precision']:.4f}"
        )
        return all_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Runs validation for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            Dict[str, float]: Dictionary containing averaged validation
                metrics such as loss, IoU, recall, precision, F1-score,
                and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        all_metrics  = {'accuracy': 0, 'precision': 0, 'recall': 0, 'iou': 0, 'f1': 0}

        with torch.no_grad():
            for features, masks in tqdm(self.val_loader, desc=f"Epoch {epoch:>3} [val]  "):
                features = features.to(self.device)
                masks    = masks.to(self.device)

                # Val always uses main head only (no aux, no deep supervision)
                outputs = self.model(features)
                loss    = self.criterion(outputs, masks, epoch=epoch)

                if not torch.isnan(loss):
                    running_loss += loss.item()

                main_out      = outputs[0] if isinstance(outputs, tuple) else outputs
                preds         = torch.sigmoid(main_out)
                batch_metrics = calculate_metrics(preds, masks)
                for key in all_metrics:
                    if key in batch_metrics:
                        all_metrics[key] += batch_metrics[key]

        n = len(self.val_loader)
        for key in all_metrics:
            all_metrics[key] /= n
        all_metrics['loss'] = running_loss / n

        logger.info(
            f"Epoch {epoch:>3} [val]   | loss={all_metrics['loss']:.4f} | "
            f"iou={all_metrics['iou']:.4f} | recall={all_metrics['recall']:.4f} | "
            f"precision={all_metrics['precision']:.4f}"
        )
        return all_metrics

    def train(self, num_epochs: int = None):
        """
        Runs the complete training and validation pipeline.

        Args:
            num_epochs (int, optional): Number of epochs to train for.
                If None, uses the configured default number of epochs.

        Returns:
            None
        """
        if num_epochs is None:
            num_epochs = self.num_epochs

        logger.info(f"Starting training for {num_epochs} epochs...")
        start_time      = time.time()
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.validate(epoch)

            self.scheduler.step()

            # Save best model
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou  = val_metrics['iou']
                self.best_epoch    = epoch
                patience_counter   = 0

                checkpoint = {
                    'epoch':                epoch,
                    'model_state_dict':     self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics':          val_metrics,
                    'train_metrics':        train_metrics,
                }
                ckpt_path = Path("checkpoints") / "best_model.pth"
                ckpt_path.parent.mkdir(exist_ok=True)
                torch.save(checkpoint, ckpt_path)
                logger.info(
                    f"  New best model saved | val_iou={self.best_val_iou:.4f}"
                )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)"
                )
                break

        elapsed = (time.time() - start_time) / 60
        logger.info(f"Training complete in {elapsed:.1f} minutes")
        logger.info(f"Best val IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch}")
