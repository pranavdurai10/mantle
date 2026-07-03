# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Utility functions and dataset class for Mantle
#               – Custom PyTorch dataset for loading images and binary masks
#               – Loss functions (Dice, BCE-Dice) and evaluation metrics
#               – Checkpoint management and visualization utilities
# =============================================================================
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
from skimage import measure

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def glob_images(directory: Path) -> List[Path]:
    """
    Case-insensitive glob for PNG files.
    Handles both .png and .PNG extensions (important on Linux/HPC).
    """
    files = sorted(
        list(directory.glob("*.png")) +
        list(directory.glob("*.PNG"))
    )
    # Deduplicate in case filesystem is case-insensitive (macOS)
    seen = set()
    unique = []
    for f in files:
        key = f.name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return sorted(unique)


class BouldersDataset(Dataset):
    """
    Dataset class for loading boulder images and masks.
    Supports both .png and .PNG extensions.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform
        self.augment    = augment

        # Case-insensitive glob for both .png and .PNG
        self.image_files = glob_images(self.images_dir)
        self.mask_files  = [
            self.masks_dir / img.name
            for img in self.image_files
        ]

        # Verify all mask files exist
        missing = [m for m in self.mask_files if not m.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} mask file(s) not found. "
                f"First missing: {missing[0]}"
            )

        logger.info(
            f"Dataset initialized with {len(self.image_files)} samples "
            f"from {images_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_files[idx]).convert('RGB')
        mask  = Image.open(self.mask_files[idx]).convert('L')

        mask = mask.resize((784, 784), Image.NEAREST)

        image_np = np.array(image)
        mask_np  = np.array(mask)

        if self.augment:
            image_np, mask_np = self._apply_augmentations(image_np, mask_np)

        image = Image.fromarray(image_np)
        mask  = Image.fromarray(mask_np)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = mask.unsqueeze(0)

        return image, mask

    def _apply_augmentations(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        from .configs import config

        # Horizontal flip only — vertical flip disabled for Mastcam ground imagery
        if np.random.random() < config.HORIZONTAL_FLIP_PROB:
            image = np.fliplr(image)
            mask  = np.fliplr(mask)

        return image, mask


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred        = torch.sigmoid(pred)
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss.
    Used for the balanced MSL boulder dataset (pos_weight=1.2).
    """
    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 1.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce  = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Move pos_weight tensor to same device as pred
        self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        bce_loss  = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def box_iou(boxA, boxB):
    """Compute IoU between two bounding boxes (min_row, min_col, max_row, max_col)."""
    (A_y0, A_x0, A_y1, A_x1) = boxA
    (B_y0, B_x0, B_y1, B_x1) = boxB

    inter_y0 = max(A_y0, B_y0)
    inter_x0 = max(A_x0, B_x0)
    inter_y1 = min(A_y1, B_y1)
    inter_x1 = min(A_x1, B_x1)

    inter_h    = max(0, inter_y1 - inter_y0)
    inter_w    = max(0, inter_x1 - inter_x0)
    inter_area = inter_h * inter_w

    areaA      = (A_y1 - A_y0) * (A_x1 - A_x0)
    areaB      = (B_y1 - B_y0) * (B_x1 - B_x0)
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def expand_bounding_boxes(
    boxes: List[Tuple[float, float, float, float]],
    expand_pct: float
):
    expanded = []
    for (minr, minc, maxr, maxc) in boxes:
        height   = maxr - minr
        width    = maxc - minc
        expand_h = height * expand_pct
        expand_w = width  * expand_pct
        expanded.append((
            minr - expand_h / 2,
            minc - expand_w / 2,
            maxr + expand_h / 2,
            maxc + expand_w / 2,
        ))
    return expanded


def compute_tp_fp_fn_iou(pred_regions, targ_regions, iou_thresh=0.5):
    pred_boxes = expand_bounding_boxes([r.bbox for r in pred_regions], 1.0)
    targ_boxes = expand_bounding_boxes([r.bbox for r in targ_regions], 1.0)

    pred_matched = [False] * len(pred_boxes)
    targ_matched = [False] * len(targ_boxes)

    for i, pbox in enumerate(pred_boxes):
        for j, tbox in enumerate(targ_boxes):
            if box_iou(pbox, tbox) > iou_thresh:
                pred_matched[i] = True
                targ_matched[j] = True

    tp = sum(pred_matched)
    fp = len(pred_matched) - tp
    fn = sum(not m for m in targ_matched)
    return tp, fp, fn


def instance_counts(pred, target):
    pred_mask    = measure.label(pred,   connectivity=2)
    pred_regions = measure.regionprops(pred_mask)
    targ_mask    = measure.label(target, connectivity=2)
    targ_regions = measure.regionprops(targ_mask)
    tp, fp, fn   = compute_tp_fp_fn_iou(pred_regions, targ_regions, iou_thresh=0.0)
    return {'tp': tp, 'fp': fp, 'fn': fn}


def instance_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    preds_binary   = (preds   > threshold).float()
    targets_binary = targets.float()
    counts = defaultdict(int)
    for p, t in zip(preds_binary, targets_binary):
        p = p.squeeze().numpy()
        t = t.squeeze().numpy()
        c = instance_counts(p, t)
        for k, i in c.items():
            counts[k] += i
    return dict(counts)


def calculate_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    preds_binary   = (preds   > threshold).float()
    targets_binary = targets.float()

    preds_flat   = preds_binary.view(-1)
    targets_flat = targets_binary.view(-1)

    tp  = (preds_flat * targets_flat).sum()
    fp  = (preds_flat * (1 - targets_flat)).sum()
    fn  = ((1 - preds_flat) * targets_flat).sum()
    tn  = ((1 - preds_flat) * (1 - targets_flat)).sum()
    eps = 1e-7

    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return {
        'accuracy':  accuracy.item(),
        'precision': precision.item(),
        'recall':    recall.item(),
        'iou':       iou.item(),
        'f1':        f1.item()
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path
) -> None:
    checkpoint = {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics':              metrics
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: Path
) -> int:
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info(f"Checkpoint loaded from {filepath}, epoch {epoch}")
    return epoch




def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0.5] = color
    return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


def draw_boxes_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    expand: float = 1.0,
    width: int = 2,
) -> np.ndarray:
    labels  = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels)
    boxes   = expand_bounding_boxes([r.bbox for r in regions], expand)

    img  = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for min_row, min_col, max_row, max_col in boxes:
        draw.rectangle(
            [(min_col, min_row), (max_col, max_row)],
            outline=color,
            width=width,
        )
    return np.asarray(img)
