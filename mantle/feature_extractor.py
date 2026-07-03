# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : DINOv2-based feature extractor for MSL Mastcam imagery
#               – Command-line interface for extraction
#               – Supports train and val splits
#               – Writes extracted features to .h5 files
# =============================================================================

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .configs import config
from .model import ConvolutionalHead, ASPPHead
from .utils import glob_images

warnings.filterwarnings("ignore", message="xFormers is not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractionDataset(Dataset):
    """
    Dataset for feature extraction without segmentation masks.

    Args: 
        images_dir (Path): Directory containing input images. 
        transform (Callable, optional): Transformations to apply to each image.

    Returns:
        None
    """
    def __init__(self, images_dir: Path, transform=None):
        self.images_dir  = Path(images_dir)
        self.image_files = glob_images(self.images_dir)
        self.transform   = transform
        logger.info(f"Feature extraction dataset: {len(self.image_files)} images from {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx].stem


class DINOv2FeatureExtractor:
    """
    Extracts and caches DINOv2 features for faster downstream training.

    Args: 
        model_name (str, optional): Name of the DINOv2 model to load. 
        device (torch.device, optional): Device for inference, defaults to CUDA.

    Returns:
        None
    """
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading {model_name}...")
        self.model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.feature_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536
        }
        self.feature_dim = self.feature_dims.get(model_name, 768)

        self.transform = transforms.Compose([
            transforms.Resize((784, 784)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info(f"Feature extractor initialized with {model_name}")

    def extract_features_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts final-layer patch token features from a batch of images.

        Args:
            images (torch.Tensor): Batch of input images. 

        Returns: 
            torch.Tensor: Extracted patch token features.
        """
        with torch.no_grad():
            features     = self.model.forward_features(images)
            patch_tokens = features['x_norm_patchtokens']
        return patch_tokens.cpu()

    def extract_and_save_features(
        self,
        data_split: str = "train",
        batch_size: int = 32,
        output_dir: Path = None,
        use_h5: bool = True,
    ):
        """
        Extract features for all images in a split and save to disk.

          Args:
            data_split (str, optional): Dataset split to process. Must be either "train" or "val". 
                Defaults to "train".
            batch_size (int, optional): Batch size used during feature extraction. 
                Defaults to 32.
            output_dir (Path, optional): Directory where extracted features will be saved. 
                Defaults to "cached_features".
            use_h5 (bool, optional): If True, saves features in HDF5 format.

        Returns:
            None
        """
        import h5py

        # Map split name to config path
        if data_split == "train":
            images_dir = config.TRAIN_IMAGES
        elif data_split == "val":
            images_dir = config.VAL_IMAGES
        else:
            raise ValueError(f"Unknown split '{data_split}'. Use 'train' or 'val'.")

        if output_dir is None:
            output_dir = Path("cached_features")
        output_dir.mkdir(exist_ok=True)

        dataset = FeatureExtractionDataset(images_dir, self.transform)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        logger.info(f"Extracting features for '{data_split}' split ({len(dataset)} images)...")

        if use_h5:
            output_file = output_dir / f"{data_split}_features.h5"
            num_patches = (config.IMAGE_SIZE[0] // 14) * (config.IMAGE_SIZE[1] // 14)

            with h5py.File(output_file, 'w') as hf:
                features_dset = hf.create_dataset(
                    'features',
                    shape=(len(dataset), num_patches, self.feature_dim),
                    dtype='float32',
                    chunks=(1, num_patches, self.feature_dim),
                    compression='gzip',
                    compression_opts=4
                )
                dt         = h5py.special_dtype(vlen=str)
                names_dset = hf.create_dataset('names', shape=(len(dataset),), dtype=dt)

                idx = 0
                for batch_images, batch_names in tqdm(loader, desc=f"Extracting {data_split}"):
                    batch_images = batch_images.to(self.device)
                    features     = self.extract_features_batch(batch_images)

                    batch_size_actual = len(batch_names)
                    features_dset[idx:idx + batch_size_actual] = features.numpy()
                    names_dset[idx:idx + batch_size_actual]    = batch_names
                    idx += batch_size_actual

                    if idx % 500 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

            logger.info(f"Saved {len(dataset)} features to {output_file}")
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Feature file size: {file_size_mb:.2f} MB (compressed)")

        else:
            output_subdir = output_dir / data_split
            output_subdir.mkdir(exist_ok=True)
            chunk_size    = 1000
            current_chunk = {}
            chunk_idx     = 0

            for batch_images, batch_names in tqdm(loader, desc=f"Extracting {data_split}"):
                batch_images = batch_images.to(self.device)
                features     = self.extract_features_batch(batch_images)

                for name, feat in zip(batch_names, features):
                    current_chunk[name] = feat.numpy()

                    if len(current_chunk) >= chunk_size:
                        chunk_file = output_subdir / f"chunk_{chunk_idx:04d}.pkl"
                        with open(chunk_file, 'wb') as f:
                            pickle.dump(current_chunk, f, protocol=4)
                        logger.info(f"Saved chunk {chunk_idx} ({len(current_chunk)} features)")
                        current_chunk = {}
                        chunk_idx    += 1
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if current_chunk:
                chunk_file = output_subdir / f"chunk_{chunk_idx:04d}.pkl"
                with open(chunk_file, 'wb') as f:
                    pickle.dump(current_chunk, f, protocol=4)
                logger.info(f"Saved final chunk {chunk_idx} ({len(current_chunk)} features)")


class CachedFeaturesDataset(Dataset):
    """
    Dataset for loading pre-computed DINOv2 features and masks.

    Args:
    features_file (Path): Path to cached feature file or feature chunks.
    masks_dir (Path): Directory containing segmentation masks.
    augment (bool, optional): Whether to apply data augmentation.
        Defaults to False.

    Returns:
        None
    """
    def __init__(
        self,
        features_file: Path,
        masks_dir: Path,
        augment: bool = False
    ):
        import h5py

        self.masks_dir    = Path(masks_dir)
        self.augment      = augment
        self.features_file = Path(features_file)

        if self.features_file.suffix == '.h5':
            with h5py.File(features_file, 'r') as hf:
                self.sample_names = [
                    name.decode() if isinstance(name, bytes) else name
                    for name in hf['names'][:]
                ]
                self.features_shape = hf['features'].shape
            self.use_h5   = True
            self.features = None

        elif (self.features_file.parent / 'chunk_0000.pkl').exists():
            self.features      = {}
            self.sample_names  = []
            chunk_dir          = self.features_file.parent
            for chunk_file in sorted(chunk_dir.glob('chunk_*.pkl')):
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    self.features.update(chunk_data)
                    self.sample_names.extend(chunk_data.keys())
            self.use_h5 = False
        else:
            with open(features_file, 'rb') as f:
                self.features = pickle.load(f)
            self.sample_names = list(self.features.keys())
            self.use_h5       = False

        logger.info(f"Loaded {len(self.sample_names)} cached features from {features_file}")

    def __len__(self):
        """
        Returns the total number of cached samples.

        Returns:
            int: Number of cached feature samples. 
        """
        return len(self.sample_names)

    def _find_mask(self, stem: str) -> Path:
        """
        Finds a mask file using case-insensitive extension matching.

        Args:
            stem (str): Filename stem of the sample.

        Returns:
            Path: Path to the corresponding mask file.

        Raises:
            FileNotFoundError: If no matching mask file is found.
        """
        for ext in ['.PNG', '.png']:
            p = self.masks_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Mask not found for {stem} in {self.masks_dir}")

    def __getitem__(self, idx):
        """
        Returns cached features and corresponding segmentation mask.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cached DINOv2 features and normalized mask tensor.
        """
        import h5py

        sample_name = self.sample_names[idx]

        if self.use_h5:
            with h5py.File(self.features_file, 'r') as hf:
                features = torch.from_numpy(hf['features'][idx].copy())
        else:
            features = torch.from_numpy(self.features[sample_name])

        # Load mask — try both .PNG and .png
        mask_path = self._find_mask(sample_name)
        mask      = Image.open(mask_path).convert('L')
        mask      = mask.resize((784, 784), Image.NEAREST)
        mask_np   = np.array(mask)

        if self.augment and np.random.random() < 0.5:
            # Horizontal flip only — vertical flip disabled for Mastcam imagery
            if np.random.random() < config.HORIZONTAL_FLIP_PROB:
                # Derive patch grid size dynamically from feature tensor shape
                num_patches  = features.shape[0]
                grid_size    = int(num_patches ** 0.5)   # 37 for 518px, 56 for 784px
                features_np  = features.numpy().reshape(grid_size, grid_size, -1)
                features_np  = np.fliplr(features_np).copy()
                features_np  = features_np.reshape(-1, features_np.shape[-1])
                features     = torch.from_numpy(features_np)
                mask_np      = np.fliplr(mask_np).copy()

        mask = torch.from_numpy(mask_np.copy()).float() / 255.0
        mask = mask.unsqueeze(0)

        return features, mask


class LightweightSegmentationModel(nn.Module):
    """
    Lightweight segmentation head for pre-computed DINOv2 features.

    Args:
        feature_dim (int, optional): Dimension of input DINOv2 feature embeddings. 
            Defaults to 384.
        num_classes (int, optional): Number of output segmentation classes.
            Defaults to 1.
        head_type (str, optional): Type of segmentation head to use. Currently only "bn" is supported. 
            Defaults to "bn".

    Returns:
        None
    """
    def __init__(
        self,
        feature_dim: int = 384,
        num_classes: int = 1,
        head_type: str = "bn"
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.patch_size  = 14
        self.img_size    = 784

        if head_type == "bn":
            self.norm     = nn.LayerNorm(feature_dim, eps=1e-6)
            self.bn       = nn.BatchNorm2d(feature_dim)
            self.conv_seg = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        else:
            raise ValueError("For cached features, only 'bn' head is supported. "
                             "Use head_type='convolutional' via CachedFeaturesSegModel in main.py.")

        logger.info(f"LightweightSegmentationModel initialized with {head_type} head")

    def forward(self, features):
        """
        Performs forward pass using cached DINOv2 features.

        Args:
            features (torch.Tensor): Input patch token features of shape `(batch_size, num_patches, feature_dim)`.

        Returns:
            torch.Tensor: Upsampled segmentation logits of shape `(batch_size, num_classes, img_size, img_size)`.
        """
        batch_size = features.shape[0]
        features   = self.norm(features)
        h_patches  = self.img_size // self.patch_size
        w_patches  = self.img_size // self.patch_size
        features   = features.reshape(batch_size, h_patches, w_patches, -1)
        features   = features.permute(0, 3, 1, 2).contiguous()
        x          = self.bn(features)
        x          = self.conv_seg(x)
        x          = torch.nn.functional.interpolate(
            x, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        )
        return x


class CachedFeaturesSegModel(nn.Module):
    """
    Wrapper for ConvolutionalHead using cached DINOv2 features.

    Args:
        feature_dim (int): Feature embedding dimension.
        num_classes (int): Number of output classes.
        img_size (int, optional): Input image size.
            Defaults to 784.
        patch_size (int, optional): DINOv2 patch size.
            Defaults to 14.

    Returns:
        None
    """
    def __init__(self, feature_dim, num_classes, img_size=784, patch_size=14):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.head = ConvolutionalHead(
            in_channels=feature_dim,
            hidden_channels=128,
            num_classes=num_classes
        )

    def forward(self, features, return_aux: bool = False):
        """
        Performs forward pass for segmentation.

        Args:
            features (torch.Tensor): Cached DINOv2 patch features.
            return_aux (bool, optional): Whether to return auxiliary
                outputs for deep supervision. Defaults to False.

        Returns:
            torch.Tensor: Segmentation predictions.
        """
        B     = features.shape[0]
        h = w = self.img_size // self.patch_size
        features = self.norm(features)
        features = features.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return self.head(features, return_aux=return_aux)


class ASPPSegModel(nn.Module):
    """
    Wrapper for ASPPHead using cached DINOv2 features.

    Args:
        feature_dim (int): Feature embedding dimension.
        num_classes (int): Number of output classes.
        img_size (int, optional): Input image size.
            Defaults to 784.
        patch_size (int, optional): DINOv2 patch size.
            Defaults to 14.

    Returns:
        None
    """
    def __init__(self, feature_dim, num_classes, img_size=784, patch_size=14):
        super().__init__()
        self.feature_dim = feature_dim
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.head = ASPPHead(
            in_channels=feature_dim,
            aspp_channels=128,
            num_classes=num_classes,
            dilations=(1, 6, 12, 18),
            dropout=0.1,
        )

    def forward(self, features):
        """
        Performs forward pass using ASPP segmentation head.

        Args:
            features (torch.Tensor): Cached DINOv2 patch features.

        Returns:
            torch.Tensor: Upsampled segmentation predictions.
        """
        B        = features.shape[0]
        h = w    = self.img_size // self.patch_size
        features = self.norm(features)
        features = features.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x        = self.head(features)
        x        = F.interpolate(
            x, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        )
        return x


def extract_all_features():
    """
    Extracts and saves DINOv2 features for train and validation splits.

    Returns:
        None
    """
    extractor = DINOv2FeatureExtractor(
        model_name=config.DINOV2_MODEL,
        device=config.DEVICE
    )
    for split in ["train", "val"]:
        extractor.extract_and_save_features(
            data_split=split,
            batch_size=32,
            output_dir=Path("cached_features_vitb_784")
        )
    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    extract_all_features()
