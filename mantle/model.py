# =============================================================================
# Author        : Pranav Durai
# Role          : Research Fellow
# Affiliation   : Stanford Center for Innovation in In Vivo Imaging
#                 Stanford University School of Medicine, Stanford, CA 94305
# Collaboration : Dr. Gary B. Doran
#                 Jet Propulsion Laboratory, California Institute of Technology,
#                 Pasadena, CA 91109
#
# Description : Mantle model architecture
#               – ConvolutionalHead v3: SE+CBAM+dilated+double-conv+deep supervision
#               – ASPPHead: multi-scale head with atrous spatial pyramid pooling
#               – DINOv2-based terrain classification model
#               – Support for multiple DINOv2 variants (ViT-S/B/L/G)
# =============================================================================

import logging
import warnings
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import config

# Suppress xFormers warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is not available")

logger = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.

    Args:
        channels (int): Number of input feature channels.
        reduction (int, optional): Channel reduction ratio used in
            the excitation MLP. Defaults to 16. 

    Returns:
        None
    """
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initializes the SEBlock attention module.

        Args:
        channels (int): Number of input feature channels.
        reduction (int, optional): Channel reduction ratio used in
            the excitation MLP. Defaults to 16. 

        Returns:
            None
        """
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies channel-wise attention to input features.

        Args:
            x (torch.Tensor): Input feature tensor of shape
                (B, C, H, W).

        Returns:
            torch.Tensor: Channel-attended feature tensor of shape
                (B, C, H, W).
        """
        B, C, _, _ = x.shape
        s = self.pool(x).view(B, C)
        s = self.fc(s).view(B, C, 1, 1)
        return x * s


class CBAMSpatialAttention(nn.Module):
    """
    CBAM spatial attention module for highlighting salient regions.

    Args:
        kernel_size (int, optional): Convolutional kernel size used for
            spatial attention generation. Defaults to 7.

    Returns:
        None
    """
    def __init__(self, kernel_size: int = 7):
        """
        Initializes the CBAM spatial attention block.

        Args:
            kernel_size (int, optional): Kernel size for the
                spatial attention mechanism. Defaults to 7.

        Returns:
            None
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies spatial attention to the input feature map.

        Args:
            x (torch.Tensor): Input feature tensor of shape
                (B, C, H, W).

        Returns:
            torch.Tensor: Spatially-attended feature tensor of shape
                (B, C, H, W).
        """
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


def double_conv(in_ch: int, out_ch: int, dilation: int = 1) -> nn.Sequential:
    """
    Creates a double convolution refinement block.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        dilation (int, optional): Dilation rate for the second convolution
            layer. Defaults to 1. 

    Returns:
        nn.Sequential: Sequential double convolution block with
            BatchNorm and ReLU activations.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1,        bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3,
                  padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvolutionalHead(nn.Module):
    """
    Enhanced convolutional segmentation head with:
    - SE channel attention      – focuses on boulder-relevant DINOv2 channels
    - CBAM spatial attention    – highlights boulder-dense spatial regions
    - Dilated conv in encoder   – wider receptive field for large boulders
    - Double-conv decoder       – 2× refinement Conv3×3 at each upsample step
    - Deep supervision          – auxiliary logits at dec1 and dec2 for
                                   multi-scale training signal (discarded at inference)
    Architecture:
        Input (B, 768, 56, 56)
          → SE channel attention
          → CBAM spatial attention
          → enc1: double_conv(768→128)             [56×56]
          → enc2: double_conv(128→64, dilation=2)  [56×56]  ← dilated
          → up1 + double_conv(64→32)               [112×112]
              └─ aux_head1 → aux_logits1  (deep supervision)
          → up2 + double_conv(32→16)               [224×224]
              └─ aux_head2 → aux_logits2  (deep supervision)
          → up3 + double_conv(16→16)               [448×448]
          → conv_final                             [448×448]
          → bilinear 448 → 784

    Args:
        in_channels (int): Number of input feature channels.
        hidden_channels (int, optional): Number of hidden channels used in
            the encoder. Defaults to 128.
        num_classes (int, optional): Number of segmentation output classes.
            Defaults to 1.

    Returns: (main_logits,) during inference
             (main_logits, aux1, aux2) during training (when return_aux=True)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 1
    ):
        """
        Initializes the enhanced convolutional segmentation head.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int, optional): Encoder hidden channel size.
                Defaults to 128.
            num_classes (int, optional): Number of output segmentation
                classes. Defaults to 1.

        Returns:
            None
        """
        super().__init__()
        mid = hidden_channels // 2   # 64

        # SE channel attention
        self.se   = SEBlock(in_channels, reduction=16)

        # CBAM spatial attention (after SE, on post-SE features)
        self.cbam = CBAMSpatialAttention(kernel_size=7)

        # Encoder with double-conv and dilation on enc2
        self.enc1 = double_conv(in_channels, hidden_channels)          # 768→128, dilation=1
        self.enc2 = double_conv(hidden_channels, mid, dilation=2)      # 128→64,  dilation=2

        self.drop = nn.Dropout2d(0.1)

        # Progressive decoder with double-conv at each stage
        # 56 → 112
        self.up1  = nn.ConvTranspose2d(mid,        mid // 2, kernel_size=2, stride=2)
        self.dec1 = double_conv(mid // 2, mid // 2)                    # 32ch

        # 112 → 224
        self.up2  = nn.ConvTranspose2d(mid // 2,   mid // 4, kernel_size=2, stride=2)
        self.dec2 = double_conv(mid // 4, mid // 4)                    # 16ch

        # 224 → 448
        self.up3  = nn.ConvTranspose2d(mid // 4,   mid // 4, kernel_size=2, stride=2)
        self.dec3 = double_conv(mid // 4, mid // 4)                    # 16ch

        # Final projection
        self.conv_final = nn.Conv2d(mid // 4, num_classes, kernel_size=1)

        # Deep supervision auxiliary heads (used only during training)
        # Bilinear upsample to full res happens in forward; these are 1×1 convs
        self.aux_head1 = nn.Conv2d(mid // 2, num_classes, kernel_size=1)  # after dec1
        self.aux_head2 = nn.Conv2d(mid // 4, num_classes, kernel_size=1)  # after dec2

        logger.info(
            f"ConvolutionalHead v3 | in={in_channels} | "
            f"SE + CBAM + dilated-enc + double-conv decoder + deep supervision"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False
    ):
        """
        Performs forward pass through the segmentation head.

        Args:
            x (torch.Tensor): Input feature tensor of shape
                (B, C, H, W).
            return_aux (bool, optional): Whether to return auxiliary
                supervision outputs. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor,
            torch.Tensor]]:
                Main segmentation logits during inference or main and
                auxiliary logits during training and `return_aux=True`.
        """
        H, W = config.IMAGE_SIZE

        # Attention gates
        x = self.se(x)
        x = self.cbam(x)

        # Encode
        x = self.drop(self.enc1(x))   # (B, 128, 56, 56)
        x = self.drop(self.enc2(x))   # (B,  64, 56, 56)

        # Decode stage 1: 56 → 112
        d1 = self.dec1(self.up1(x))   # (B,  32, 112, 112)

        # Decode stage 2: 112 → 224
        d2 = self.dec2(self.up2(d1))  # (B,  16, 224, 224)

        # Decode stage 3: 224 → 448
        d3 = self.dec3(self.up3(d2))  # (B,  16, 448, 448)

        # Main logits → upsample to full resolution
        main = F.interpolate(
            self.conv_final(d3),
            size=(H, W), mode='bilinear', align_corners=False
        )

        if return_aux:
            aux1 = F.interpolate(
                self.aux_head1(d1),
                size=(H, W), mode='bilinear', align_corners=False
            )
            aux2 = F.interpolate(
                self.aux_head2(d2),
                size=(H, W), mode='bilinear', align_corners=False
            )
            return main, aux1, aux2

        return main


class ASPPModule(nn.Module):
    """
    Single ASPP branch using atrous convolution, BatchNorm, and ReLU.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dilation (int): Dilation rate for atrous convolution.

    Returns:
        None
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        """
        Initializes an ASPP branch module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dilation (int): Dilation factor for the convolution layer.

        Returns:
            None
        """
        super().__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, bias=False
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation, bias=False
            )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the ASPP branch.

        Args:
            x (torch.Tensor): Input feature tensor of shape
                (B, C, H, W).

        Returns:
            torch.Tensor: Processed feature tensor after atrous
                convolution, normalization and activation. 
        """
        return self.relu(self.bn(self.conv(x)))


class ASPPHead(nn.Module):
    """
    Atrous Spatial Pyramid Pooling segmentation head.

    Args:
        in_channels (int, optional): Number of input feature channels.
            Defaults to 384.
        aspp_channels (int, optional): Number of channels used in ASPP
            branches. Defaults to 128.
        num_classes (int, optional): Number of segmentation output classes.
            Defaults to 1.
        dilations (tuple, optional): Dilation rates for ASPP branches.
            Defaults to (1, 6, 12, 18).
        dropout (float, optional): Dropout probability applied after ASPP
            projection. Defaults to 0.1.

    Returns:
        None
    """

    def __init__(
        self,
        in_channels:  int = 384,
        aspp_channels: int = 128,
        num_classes:  int = 1,
        dilations:    tuple = (1, 6, 12, 18),
        dropout:      float = 0.1,
    ):
        """
        Initializes the ASPP segmentation head.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 384.
            aspp_channels (int, optional): Number of channels in each ASPP
                branch. Defaults to 128.
            num_classes (int, optional): Number of output segmentation
                classes. Defaults to 1.
            dilations (tuple, optional): Atrous convolution dilation rates.
                Defaults to (1, 6, 12, 18).
            dropout (float, optional): Dropout probability for ASPP feature
                projection. Defaults to 0.1.

        Returns:
            None
        """
        super().__init__()

        # Parallel ASPP branches
        self.branches = nn.ModuleList([
            ASPPModule(in_channels, aspp_channels, d) for d in dilations
        ])

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )

        # Project concatenated branches back to aspp_channels
        n_branches     = len(dilations) + 1          # +1 for global pool
        self.project   = nn.Sequential(
            nn.Conv2d(aspp_channels * n_branches, aspp_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        logger.info(
            f"ASPPHead initialized | in_channels={in_channels} | "
            f"aspp_channels={aspp_channels} | dilations={dilations} | "
            f"num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the ASPP segmentation head.

        Args:
            x (torch.Tensor): Input feature tensor of shape
                (B, C, H, W).

        Returns:
            torch.Tensor: Segmentation logits tensor.
        """
        h, w = x.shape[2], x.shape[3]

        # Run all parallel branches
        branch_outs = [b(x) for b in self.branches]

        # Global pool branch — upsample back to spatial size
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode='bilinear', align_corners=False)
        branch_outs.append(gp)

        # Concatenate, project, decode
        x = torch.cat(branch_outs, dim=1)
        x = self.project(x)
        x = self.decoder(x)
        return x


class DINOv2Classification(nn.Module):
    """
    DINOv2-based classification model for terrain type classification.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        num_classes: int = 7,
        freeze_backbone: bool = True,
        class_names: List[str] = None
    ):
        """
        Initialize DINOv2 classification model.
        
        Args:
            model_name: DINOv2 model variant
            num_classes: Number of terrain classes
            freeze_backbone: Whether to freeze DINOv2 backbone
            class_names: List of class names
        """
        super().__init__()
        
        # Load DINOv2 model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available")
            self.transformer = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True
            )
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("DINOv2 backbone frozen for classification")
        
        # Create classification head - matching experimental code exactly
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Store class names
        if class_names is None:
            class_names = [
                'crater', 'dark_dune', 'slope_streak', 
                'bright_dune', 'impact_ejecta', 'swiss_cheese', 'spider'
            ]
        self.class_names = class_names
        
        logger.info(
            f"DINOv2 Classification model initialized with {model_name}, "
            f"num_classes={num_classes}"
        )
    
    def _get_feature_dim(self, model_name: str) -> int:
        """Get feature dimension for different DINOv2 variants."""
        feature_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536
        }
        return feature_dims.get(model_name, 384)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        SIMPLIFIED to match experimental code that works.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Class logits of shape (B, num_classes)
        """
        # Get features directly from transformer (returns CLS token)
        x = self.transformer(x)
        
        # Apply normalization (matching experimental code)
        x = self.transformer.norm(x)
        
        # Pass through classifier
        x = self.classifier(x)
        
        return x


def create_classification_model(
    model_name: str = "dinov2_vits14",
    num_classes: int = 7,
    freeze_backbone: bool = True,
    class_names: List[str] = None,
    device: Optional[torch.device] = None
) -> DINOv2Classification:
    """
    Create and initialize DINOv2 classification model.
    
    Args:
        model_name: DINOv2 model variant
        num_classes: Number of terrain classes
        freeze_backbone: Whether to freeze backbone
        class_names: List of class names
        device: Device to place model on
    
    Returns:
        Initialized classification model
    """
    model = DINOv2Classification(
        model_name=model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        class_names=class_names
    )
    
    if device:
        model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    logger.info(
        f"Classification model created with {total_params:,} total parameters, "
        f"{trainable_params:,} trainable"
    )
    
    return model
