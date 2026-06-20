"""
ResNet18-UNet + ASPP Head
─────────────────────────
Encoder:  ResNet18 (pretrained on ImageNet)
Decoder:  4 up-blocks with skip connections
Head:     ASPP module (inspired by DeSTSeg) for multi-scale feature fusion
Output:   (B, num_classes, H, W) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class DecoderBlock(nn.Module):
    """Upsample + concat skip + conv-bn-relu × 2."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── ASPP Module ──────────────────────────────────

class ASPPConv(nn.Module):
    """Single ASPP branch: atrous conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch of ASPP."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        out = self.pool(x)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (DeepLab v3 style).
    Captures multi-scale context — crucial for detecting disease spots
    of varying sizes.

    Branches: 1×1 conv, 3×3 dilated (rate 6, 12, 18), global pool.
    All concatenated → 1×1 projection → dropout.
    """

    def __init__(self, in_ch: int, out_ch: int = 256, rates: tuple = (6, 12, 18)):
        super().__init__()
        branch_ch = out_ch // (len(rates) + 2)  # distribute channels evenly
        # Adjust so total matches out_ch after concat
        branch_ch = out_ch

        self.branches = nn.ModuleList()
        # 1×1 conv
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        ))
        # Atrous convolutions at different rates
        for rate in rates:
            self.branches.append(ASPPConv(in_ch, branch_ch, dilation=rate))
        # Global pooling
        self.branches.append(ASPPPooling(in_ch, branch_ch))

        # Project concatenated branches back to out_ch
        total_ch = branch_ch * len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        return self.project(x)


# ── Segmentation Head ────────────────────────────

class SegmentationHead(nn.Module):
    """
    Post-decoder segmentation head inspired by DeSTSeg:
      residual block → residual block → ASPP → 1×1 classifier

    Takes decoder output and refines it with multi-scale context
    before final classification.
    """

    def __init__(self, in_ch: int, mid_ch: int = 64, num_classes: int = 2,
                 aspp_rates: tuple = (6, 12, 18)):
        super().__init__()

        # Two residual-style conv blocks for feature refinement
        self.refine = nn.Sequential(
            # Block 1
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # ASPP for multi-scale context
        self.aspp = ASPP(mid_ch, out_ch=mid_ch, rates=aspp_rates)

        # Final classifier
        self.classifier = nn.Conv2d(mid_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refine(x)
        x = self.aspp(x)
        return self.classifier(x)


# ── Full Model ───────────────────────────────────

class ResNet18UNet(nn.Module):
    """
    Encoder channels (ResNet18): 64 → 64 → 128 → 256 → 512
    Decoder mirrors with skip connections.
    Head: SegmentationHead with ASPP (when use_aspp=True) or simple 1×1 conv.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 3,
        use_aspp: bool = True,
        aspp_rates: tuple = (6, 12, 18),
    ):
        super().__init__()
        self.use_aspp = use_aspp

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Encoder stages
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.pool0 = backbone.maxpool                                            # /4
        self.enc1 = backbone.layer1   # /4,  64
        self.enc2 = backbone.layer2   # /8,  128
        self.enc3 = backbone.layer3   # /16, 256
        self.enc4 = backbone.layer4   # /32, 512

        # Adapt first conv if in_channels != 3
        if in_channels != 3:
            self.enc0[0] = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  64,  64)

        # Final upsample to full resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # Segmentation head
        if use_aspp:
            self.head = SegmentationHead(
                in_ch=32, mid_ch=64, num_classes=num_classes, aspp_rates=aspp_rates,
            )
        else:
            self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)      # /2,  64
        p0 = self.pool0(e0)    # /4
        e1 = self.enc1(p0)     # /4,  64
        e2 = self.enc2(e1)     # /8,  128
        e3 = self.enc3(e2)     # /16, 256
        e4 = self.enc4(e3)     # /32, 512

        # Decoder
        d4 = self.dec4(e4, e3)  # /16, 256
        d3 = self.dec3(d4, e2)  # /8,  128
        d2 = self.dec2(d3, e1)  # /4,  64
        d1 = self.dec1(d2, e0)  # /2,  64

        # Final
        out = self.final_up(d1)  # /1, 32
        out = self.head(out)     # /1, num_classes

        # Ensure output matches input spatial dims
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out
