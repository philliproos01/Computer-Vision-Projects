"""
timm-backed geometry-first document unwrapper.

This model predicts a dense sampling grid, not a texture. The dewarped image is
created by torch.nn.functional.grid_sample(input_image, predicted_grid), which
keeps the task focused on non-rigid geometry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_CACHE = Path.cwd() / ".cache"
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def create_base_grid(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a grid_sample grid shaped [B, H, W, 2] in normalized [-1, 1] coords."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)


def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((x, skip), dim=1))


class ResidualRefineBlock(nn.Module):
    """Zero-initialized high-resolution refinement that starts as identity."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TimmGeometryUnwrapper(nn.Module):
    """
    EfficientNet/Swin/etc. encoder from timm plus U-Net decoder.

    The final head predicts residual flow over an identity grid:
        grid = identity_grid + tanh(flow_head(features)) * max_displacement

    Output dict:
        image: dewarped RGB tensor, same normalization as input
        grid:  [B, H, W, 2] normalized grid used by grid_sample
        flow:  [B, 2, H, W] normalized residual displacement
        uv:    [B, 2, H, W] grid converted from [-1, 1] to [0, 1]
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        max_displacement: float = 2.0,
        decoder_refine_blocks: int = 0,
        head_channels: int = 32,
        head_depth: int = 1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("Install timm to use TimmGeometryUnwrapper: pip install timm") from exc

        self.backbone_name = backbone
        self.max_displacement = float(max_displacement)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=3,
            out_indices=(0, 1, 2, 3, 4),
        )
        encoder_channels = list(self.encoder.feature_info.channels())
        if len(encoder_channels) < 5:
            raise ValueError(f"{backbone} did not expose 5 feature maps via timm features_only")

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        blocks: List[nn.Module] = []
        in_channels = encoder_channels[-1]
        for skip_channels, out_channels in zip(reversed(encoder_channels[:-1]), decoder_channels):
            blocks.append(UpBlock(in_channels, skip_channels, out_channels))
            in_channels = out_channels
        self.decoder = nn.ModuleList(blocks)

        refine_layers: List[nn.Module] = []
        for _ in range(max(0, int(decoder_refine_blocks))):
            refine_layers.append(ResidualRefineBlock(decoder_channels[-1]))
        self.refine = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()

        # CoordConv-style channels tell the head where each output pixel lives
        # in the flat document canvas.
        head_channels = int(head_channels)
        head_depth = max(1, int(head_depth))
        head_in = decoder_channels[-1] + 2
        head_layers: List[nn.Module] = [ConvBlock(head_in, head_channels)]
        for _ in range(head_depth - 1):
            head_layers.append(ResidualRefineBlock(head_channels))
        head_layers.append(nn.Conv2d(head_channels, 2, kernel_size=3, padding=1))
        self.head = nn.Sequential(*head_layers)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, return_debug: bool = False) -> Dict[str, Any]:
        batch_size, _, height, width = x.shape
        features = self.encoder(x)
        decoded = features[-1]
        decoder_features: List[torch.Tensor] = []
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            decoded = block(decoded, skip)
            if return_debug:
                decoder_features.append(decoded)
        decoded = F.interpolate(decoded, size=(height, width), mode="bilinear", align_corners=False)
        decoded = self.refine(decoded)

        base_grid = create_base_grid(batch_size, height, width, x.device, x.dtype)
        coords = base_grid.permute(0, 3, 1, 2)
        raw_flow = self.head(torch.cat((decoded, coords), dim=1))
        flow = torch.tanh(raw_flow) * self.max_displacement
        grid = base_grid + flow.permute(0, 2, 3, 1)

        image = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        uv = ((grid.permute(0, 3, 1, 2) + 1.0) * 0.5).clamp(0.0, 1.0)
        outputs: Dict[str, Any] = {
            "image": image,
            "grid": grid,
            "flow": flow,
            "uv": uv,
        }
        if return_debug:
            outputs["debug"] = {
                "encoder_features": [feature.detach() for feature in features],
                "decoder_features": [feature.detach() for feature in decoder_features],
                "decoded_full_resolution": decoded.detach(),
                "base_grid": base_grid.detach(),
                "coords": coords.detach(),
                "raw_flow": raw_flow.detach(),
            }
        return outputs


def build_model(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    max_displacement: float = 2.0,
    decoder_refine_blocks: int = 0,
    head_channels: int = 32,
    head_depth: int = 1,
    freeze_encoder: bool = False,
) -> TimmGeometryUnwrapper:
    return TimmGeometryUnwrapper(
        backbone=backbone,
        pretrained=pretrained,
        max_displacement=max_displacement,
        decoder_refine_blocks=decoder_refine_blocks,
        head_channels=head_channels,
        head_depth=head_depth,
        freeze_encoder=freeze_encoder,
    )
