"""
Train the timm-based geometry-first document unwrapper.

This training script uses UV maps as direct supervision for the predicted
sampling grid. The RGB image is still the only model input; UV is a training
target that teaches the model the dense non-rigid correspondence.

Recommended first full run:
    .\\venv\\Scripts\\python.exe train.py --lr 1e-4 --epochs 30 --batch-size 4

To try the more aggressive learning rate the project brief mentioned:
    .\\venv\\Scripts\\python.exe train.py --lr 1e-3 --encoder-lr-scale 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_loader import get_dataloaders
from model import build_model, create_base_grid, denormalize_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train timm geometry-first document unwrapper")
    parser.add_argument("--data-dir", default="auto")
    parser.add_argument("--output-dir", default="timm_geometry_outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--max-displacement", type=float, default=2.0)
    parser.add_argument(
        "--decoder-refine-blocks",
        type=int,
        default=0,
        help="Extra full-resolution residual refinement blocks after the U-Net decoder. Use 3 for sharper local geometry.",
    )
    parser.add_argument(
        "--head-channels",
        type=int,
        default=32,
        help="Channels in the final flow/grid prediction head. Larger values add local geometry capacity.",
    )
    parser.add_argument(
        "--head-depth",
        type=int,
        default=1,
        help="Number of flow/grid head blocks before the final 2-channel prediction.",
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="Use 1e-4 or try 1e-3")
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.25,
        help="Pretrained encoder LR = lr * this scale.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--grid-weight", type=float, default=10.0)
    parser.add_argument(
        "--grid-grad-weight",
        type=float,
        default=0.0,
        help="Match local x/y derivatives of predicted and UV-supervised grids to preserve word geometry.",
    )
    parser.add_argument(
        "--grid-grad-target-smooth",
        type=int,
        default=0,
        help="Odd kernel size for smoothing only the target grid used by grid-gradient loss. 0 disables it.",
    )
    parser.add_argument(
        "--grid-grad-target-smooth-passes",
        type=int,
        default=1,
        help="Number of mask-normalized smoothing passes for the grid-gradient target.",
    )
    parser.add_argument(
        "--edge-grid-weight",
        type=float,
        default=0.0,
        help="Extra grid supervision weighted toward strong UV-oracle text/document edges.",
    )
    parser.add_argument("--recon-weight", type=float, default=0.25)
    parser.add_argument("--ssim-weight", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.10)
    parser.add_argument(
        "--oracle-recon-weight",
        type=float,
        default=0.0,
        help="Photometric loss against a UV-grid-sampled oracle from the input image.",
    )
    parser.add_argument(
        "--oracle-ssim-weight",
        type=float,
        default=0.0,
        help="SSIM loss against a UV-grid-sampled oracle from the input image.",
    )
    parser.add_argument(
        "--oracle-edge-weight",
        type=float,
        default=0.0,
        help="Edge loss against a UV-grid-sampled oracle from the input image.",
    )
    parser.add_argument("--smoothness-weight", type=float, default=0.005)
    parser.add_argument(
        "--foldover-weight",
        type=float,
        default=0.0,
        help="Penalize collapsed or flipped local sampling-grid cells.",
    )
    parser.add_argument(
        "--jacobian-min",
        type=float,
        default=0.02,
        help="Minimum normalized local area allowed by the foldover loss.",
    )
    parser.add_argument(
        "--bend-weight",
        type=float,
        default=0.0,
        help="Second-order grid bending regularization for smoother local geometry.",
    )
    parser.add_argument(
        "--mask-sample-weight",
        type=float,
        default=1.0,
        help="Penalize predicted grids that sample background instead of document pixels.",
    )
    parser.add_argument(
        "--mask-source",
        choices=("border", "uv", "none"),
        default="border",
        help="Source mask used by mask-sample loss. 'uv' is often cleaner than border masks.",
    )
    parser.add_argument(
        "--inverse-grid-min-weight",
        type=float,
        default=0.05,
        help="Minimum bilinear splat weight for a UV target-grid cell to be trusted before hole filling.",
    )
    parser.add_argument("--no-uv-normalize", dest="uv_normalize", action="store_false", default=True)
    parser.add_argument("--no-flip-uv-v", dest="flip_uv_v", action="store_false", default=True)

    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load model weights from --resume but start a fresh optimizer.",
    )
    parser.add_argument(
        "--allow-partial-resume",
        action="store_true",
        help="Load checkpoint weights with strict=False, useful after adding decoder refinement blocks.",
    )
    parser.add_argument(
        "--reset-best-on-resume",
        action="store_true",
        help="When changing the loss/target definition, ignore the resumed checkpoint's old best validation loss.",
    )
    parser.add_argument(
        "--reset-history-on-resume",
        action="store_true",
        help="Start fresh loss curves after loading weights from --resume.",
    )
    parser.add_argument(
        "--restart-epoch-on-resume",
        action="store_true",
        help="Use resumed weights as initialization but start epoch numbering from 1.",
    )
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--preview-samples", type=int, default=3)
    return parser.parse_args()


def resolve_data_dir(data_dir: str) -> Path:
    if data_dir != "auto":
        path = Path(data_dir)
        if not path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        return path
    for path in (Path("renders/synthetic_data_pitch_sweep"), Path("renders/renders/synthetic_data_pitch_sweep")):
        if (path / "rgb").exists() and (path / "uv").exists():
            return path
    raise FileNotFoundError("Could not find synthetic_data_pitch_sweep dataset.")


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if mask.shape[-2:] != values.shape[-2:]:
        mask = F.interpolate(mask, size=values.shape[-2:], mode="nearest")
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape[1] == 1 and values.shape[1] > 1:
        mask = mask.expand(-1, values.shape[1], -1, -1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def charbonnier(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_mean(torch.sqrt((pred - target).pow(2) + 1e-6), mask)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def _window(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(self.window_size, device=device, dtype=dtype) - self.window_size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        kernel = gaussian[:, None] * gaussian[None, :]
        return kernel.view(1, 1, self.window_size, self.window_size).expand(channels, 1, -1, -1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        channels = pred.shape[1]
        window = self._window(channels, pred.device, pred.dtype)
        pad = self.window_size // 2
        mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_y = F.conv2d(target, window, padding=pad, groups=channels)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x2
        sigma_y = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y2
        sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        return 1.0 - masked_mean(ssim_map, mask)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    return torch.cat((F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)), dim=1)


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def grid_gradient_loss(
    pred_grid: torch.Tensor,
    target_grid: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Supervise local grid derivatives so text rows/columns keep their shape."""
    pred = pred_grid.permute(0, 3, 1, 2)
    target = target_grid.permute(0, 3, 1, 2)
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    if mask is None:
        mask_x = None
        mask_y = None
    else:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    return charbonnier(pred_dx, target_dx, mask_x) + charbonnier(pred_dy, target_dy, mask_y)


def smooth_grid_target(
    grid: torch.Tensor,
    mask: Optional[torch.Tensor],
    kernel_size: int = 0,
    passes: int = 1,
) -> torch.Tensor:
    """Mask-normalized smoothing for derivative targets only.

    Direct grid loss still uses the raw UV-derived target. This function only
    removes speckle from the target used for local derivative supervision.
    """
    kernel_size = int(kernel_size)
    passes = max(0, int(passes))
    if kernel_size <= 1 or passes <= 0:
        return grid
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2

    grid_chw = grid.permute(0, 3, 1, 2)
    if mask is None:
        smoothed = grid_chw
        for _ in range(passes):
            smoothed = F.avg_pool2d(smoothed, kernel_size, stride=1, padding=padding)
        return smoothed.permute(0, 2, 3, 1)

    valid = mask[:, :1].to(device=grid.device, dtype=grid.dtype)
    if valid.shape[-2:] != grid_chw.shape[-2:]:
        valid = F.interpolate(valid, size=grid_chw.shape[-2:], mode="nearest")
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=grid.device, dtype=grid.dtype)

    smoothed = grid_chw
    for _ in range(passes):
        counts = F.conv2d(valid, kernel, padding=padding).clamp_min(1.0)
        sx = F.conv2d(smoothed[:, 0:1] * valid, kernel, padding=padding) / counts
        sy = F.conv2d(smoothed[:, 1:2] * valid, kernel, padding=padding) / counts
        candidate = torch.cat((sx, sy), dim=1)
        smoothed = torch.where(valid.bool().expand_as(smoothed), candidate, smoothed)
    return smoothed.permute(0, 2, 3, 1)


def normalized_edge_weight(image01: torch.Tensor) -> torch.Tensor:
    """Return a detached [B,1,H,W] weight map emphasizing text/document edges."""
    edges = sobel_edges(image01.float())
    magnitude = torch.sqrt(edges[:, 0:1].pow(2) + edges[:, 1:2].pow(2) + 1e-8)
    flat = magnitude.flatten(1)
    scale = torch.quantile(flat, 0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-4)
    return (magnitude / scale).clamp(0.0, 1.0).to(dtype=image01.dtype).detach()


def grid_cell_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return mask for grid cells whose four corners are valid."""
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    return (
        mask[:, :, :-1, :-1]
        * mask[:, :, :-1, 1:]
        * mask[:, :, 1:, :-1]
        * mask[:, :, 1:, 1:]
    )


def foldover_loss(grid: torch.Tensor, mask: Optional[torch.Tensor], min_area: float = 0.02) -> torch.Tensor:
    """
    Penalize local sampling-grid cells that collapse or flip.

    The raw determinant is tiny in normalized [-1, 1] coordinates, so it is
    scaled such that an identity grid has area close to 1.
    """
    height, width = grid.shape[1:3]
    dx_du = grid[:, :-1, 1:, 0] - grid[:, :-1, :-1, 0]
    dy_du = grid[:, :-1, 1:, 1] - grid[:, :-1, :-1, 1]
    dx_dv = grid[:, 1:, :-1, 0] - grid[:, :-1, :-1, 0]
    dy_dv = grid[:, 1:, :-1, 1] - grid[:, :-1, :-1, 1]
    determinant = dx_du * dy_dv - dx_dv * dy_du
    determinant = determinant * ((height - 1) * (width - 1) / 4.0)
    penalty = F.relu(float(min_area) - determinant).unsqueeze(1)
    return masked_mean(penalty, grid_cell_mask(mask) if mask is not None else None)


def grid_bending_energy(grid: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Second-order smoothness for the sampling grid."""
    grid_chw = grid.permute(0, 3, 1, 2)
    dxx = grid_chw[:, :, :, 2:] - 2.0 * grid_chw[:, :, :, 1:-1] + grid_chw[:, :, :, :-2]
    dyy = grid_chw[:, :, 2:, :] - 2.0 * grid_chw[:, :, 1:-1, :] + grid_chw[:, :, :-2, :]
    if mask is None:
        return dxx.abs().mean() + dyy.abs().mean()
    mask_x = mask[:, :, :, 1:-1]
    mask_y = mask[:, :, 1:-1, :]
    return masked_mean(dxx.abs(), mask_x) + masked_mean(dyy.abs(), mask_y)


def fill_grid_holes(grid: torch.Tensor, valid: torch.Tensor, iterations: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel = torch.ones((1, 1, 3, 3), device=grid.device, dtype=grid.dtype)
    for _ in range(iterations):
        missing = valid < 0.5
        if not bool(missing.any()):
            break
        counts = F.conv2d(valid, kernel, padding=1)
        sx = F.conv2d(grid[:, 0:1] * valid, kernel, padding=1)
        sy = F.conv2d(grid[:, 1:2] * valid, kernel, padding=1)
        candidate = torch.cat((sx, sy), dim=1) / counts.clamp_min(1.0)
        fill = missing & (counts > 0)
        grid = torch.where(fill.expand_as(grid), candidate, grid)
        valid = torch.where(fill, torch.ones_like(valid), valid)
    return grid, valid


@torch.no_grad()
def make_inverse_sampling_grid(
    uv: torch.Tensor,
    uv_mask: torch.Tensor,
    normalize_uv: bool = True,
    flip_v: bool = True,
    min_splat_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dataset UV maps are input-pixel -> flat-paper coordinates. grid_sample needs
    flat-output-pixel -> input-pixel coordinates, so we scatter source pixel
    locations into UV space and fill holes.
    """
    batch_size, _, height, width = uv.shape
    source_grid = create_base_grid(batch_size, height, width, uv.device, uv.dtype)
    target_grids = []
    target_masks = []

    for idx in range(batch_size):
        u = uv[idx, 0].float()
        v = uv[idx, 1].float()
        valid = uv_mask[idx, 0].bool() & torch.isfinite(u) & torch.isfinite(v)
        if int(valid.sum()) < 16:
            identity = source_grid[idx].permute(2, 0, 1).unsqueeze(0)
            target_grids.append(identity)
            target_masks.append(torch.ones((1, 1, height, width), device=uv.device, dtype=uv.dtype))
            continue

        if normalize_uv:
            u_valid = u[valid]
            v_valid = v[valid]
            u = (u - u_valid.min()) / (u_valid.max() - u_valid.min()).clamp_min(1e-6)
            v = (v - v_valid.min()) / (v_valid.max() - v_valid.min()).clamp_min(1e-6)
        if flip_v:
            v = 1.0 - v

        valid = valid & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        fx = u[valid] * (width - 1)
        fy = v[valid] * (height - 1)
        x0 = torch.floor(fx).long().clamp(0, width - 1)
        y0 = torch.floor(fy).long().clamp(0, height - 1)
        x1 = (x0 + 1).clamp(0, width - 1)
        y1 = (y0 + 1).clamp(0, height - 1)
        wx = (fx - x0.float()).clamp(0.0, 1.0)
        wy = (fy - y0.float()).clamp(0.0, 1.0)
        weights = (
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        )

        source_values = source_grid[idx][valid].float()
        accum = torch.zeros((height * width, 2), device=uv.device)
        counts = torch.zeros((height * width, 1), device=uv.device)
        for xs, ys, weight in weights:
            flat_index = ys * width + xs
            weight = weight.unsqueeze(1)
            accum.index_add_(0, flat_index, source_values * weight)
            counts.index_add_(0, flat_index, weight)

        grid = (accum / counts.clamp_min(1.0)).view(height, width, 2).permute(2, 0, 1).unsqueeze(0)
        valid_out = (
            counts.view(height, width, 1) > float(min_splat_weight)
        ).float().permute(2, 0, 1).unsqueeze(0)
        grid, valid_out = fill_grid_holes(grid.to(uv.dtype), valid_out.to(uv.dtype))
        identity = source_grid[idx].permute(2, 0, 1).unsqueeze(0)
        grid = torch.where(valid_out.bool().expand_as(grid), grid, identity)
        target_grids.append(grid)
        target_masks.append(valid_out)

    return (
        torch.cat(target_grids, dim=0).permute(0, 2, 3, 1).to(uv.dtype),
        torch.cat(target_masks, dim=0).to(uv.dtype),
    )


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], args, ssim: SSIMLoss, device):
    target_grid, target_mask = make_inverse_sampling_grid(
        batch["uv"].to(device),
        batch["uv_mask"].to(device),
        normalize_uv=args.uv_normalize,
        flip_v=args.flip_uv_v,
        min_splat_weight=args.inverse_grid_min_weight,
    )
    pred_grid_chw = outputs["grid"].permute(0, 3, 1, 2)
    target_grid_chw = target_grid.permute(0, 3, 1, 2)
    target_grid_for_grad = smooth_grid_target(
        target_grid,
        target_mask,
        kernel_size=getattr(args, "grid_grad_target_smooth", 0),
        passes=getattr(args, "grid_grad_target_smooth_passes", 1),
    )

    pred01 = denormalize_image(outputs["image"])
    target01 = denormalize_image(batch["ground_truth"].to(device))
    oracle_norm = F.grid_sample(
        batch["rgb"].to(device),
        target_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    oracle01 = denormalize_image(oracle_norm)
    mask_source = getattr(args, "mask_source", "border")
    if mask_source == "uv":
        source_mask = batch.get("uv_mask")
    elif mask_source == "none":
        source_mask = None
    else:
        source_mask = batch.get("border")
    if source_mask is not None and getattr(args, "mask_sample_weight", 0.0) != 0.0:
        source_mask = source_mask.to(device=device, dtype=pred01.dtype)
        sampled_source_mask = F.grid_sample(
            source_mask,
            outputs["grid"],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        mask_sample_loss = charbonnier(sampled_source_mask, torch.ones_like(sampled_source_mask), target_mask)
    else:
        mask_sample_loss = pred_grid_chw.new_tensor(0.0)
    edge_weight = normalized_edge_weight(oracle01)

    losses = {
        "grid": charbonnier(pred_grid_chw, target_grid_chw, target_mask),
        "grid_grad": grid_gradient_loss(outputs["grid"], target_grid_for_grad, target_mask),
        "edge_grid": charbonnier(pred_grid_chw, target_grid_chw, target_mask * edge_weight),
        "recon": charbonnier(pred01, target01, target_mask),
        "ssim": ssim(pred01, target01, target_mask),
        "edge": charbonnier(sobel_edges(pred01), sobel_edges(target01), target_mask),
        "oracle_recon": charbonnier(pred01, oracle01, target_mask),
        "oracle_ssim": ssim(pred01, oracle01, target_mask),
        "oracle_edge": charbonnier(sobel_edges(pred01), sobel_edges(oracle01), target_mask),
        "smoothness": flow_smoothness(outputs["flow"]),
        "foldover": foldover_loss(outputs["grid"], target_mask, min_area=args.jacobian_min),
        "bend": grid_bending_energy(outputs["grid"], target_mask),
        "mask_sample": mask_sample_loss,
    }
    total = (
        args.grid_weight * losses["grid"]
        + getattr(args, "grid_grad_weight", 0.0) * losses["grid_grad"]
        + getattr(args, "edge_grid_weight", 0.0) * losses["edge_grid"]
        + args.recon_weight * losses["recon"]
        + args.ssim_weight * losses["ssim"]
        + args.edge_weight * losses["edge"]
        + args.oracle_recon_weight * losses["oracle_recon"]
        + args.oracle_ssim_weight * losses["oracle_ssim"]
        + args.oracle_edge_weight * losses["oracle_edge"]
        + args.smoothness_weight * losses["smoothness"]
        + args.foldover_weight * losses["foldover"]
        + args.bend_weight * losses["bend"]
        + args.mask_sample_weight * losses["mask_sample"]
    )
    metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
    metrics["total"] = float(total.detach().cpu())
    metrics["ssim_score"] = 1.0 - metrics["ssim"]
    metrics["oracle_ssim_score"] = 1.0 - metrics["oracle_ssim"]
    return total, metrics


def limited(loader: Iterable, limit: Optional[int]) -> Iterable:
    for idx, batch in enumerate(loader):
        if limit is not None and idx >= limit:
            break
        yield idx, batch


def average(metrics_sum: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: value / max(count, 1) for key, value in metrics_sum.items()}


def add_metrics(metrics_sum: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        metrics_sum[key] = metrics_sum.get(key, 0.0) + value


def optimizer_for(model: nn.Module, args) -> torch.optim.Optimizer:
    encoder_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(parameter)
        else:
            other_params.append(parameter)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.lr * args.encoder_lr_scale})
    if other_params:
        groups.append({"params": other_params, "lr": args.lr})
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def train_epoch(model, loader, optimizer, scaler, ssim, args, device, epoch):
    model.train()
    sums: Dict[str, float] = {}
    steps = 0
    start = time.time()
    use_amp = bool(args.amp and device.type == "cuda")

    for batch_idx, batch in limited(loader, args.limit_train_batches):
        rgb = batch["rgb"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(rgb)
            loss, metrics = compute_losses(outputs, batch, args, ssim, device)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        add_metrics(sums, metrics)
        steps += 1
        if batch_idx % 25 == 0:
            print(
                f"epoch {epoch:03d} step {batch_idx:04d} "
                f"total={metrics['total']:.4f} grid={metrics['grid']:.4f} "
                f"grid_grad={metrics['grid_grad']:.4f} "
                f"mask={metrics['mask_sample']:.4f} "
                f"fold={metrics['foldover']:.4f} "
                f"ssim={metrics['ssim_score']:.4f} ({time.time() - start:.1f}s)"
            )
    return average(sums, steps)


@torch.no_grad()
def validate(model, loader, ssim, args, device):
    model.eval()
    sums: Dict[str, float] = {}
    steps = 0
    for _, batch in limited(loader, args.limit_val_batches):
        rgb = batch["rgb"].to(device)
        outputs = model(rgb)
        _, metrics = compute_losses(outputs, batch, args, ssim, device)
        add_metrics(sums, metrics)
        steps += 1
    return average(sums, steps)


def to_uint8(image01: torch.Tensor) -> np.ndarray:
    arr = image01.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


@torch.no_grad()
def save_preview(path: Path, model, batch, args, device) -> None:
    model.eval()
    rgb = batch["rgb"].to(device)
    outputs = model(rgb)
    inputs = denormalize_image(rgb)
    preds = denormalize_image(outputs["image"])
    targets = denormalize_image(batch["ground_truth"].to(device))
    target_grid, target_mask = make_inverse_sampling_grid(
        batch["uv"].to(device),
        batch["uv_mask"].to(device),
        normalize_uv=args.uv_normalize,
        flip_v=args.flip_uv_v,
        min_splat_weight=args.inverse_grid_min_weight,
    )
    oracle = F.grid_sample(rgb, target_grid, mode="bilinear", padding_mode="border", align_corners=True)
    oracle = denormalize_image(oracle)

    rows = []
    count = min(args.preview_samples, rgb.shape[0])
    for idx in range(count):
        diff = (preds[idx] - oracle[idx]).abs().mul(2.0).clamp(0.0, 1.0)
        row = np.concatenate(
            [
                to_uint8(inputs[idx]),
                to_uint8(preds[idx]),
                to_uint8(oracle[idx]),
                to_uint8(diff),
                to_uint8(targets[idx]),
            ],
            axis=1,
        )
        rows.append(row)
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def save_history(history: Dict[str, list], output_dir: Path) -> None:
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        epochs = history["epoch"]
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, history["train_total"], label="train total")
        plt.plot(epochs, history["val_total"], label="val total")
        plt.plot(epochs, history["val_grid"], label="val grid")
        if "val_grid_grad" in history:
            plt.plot(epochs[-len(history["val_grid_grad"]):], history["val_grid_grad"], label="val grid grad")
        if "val_edge_grid" in history:
            plt.plot(epochs[-len(history["val_edge_grid"]):], history["val_edge_grid"], label="val edge grid")
        if "val_mask_sample" in history:
            plt.plot(epochs[-len(history["val_mask_sample"]):], history["val_mask_sample"], label="val mask sample")
        if "val_foldover" in history:
            plt.plot(epochs[-len(history["val_foldover"]):], history["val_foldover"], label="val foldover")
        if "val_oracle_ssim_score" in history:
            plt.plot(
                epochs[-len(history["val_oracle_ssim_score"]):],
                history["val_oracle_ssim_score"],
                label="val oracle SSIM",
            )
        plt.plot(epochs, history["val_ssim_score"], label="val SSIM")
        plt.xlabel("epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curves.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"Could not save loss curve: {exc}")


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, args, history) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
            "model_file": "model.py",
            "loader_file": "dataset_loader.py",
        },
        path,
    )


def load_compatible_model_state(model: nn.Module, checkpoint_state: Dict[str, torch.Tensor]) -> Tuple[list, list]:
    """Load only checkpoint tensors whose names and shapes match the current model."""
    current_state = model.state_dict()
    compatible = {}
    skipped = []
    for name, tensor in checkpoint_state.items():
        if name in current_state and current_state[name].shape == tensor.shape:
            compatible[name] = tensor
        else:
            skipped.append(name)
    missing = [name for name in current_state if name not in compatible]
    current_state.update(compatible)
    model.load_state_dict(current_state)
    return missing, skipped


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    print(f"Using data: {data_dir}")
    print(f"Using device: {device}")
    print(f"Backbone: timm/{args.backbone}, pretrained={args.pretrained}")
    print(f"Learning rate: decoder/head={args.lr:g}, encoder={args.lr * args.encoder_lr_scale:g}")

    train_loader, val_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        train_split=args.train_split,
        use_depth=False,
        use_uv=True,
        use_border=True,
        img_size=tuple(args.img_size),
        num_workers=args.num_workers,
        shuffle=True,
        random_seed=args.seed,
    )

    model = build_model(
        backbone=args.backbone,
        pretrained=args.pretrained,
        max_displacement=args.max_displacement,
        decoder_refine_blocks=args.decoder_refine_blocks,
        head_channels=args.head_channels,
        head_depth=args.head_depth,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    optimizer = optimizer_for(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    ssim = SSIMLoss()

    history = {
        "epoch": [],
        "train_total": [],
        "val_total": [],
        "val_grid": [],
        "val_grid_grad": [],
        "val_edge_grid": [],
        "val_mask_sample": [],
        "val_foldover": [],
        "val_oracle_ssim_score": [],
        "val_ssim_score": [],
    }
    start_epoch = 1
    best_val = math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if args.allow_partial_resume:
            missing, skipped = load_compatible_model_state(model, checkpoint["model_state_dict"])
            print(f"Partially loaded compatible model weights: missing={len(missing)}, skipped={len(skipped)}")
            if missing:
                print(f"  First missing keys: {missing[:5]}")
            if skipped:
                print(f"  First skipped incompatible keys: {skipped[:5]}")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_weights_only or args.allow_partial_resume:
            print("Started a fresh optimizer after loading model weights.")
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.reset_history_on_resume:
            print("Reset training history after resume.")
        else:
            history = checkpoint.get("history", history)
        if args.reset_best_on_resume:
            best_val = math.inf
            print("Reset best validation loss after resume.")
        else:
            best_val = float(checkpoint.get("best_val_loss", math.inf))
        if args.restart_epoch_on_resume:
            start_epoch = 1
            print("Restarted epoch numbering after resume.")
        else:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    preview_batch = next(iter(val_loader))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, ssim, args, device, epoch)
        val_metrics = validate(model, val_loader, ssim, args, device)
        scheduler.step(val_metrics["total"])

        history["epoch"].append(epoch)
        history["train_total"].append(train_metrics["total"])
        history["val_total"].append(val_metrics["total"])
        history["val_grid"].append(val_metrics["grid"])
        history.setdefault("val_grid_grad", []).append(val_metrics["grid_grad"])
        history.setdefault("val_edge_grid", []).append(val_metrics["edge_grid"])
        history.setdefault("val_mask_sample", []).append(val_metrics["mask_sample"])
        history.setdefault("val_foldover", []).append(val_metrics["foldover"])
        history.setdefault("val_oracle_ssim_score", []).append(val_metrics["oracle_ssim_score"])
        history["val_ssim_score"].append(val_metrics["ssim_score"])

        print(
            f"epoch {epoch:03d} summary: "
            f"train={train_metrics['total']:.4f} "
            f"val={val_metrics['total']:.4f} "
            f"val_grid={val_metrics['grid']:.4f} "
            f"val_grid_grad={val_metrics['grid_grad']:.4f} "
            f"val_edge_grid={val_metrics['edge_grid']:.4f} "
            f"val_mask={val_metrics['mask_sample']:.4f} "
            f"val_fold={val_metrics['foldover']:.4f} "
            f"val_oracle_ssim={val_metrics['oracle_ssim_score']:.4f} "
            f"val_ssim={val_metrics['ssim_score']:.4f}"
        )

        save_checkpoint(output_dir / "latest_timm_geometry.pth", model, optimizer, epoch, best_val, args, history)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(output_dir / "best_timm_geometry.pth", model, optimizer, epoch, best_val, args, history)
            save_preview(output_dir / "best_preview.png", model, preview_batch, args, device)
            print(f"Saved new best: {output_dir / 'best_timm_geometry.pth'}")
        save_history(history, output_dir)

    print("Training complete.")
    print(f"Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
