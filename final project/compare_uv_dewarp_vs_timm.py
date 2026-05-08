from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from dataset_loader_geometry import DocumentDataset
from inference_timm_geometry import (
    _resize_rgb,
    _tensor_rgb01_to_uint8,
    _uv_grid_lines,
    preprocess,
    render_with_grid,
)
from model_timm_geometry import build_model, denormalize_image
from train_timm_geometry import choose_device, make_inverse_sampling_grid, resolve_data_dir
from uv_dewarp import dewarp_with_uv, load_rgb, load_uv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare timm model dewarp against UV dewarp oracle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="renders/synthetic_data_pitch_sweep")
    parser.add_argument("--output-dir", default="compare_uv_vs_timm")
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--output-size", type=int, default=768)
    parser.add_argument("--samples", nargs="*", default=None, help="Optional RGB filenames or full paths.")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint["args"]
    model = build_model(
        backbone=ckpt_args.get("backbone", "efficientnet_b0"),
        pretrained=False,
        max_displacement=float(ckpt_args.get("max_displacement", 2.0)),
        decoder_refine_blocks=int(ckpt_args.get("decoder_refine_blocks", 0)),
        head_channels=int(ckpt_args.get("head_channels", 32)),
        head_depth=int(ckpt_args.get("head_depth", 1)),
        freeze_encoder=bool(ckpt_args.get("freeze_encoder", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, ckpt_args


def resolve_samples(data_dir: Path, samples: Iterable[str] | None, count: int) -> List[Path]:
    if samples:
        paths = []
        for sample in samples:
            path = Path(sample)
            if not path.exists():
                path = data_dir / "rgb" / sample
            if not path.exists():
                raise FileNotFoundError(f"Could not find sample: {sample}")
            paths.append(path)
        return paths[:count]
    return sorted((data_dir / "rgb").glob("*.jpg"))[:count]


def sobel_edges(image01: torch.Tensor) -> torch.Tensor:
    if image01.ndim == 3:
        image01 = image01.unsqueeze(0)
    gray = image01[:, 0:1] * 0.299 + image01[:, 1:2] * 0.587 + image01[:, 2:3] * 0.114
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image01.device,
        dtype=image01.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image01.device,
        dtype=image01.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def simple_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean()
    muy = y.mean()
    varx = ((x - mux) ** 2).mean()
    vary = ((y - muy) ** 2).mean()
    cov = ((x - mux) * (y - muy)).mean()
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (varx + vary + c2))
    return float(score.detach().cpu())


def edge_f1(model01: torch.Tensor, oracle01: torch.Tensor) -> Tuple[float, float]:
    model_edges = sobel_edges(model01)
    oracle_edges = sobel_edges(oracle01)
    m_thr = torch.quantile(model_edges.flatten(), 0.85)
    o_thr = torch.quantile(oracle_edges.flatten(), 0.85)
    m = model_edges >= m_thr
    o = oracle_edges >= o_thr
    tp = (m & o).sum().float()
    precision = tp / m.sum().clamp_min(1)
    recall = tp / o.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return float(f1.cpu()), float((model_edges - oracle_edges).abs().mean().cpu())


def grid_metrics(pred_grid: torch.Tensor, target_grid: torch.Tensor, target_mask: torch.Tensor) -> dict:
    mask = target_mask[:, 0].bool()
    delta = pred_grid - target_grid
    height, width = pred_grid.shape[1:3]
    dx_px = delta[..., 0] * (width - 1) / 2.0
    dy_px = delta[..., 1] * (height - 1) / 2.0
    endpoint = torch.sqrt(dx_px * dx_px + dy_px * dy_px + 1e-8)[mask]
    in_bounds = (
        (pred_grid[..., 0] >= -1.0)
        & (pred_grid[..., 0] <= 1.0)
        & (pred_grid[..., 1] >= -1.0)
        & (pred_grid[..., 1] <= 1.0)
    )
    return {
        "grid_endpoint_px_mean": float(endpoint.mean().cpu()),
        "grid_endpoint_px_p50": float(torch.quantile(endpoint, 0.50).cpu()),
        "grid_endpoint_px_p90": float(torch.quantile(endpoint, 0.90).cpu()),
        "grid_endpoint_px_p95": float(torch.quantile(endpoint, 0.95).cpu()),
        "grid_endpoint_px_max": float(endpoint.max().cpu()),
        "grid_in_bounds_ratio": float(in_bounds.float().mean().cpu()),
    }


def add_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font) -> None:
    draw.rectangle((xy[0], xy[1], xy[0] + 260, xy[1] + 24), fill=(255, 255, 255))
    draw.text((xy[0] + 4, xy[1] + 4), text, fill=(0, 0, 0), font=font)


def save_panel(
    path: Path,
    title: str,
    panels: List[Tuple[str, np.ndarray]],
    cell: Tuple[int, int] = (280, 280),
) -> None:
    cols = 3
    rows = math.ceil(len(panels) / cols)
    label_h = 30
    title_h = 36
    canvas = Image.new("RGB", (cols * cell[0], title_h + rows * (cell[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((8, 8), title, fill=(0, 0, 0), font=title_font)
    for idx, (label, arr) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = col * cell[0]
        y = title_h + row * (cell[1] + label_h)
        canvas.paste(Image.fromarray(_resize_rgb(arr, cell)), (x, y))
        add_label(draw, (x, y + cell[1]), label, font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, ckpt_args = load_model(Path(args.checkpoint), device)
    img_size = tuple(args.img_size or ckpt_args.get("img_size", (384, 384)))
    out_size = int(args.output_size)
    samples = resolve_samples(data_dir, args.samples, args.num_samples)
    dataset = DocumentDataset(
        data_dir=str(data_dir),
        use_uv=True,
        use_border=True,
        img_size=img_size,
    )
    by_name = {name: idx for idx, name in enumerate(dataset.samples)}

    all_metrics = []
    for sample_path in samples:
        stem = sample_path.stem
        if stem not in by_name:
            print(f"Skipping unmatched sample: {sample_path}")
            continue
        batch = dataset[by_name[stem]]
        rgb_tensor = preprocess(sample_path, img_size, device)
        with torch.no_grad():
            outputs = model(rgb_tensor, return_debug=False)
            model01 = render_with_grid(sample_path, data_dir, outputs["grid"], (out_size, out_size), device, False, 0.35)
            target_grid, target_mask = make_inverse_sampling_grid(
                batch["uv"].unsqueeze(0).to(device),
                batch["uv_mask"].unsqueeze(0).to(device),
                normalize_uv=True,
                min_splat_weight=0.05,
            )
        rgb = load_rgb(sample_path)
        uv, fg = load_uv(data_dir / "uv" / f"{stem}.png")
        uv_oracle = dewarp_with_uv(rgb, uv, out_size=out_size, mask=fg, flip_v=True)
        uv_oracle01 = torch.from_numpy(uv_oracle.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        gt = np.asarray(Image.open(data_dir / "ground_truth" / f"{stem}.png").convert("RGB").resize((out_size, out_size)))
        gt01 = torch.from_numpy(gt.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        f1, edge_l1 = edge_f1(model01, uv_oracle01)
        metrics = {
            "sample": stem,
            **grid_metrics(outputs["grid"], target_grid, target_mask),
            "model_vs_uv_l1": float((model01 - uv_oracle01).abs().mean().cpu()),
            "model_vs_uv_ssim_global": simple_ssim(model01, uv_oracle01),
            "model_vs_uv_edge_f1_top15pct": f1,
            "model_vs_uv_edge_l1": edge_l1,
            "uv_oracle_vs_gt_ssim_global": simple_ssim(uv_oracle01, gt01),
            "model_vs_gt_ssim_global": simple_ssim(model01, gt01),
        }
        all_metrics.append(metrics)

        model_uint8 = _tensor_rgb01_to_uint8(model01)
        uv_uint8 = uv_oracle
        diff = np.clip(np.abs(model_uint8.astype(np.int16) - uv_uint8.astype(np.int16)) * 2, 0, 255).astype(np.uint8)
        pred_grid_lines = _uv_grid_lines((outputs["grid"].permute(0, 3, 1, 2) + 1.0) * 0.5, (out_size, out_size))
        target_grid_lines = _uv_grid_lines((target_grid.permute(0, 3, 1, 2) + 1.0) * 0.5, (out_size, out_size))
        edge_model = _tensor_rgb01_to_uint8(sobel_edges(model01).repeat(1, 3, 1, 1) / sobel_edges(model01).max().clamp_min(1e-6))
        edge_uv = _tensor_rgb01_to_uint8(sobel_edges(uv_oracle01).repeat(1, 3, 1, 1) / sobel_edges(uv_oracle01).max().clamp_min(1e-6))
        input_view = np.asarray(Image.open(sample_path).convert("RGB"))

        save_panel(
            output_dir / f"{stem}_comparison.png",
            f"{stem} | endpoint mean {metrics['grid_endpoint_px_mean']:.1f}px, p90 {metrics['grid_endpoint_px_p90']:.1f}px",
            [
                ("input warped", input_view),
                ("timm model output", model_uint8),
                ("uv_dewarp oracle", uv_uint8),
                ("ground truth", gt),
                ("model vs uv diff x2", diff),
                ("predicted grid lines", pred_grid_lines),
                ("target uv grid lines", target_grid_lines),
                ("model edges", edge_model),
                ("uv oracle edges", edge_uv),
            ],
        )

    summary = {}
    if all_metrics:
        keys = [k for k in all_metrics[0] if k != "sample"]
        summary = {
            key: float(np.mean([m[key] for m in all_metrics]))
            for key in keys
        }
    (output_dir / "metrics.json").write_text(
        json.dumps({"samples": all_metrics, "mean": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"samples": all_metrics, "mean": summary}, indent=2))
    print(f"Saved comparison panels to {output_dir}")


if __name__ == "__main__":
    main()
