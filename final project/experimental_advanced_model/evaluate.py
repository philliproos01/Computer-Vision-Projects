"""
Evaluate a checkpoint from train.py.

Example:
    .\\venv\\Scripts\\python.exe evaluate.py ^
        --checkpoint timm_geometry_outputs\\best_timm_geometry.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from dataset_loader import get_dataloaders
from model import build_model
from train import (
    SSIMLoss,
    choose_device,
    compute_losses,
    resolve_data_dir,
    save_preview,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate timm geometry unwrapper")
    parser.add_argument("--checkpoint", default="timm_geometry_outputs/best_timm_geometry.pth")
    parser.add_argument("--data-dir", default="auto")
    parser.add_argument("--output-dir", default="timm_geometry_eval")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--preview-samples", type=int, default=3)
    return parser.parse_args()


def add_metrics(running: Dict[str, float], values: Dict[str, float]) -> None:
    for key, value in values.items():
        running[key] = running.get(key, 0.0) + float(value)


@torch.no_grad()
def evaluate(model, loader, loss_args, device: torch.device, limit: Optional[int]) -> Dict[str, float]:
    model.eval()
    ssim = SSIMLoss()
    running: Dict[str, float] = {}
    steps = 0
    for batch_idx, batch in enumerate(loader):
        if limit is not None and batch_idx >= limit:
            break
        rgb = batch["rgb"].to(device)
        outputs = model(rgb)
        _, metrics = compute_losses(outputs, batch, loss_args, ssim, device)
        add_metrics(running, metrics)
        steps += 1
    return {key: value / max(steps, 1) for key, value in running.items()}


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint["args"]

    seed = args.seed if args.seed is not None else int(ckpt_args.get("seed", 42))
    set_seed(seed)
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = tuple(args.img_size or ckpt_args.get("img_size", (256, 256)))
    batch_size = int(args.batch_size or ckpt_args.get("batch_size", 4))
    train_split = float(args.train_split or ckpt_args.get("train_split", 0.8))
    device = choose_device(args.device)

    _, val_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        train_split=train_split,
        use_depth=False,
        use_uv=True,
        use_border=True,
        img_size=img_size,
        num_workers=args.num_workers,
        shuffle=False,
        random_seed=seed,
    )

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

    loss_args = argparse.Namespace(**ckpt_args)
    metrics = evaluate(model, val_loader, loss_args, device, args.limit_val_batches)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "data_dir": str(data_dir),
                "img_size": list(img_size),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    preview_batch = next(iter(val_loader))
    loss_args.preview_samples = args.preview_samples
    save_preview(output_dir / "validation_preview.png", model, preview_batch, loss_args, device)

    print(f"Total loss: {metrics['total']:.4f}")
    print(f"Grid loss: {metrics['grid']:.4f}")
    print(f"SSIM: {metrics['ssim_score']:.4f}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
