"""
Run inference with a checkpoint from train.py.

Example:
    .\\venv\\Scripts\\python.exe inference.py ^
        --checkpoint timm_geometry_outputs\\best_timm_geometry.pth
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import torch

from model import IMAGENET_MEAN, IMAGENET_STD, build_model, create_base_grid, denormalize_image
from train import choose_device, resolve_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dewarp one image with the timm geometry model")
    parser.add_argument("--checkpoint", default="timm_geometry_outputs/best_timm_geometry.pth")
    parser.add_argument("--input", default=None)
    parser.add_argument("--data-dir", default="auto")
    parser.add_argument("--output", default="timm_dewarped.png")
    parser.add_argument("--uv-output", default="timm_predicted_uv.png")
    parser.add_argument("--flow-output", default="timm_flow_magnitude.png")
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional high-resolution output size. Predicts at checkpoint size, then samples from the original image.",
    )
    parser.add_argument(
        "--mask-cleanup",
        action="store_true",
        help="Sample the border mask through the predicted grid and whiten non-document regions.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.35,
        help="Threshold for mask cleanup after sampling the border mask.",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply document-style contrast and sharpening to the final dewarped image.",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="Average predicted grids from the image and its horizontal flip.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory for step-by-step debug images and a labeled debug panel.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def default_input(data_dir: Path) -> Path:
    first = next(iter(sorted((data_dir / "rgb").glob("*.jpg"))), None)
    if first is None:
        raise FileNotFoundError(f"No jpg samples found in {data_dir / 'rgb'}")
    return first


def preprocess(path: Path, img_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    height, width = img_size
    image = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def image_to_normalized_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(device)


def load_border_mask(input_path: Path, data_dir: Path, size_wh: Tuple[int, int], device: torch.device) -> torch.Tensor | None:
    stem = input_path.stem
    candidates = [
        data_dir / "border" / f"{stem}.png",
        data_dir / "border" / f"{stem}.jpg",
        data_dir / "border" / f"{stem}.jpeg",
    ]
    mask_path = next((path for path in candidates if path.exists()), None)
    if mask_path is None:
        return None
    mask = Image.open(mask_path).convert("L").resize(size_wh, Image.BILINEAR)
    arr = np.asarray(mask).astype(np.float32) / 255.0
    return torch.from_numpy(arr).view(1, 1, size_wh[1], size_wh[0]).to(device)


def upsample_grid(grid: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    height, width = output_size
    grid_chw = grid.permute(0, 3, 1, 2)
    grid_chw = torch.nn.functional.interpolate(
        grid_chw,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )
    return grid_chw.permute(0, 2, 3, 1)


def unflip_grid(grid: torch.Tensor) -> torch.Tensor:
    """Convert a grid predicted from a horizontally flipped input back to original-image coordinates."""
    unflipped = torch.flip(grid, dims=(2,))
    unflipped = unflipped.clone()
    unflipped[..., 0] = -unflipped[..., 0]
    return unflipped


def predict_grid(model, rgb: torch.Tensor, use_tta_flip: bool, return_debug: bool) -> dict:
    outputs = model(rgb, return_debug=return_debug)
    if not use_tta_flip:
        return outputs

    flipped_rgb = torch.flip(rgb, dims=(3,))
    flipped_outputs = model(flipped_rgb, return_debug=False)
    averaged_grid = 0.5 * (outputs["grid"] + unflip_grid(flipped_outputs["grid"]))
    base_grid = create_base_grid(
        averaged_grid.shape[0],
        averaged_grid.shape[1],
        averaged_grid.shape[2],
        averaged_grid.device,
        averaged_grid.dtype,
    )
    outputs = dict(outputs)
    outputs["grid"] = averaged_grid
    outputs["flow"] = (averaged_grid - base_grid).permute(0, 3, 1, 2)
    outputs["uv"] = (averaged_grid.permute(0, 3, 1, 2) + 1.0) * 0.5
    outputs["image"] = torch.nn.functional.grid_sample(
        rgb,
        averaged_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return outputs


def render_with_grid(
    input_path: Path,
    data_dir: Path,
    grid: torch.Tensor,
    output_size: Tuple[int, int],
    device: torch.device,
    mask_cleanup: bool,
    mask_threshold: float,
) -> torch.Tensor:
    height, width = output_size
    source = Image.open(input_path).convert("RGB").resize((width, height), Image.BILINEAR)
    source_tensor = image_to_normalized_tensor(source, device)
    high_grid = upsample_grid(grid, output_size)
    rendered = torch.nn.functional.grid_sample(
        source_tensor,
        high_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    rendered01 = denormalize_image(rendered)

    if mask_cleanup:
        source_mask = load_border_mask(input_path, data_dir, (width, height), device)
        if source_mask is not None:
            sampled_mask = torch.nn.functional.grid_sample(
                source_mask,
                high_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            keep = (sampled_mask >= mask_threshold).to(rendered01.dtype)
            rendered01 = rendered01 * keep + (1.0 - keep)
    return rendered01


def enhance_document(image01: torch.Tensor) -> torch.Tensor:
    arr = _tensor_rgb01_to_uint8(image01)
    image = Image.fromarray(arr).convert("RGB")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(image01.device)


def save_rgb(path: Path, image01: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = image01.detach().cpu().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_uv(path: Path, uv: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uv = uv.detach().cpu().clamp(0.0, 1.0).squeeze(0)
    rgb = torch.cat((uv, torch.zeros_like(uv[:1])), dim=0).permute(1, 2, 0).numpy()
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_flow(path: Path, flow: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    magnitude = flow.detach().cpu().squeeze(0).pow(2).sum(dim=0).sqrt()
    magnitude = magnitude / magnitude.max().clamp_min(1e-6)
    Image.fromarray((magnitude.numpy() * 255.0 + 0.5).astype(np.uint8)).save(path)


def _slug(label: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean or "debug"


def _resize_rgb(arr: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(arr).convert("RGB").resize(size_wh, Image.BILINEAR))


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi - lo < 1e-6:
        lo = float(values.min())
        hi = float(values.max())
    if hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _heatmap(values: np.ndarray) -> np.ndarray:
    v = _normalize01(values)
    red = np.clip(1.55 * v - 0.20, 0.0, 1.0)
    green = np.clip(1.35 - np.abs(v - 0.50) * 2.70, 0.0, 1.0)
    blue = np.clip(1.20 - 1.55 * v, 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255.0 + 0.5).astype(np.uint8)


def _signed_map(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(np.abs(values), 99.0))
    if scale < 1e-6:
        scale = float(np.abs(values).max())
    if scale < 1e-6:
        scaled = np.zeros_like(values, dtype=np.float32)
    else:
        scaled = np.clip(values / scale, -1.0, 1.0)
    mag = np.abs(scaled)
    red = 0.5 + 0.5 * np.maximum(scaled, 0.0)
    green = 0.5 - 0.5 * mag
    blue = 0.5 + 0.5 * np.maximum(-scaled, 0.0)
    return (np.stack((red, green, blue), axis=-1) * 255.0 + 0.5).astype(np.uint8)


def _tensor_rgb01_to_uint8(image01: torch.Tensor) -> np.ndarray:
    tensor = image01.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    arr = tensor.permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _uv_to_rgb(uv: torch.Tensor, size_wh: Tuple[int, int]) -> np.ndarray:
    uv_cpu = uv.detach().cpu().clamp(0.0, 1.0)
    if uv_cpu.ndim == 4:
        uv_cpu = uv_cpu[0]
    zeros = torch.zeros_like(uv_cpu[:1])
    rgb = torch.cat((uv_cpu[:2], zeros), dim=0).permute(1, 2, 0).numpy()
    return _resize_rgb((rgb * 255.0 + 0.5).astype(np.uint8), size_wh)


def _feature_to_rgb(feature: torch.Tensor, size_wh: Tuple[int, int]) -> np.ndarray:
    feature_cpu = feature.detach().float().cpu()
    if feature_cpu.ndim == 4:
        feature_cpu = feature_cpu[0]
    heat = feature_cpu.abs().mean(dim=0).numpy()
    return _resize_rgb(_heatmap(heat), size_wh)


def _tensor_channel_to_rgb(tensor: torch.Tensor, channel: int, size_wh: Tuple[int, int]) -> np.ndarray:
    tensor_cpu = tensor.detach().float().cpu()
    if tensor_cpu.ndim == 4:
        tensor_cpu = tensor_cpu[0]
    channel = min(channel, tensor_cpu.shape[0] - 1)
    return _resize_rgb(_signed_map(tensor_cpu[channel].numpy()), size_wh)


def _flow_magnitude_to_rgb(flow: torch.Tensor, size_wh: Tuple[int, int]) -> np.ndarray:
    flow_cpu = flow.detach().float().cpu()
    if flow_cpu.ndim == 4:
        flow_cpu = flow_cpu[0]
    magnitude = flow_cpu.pow(2).sum(dim=0).sqrt().numpy()
    return _resize_rgb(_heatmap(magnitude), size_wh)


def _uv_grid_lines(uv: torch.Tensor, size_wh: Tuple[int, int], divisions: int = 16) -> np.ndarray:
    uv_cpu = uv.detach().float().cpu().clamp(0.0, 1.0)
    if uv_cpu.ndim == 4:
        uv_cpu = uv_cpu[0]
    u = uv_cpu[0].numpy()
    v = uv_cpu[1].numpy()

    def near_integer(values: np.ndarray) -> np.ndarray:
        frac = np.mod(values * divisions, 1.0)
        return np.minimum(frac, 1.0 - frac) < 0.018

    u_lines = near_integer(u)
    v_lines = near_integer(v)
    grid = np.full((u.shape[0], u.shape[1], 3), 255, dtype=np.uint8)
    grid[u_lines] = (220, 50, 45)
    grid[v_lines] = (35, 90, 220)
    grid[u_lines & v_lines] = (20, 20, 20)
    return _resize_rgb(grid, size_wh)


def _load_dataset_references(
    input_path: Path,
    data_dir: Path,
    size_wh: Tuple[int, int],
) -> List[Tuple[str, np.ndarray]]:
    references: List[Tuple[str, np.ndarray]] = []
    stem = input_path.stem

    gt_path = data_dir / "ground_truth" / f"{stem}.png"
    if gt_path.exists():
        gt = np.asarray(Image.open(gt_path).convert("RGB").resize(size_wh, Image.BILINEAR))
        references.append(("ground_truth", gt))

    uv_path = data_dir / "uv" / f"{stem}.png"
    if uv_path.exists():
        try:
            from uv_dewarp import dewarp_with_uv, load_rgb, load_uv

            rgb = load_rgb(input_path)
            uv, fg_mask = load_uv(uv_path)
            if uv.shape[:2] == rgb.shape[:2]:
                oracle = dewarp_with_uv(rgb, uv, out_size=size_wh[1], mask=fg_mask, flip_v=True)
                oracle = _resize_rgb(oracle, size_wh)
                references.append(("uv_dewarp_oracle", oracle))
        except Exception as exc:
            print(f"Skipping UV oracle debug image: {exc}")

    return references


def _draw_panel(path: Path, panels: List[Tuple[str, np.ndarray]], cell_size: Tuple[int, int] = (240, 240)) -> None:
    if not panels:
        return
    cols = min(4, len(panels))
    rows = int(math.ceil(len(panels) / cols))
    label_h = 34
    width = cols * cell_size[0]
    height = rows * (cell_size[1] + label_h)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font = None
    for candidate in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(candidate, 15)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for idx, (label, arr) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = col * cell_size[0]
        y = row * (cell_size[1] + label_h)
        image = Image.fromarray(_resize_rgb(arr, cell_size))
        canvas.paste(image, (x, y))
        short = label if len(label) <= 30 else f"{label[:27]}..."
        bbox = draw.textbbox((0, 0), short, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + max(4, (cell_size[0] - text_w) // 2), y + cell_size[1] + 8), short, fill=(0, 0, 0), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_debug_images(
    debug_dir: Path,
    input_path: Path,
    data_dir: Path,
    rgb: torch.Tensor,
    outputs: dict,
    dewarped: torch.Tensor,
    img_size: Tuple[int, int],
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    height, width = img_size
    size_wh = (width, height)
    debug = outputs.get("debug", {})
    panels: List[Tuple[str, np.ndarray]] = []
    written: List[str] = []
    counter = 0

    def add(label: str, arr: np.ndarray) -> None:
        nonlocal counter
        arr = _resize_rgb(arr, size_wh)
        out_path = debug_dir / f"{counter:02d}_{_slug(label)}.png"
        Image.fromarray(arr).save(out_path)
        written.append(str(out_path))
        panels.append((label, arr))
        counter += 1

    add("input_warped", _tensor_rgb01_to_uint8(denormalize_image(rgb)))

    for idx, feature in enumerate(debug.get("encoder_features", [])):
        add(f"encoder_{idx}_feature_heatmap", _feature_to_rgb(feature, size_wh))

    for idx, feature in enumerate(debug.get("decoder_features", [])):
        add(f"decoder_{idx}_feature_heatmap", _feature_to_rgb(feature, size_wh))

    if "decoded_full_resolution" in debug:
        add("decoder_full_resolution_heatmap", _feature_to_rgb(debug["decoded_full_resolution"], size_wh))
    if "raw_flow" in debug:
        add("raw_flow_x_signed", _tensor_channel_to_rgb(debug["raw_flow"], 0, size_wh))
        add("raw_flow_y_signed", _tensor_channel_to_rgb(debug["raw_flow"], 1, size_wh))

    add("predicted_flow_x_signed", _tensor_channel_to_rgb(outputs["flow"], 0, size_wh))
    add("predicted_flow_y_signed", _tensor_channel_to_rgb(outputs["flow"], 1, size_wh))
    add("predicted_flow_magnitude", _flow_magnitude_to_rgb(outputs["flow"], size_wh))
    add("predicted_uv_rg", _uv_to_rgb(outputs["uv"], size_wh))
    add("predicted_sampling_grid_lines", _uv_grid_lines(outputs["uv"], size_wh))

    model_output = _tensor_rgb01_to_uint8(dewarped)
    add("model_dewarped_output", model_output)

    references = _load_dataset_references(input_path, data_dir, size_wh)
    for label, arr in references:
        add(label, arr)
        if label == "ground_truth":
            diff = np.abs(model_output.astype(np.int16) - arr.astype(np.int16))
            add("model_vs_ground_truth_abs_diff_x2", np.clip(diff * 2, 0, 255).astype(np.uint8))

    panel_path = debug_dir / "debug_panel.png"
    _draw_panel(panel_path, panels)
    written.append(str(panel_path))
    (debug_dir / "debug_manifest.txt").write_text("\n".join(written) + "\n", encoding="utf-8")
    return panel_path


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    input_path = Path(args.input) if args.input else default_input(data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    img_size = tuple(ckpt_args.get("img_size", (256, 256)))

    device = choose_device(args.device)
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

    with torch.no_grad():
        rgb = preprocess(input_path, img_size, device)
        outputs = predict_grid(model, rgb, args.tta_flip, return_debug=args.debug_dir is not None)
        if args.output_size is not None:
            output_size = tuple(args.output_size)
            dewarped = render_with_grid(
                input_path=input_path,
                data_dir=data_dir,
                grid=outputs["grid"],
                output_size=output_size,
                device=device,
                mask_cleanup=args.mask_cleanup,
                mask_threshold=args.mask_threshold,
            )
        else:
            output_size = img_size
            dewarped = denormalize_image(outputs["image"])
            if args.mask_cleanup:
                high_grid = outputs["grid"]
                source_mask = load_border_mask(input_path, data_dir, (img_size[1], img_size[0]), device)
                if source_mask is not None:
                    sampled_mask = torch.nn.functional.grid_sample(
                        source_mask,
                        high_grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
                    keep = (sampled_mask >= args.mask_threshold).to(dewarped.dtype)
                    dewarped = dewarped * keep + (1.0 - keep)
        if args.enhance:
            dewarped = enhance_document(dewarped)

    save_rgb(Path(args.output), dewarped)
    save_uv(Path(args.uv_output), outputs["uv"])
    save_flow(Path(args.flow_output), outputs["flow"])
    if args.debug_dir is not None:
        panel_path = save_debug_images(Path(args.debug_dir), input_path, data_dir, rgb, outputs, dewarped, output_size)
        print(f"Saved debug panel: {panel_path}")
    print(f"Input: {input_path}")
    print(f"Saved dewarped image: {args.output}")
    print(f"Saved predicted UV map: {args.uv_output}")
    print(f"Saved flow magnitude: {args.flow_output}")


if __name__ == "__main__":
    main()
