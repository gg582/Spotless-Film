#!/usr/bin/env python3
"""Batch dust removal for scanned film photos.

Pipeline (adapted from Spotless-Film):
  1. UNet dust-probability map (1024x1024, from CoreML-converted weights)
  2. White-hat baseline subtraction (morphological opening) + adaptive delta
  3. Small-component cleanup, 5x5 dilation
  4. CV2 TELEA inpainting, blended back

Usage:
    python batch_dust_removal.py <input_dir> <output_dir> [--weights PATH]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from image_processing import UNet  # noqa: E402


def load_model(weights_path: str) -> UNet:
    device = torch.device("cpu")
    model = UNet()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_dust_probability(model: UNet, image_path: Path) -> np.ndarray:
    """Probability map at the image's original resolution."""
    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    img_1024 = img.resize((1024, 1024), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(img_1024, dtype=np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)
    pred = pred.squeeze().numpy().astype(np.float32)
    return cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def build_dust_mask(pred: np.ndarray) -> np.ndarray:
    """White-hat on the probability map: dust = local positive anomaly."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    baseline = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    diff = pred - baseline
    delta = max(0.05, min(float(np.percentile(diff, 99)), 0.2))
    mask = (diff > delta).astype(np.uint8) * 255
    # drop noise specks smaller than 3 px
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            mask[labels == i] = 0
    return mask


def remove_dust(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    inpainted = cv2.inpaint(image_bgr, mask_d, 5, cv2.INPAINT_TELEA)
    m = (mask_d > 0).astype(np.float32)[:, :, None]
    blended = image_bgr.astype(np.float32) * (1 - m) + inpainted.astype(np.float32) * m
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--weights", default=str(Path(__file__).parent / "src" / "weights" / "unetdust_coreml.pth"))
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = load_model(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = sorted(f for f in args.input_dir.iterdir() if f.suffix.lower() in exts)
    print(f"Found {len(files)} images in {args.input_dir}")

    failed = []
    for i, path in enumerate(files, 1):
        out_path = args.output_dir / (path.stem + ".JPG")
        if out_path.exists():
            print(f"[{i}/{len(files)}] skip (exists): {path.name}")
            continue
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("cv2.imread failed")
            pred = predict_dust_probability(model, path)
            mask = build_dust_mask(pred)
            coverage = (mask > 0).mean() * 100
            result = remove_dust(image, mask)
            cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            print(f"[{i}/{len(files)}] {path.name}: dust {coverage:.2f}% -> {out_path.name}")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED {path.name}: {e}")
            failed.append(path.name)

    print(f"\nDone. {len(files) - len(failed)}/{len(files)} succeeded.")
    if failed:
        print("Failed files:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
