#!/usr/bin/env python3
"""Batch dust removal for scanned film photos.

Two-step pipeline:
  1. Bright dust: UNet dust-probability map (1024x1024, from CoreML-converted
     weights), white-hat baseline subtraction + adaptive delta.
  2. Dark dust: classical black-hat detection (15px and 31px kernels) gated to
     locally smooth areas, so compact dark specks on bright/smooth surfaces
     (e.g. ice cream, sky) are caught without firing on texture or shading.
  3. Union mask: small-component cleanup, 5x5 dilation, CV2 TELEA inpainting.

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
    return _clean_components(mask, min_area=3)


def build_dark_dust_mask(gray: np.ndarray, budget: float = 0.005) -> np.ndarray:
    """Compact dark specks on brighter, locally smooth surroundings.

    The UNet is trained on bright dust only and responds unreliably on
    inverted input, so dark dust is detected classically: darker than a
    15px neighborhood (morphological black-hat), on a bright median
    background, inside a locally flat region. High-frequency image texture
    (sea sparkles, gravel) can look similar locally, so thresholds
    auto-tighten until the detections fit a per-image coverage budget.
    Very dark (near-black) thin structures such as hairs are always allowed.
    """
    bh15 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ref = cv2.medianBlur(gray, 25)
    g = gray.astype(np.float32) / 255.0
    win = 41
    mean = cv2.boxFilter(g, -1, (win, win))
    mean2 = cv2.boxFilter(g * g, -1, (win, win))
    local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0))
    # thin near-black structures (hairs/fibers) on a bright background;
    # the bright-reference requirement keeps this from firing on dark areas
    hair = (gray < 60) & (bh15 > 25) & (ref > 110) & (local_std < 0.06)

    for bh_t, ref_t, std_t in [(20, 140, 0.045), (28, 150, 0.035),
                               (38, 165, 0.028), (50, 180, 0.022)]:
        base = ((bh15 > bh_t) & (ref > ref_t) & (local_std < std_t))
        # thin near-black structures (hairs/fibers) on a bright background,
        # with a looser smoothness gate since they often lie near object edges
        hair = (gray < 130) & (bh15 > 25) & (ref > 110) & (local_std < 0.12)
        cand = cv2.bitwise_or(base.astype(np.uint8) * 255, hair.astype(np.uint8) * 255)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # sparsity: real dust and single hairs are isolated; image texture
        # (sea sparkles) forms dense clusters. The high threshold keeps long
        # thin hairs (which neighbor themselves) while cutting texture.
        neighbor_count = cv2.boxFilter((cand > 0).astype(np.float32), -1, (31, 31),
                                       normalize=False)
        cand[(cand > 0) & (neighbor_count > 400)] = 0
        if (cand > 0).mean() > budget:
            continue  # skip expensive component labeling; tighten thresholds
        mask = _clean_components(cand, min_area=5, max_area=1500)
        if (mask > 0).mean() <= budget:
            return mask
    return np.zeros_like(gray)


def _clean_components(mask: np.ndarray, min_area: int, max_area: int = 0) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or (max_area and area > max_area):
            mask[labels == i] = 0
    return mask


def remove_dust(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # bridge fragments of the same defect (e.g. a hair caught in sections)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pred = predict_dust_probability(model, path)
            mask_bright = build_dust_mask(pred)
            mask_dark = build_dark_dust_mask(gray)
            coverage = ((mask_bright > 0).mean() + (mask_dark > 0).mean()) * 100
            result = remove_dust(image, cv2.bitwise_or(mask_bright, mask_dark))
            cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            print(f"[{i}/{len(files)}] {path.name}: dust {coverage:.2f}% "
                  f"(bright {(mask_bright > 0).mean() * 100:.2f}% + dark {(mask_dark > 0).mean() * 100:.2f}%)")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAILED {path.name}: {e}")
            failed.append(path.name)

    print(f"\nDone. {len(files) - len(failed)}/{len(files)} succeeded.")
    if failed:
        print("Failed files:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
