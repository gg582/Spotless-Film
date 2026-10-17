#!/usr/bin/env python3
"""Convert the UNetDust CoreML model (.mlmodelc) to a PyTorch .pth state dict.

The upstream app ships its dust-detection UNet only as a compiled CoreML
model (UNetDust.mlmodelc). Its weights/weight.bin stores every tensor as
fp16 behind a 20-byte entry header (0xDEADBEEF magic, version, byte size,
reserved field, 4-byte sub-field), and small tensors (e.g. the final bias)
may be inlined as hex-float literals in model.mil. This script recovers the
full state dict for the UNet architecture in src/image_processing.py.

Usage:
    # extract UNetDust.mlmodelc from the macOS release DMG, then:
    python tools/convert_coreml_weights.py <path-to-UNetDust.mlmodelc> <output.pth>
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch

ENTRY_HEADER = 20  # 0xDEADBEEF magic + version + data size + reserved + 4-byte field


def parse_mil(mil_path: Path):
    mil = mil_path.read_text()

    blob_pattern = re.compile(
        r'tensor<fp16, \[([0-9, ]*)\]> [a-zA-Z0-9_]+? = const\(\)\[name = tensor<string, \[\]>\("'
        r'([a-zA-Z0-9_]+?)"\), val = tensor<fp16, \[[0-9, ]*\]>\(BLOBFILE\(path = .*?'
        r"offset = tensor<uint64, \[\]>\(([0-9]+)\)\)\)"
    )
    entries = []
    for m in blob_pattern.finditer(mil):
        shape = tuple(int(s) for s in m.group(1).split(","))
        entries.append((m.group(2), int(m.group(3)), shape))

    inline_pattern = re.compile(
        r'tensor<fp16, \[([0-9, ]*)\]> [a-zA-Z0-9_]+? = const\(\)\[name = tensor<string, \[\]>\("'
        r'([a-zA-Z0-9_]+?)"\), val = tensor<fp16, \[[0-9, ]*\]>\(\[([^\]]*)\]\)\]'
    )
    inline = []
    for m in inline_pattern.finditer(mil):
        shape = tuple(int(s) for s in m.group(1).split(","))
        vals = [float.fromhex(v.strip()) for v in m.group(3).split(",")]
        inline.append((m.group(2), shape, vals))

    return mil, entries, inline


def to_state_dict_key(name: str) -> str:
    # enc1_0_weight_to_fp16 -> enc1.0.weight
    key = re.sub(r"_(\d+)_", r".\1.", name)
    key = key.replace("_weight", ".weight").replace("_bias", ".bias")
    if key.endswith("_to_fp16"):
        key = key[: -len("_to_fp16")]
    return key


def convert(modelc_dir: Path, out_path: Path):
    mil, entries, inline = parse_mil(modelc_dir / "model.mil")
    blob = (modelc_dir / "weights" / "weight.bin").read_bytes()

    state_dict = {}
    for name, mil_offset, shape in entries:
        n = int(np.prod(shape))
        offset = mil_offset + ENTRY_HEADER
        if blob[mil_offset : mil_offset + 4] != b"\xef\xbe\xad\xde":
            raise ValueError(f"{name}: missing 0xDEADBEEF magic at entry offset {mil_offset}")
        arr = np.frombuffer(blob, dtype="<f2", count=n, offset=offset).astype(np.float32)
        if arr.size != n:
            raise ValueError(f"{name}: parsed {arr.size} elements, expected {n}")
        if not np.isfinite(arr).all() or np.abs(arr).max() > 10:
            raise ValueError(f"{name}: implausible values — entry header size may differ")
        state_dict[to_state_dict_key(name)] = torch.from_numpy(arr.reshape(shape))

    for name, shape, vals in inline:
        key = to_state_dict_key(name)
        if key not in state_dict:
            state_dict[key] = torch.tensor(np.array(vals, dtype=np.float32).reshape(shape))

    if len(state_dict) != 46:  # 23 conv layers x (weight + bias)
        raise ValueError(f"expected 46 tensors, got {len(state_dict)}")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from image_processing import UNet

    model = UNet()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()

    torch.save(model.state_dict(), out_path)
    print(f"Saved {len(state_dict)} tensors -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
