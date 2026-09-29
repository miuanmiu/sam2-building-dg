"""Evaluate a released checkpoint on the held-out Kitsap test set.

This is a self-contained, CPU-runnable entry point: it rebuilds the SAM 2 tiny
image encoder plus the FPN head, loads one of the released best-per-seed
checkpoints, and reports the dataset-level (aggregate) IoU that the manuscript
tabulates.

Example
-------
python scripts/eval_checkpoint.py \
    --checkpoint checkpoints/best_sam2_whu_4city_800_20ep_fosmix_s43.pth \
    --manifest data/splits_whu/whu_kitsap_manifest.csv \
    --sam2-weights models/sam2/sam2.1_hiera_tiny.pt

Verified values (seed 43, 207 Kitsap tiles, threshold 0.5):
    Basic     0.7269    FOSMix  0.7565    MixStyle 0.7161
which reproduces the per-seed numbers reported in the paper.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class FPNHead(nn.Module):
    """Three-scale FPN head used in the paper (1x1 lateral, top-down sum, 3x3)."""

    def __init__(self, in_channels: list[int], hidden: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, hidden, kernel_size=1) for c in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feats: list[torch.Tensor], target_size) -> torch.Tensor:
        lat = [conv(f) for conv, f in zip(self.lateral, feats)]
        x = lat[-1]
        for f in reversed(lat[:-1]):
            x = F.interpolate(x, size=f.shape[-2:], mode="bilinear",
                              align_corners=False) + f
        x = self.fuse(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


class SAM2FPN(nn.Module):
    def __init__(self, sam2_weights: str, device: str = "cpu") -> None:
        super().__init__()
        try:
            from sam2.build_sam import build_sam2
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "SAM 2 is not importable. Install it with "
                "`pip install git+https://github.com/facebookresearch/sam2.git` "
                "and run this script from a checkout that provides `sam2`."
            ) from exc
        self.sam2 = build_sam2(SAM2_CFG, sam2_weights, device=device, mode="eval")
        for param in self.sam2.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512, device=device)
            channels = [f.shape[1] for f in self.sam2.forward_image(dummy)["backbone_fpn"]]
        self.head = FPNHead(channels)
        self.register_buffer("norm_mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        feats = self.sam2.forward_image(x)["backbone_fpn"]
        return self.head(feats, target_size=(x.shape[-2], x.shape[-1]))


def resolve(path: str, data_root: str) -> str:
    """Accept repo-relative paths, the release archive layout, or a data root."""
    if os.path.exists(path):
        return path
    marker = "WHU_Mix_cities/"
    if marker in path:
        tail = path.replace("\\", "/").split(marker, 1)[1]
        candidate = os.path.join(data_root, tail.replace("/", os.sep))
        if os.path.exists(candidate):
            return candidate
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sam2-weights", required=True)
    parser.add_argument("--data-root", default=os.path.join("data", "raw", "WHU_Mix_cities"),
                        help="directory that contains <city>/image and <city>/label")
    parser.add_argument("--city", default="kitsap")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = SAM2FPN(args.sam2_weights, device=args.device)
    blob = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = blob.get("model_state_dict", blob)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("city", args.city) == args.city]
    if args.limit:
        rows = rows[: args.limit]

    tp = fp = fn = 0
    for row in rows:
        image = np.asarray(
            Image.open(resolve(row["image_path"], args.data_root)).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        gt = np.asarray(
            Image.open(resolve(row["mask_path"], args.data_root)).convert("L"),
            dtype=np.float32,
        ) / 255.0 > 0.5
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(args.device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        pred = prob > args.threshold
        tp += int(np.logical_and(pred, gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred, gt).sum())

    union = tp + fp + fn
    iou = tp / union if union else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    print(f"tiles: {len(rows)}  IoU {iou:.4f}  precision {precision:.4f}  "
          f"recall {recall:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
