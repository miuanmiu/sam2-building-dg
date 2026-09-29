#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download the SAM 2.1 checkpoints used by the paper (Meta, Apache-2.0).

Official URLs (facebookresearch/sam2):
    sam2.1_hiera_tiny.pt       https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
    sam2.1_hiera_base_plus.pt  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

Usage:
    python scripts/download_sam2_weights.py --out models/sam2
    python scripts/download_sam2_weights.py --out models/sam2 --model base_plus
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URLS = {
    "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
}
SHA256 = {"tiny": None, "base_plus": None}  # TODO: pin after first verified download


def _download(url: str, dst: Path) -> None:
    print(f"downloading {url}\n  -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, dst.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"done ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(URLS), default="tiny")
    ap.add_argument("--out", type=Path, default=Path("models/sam2"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"sam2.1_hiera_{'tiny' if args.model == 'tiny' else 'base_plus'}.pt"
    dst = args.out / name
    if dst.exists() and dst.stat().st_size > 1e6:
        print(f"{dst} already exists, skipping.")
        return
    _download(URLS[args.model], dst)
    expected = SHA256[args.model]
    if expected:
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        if digest != expected:
            sys.exit(f"SHA256 mismatch: {digest}")
        print("SHA256 OK")


if __name__ == "__main__":
    main()
