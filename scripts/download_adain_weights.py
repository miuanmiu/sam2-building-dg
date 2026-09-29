#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download the AdaIN weights (VGG encoder + decoder) used by the BSM baseline.

Source: naoto0804/pytorch-AdaIN (Huang & Belongie, ICCV 2017). The official
model files and URLs are listed in that repository's README:

    vgg_normalised.pth  https://www.dropbox.com/s/l6nvfeyz4i0hsxx/vgg_normalised.pth?dl=1
    decoder.pth         https://www.dropbox.com/s/2j2b6q0m9w0kpy6/decoder.pth?dl=1

Usage:
    python scripts/download_adain_weights.py --out checkpoints/adain_weights
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FILES = {
    "vgg_normalised.pth": "https://www.dropbox.com/s/l6nvfeyz4i0hsxx/vgg_normalised.pth?dl=1",
    "decoder.pth": "https://www.dropbox.com/s/2j2b6q0m9w0kpy6/decoder.pth?dl=1",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/adain_weights"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = args.out / name
        if dst.exists() and dst.stat().st_size > 1e6:
            print(f"{dst} already exists, skipping.")
            continue
        print(f"downloading {url}\n  -> {dst}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, dst.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"done ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
