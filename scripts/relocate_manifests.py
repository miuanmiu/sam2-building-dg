#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rewrite absolute image/mask paths in WHU split manifests to a new data root.

The released data/splits_whu/*.csv were generated on the author's machine and
contain drive-rooted absolute paths such as:

    C:/datasets/WHU_Mix_cities/dunedin/image/dunedin_650.tif

This script rewrites the drive-rooted prefix to any local root and normalizes
separators, so the same splits work on any machine. It preserves all other
columns.

Examples:
    # dry-run style preview (writes to a separate --out dir):
    python scripts/relocate_manifests.py --src data/splits_whu --out preview \
        --new-root /mnt/data/WHU_Mix_cities

    # rewrite the shipped manifests in place:
    python scripts/relocate_manifests.py --src data/splits_whu \
        --new-root <YOUR_DATA_ROOT>
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def _norm(text: str) -> str:
    return text.replace("\\", "/")


def rewrite_path(path: str, old_root: str | None, new_root: str) -> str:
    norm = _norm(path)
    new = _norm(new_root).rstrip("/")
    if old_root:
        old = _norm(old_root).rstrip("/")
        if norm.lower() == old.lower() or norm.lower().startswith(old.lower() + "/"):
            rel = norm[len(old):].lstrip("/")
            return f"{new}/{rel}"
        return path
    # Default: replace the drive-rooted first component, e.g. "D:/Datasets".
    m = re.match(r"^[A-Za-z]:/([^/]+)", norm)
    if m:
        return f"{new}/{norm[len(m.group(0)):].lstrip('/')}"
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="manifest directory or a single CSV")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: same as --src, i.e. in place)")
    ap.add_argument("--old-root", default=None,
                    help="old absolute root to replace (auto-detected if omitted)")
    ap.add_argument("--new-root", required=True,
                    help="new data root, e.g. /mnt/data/WHU_Mix_cities or C:/data/WHU_Mix_cities")
    args = ap.parse_args()

    files = sorted(args.src.rglob("*.csv")) if args.src.is_dir() else [args.src]
    if not files:
        sys.exit(f"no CSVs found under {args.src}")

    for src in files:
        if args.out is None:
            dst = src
        else:
            dst = args.out / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        changed = 0
        for r in rows:
            for col in ("image_path", "mask_path"):
                if col in r and r[col]:
                    new = rewrite_path(r[col], args.old_root, args.new_root)
                    if new != r[col]:
                        changed += 1
                    r[col] = new
        with dst.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{src.name}: {changed} paths rewritten -> {dst}")


if __name__ == "__main__":
    main()
