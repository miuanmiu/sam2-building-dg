# SAM 2 Building DG

Reproducible multi-seed implementation for:

> **Dual-Level Frequency- and Feature-Domain Mixing for SAM 2 Fine-Tuning in
> Cross-City Building Extraction** (submitted to IEEE Access)

Repository: <https://github.com/miuanmiu/sam2-building-dg>
Checkpoints: <https://github.com/miuanmiu/sam2-building-dg/releases/tag/v1.0-checkpoints>
Anonymous review mirror: <https://anonymous.4open.science/r/sam2-building-dg-A242/>

All methods fine-tune the SAM 2 tiny image encoder with a lightweight FPN
segmentation head on 512x512 RGB patches. Training-only augmentations:

| Config | Input level | Feature level | Script |
|---|---|---|---|
| Basic | – | – | `src/train_sam2_building.py` |
| MixStyle | – | MixStyle (p=0.5, alpha=0.1) | `src/train_sam2_building.py --mixstyle` |
| FOSMix | FOSMix (re-impl. of TGRS 2024) | – | `src/train_sam2_building_fosmix.py` |
| Dual | FOSMix | MixStyle | `src/train_sam2_building_fosmix.py --mixstyle` |
| BSM | BSM (re-impl.) | – | `src/train_sam2_building_bsm.py` |
| FACT / DSU | FACT | DSU | `src/train_sam2_dg.py --dg fact\|dsu` |

## Repository layout

```
src/                    training / evaluation code
  train_sam2_building.py        Basic / MixStyle
  train_sam2_building_fosmix.py FOSMix / Dual (with --mixstyle)
  train_sam2_building_bsm.py    BSM baseline
  train_sam2_dg.py              FACT / DSU baselines
  train_cnn_baseline.py         SegFormer-B0 / DeepLabV3+ baselines
  fosmix_module.py, bsm_module.py, losses.py, metrics.py, ...
data/splits_whu/        train/val/test manifests (paths must be relocated)
records/                per-epoch training & test CSVs used in the paper
scripts/                download + path-relocation helpers
```

## Requirements

- Python 3.12, CUDA GPU with >= 8 GB VRAM (batch size 2)
- `pip install -r requirements.txt`
- Install the official SAM 2.1 package from source (the experiments used the
  facebookresearch/sam2 repository as of August 2026; pin a specific commit
  for full reproducibility):
  `pip install "git+https://github.com/facebookresearch/sam2.git"`
- Download SAM 2.1 weights:
  `python scripts/download_sam2_weights.py --out models/sam2`
- BSM baseline only - download the AdaIN weights:
  `python scripts/download_adain_weights.py --out checkpoints/adain_weights`

## Data

- **WHU-Mix** (Luo et al., JSTARS 2023): obtain from the authors.
- **Kitsap held-out test**: INRIA Aerial Image Labeling benchmark.
- **Christchurch external test**: WHU Building dataset.

The shipped manifests point at the repo-relative root `data/raw/`, so the
simplest layout is:

```
data/raw/WHU_Mix_cities/<city>/image/*.tif
data/raw/WHU_Mix_cities/<city>/label/*.tif
data/raw/WHU_Building/...
```

If you keep the data somewhere else, point the manifests at your root without
editing files by hand:

```
python scripts/relocate_manifests.py \
  --src data/splits_whu --out data/splits_whu \
  --new-root <YOUR_DATA_ROOT>
```

## Reproducing the paper

Standard protocol (four-city-800, Kitsap held-out), one seed:

```
python src/train_sam2_building.py \
  --finetune-encoder --epochs 20 --batch-size 2 --num-workers 4 \
  --seed 42 --ckpt-dir checkpoints \
  --manifest-path data/splits_whu/whu_4city_train_800_manifest.csv \
  --eval-manifest-path data/splits_whu/whu_4city_val_100_manifest.csv \
  --test-manifest-path data/splits_whu/whu_kitsap_manifest.csv \
  --eval-split val --eval-max 100 --save-every-epochs 1 \
  --run-name sam2_whu_4city_800_20ep_basic_s42
```

- **MixStyle**: same, add `--mixstyle --mixstyle-alpha 0.1`.
- **FOSMix**: use `src/train_sam2_building_fosmix.py`, add `--fosmix-res 512`.
- **Dual**: FOSMix script + `--mixstyle --mixstyle-alpha 0.1`.
- **LODO**: use `lodo_train_{city}_manifest.csv` / `lodo_val_{city}_manifest.csv`
  / `lodo_test_{city}_manifest.csv` and run name `lodo_{city}_{method}_s{seed}`.
- **FACT/DSU**: `python src/train_sam2_dg.py --dg fact --run-name
  sam2_whu_4city_800_20ep_fact_s42 ...` (same manifest arguments).
- Repeat seeds 42-46; tables report five-seed means.

Paper table mapping: Table II (Basic/FOSMix/MixStyle/Dual), Table III
(BSM/FACT/DSU), Table V (FOSMix ablations), Table XIII (SegFormer-B0 /
DeepLabV3+ via `src/train_cnn_baseline.py`).

## Per-epoch records

- `records/<run>_training_history.csv`: `epoch,train_loss,val_iou,val_f1,val_precision,val_recall,best_iou,elapsed_s`
- `records/<run>_test_generalization.csv`: `epoch,iou,f1,precision,recall`

Per-epoch CSVs are included for every run used in the paper's main tables.
A small number of early seed-42 runs (Potsdam FOSMix, Wuxi Basic/FOSMix, and
the four LODO MixStyle runs) no longer have their per-epoch logs; those runs
are documented by their best source-validation checkpoints (to be archived on
Zenodo; DOI added after upload) and the full-test values reported in Table II.
Every value in the paper is reproducible from the released checkpoints, whose
SHA256 checksums are listed in `records/checkpoints_manifest.csv`.

## Checkpoints

Best-per-seed checkpoints (`best_<run>.pth`, 141 files, 20.0 GB) are attached as
split parts to the GitHub release tagged `v1.0-checkpoints`:

```
all_checkpoints.zip.part01 ... all_checkpoints.zip.part10   (2 GiB each; the last part is smaller)
README_parts.txt          merge instructions
parts_sha256.csv          per-part SHA256 checksums
checkpoints_manifest.csv  per-checkpoint SHA256 checksums
```

Reassemble the archive and verify it:

```
# Linux / macOS
cat all_checkpoints.zip.part* > all_checkpoints.zip

# Windows (cmd)
copy /b all_checkpoints.zip.part01+all_checkpoints.zip.part02+all_checkpoints.zip.part03+all_checkpoints.zip.part04+all_checkpoints.zip.part05+all_checkpoints.zip.part06+all_checkpoints.zip.part07+all_checkpoints.zip.part08+all_checkpoints.zip.part09+all_checkpoints.zip.part10 all_checkpoints.zip
```

SHA256 of the reassembled archive:

```
d917099a97d069a36a5f6ebfe557cfb80fd8def7b7abd21a42edbcea4f02b20e
```

`records/checkpoints_manifest.csv` lists every checkpoint inside the archive with
its own SHA256 checksum.

## Attribution and licenses

- SAM 2 (Meta): Apache-2.0
- FOSMix: re-implementation of https://github.com/Reo-I/FOSMix (Iizuka et al., IEEE TGRS 2024)
- MixStyle: Zhou et al., ICLR 2021
- BSM / AdaIN: re-implementation based on WHU-Mix (Luo et al., JSTARS 2023) and naoto0804/pytorch-AdaIN
- FACT / DSU: re-implementations of Xu et al., CVPR 2023 and Li et al., ICLR 2022
- Datasets: see the original distribution pages; raw imagery is not redistributed here

Code license: MIT (see LICENSE).
