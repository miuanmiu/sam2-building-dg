
import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sam2.build_sam import build_sam2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import BCEDiceLoss  # noqa: E402
from src.bsm_module import (  # noqa: E402
    BSMStyleTransfer,
    bsm_color_augment,
    bsm_geometric_augment,
)

MODEL_DIR = Path(os.environ.get("SAM2_MODEL_DIR", "models/sam2"))
MODEL_CFG = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", MODEL_DIR / "sam2.1_hiera_tiny.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", MODEL_DIR / "sam2.1_hiera_base_plus.pt"),
}

DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "splits_whu" / "whu_mix_4city_manifest.csv"
DEFAULT_EVAL_MANIFEST = PROJECT_ROOT / "data" / "splits_whu" / "whu_classic_manifest.csv"
RANDOM_SEED = 42
SEED = RANDOM_SEED  # set from --seed in main()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TileDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str | None = None,
        cities: list[str] | None = None,
        max_per_city: int | None = None,
        augment: bool = False,
        augment_level: str = "basic",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.augment = augment
        self.augment_level = augment_level
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if split is not None:
            rows = [r for r in rows if r["split"].strip().lower() == split]
        if cities is not None:
            allowed = {c.lower() for c in cities}
            rows = [r for r in rows if r["city"].strip().lower() in allowed]
        if max_per_city is not None:
            by_city: dict[str, list[dict]] = {}
            for r in rows:
                by_city.setdefault(r["city"], []).append(r)
            rng = random.Random(SEED)
            capped: list[dict] = []
            for city, city_rows in sorted(by_city.items()):
                capped.extend(rng.sample(city_rows, min(max_per_city, len(city_rows))))
            rows = capped
        missing = [
            r for r in rows
            if not (Path(r["image_path"]).exists() and Path(r["mask_path"]).exists())
        ]
        if missing:
            rows = [r for r in rows if r not in missing]
        if not rows:



            pass
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0)        # [1,H,W]
        if self.augment:
            if self.augment_level == "bsm":
                # BSMA?CA?p=0.5 ?Fig.5?
                if random.random() < 0.5:
                    image, mask = bsm_geometric_augment(image, mask)
                if random.random() < 0.5:
                    image = bsm_color_augment(image)
            else:
                if random.random() < 0.5:
                    image = torch.flip(image, dims=[2])
                    mask = torch.flip(mask, dims=[2])
                if random.random() < 0.5:
                    image = torch.flip(image, dims=[1])
                    mask = torch.flip(mask, dims=[1])
                k = random.randint(0, 3)
                if k:
                    image = torch.rot90(image, k, dims=[1, 2])
                    mask = torch.rot90(mask, k, dims=[1, 2])
                if self.augment_level == "strong":
                    from torchvision.transforms import functional as TF
                    angle = random.uniform(-15, 15)
                    scale = random.uniform(0.85, 1.15)
                    image = TF.affine(
                        image, angle=angle, translate=[0, 0], scale=scale, shear=0,
                        interpolation=TF.InterpolationMode.BILINEAR, fill=0.0,
                    )
                    mask = TF.affine(
                        mask, angle=angle, translate=[0, 0], scale=scale, shear=0,
                        interpolation=TF.InterpolationMode.NEAREST, fill=0.0,
                    )
                    image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                    image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                    image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        return image, mask, row["tile_name"], row["city"]


class FPNHead(nn.Module):
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

    def forward(self, feats: list[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        lat = [conv(f) for conv, f in zip(self.lateral, feats)]
        x = lat[-1]
        for f in reversed(lat[:-1]):
            x = F.interpolate(x, size=f.shape[-2:], mode="bilinear", align_corners=False) + f
        x = self.fuse(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


class SAM2Segmentor(nn.Module):
    def __init__(self, model_name: str = "tiny", finetune_encoder: bool = False) -> None:
        super().__init__()
        cfg, ckpt = MODEL_CFG[model_name]
        self.sam2 = build_sam2(cfg, str(ckpt), device="cuda", mode="eval")
        for p in self.sam2.parameters():
            p.requires_grad_(False)

        if finetune_encoder:
            for p in self.sam2.image_encoder.parameters():
                p.requires_grad_(True)
            decoder = getattr(self.sam2, "sam_mask_decoder", None)
            if decoder is not None:
                for name in ("conv_s0", "conv_s1"):
                    mod = getattr(decoder, name, None)
                    if mod is not None:
                        for p in mod.parameters():
                            p.requires_grad_(True)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512, device="cuda")
            backbone_out = self.sam2.forward_image(dummy)
            in_channels = [f.shape[1] for f in backbone_out["backbone_fpn"]]
        self.head = FPNHead(in_channels)

        self.register_buffer("norm_mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        backbone_out = self.sam2.forward_image(x)
        return self.head(backbone_out["backbone_fpn"], target_size=(x.shape[-2], x.shape[-1]))


def atomic_torch_save(content: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    torch.save(content, tmp)
    os.replace(tmp, output_path)


def save_resume_checkpoint(
    path: Path,
    epoch: int,
    next_batch: int,
    global_step: int,
    epoch_loss_sum: float,
    epoch_sample_count: int,
    best_val_iou: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> None:
    training_arguments = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }
    atomic_torch_save(
        {
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_sample_count": epoch_sample_count,
            "best_val_iou": best_val_iou,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "training_arguments": training_arguments,
        },
        path,
    )


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict:
    state = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    return state


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    tp = fp = fn = 0
    per_city: dict[str, list[float]] = {}
    for images, masks, names, cities in tqdm(data_loader, desc="eval"):
        images = images.to(device)
        masks = masks.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        preds = (torch.sigmoid(logits.float()) > 0.5).float()
        gt = masks.float()
        tp += int(((preds == 1) & (gt == 1)).sum())
        fp += int(((preds == 1) & (gt == 0)).sum())
        fn += int(((preds == 0) & (gt == 1)).sum())
        for c, p, g in zip(cities, preds, gt):
            inter = int(((p == 1) & (g == 1)).sum())
            union = int(((p == 1) | (g == 1)).sum())
            per_city.setdefault(c, []).append(inter / union if union else 1.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_city": {c: float(np.mean(v)) for c, v in per_city.items()},
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 fine-tuning with BSM on WHU tiles")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--eval-manifest-path", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--eval-max", type=int, default=1000)
    parser.add_argument("--train-max-per-city", type=int, default=1000)
    parser.add_argument("--model", choices=list(MODEL_CFG), default="tiny")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--augmentation", choices=["basic", "strong", "bsm"], default="basic")
    parser.add_argument("--bsm-weights-dir", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "adain_weights")
    parser.add_argument("--bsm-size", type=int, default=512)
    parser.add_argument("--bsm-alpha", type=float, default=0.5)
    parser.add_argument("--bsm-prob", type=float, default=0.5)
    parser.add_argument("--run-name", type=str, default="sam2_whu_tiny")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every-batches", type=int, default=200)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    global SEED
    SEED = args.seed
    if args.resume is not None and args.num_workers != 0:
        raise ValueError("error")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("error")

    resume_path = args.resume
    ckpt_root = args.ckpt_dir if args.ckpt_dir else PROJECT_ROOT / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / f"resume_{args.run_name}.pth"
    best_path = ckpt_root / f"best_{args.run_name}.pth"
    history_path = PROJECT_ROOT / "logs" / f"{args.run_name}_training_history.csv"
    eval_summary_path = PROJECT_ROOT / "outputs" / f"{args.run_name}_final_eval.txt"

    print("=" * 50)
    print("=" * 50)
    if args.augmentation == "bsm":

        args.manifest_path, max_per_city=args.train_max_per_city,


    eval_dataset = TileDataset(args.eval_manifest_path, split=args.eval_split, max_per_city=args.eval_max)
    for label, ds in (("train", train_dataset), ("eval", eval_dataset)):
        counts = {}
        for r in ds.rows:
            counts[r["city"]] = counts.get(r["city"], 0) + 1

    model = SAM2Segmentor(model_name=args.model, finetune_encoder=args.finetune_encoder)
    model.to(device)

    bsm_net = None
    if args.augmentation == "bsm":
        bsm_net = BSMStyleTransfer(args.bsm_weights_dir, device=device)

    head_params = list(model.head.parameters())
    encoder_params = [p for p in model.sam2.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.learning_rate},
            {"params": encoder_params, "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = BCEDiceLoss()

    start_epoch = 0
    next_batch = 0
    global_step = 0
    epoch_loss_sum = 0.0
    epoch_sample_count = 0
    best_val_iou = 0.0

    if resume_path is not None:
        state = load_resume_checkpoint(resume_path, model, optimizer, scaler)
        start_epoch = state["epoch"]
        next_batch = state["next_batch"]
        global_step = state["global_step"]
        epoch_loss_sum = state["epoch_loss_sum"]
        epoch_sample_count = state["epoch_sample_count"]
        best_val_iou = state["best_val_iou"]
        if best_path.exists():
            try:
                best_state = torch.load(best_path, map_location="cpu", weights_only=False)
                best_iou_from_file = best_state.get("metrics", {}).get("iou", 0.0)
                best_val_iou = max(best_val_iou, best_iou_from_file)
            except Exception as exc:




                pass
    if history_path.exists():
        with history_path.open("r", encoding="utf-8-sig", newline="") as f:
            history_rows = list(csv.DictReader(f))

    interrupted = False

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, on_interrupt)
    start_time = time.time()

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            progress = tqdm(loader, desc="epoch")

            for batch_idx, (images, masks, _, _) in enumerate(progress):
                if resume_path is not None and batch_idx < next_batch:
                    continue
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                if bsm_net is not None:
                    images = bsm_net.maybe_mix(
                        images, prob=args.bsm_prob, alpha=args.bsm_alpha, size=args.bsm_size
                    )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                epoch_loss_sum += float(loss.detach()) * images.size(0)
                epoch_sample_count += images.size(0)
                progress.set_postfix(loss=f"{loss.item():.4f}")

                if (
                    args.save_every_batches
                    and global_step % args.save_every_batches == 0
                ):
                    save_resume_checkpoint(
                        checkpoint_path, epoch, batch_idx + 1, global_step,
                        epoch_loss_sum, epoch_sample_count, best_val_iou,
                        model, optimizer, scaler, args,
                    )
                if args.max_train_batches and batch_idx + 1 >= args.max_train_batches:
                    break
                if interrupted:
                    break

            # 
            save_resume_checkpoint(
                checkpoint_path, epoch + 1, 0, global_step,
                epoch_loss_sum, epoch_sample_count, best_val_iou,
                model, optimizer, scaler, args,
            )
            resume_path = checkpoint_path
            train_loss = epoch_loss_sum / max(epoch_sample_count, 1)
            row = {
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "val_iou": "",
                "val_f1": "",
                "val_precision": "",
                "val_recall": "",
                "best_iou": f"{best_val_iou:.4f}",
                "elapsed_s": f"{time.time() - start_time:.0f}",
            }

            if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                metrics = evaluate(model, eval_loader, device)
                row.update({
                    "val_iou": f"{metrics['iou']:.4f}",
                    "val_f1": f"{metrics['f1']:.4f}",
                    "val_precision": f"{metrics['precision']:.4f}",
                    "val_recall": f"{metrics['recall']:.4f}",
                })
                print(
                    f"Epoch {epoch + 1}: loss={train_loss:.4f} | "
                    f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f}"
                )
                if metrics["iou"] > best_val_iou:
                    best_val_iou = metrics["iou"]
                    row["best_iou"] = f"{best_val_iou:.4f}"
                    atomic_torch_save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "metrics": metrics,
                        },
                        best_path,
                    )
            else:

                pass
            history_rows.append(row)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                writer.writeheader()
                writer.writerows(history_rows)

            if interrupted:
                break

    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    metrics = evaluate(model, eval_loader, device)
    lines = []
    lines.append("")
    lines.append(f"IoU:      {metrics['iou']:.4f}")
    lines.append(f"F1:       {metrics['f1']:.4f}")
    lines.append(f"Precision:{metrics['precision']:.4f}")
    lines.append(f"Recall:   {metrics['recall']:.4f}")
    lines.append(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    lines.append("")
    for c, v in sorted(metrics["per_city"].items()):
        lines.append(f"  {c:<16} {v:.4f}")
    report = "\n".join(lines)
    print(report)
    eval_summary_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
