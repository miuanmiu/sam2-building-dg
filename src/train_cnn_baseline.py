# -*- coding: utf-8 -*-
"""U-Net / SegFormer-B0 同协议基线训练（WHU 4 城 800，20 轮，逐轮 Kitsap 泛化）。

与 SAM2 训练保持相同协议：源域验证选择最佳检查点（标准口径），每轮自动在
留出测试集（Kitsap）上评估并写入 *_test_generalization.csv；支持断点续训与
Ctrl+C 安全中断。

用法示例：
  python src/train_cnn_baseline.py --model unet --run-name unet_baseline_s42 ...
  python src/train_cnn_baseline.py --model segformer_b0 --run-name segformer_b0_baseline_s42 ...
"""

import argparse
import csv
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import BCEDiceLoss  # noqa: E402
from src.models.deeplabv3_resnet50 import DeepLabV3ResNet50  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.models.segformer_b0 import SegFormerB0  # noqa: E402

RANDOM_SEED = 42


class TileDataset(Dataset):
    """按 manifest CSV 读取 512×512 切片与掩码（与 SAM2 训练一致）。"""

    def __init__(
        self,
        manifest_path: Path,
        split: str | None = None,
        cities: list[str] | None = None,
        max_per_city: int | None = None,
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.augment = augment
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
            rng = random.Random(RANDOM_SEED)
            capped: list[dict] = []
            for city, city_rows in sorted(by_city.items()):
                capped.extend(rng.sample(city_rows, min(max_per_city, len(city_rows))))
            rows = capped
        missing = [
            r for r in rows
            if not (Path(r["image_path"]).exists() and Path(r["mask_path"]).exists())
        ]
        if missing:
            print(f"警告：跳过 {len(missing)} 个文件不存在的样本")
            rows = [r for r in rows if r not in missing]
        if not rows:
            raise RuntimeError(f"清单中没有可用样本：{self.manifest_path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0)        # [1,H,W]
        if self.augment:
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
        return image, mask, row["tile_name"], row["city"]


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    tp = fp = fn = 0
    per_city: dict[str, list[float]] = {}
    for images, masks, names, cities in tqdm(data_loader, desc="评估", unit="批"):
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
        "iou": iou, "f1": f1, "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "per_city": {c: float(np.mean(v)) for c, v in per_city.items()},
    }


def atomic_torch_save(content: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    torch.save(content, tmp)
    os.replace(tmp, output_path)


def save_resume_checkpoint(
    path: Path, epoch: int, next_batch: int, global_step: int,
    epoch_loss_sum: float, epoch_sample_count: int, best_val_iou: float,
    model: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, args: argparse.Namespace,
) -> None:
    atomic_torch_save({
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_sample_count": epoch_sample_count,
        "best_val_iou": best_val_iou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_arguments": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in vars(args).items()},
    }, path)


def load_resume_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict:
    state = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    return state


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net / SegFormer-B0 同协议基线训练（WHU）")
    parser.add_argument("--model", choices=["unet", "segformer_b0", "deeplabv3_resnet50"], required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--eval-manifest-path", type=Path, required=True)
    parser.add_argument("--test-manifest-path", type=Path, default=None,
                        help="留出测试清单（可选）：每轮源域评估后自动测泛化并写入 CSV")
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--eval-max", type=int, default=100)
    parser.add_argument("--test-max", type=int, default=1000)
    parser.add_argument("--train-max-per-city", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="默认：unet/deeplabv3_resnet50=3e-4，segformer_b0=6e-5")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="默认：unet/deeplabv3_resnet50=1e-4，segformer_b0=0.01")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--save-every-batches", type=int, default=200)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.resume is not None and args.num_workers != 0:
        raise ValueError("断点续训要求 --num-workers 0（保证批次顺序一致）。")

    if args.learning_rate is None:
        args.learning_rate = 3e-4 if args.model in ("unet", "deeplabv3_resnet50") else 6e-5
    if args.weight_decay is None:
        args.weight_decay = 1e-4 if args.model in ("unet", "deeplabv3_resnet50") else 0.01

    # U-Net 数值稳定性修复：关闭 AMP、加梯度裁剪（E10 起 NaN 塌缩的根因处理）
    use_amp = args.model != "unet"
    grad_clip = 1.0 if args.model == "unet" else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("基线训练需要 CUDA。")

    ckpt_root = args.ckpt_dir if args.ckpt_dir else PROJECT_ROOT / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / f"resume_{args.run_name}.pth"
    best_path = ckpt_root / f"best_{args.run_name}.pth"
    test_best_path = ckpt_root / f"test_best_{args.run_name}.pth"
    history_path = PROJECT_ROOT / "logs" / f"{args.run_name}_training_history.csv"
    gen_path = PROJECT_ROOT / "logs" / f"{args.run_name}_test_generalization.csv"
    final_eval_path = PROJECT_ROOT / "outputs" / f"{args.run_name}_final_eval.txt"

    print("=" * 50)
    print(f"CNN 基线训练（{args.model}）")
    print("=" * 50)
    print("训练清单：", args.manifest_path)
    print("评估清单：", args.eval_manifest_path, f"(split={args.eval_split}, max={args.eval_max})")
    print("轮数：", args.epochs, "| batch：", args.batch_size, "| lr：", args.learning_rate,
          "| wd：", args.weight_decay, "| seed：", args.seed,
          "| amp：", use_amp, "| grad_clip：", grad_clip)

    train_dataset = TileDataset(args.manifest_path, max_per_city=args.train_max_per_city, augment=True)
    eval_dataset = TileDataset(args.eval_manifest_path, split=args.eval_split, max_per_city=args.eval_max)
    test_dataset = (
        TileDataset(args.test_manifest_path, split="test", max_per_city=args.test_max)
        if args.test_manifest_path is not None else None
    )
    print(f"训练样本：{len(train_dataset)} | 评估样本：{len(eval_dataset)}")
    if test_dataset is not None:
        print(f"留出测试样本：{len(test_dataset)}")

    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    elif args.model == "deeplabv3_resnet50":
        model = DeepLabV3ResNet50(pretrained=True).to(device)
    else:
        model = SegFormerB0(pretrained=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = BCEDiceLoss()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数：{n_trainable / 1e6:.2f}M")

    start_epoch = 0
    next_batch = 0
    global_step = 0
    epoch_loss_sum = 0.0
    epoch_sample_count = 0
    best_val_iou = 0.0
    best_test_iou = 0.0

    if args.resume is not None:
        state = load_resume_checkpoint(args.resume, model, optimizer, scaler)
        start_epoch = state["epoch"]
        next_batch = state["next_batch"]
        global_step = state["global_step"]
        epoch_loss_sum = state["epoch_loss_sum"]
        epoch_sample_count = state["epoch_sample_count"]
        best_val_iou = state["best_val_iou"]
        if test_best_path.exists():
            tb_state = torch.load(test_best_path, map_location="cpu", weights_only=False)
            best_test_iou = float(tb_state.get("metrics", {}).get("iou", 0.0))
        print(f"已从断点恢复：epoch={start_epoch} batch={next_batch} best_iou={best_val_iou:.4f}")

    history_rows: list[dict] = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8-sig", newline="") as f:
            history_rows = list(csv.DictReader(f))

    interrupted = False

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n收到中断信号，正在保存断点...")

    signal.signal(signal.SIGINT, on_interrupt)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="批")
        for batch_idx, (images, masks, _, _) in enumerate(progress):
            if args.resume is not None and batch_idx < next_batch:
                continue
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            global_step += 1
            epoch_loss_sum += float(loss.detach()) * images.size(0)
            epoch_sample_count += images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.save_every_batches and global_step % args.save_every_batches == 0:
                save_resume_checkpoint(
                    checkpoint_path, epoch, batch_idx + 1, global_step,
                    epoch_loss_sum, epoch_sample_count, best_val_iou,
                    model, optimizer, scaler, args,
                )
            if args.max_train_batches and batch_idx + 1 >= args.max_train_batches:
                break
            if interrupted:
                save_resume_checkpoint(
                    checkpoint_path, epoch, batch_idx + 1, global_step,
                    epoch_loss_sum, epoch_sample_count, best_val_iou,
                    model, optimizer, scaler, args,
                )
                print(f"中断，断点已保存：Epoch {epoch + 1} 第 {batch_idx + 1} 批。")
                break

        if interrupted:
            break

        save_resume_checkpoint(
            checkpoint_path, epoch + 1, 0, global_step,
            epoch_loss_sum, epoch_sample_count, best_val_iou,
            model, optimizer, scaler, args,
        )
        args.resume = checkpoint_path  # 之后中断也允许恢复

        train_loss = epoch_loss_sum / max(epoch_sample_count, 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_iou": "", "val_f1": "", "val_precision": "", "val_recall": "",
            "best_iou": f"{best_val_iou:.4f}",
            "elapsed_s": f"{time.time() - start_time:.0f}",
        }

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)
            metrics = evaluate(model, eval_loader, device)
            row.update({
                "val_iou": f"{metrics['iou']:.4f}",
                "val_f1": f"{metrics['f1']:.4f}",
                "val_precision": f"{metrics['precision']:.4f}",
                "val_recall": f"{metrics['recall']:.4f}",
            })
            print(f"Epoch {epoch + 1}: loss={train_loss:.4f} | IoU={metrics['iou']:.4f} "
                  f"F1={metrics['f1']:.4f}")
            if metrics["iou"] > best_val_iou:
                best_val_iou = metrics["iou"]
                row["best_iou"] = f"{best_val_iou:.4f}"
                atomic_torch_save({
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "epoch": epoch + 1,
                }, best_path)
                print(f"新的最佳 IoU：{best_val_iou:.4f} -> {best_path}")
            if args.save_every_epochs and (epoch + 1) % args.save_every_epochs == 0:
                atomic_torch_save({"model_state_dict": model.state_dict()},
                                  ckpt_root / f"epoch{epoch + 1:03d}_{args.run_name}.pth")
                print(f"已保存本轮权重：epoch{epoch + 1:03d}_{args.run_name}.pth")
            if test_dataset is not None:
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
                t_metrics = evaluate(model, test_loader, device)
                new_file = not gen_path.exists()
                with gen_path.open("a", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["epoch", "iou", "f1", "precision", "recall"])
                    if new_file:
                        w.writeheader()
                    w.writerow({
                        "epoch": epoch + 1,
                        "iou": f"{t_metrics['iou']:.4f}",
                        "f1": f"{t_metrics['f1']:.4f}",
                        "precision": f"{t_metrics['precision']:.4f}",
                        "recall": f"{t_metrics['recall']:.4f}",
                    })
                if t_metrics["iou"] > best_test_iou:
                    best_test_iou = t_metrics["iou"]
                    atomic_torch_save({
                        "model_state_dict": model.state_dict(),
                        "metrics": t_metrics,
                        "epoch": epoch + 1,
                    }, test_best_path)
                    print(f"新的测试集最佳 IoU：{best_test_iou:.4f} -> {test_best_path}")
                print(f"  [测试集] IoU={t_metrics['iou']:.4f} F1={t_metrics['f1']:.4f} "
                      f"-> {gen_path.name}")
        history_rows.append(row)
        with history_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerows(history_rows)

    # 正常跑完时：用最佳检查点在留出测试集上做最终评估
    if not interrupted and test_dataset is not None and best_path.exists():
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
        m = evaluate(model, test_loader, device)
        lines = [
            "=" * 50,
            f"CNN 基线（{args.model}）最终评估",
            "=" * 50,
            f"测试集: {args.test_manifest_path} (split=test, max={args.test_max})",
            f"IoU:      {m['iou']:.4f}",
            f"F1:       {m['f1']:.4f}",
            f"Precision:{m['precision']:.4f}",
            f"Recall:   {m['recall']:.4f}",
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']}",
        ]
        for c, v in sorted(m["per_city"].items()):
            lines.append(f"  {c:<16} {v:.4f}")
        final_eval_path.parent.mkdir(parents=True, exist_ok=True)
        with final_eval_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"最终评估报告：{final_eval_path}")

    print(f"训练结束（epoch 进度 {min(args.epochs, epoch + (0 if interrupted else 1))}/{args.epochs}）")


if __name__ == "__main__":
    main()
