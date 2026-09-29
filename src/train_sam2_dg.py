# -*- coding: utf-8 -*-
"""DSU / FACT 鍚屽崗璁缁冿紙SAM2 tiny + FPN锛? 鍩?800 鎴?LODO锛夈€?
鐢ㄦ硶绀轰緥锛堢敱 run_all_planned.ps1 璋冪敤锛夛細
  python train_sam2_dg.py --dg dsu --seed 42 --run-name sam2_whu_4city_800_20ep_dsu_s42 ...

杈撳嚭涓庝富瀹為獙瀹屽叏涓€鑷达細logs/<run>_training_history.csv銆?logs/<run>_test_generalization.csv銆乥est/resume/epoch 妫€鏌ョ偣銆?"""

import argparse
import csv
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import BCEDiceLoss  # noqa: E402
from src.train_sam2_building import (  # noqa: E402
    SAM2Segmentor,
    TileDataset,
    atomic_torch_save,
    evaluate,
    load_resume_checkpoint,
    save_resume_checkpoint,
)

from src.dg_modules import DSU, FACT  # noqa: E402


class DGModel(SAM2Segmentor):
    """鍦?SAM2Segmentor 涓婂彔鍔?DSU锛堢壒寰佺骇锛夋垨 FACT锛堣緭鍏ョ骇锛夈€?""

    def __init__(self, dg: str = "dsu", **kwargs) -> None:
        super().__init__(**kwargs)
        self.dg_name = dg
        self.dsu = DSU() if dg == "dsu" else None
        self.fact = FACT() if dg == "fact" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fact is not None and self.training:
            x = self.fact(x)
        x = (x - self.norm_mean) / self.norm_std
        feats = self.sam2.forward_image(x)["backbone_fpn"]
        if self.dsu is not None:
            feats = [self.dsu(f) for f in feats]
        return self.head(feats, target_size=(x.shape[-2], x.shape[-1]))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 寰皟璁粌锛圖SU/FACT锛?)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--eval-manifest-path", type=Path, required=True)
    parser.add_argument("--test-manifest-path", type=Path, default=None)
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--eval-max", type=int, default=100)
    parser.add_argument("--test-max", type=int, default=1000)
    parser.add_argument("--train-max-per-city", type=int, default=1000)
    parser.add_argument("--model", choices=["tiny", "base_plus"], default="tiny")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--dg", choices=["dsu", "fact"], default="dsu")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every-batches", type=int, default=200)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.resume is not None and args.num_workers != 0:
        raise ValueError("鏂偣缁瑕佹眰 --num-workers 0锛堜繚璇佹壒娆￠『搴忎竴鑷达級銆?)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("闇€瑕?CUDA銆?)

    resume_path = args.resume
    ckpt_root = args.ckpt_dir if args.ckpt_dir else PROJECT_ROOT / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / f"resume_{args.run_name}.pth"
    best_path = ckpt_root / f"best_{args.run_name}.pth"
    test_best_path = ckpt_root / f"test_best_{args.run_name}.pth"
    history_path = PROJECT_ROOT / "logs" / f"{args.run_name}_training_history.csv"

    train_dataset = TileDataset(
        args.manifest_path, max_per_city=args.train_max_per_city,
        augment=True, augment_level="basic",
    )
    eval_dataset = TileDataset(args.eval_manifest_path, split=args.eval_split, max_per_city=args.eval_max)
    test_dataset = (
        TileDataset(args.test_manifest_path, split="test", max_per_city=args.test_max)
        if args.test_manifest_path is not None
        else None
    )
    print(f"[{args.dg}] 璁粌鏍锋湰锛歿len(train_dataset)} | 璇勪及鏍锋湰锛歿len(eval_dataset)} | seed={args.seed}")

    model = DGModel(dg=args.dg, model_name=args.model, finetune_encoder=args.finetune_encoder)
    model.to(device)

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
    best_test_iou = 0.0

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
                print(f"璀﹀憡锛氳鍙栨渶浣虫ā鍨嬩俊鎭け璐ワ紙{exc}锛夛紝娌跨敤鏂偣涓殑 best IoU")
        if test_best_path.exists():
            try:
                tb_state = torch.load(test_best_path, map_location="cpu", weights_only=False)
                best_test_iou = float(tb_state.get("metrics", {}).get("iou", 0.0))
            except Exception as exc:
                print(f"璀﹀憡锛氳鍙栨祴璇曢泦鏈€浣虫ā鍨嬩俊鎭け璐ワ紙{exc}锛夛紝娌跨敤 0")
        print(f"宸蹭粠鏂偣鎭㈠锛歟poch={start_epoch} batch={next_batch} best_iou={best_val_iou:.4f}")

    history_rows: list[dict] = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8-sig", newline="") as f:
            history_rows = list(csv.DictReader(f))

    interrupted = False

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n鏀跺埌涓柇淇″彿锛屾鍦ㄤ繚瀛樻柇鐐?..")

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
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="鎵?)

            for batch_idx, (images, masks, _, _) in enumerate(progress):
                if resume_path is not None and batch_idx < next_batch:
                    continue
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

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
                    print(f"鏀跺埌涓柇淇″彿锛屾柇鐐瑰凡淇濆瓨锛氫笅娆′粠 Epoch {epoch + 1} 绗?{batch_idx + 1} 鎵圭户缁€?)
                    break

            if interrupted:
                break

            save_resume_checkpoint(
                checkpoint_path, epoch + 1, 0, global_step,
                epoch_loss_sum, epoch_sample_count, best_val_iou,
                model, optimizer, scaler, args,
            )
            resume_path = checkpoint_path
            next_batch = 0  # 淇锛氭柊杞涓嶅啀璺宠繃鍓?next_batch 涓?batch

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
                eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                metrics = evaluate(model, eval_loader, device)
                row.update({
                    "val_iou": f"{metrics['iou']:.4f}",
                    "val_f1": f"{metrics['f1']:.4f}",
                    "val_precision": f"{metrics['precision']:.4f}",
                    "val_recall": f"{metrics['recall']:.4f}",
                })
                print(f"Epoch {epoch + 1}: loss={train_loss:.4f} | IoU={metrics['iou']:.4f}")
                if metrics["iou"] > best_val_iou:
                    best_val_iou = metrics["iou"]
                    row["best_iou"] = f"{best_val_iou:.4f}"
                    atomic_torch_save(
                        {"model_state_dict": model.state_dict(), "args": vars(args), "metrics": metrics},
                        best_path,
                    )
                    print(f"鏂扮殑鏈€浣?IoU锛歿best_val_iou:.4f} -> {best_path}")
                if test_dataset is not None:
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                    test_metrics = evaluate(model, test_loader, device)
                    test_csv = PROJECT_ROOT / "logs" / f"{args.run_name}_test_generalization.csv"
                    test_csv.parent.mkdir(parents=True, exist_ok=True)
                    test_row = {
                        "epoch": epoch + 1,
                        "iou": f"{test_metrics['iou']:.4f}",
                        "f1": f"{test_metrics['f1']:.4f}",
                        "precision": f"{test_metrics['precision']:.4f}",
                        "recall": f"{test_metrics['recall']:.4f}",
                    }
                    write_header = not test_csv.exists()
                    with test_csv.open("a", encoding="utf-8-sig", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(test_row.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(test_row)
                    print(f"  [娴嬭瘯闆哴 IoU={test_metrics['iou']:.4f} -> {test_csv.name}")
                    if test_metrics["iou"] > best_test_iou:
                        best_test_iou = test_metrics["iou"]
                        atomic_torch_save(
                            {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "args": vars(args), "metrics": test_metrics},
                            test_best_path,
                        )
                        print(f"鏂扮殑娴嬭瘯闆嗘渶浣?IoU锛歿best_test_iou:.4f} -> {test_best_path}")
            else:
                print(f"Epoch {epoch + 1}: loss={train_loss:.4f}锛堟湰杞笉璇勪及锛?)

            history_rows.append(row)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                writer.writeheader()
                writer.writerows(history_rows)

            if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                epoch_path = ckpt_root / f"epoch{epoch + 1:03d}_{args.run_name}.pth"
                atomic_torch_save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "args": vars(args)}, epoch_path)
                print(f"宸蹭繚瀛樻湰杞潈閲嶏細{epoch_path.name}")

            if interrupted:
                print("璁粌宸蹭腑鏂紝鏂偣宸蹭繚瀛樸€?)
                break

    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    if interrupted:
        print("璁粌宸蹭腑鏂紝璺宠繃鏈€缁堣瘎浼般€?)
        return

    print("璁粌瀹屾垚銆?)


if __name__ == "__main__":
    main()
