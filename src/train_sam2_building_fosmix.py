"""SAM2 寰皟璁粌锛氫簩鍊煎缓绛戠墿鎻愬彇銆?
鍦?SAM2 棰勮缁?image encoder 杈撳嚭鐨勫灏哄害鐗瑰緛涓婃帴涓€涓交閲?FPN 鍒嗗壊澶达紝
鐢?WHU 澶氬煄甯傚垏鐗囪缁冿紝璇勪及鏃跺湪鍗曠嫭鍩庡競涓婃姤鍛?IoU/F1銆?
鐗圭偣锛?  - 榛樿鍐荤粨 SAM2 缂栫爜鍣紝鍙缁冨垎鍓插ご锛堟樉瀛樺弸濂姐€佺ǔ瀹氾級锛?    鍔?--finetune-encoder 鍚庡悓鏃跺井璋?image encoder锛堜笂闄愭洿楂橈紝鏄惧瓨鍗犵敤鏇村ぇ锛夈€?  - 鏀寔 Ctrl+C 瀹夊叏鍋滄锛屾柇鐐圭画璁敤 --resume锛堢画璁椂瑕佹眰 --num-workers 0锛夈€?
鐢ㄦ硶绀轰緥锛?  python src/train_sam2_building.py
  python src/train_sam2_building.py --finetune-encoder --batch-size 1
  python src/train_sam2_building.py --resume checkpoints\resume_sam2_whu_tiny.pth --num-workers 0
"""

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
from src.fosmix_module import FOSMixAugment  # noqa: E402
from src.train_sam2_building import MixStyle  # noqa: E402

MODEL_DIR = Path(os.environ.get("SAM2_MODEL_DIR", "models/sam2"))
MODEL_CFG = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", MODEL_DIR / "sam2.1_hiera_tiny.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", MODEL_DIR / "sam2.1_hiera_base_plus.pt"),
}

DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "data" / "splits_whu" / "whu_mix_4city_manifest.csv"
DEFAULT_EVAL_MANIFEST = PROJECT_ROOT / "data" / "splits_whu" / "whu_classic_manifest.csv"
RANDOM_SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TileDataset(Dataset):
    """鎸?manifest CSV 璇诲彇 512脳512 鍒囩墖涓庢帺鐮併€?""

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
            print(f"璀﹀憡锛氳烦杩?{len(missing)} 涓枃浠朵笉瀛樺湪鐨勬牱鏈紙渚嬪 {missing[0]['tile_name']}锛?)
            rows = [r for r in rows if r not in missing]
        if not rows:
            raise RuntimeError(f"娓呭崟涓病鏈夊彲鐢ㄦ牱鏈細{self.manifest_path}")
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
    """杞婚噺 FPN 鍒嗗壊澶达細澶氬昂搴︾壒寰佽瀺鍚堝悗杈撳嚭 1 閫氶亾 logits銆?""

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
    """SAM2 image encoder + FPN 鍒嗗壊澶达紙鍙€?FPN 鐗瑰緛绾?MixStyle锛夈€?""

    def __init__(
        self,
        model_name: str = "tiny",
        finetune_encoder: bool = False,
        mixstyle: bool = False,
        mixstyle_alpha: float = 0.1,
    ) -> None:
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
        self.mixstyle = MixStyle(alpha=mixstyle_alpha) if mixstyle else None

        self.register_buffer("norm_mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        backbone_out = self.sam2.forward_image(x)
        feats = backbone_out["backbone_fpn"]
        if self.mixstyle is not None:
            feats = [self.mixstyle(f) for f in feats]
        return self.head(feats, target_size=(x.shape[-2], x.shape[-1]))


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
    mixer: nn.Module | None = None,
    mask_optimizer: torch.optim.Optimizer | None = None,
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
            "mixer_state_dict": mixer.state_dict() if mixer is not None else None,
            "mask_optimizer_state_dict": mask_optimizer.state_dict() if mask_optimizer is not None else None,
            "training_arguments": training_arguments,
        },
        path,
    )


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    mixer: nn.Module | None = None,
    mask_optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    state = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    if mixer is not None and state.get("mixer_state_dict") is not None:
        mixer.load_state_dict(state["mixer_state_dict"])
        if mask_optimizer is not None and state.get("mask_optimizer_state_dict") is not None:
            mask_optimizer.load_state_dict(state["mask_optimizer_state_dict"])
    return state


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    tp = fp = fn = 0
    per_city: dict[str, list[float]] = {}
    for images, masks, names, cities in tqdm(data_loader, desc="璇勪及", unit="鎵?):
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
    parser = argparse.ArgumentParser(description="SAM2 寰皟璁粌锛圵HU 寤虹瓚鐗╂彁鍙栵級")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--eval-manifest-path", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--eval-max", type=int, default=1000, help="璇勪及鍩庡競鏈€澶氭娊澶氬皯寮?)
    parser.add_argument("--test-manifest-path", type=Path, default=None,
                        help="鐣欏嚭娴嬭瘯娓呭崟锛堝彲閫夛級锛氭瘡杞簮鍩熻瘎浼板悗鑷姩娴嬫硾鍖栧苟鍐欏叆 CSV")
    parser.add_argument("--test-max", type=int, default=1000, help="娴嬭瘯闆嗘瘡鍩庡競鏈€澶氭娊澶氬皯寮?)
    parser.add_argument("--train-max-per-city", type=int, default=1000, help="璁粌闆嗘瘡鍩庡競鏈€澶氭娊澶氬皯寮?)
    parser.add_argument("--model", choices=list(MODEL_CFG), default="tiny")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="鍒嗗壊澶村涔犵巼")
    parser.add_argument("--encoder-lr", type=float, default=1e-5, help="寰皟缂栫爜鍣ㄦ椂鐨勫涔犵巼")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--finetune-encoder", action="store_true", help="鍚屾椂寰皟 SAM2 image encoder")
    parser.add_argument("--augmentation", choices=["basic", "strong"], default="basic",
                        help="鏁版嵁澧炲己绾у埆锛歜asic=缈昏浆/鏃嬭浆90锛泂trong=鍐嶅姞浠垮皠+棰滆壊鎶栧姩")
    parser.add_argument("--mask-lr", type=float, default=1e-4, help="FOSMix mask 缃戠粶瀛︿範鐜囷紙璁烘枃榛樿 1e-4锛?)
    parser.add_argument("--no-mask-net", action="store_true", help="娑堣瀺锛氱鐢ㄦ帺鐮佺綉缁滐紝浠?Full Mix锛堟棤 Optimal Mix / 涓€鑷存€э級")
    parser.add_argument("--mask-lasso-weight", type=float, default=5.0, help="FOSMix mask 绋€鐤忔鍒欐潈閲嶏紙璁烘枃榛樿 5.0锛?)
    parser.add_argument("--mask-variance-weight", type=float, default=0.1, help="FOSMix mask 鏂瑰樊姝ｅ垯鏉冮噸锛堣鏂囬粯璁?0.1锛?)
    parser.add_argument("--consistency-weight", type=float, default=1e-4, help="FOSMix 涓€鑷存€ф崯澶辨潈閲嶏紙璁烘枃榛樿 1e-4锛?)
    parser.add_argument("--fosmix-res", type=int, default=512, help="FOSMix 娣峰悎鍒嗚鲸鐜囷紙璁烘枃璁粌鐢?512锛屾贩鍚堝悗鏀惧ぇ鍥炲師灏哄锛?)
    parser.add_argument("--mixstyle", action="store_true",
                        help="鍙屽眰绾ц仈鍚堬細鍦?FPN 澶氬昂搴︾壒寰佷笂鍚敤 MixStyle锛坧=0.5, alpha=0.1锛?)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1, help="MixStyle Beta 鍒嗗竷鍙傛暟锛堥粯璁?0.1锛?)
    parser.add_argument("--ckpt-dir", type=Path, default=None,
                        help="妫€鏌ョ偣淇濆瓨鐩綍锛堥粯璁わ細椤圭洰 checkpoints 鐩綍锛?)
    parser.add_argument("--seed", type=int, default=42, help="闅忔満绉嶅瓙锛堣鏂囬粯璁?42锛?)
    parser.add_argument("--save-every-epochs", type=int, default=0,
                        help="姣?N 杞繚瀛樹竴浠?epoch{缂栧彿}_<run鍚?.pth锛?=涓嶄繚瀛橈級")
    parser.add_argument("--init-ckpt", type=Path, default=None,
                        help="浠庢寚瀹氭潈閲嶅垵濮嬪寲锛堝彧鍔犺浇妯″瀷鏉冮噸锛夛紝閰嶅悎 --init-epoch 浣跨敤")
    parser.add_argument("--init-epoch", type=int, default=1,
                        help="鍒濆鍖栨潈閲嶅搴旂殑宸插畬鎴愯疆鏁帮紙濡?2 = 浠庣 2 杞ā鍨嬬户缁紝涓嬩竴杞樉绀?Epoch 3锛?)
    parser.add_argument("--run-name", type=str, default="sam2_whu_tiny")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every-batches", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None, help="姣忚疆鏈€澶氳缁冩壒娆℃暟锛堝揩閫熸祴璇曠敤锛?)
    parser.add_argument("--eval-every", type=int, default=1, help="姣忓灏戣疆璇勪及涓€娆?)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.init_ckpt is not None and args.resume is not None:
        raise ValueError("--init-ckpt 涓?--resume 涓嶈兘鍚屾椂浣跨敤")
    if args.batch_size < 2:
        raise ValueError("FOSMix 闇€瑕?--batch-size >= 2锛堢敤鍚屾壒鍙︿竴寮犲浘鍋氬弬鑰冿級")
    if args.resume is not None and args.num_workers != 0:
        raise ValueError("鏂偣缁瑕佹眰 --num-workers 0锛堜繚璇佹壒娆￠『搴忎竴鑷达級銆?)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM2 寰皟闇€瑕?CUDA銆?)

    resume_path = args.resume
    ckpt_root = args.ckpt_dir if args.ckpt_dir else PROJECT_ROOT / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_root / f"resume_{args.run_name}.pth"
    best_path = ckpt_root / f"best_{args.run_name}.pth"
    test_best_path = ckpt_root / f"test_best_{args.run_name}.pth"
    history_path = PROJECT_ROOT / "logs" / f"{args.run_name}_training_history.csv"
    eval_summary_path = PROJECT_ROOT / "outputs" / f"{args.run_name}_final_eval.txt"

    print("=" * 50)
    print(f"SAM2 寰皟璁粌锛坽args.model}锛?)
    print("=" * 50)
    print("璁粌娓呭崟锛?, args.manifest_path)
    print("璇勪及娓呭崟锛?, args.eval_manifest_path, f"(split={args.eval_split}, max={args.eval_max})")
    print("妯″瀷锛?, args.model, "| 寰皟缂栫爜鍣細", args.finetune_encoder)
    print("FOSMix锛歮ask_lr=", args.mask_lr, "lasso=", args.mask_lasso_weight, "var=", args.mask_variance_weight, "consistency=", args.consistency_weight, "res=", args.fosmix_res)
    print("MixStyle锛?, f"鍚敤 alpha={args.mixstyle_alpha}" if args.mixstyle else "鍏抽棴")
    print("杞暟锛?, args.epochs, "| batch锛?, args.batch_size, "| workers锛?, args.num_workers)

    train_dataset = TileDataset(
        args.manifest_path, max_per_city=args.train_max_per_city,
        augment=True, augment_level=args.augmentation,
    )
    eval_dataset = TileDataset(args.eval_manifest_path, split=args.eval_split, max_per_city=args.eval_max)
    test_dataset = (
        TileDataset(args.test_manifest_path, split="test", max_per_city=args.test_max)
        if args.test_manifest_path is not None
        else None
    )
    print(f"璁粌鏍锋湰锛歿len(train_dataset)} | 璇勪及鏍锋湰锛歿len(eval_dataset)}")
    for label, ds in (("璁粌鎸夊煄甯?, train_dataset), ("璇勪及鎸夊煄甯?, eval_dataset)):
        counts = {}
        for r in ds.rows:
            counts[r["city"]] = counts.get(r["city"], 0) + 1
        print(label + "锛?, ", ".join(f"{c}={n}" for c, n in sorted(counts.items())))

    model = SAM2Segmentor(
        model_name=args.model,
        finetune_encoder=args.finetune_encoder,
        mixstyle=args.mixstyle,
        mixstyle_alpha=args.mixstyle_alpha,
    )
    model.to(device)

    mixer = FOSMixAugment(use_mask_net=not args.no_mask_net).to(device)
    if args.no_mask_net:
        print("FOSMix 娑堣瀺妯″紡锛?-no-mask-net锛屼粎 Full Mix锛堟帺鐮佺綉缁滃凡绂佺敤锛?)
    else:
        print("FOSMix mask 缃戠粶宸插姞杞斤細resnet18 ImageNet 棰勮缁?)

    head_params = list(model.head.parameters())
    encoder_params = [p for p in model.sam2.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.learning_rate},
            {"params": encoder_params, "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )

    mask_optimizer = (
        torch.optim.AdamW(mixer.parameters(), lr=args.mask_lr, weight_decay=args.weight_decay)
        if not args.no_mask_net
        else None
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

    if args.init_ckpt is not None:
        init_state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(init_state["model_state_dict"])
        init_metrics = init_state.get("metrics", {}) or {}
        best_val_iou = float(init_metrics.get("iou", 0.0))
        start_epoch = max(0, args.init_epoch)
        next_batch = 0
        print(
            f"宸蹭粠 {args.init_ckpt} 鍒濆鍖栵紙绗?{args.init_epoch} 杞ā鍨嬶紝best IoU={best_val_iou:.4f}锛夛紝"
            f"涓嬩竴杞粠 Epoch {start_epoch + 1} 寮€濮?
        )
        if history_path.exists():
            with history_path.open("r", encoding="utf-8-sig", newline="") as f:
                old_rows = list(csv.DictReader(f))
            old_rows = [r for r in old_rows if int(float(r["epoch"])) < args.init_epoch]
            with history_path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(old_rows[0].keys()) if old_rows else ["epoch"])
                w.writeheader()
                w.writerows(old_rows)

    if resume_path is not None:
        state = load_resume_checkpoint(resume_path, model, optimizer, scaler, mixer, mask_optimizer)
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

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"鍙缁冨弬鏁帮細{n_trainable / 1e6:.2f}M")

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
            mixer.train()
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
                if mask_optimizer is not None:
                    mask_optimizer.zero_grad(set_to_none=True)

                # FOSMix锛氬悓鎵瑰唴婊氫竴寮犲浘浣滃弬鑰冿紝榛樿 512 鍒嗚鲸鐜囧仛 DCT 棰戠巼娣峰悎鍐嶆斁澶у洖鍘熷昂瀵?                mix_input = images
                if args.fosmix_res and (images.shape[-2] != args.fosmix_res or images.shape[-1] != args.fosmix_res):
                    mix_input = F.interpolate(
                        images, size=(args.fosmix_res, args.fosmix_res),
                        mode="bilinear", align_corners=False,
                    )
                if args.no_mask_net:
                    # 娑堣瀺锛氫粎 Full Mix锛屾棤鎺╃爜缃戠粶 / 鏃?Optimal Mix / 鏃犱竴鑷存€?                    full_image, _, _ = mixer(
                        mix_input, torch.roll(mix_input, shifts=1, dims=0)
                    )
                    if args.fosmix_res and (full_image.shape[-2] != images.shape[-2] or full_image.shape[-1] != images.shape[-1]):
                        full_image = F.interpolate(
                            full_image, size=(images.shape[-2], images.shape[-1]),
                            mode="bilinear", align_corners=False,
                        )
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images)
                        full_logits = model(full_image)
                        seg_loss = criterion(logits, masks) + criterion(full_logits, masks)
                    loss = seg_loss
                else:
                    optimal_image, full_image, freq_mask = mixer(
                        mix_input, torch.roll(mix_input, shifts=1, dims=0)
                    )
                    if args.fosmix_res and (optimal_image.shape[-2] != images.shape[-2] or optimal_image.shape[-1] != images.shape[-1]):
                        optimal_image = F.interpolate(
                            optimal_image, size=(images.shape[-2], images.shape[-1]),
                            mode="bilinear", align_corners=False,
                        )
                        full_image = F.interpolate(
                            full_image, size=(images.shape[-2], images.shape[-1]),
                            mode="bilinear", align_corners=False,
                        )

                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images)
                        opt_logits = model(optimal_image)
                        full_logits = model(full_image)
                        seg_loss = (
                            criterion(logits, masks)
                            + criterion(opt_logits, masks)
                            + criterion(full_logits, masks)
                        )
                        mask_reg = (
                            args.mask_lasso_weight * freq_mask.mean()
                            + args.mask_variance_weight / (freq_mask.var(unbiased=False) + 1e-6)
                        )
                        consistency = args.consistency_weight * F.l1_loss(opt_logits, full_logits)
                        loss = seg_loss + mask_reg + consistency
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                if mask_optimizer is not None:
                    scaler.step(mask_optimizer)
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
                        model, optimizer, scaler, args, mixer, mask_optimizer,
                    )
                if args.max_train_batches and batch_idx + 1 >= args.max_train_batches:
                    break
                if interrupted:
                    save_resume_checkpoint(
                        checkpoint_path, epoch, batch_idx + 1, global_step,
                        epoch_loss_sum, epoch_sample_count, best_val_iou,
                        model, optimizer, scaler, args, mixer, mask_optimizer,
                    )
                    print(f"鏀跺埌涓柇淇″彿锛屾柇鐐瑰凡淇濆瓨锛氫笅娆′粠 Epoch {epoch + 1} 绗?{batch_idx + 1} 鎵圭户缁€?)
                    break

            if interrupted:
                break

            # 姣?5 杞繚瀛樹竴娆℃柇鐐癸紙鏈€鍚庝竴杞繀瀛橈級
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                save_resume_checkpoint(
                    checkpoint_path, epoch + 1, 0, global_step,
                    epoch_loss_sum, epoch_sample_count, best_val_iou,
                    model, optimizer, scaler, args, mixer, mask_optimizer,
                )
            resume_path = checkpoint_path  # 涔嬪悗涓柇涔熷厑璁告仮澶?            next_batch = 0  # 淇锛氭柊杞涓嶅啀璺宠繃鍓?next_batch 涓?batch

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
                    print(f"鏂扮殑鏈€浣?IoU锛歿best_val_iou:.4f} -> {best_path}")
                if test_dataset is not None:
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                    )
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
                    print(
                        f"  [娴嬭瘯闆哴 IoU={test_metrics['iou']:.4f} F1={test_metrics['f1']:.4f} -> {test_csv.name}"
                    )
                    if test_metrics["iou"] > best_test_iou:
                        best_test_iou = test_metrics["iou"]
                        print(f"鏂扮殑娴嬭瘯闆嗘渶浣?IoU锛歿best_test_iou:.4f}锛坱est_best 涓嶅啀淇濆瓨锛?)
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
                atomic_torch_save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                    },
                    epoch_path,
                )
                print(f"宸蹭繚瀛樻湰杞潈閲嶏細{epoch_path.name}")

    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    if interrupted:
        print("璁粌宸蹭腑鏂紝璺宠繃鏈€缁堣瘎浼般€?)
        return

    # 鏈€缁堣瘎浼?    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    metrics = evaluate(model, eval_loader, device)
    lines = ["=" * 50, f"SAM2 ({args.model}) WHU 鏈€缁堣瘎浼?, "=" * 50]
    lines.append(f"娴嬭瘯闆? {args.eval_manifest_path} (split={args.eval_split}, max={args.eval_max})")
    lines.append(f"IoU:      {metrics['iou']:.4f}")
    lines.append(f"F1:       {metrics['f1']:.4f}")
    lines.append(f"Precision:{metrics['precision']:.4f}")
    lines.append(f"Recall:   {metrics['recall']:.4f}")
    lines.append(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    lines.append("\n--- 鎸夊煄甯?---")
    for c, v in sorted(metrics["per_city"].items()):
        lines.append(f"  {c:<16} {v:.4f}")
    report = "\n".join(lines)
    print(report)
    eval_summary_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary_path.write_text(report, encoding="utf-8")
    print(f"\n鏈€缁堣瘎浼版姤鍛婏細{eval_summary_path}")
    print(f"鏈€浣虫ā鍨嬶細{best_path}")


if __name__ == "__main__":
    main()
