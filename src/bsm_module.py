"""Batch Style Mixing (BSM) 复现模块

论文: A Diverse Large-Scale Building Dataset and a Novel Plug-and-Play
Domain Generalization Method for Building Extraction (IEEE TGRS 2023,
arXiv:2208.10004)

风格转移网络结构与预训练权重来自:
Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive
Instance Normalization" (ICCV 2017); 参考实现 naoto0804/pytorch-AdaIN。

用法（训练脚本内）:
    net = BSMStyleTransfer(weights_dir, device="cuda")
    # 批次级风格混合（论文第 2 步：shuffle + AdaIN, p=0.5, alpha=0.5）
    images = net.maybe_mix(images, prob=0.5, alpha=0.5, size=512)

几何/颜色增强是逐样本的:
    image, mask = bsm_geometric_augment(image, mask)   # p=0.5 由调用方控制
    image = bsm_color_augment(image)                   # p=0.5 由调用方控制
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    """按 batch/channel 计算 feature 的均值与标准差（AdaIN 用）。"""
    n, c = feat.size()[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor):
    """AdaIN: 把 content 的 channel 统计量替换为 style 的。"""
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _build_vgg_encoder() -> nn.Sequential:
    """与 naoto0804/pytorch-AdaIN net.py 完全一致的 VGG 编码器。"""
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-4
    )


def _build_decoder() -> nn.Sequential:
    """与 naoto0804/pytorch-AdaIN net.py 完全一致的解码器。"""
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


class BSMStyleTransfer(nn.Module):
    """论文 BSM 中的风格转移网络：VGG 编码器 + AdaIN + 解码器（权重固定）。"""

    def __init__(self, weights_dir: Path, device: torch.device | str = "cuda") -> None:
        super().__init__()
        weights_dir = Path(weights_dir)
        vgg = _build_vgg_encoder()
        decoder = _build_decoder()
        vgg.load_state_dict(
            torch.load(weights_dir / "vgg_normalised.pth", map_location="cpu", weights_only=False)
        )
        decoder.load_state_dict(
            torch.load(weights_dir / "decoder.pth", map_location="cpu", weights_only=False)
        )
        # BSM 只取到 relu4-1（与 pytorch-AdaIN test.py 一致），解码器再放大 8 倍回原尺寸
        self.vgg = nn.Sequential(*list(vgg.children())[:31]).to(device).eval()
        self.decoder = decoder.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def mix(
        self,
        images: torch.Tensor,
        alpha: float = 0.5,
        size: int = 512,
    ) -> torch.Tensor:
        """论文 Eq.(1)+(2)：批次内 shuffle 配对，AdaIN 后按 alpha 线性插值再解码。"""
        if images.size(0) < 2:
            return images
        orig_size = (images.shape[-2], images.shape[-1])
        if size and (orig_size[0] != size or orig_size[1] != size):
            images = F.interpolate(
                images, size=(size, size), mode="bilinear", align_corners=False
            )
        perm = torch.randperm(images.size(0), device=images.device)
        style = images[perm]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            content_f = self.vgg(images)
            style_f = self.vgg(style)
        content_f = content_f.float()
        style_f = style_f.float()
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = alpha * feat + (1.0 - alpha) * content_f
        out = self.decoder(feat).float().clamp_(0.0, 1.0)
        if size and (out.shape[-2] != orig_size[0] or out.shape[-1] != orig_size[1]):
            out = F.interpolate(
                out, size=orig_size, mode="bilinear", align_corners=False
            )
        return out

    def maybe_mix(
        self,
        images: torch.Tensor,
        prob: float = 0.5,
        alpha: float = 0.5,
        size: int = 512,
    ) -> torch.Tensor:
        if random.random() < prob:
            return self.mix(images, alpha=alpha, size=size)
        return images


def bsm_geometric_augment(
    image: torch.Tensor,
    mask: torch.Tensor,
    angle_range: float = 30.0,
    scale_range: tuple[float, float] = (0.5, 2.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """论文 GA：水平/垂直翻转、±30° 旋转、0.5–2.0 缩放并裁剪/填充回原尺寸。
    图像与 mask 使用完全相同的变换。"""
    if random.random() < 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])
    if random.random() < 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])
    angle = random.uniform(-angle_range, angle_range)
    scale = random.uniform(*scale_range)
    image = TF.affine(
        image, angle=angle, translate=[0, 0], scale=scale, shear=0,
        interpolation=TF.InterpolationMode.BILINEAR, fill=0.0,
    )
    mask = TF.affine(
        mask, angle=angle, translate=[0, 0], scale=scale, shear=0,
        interpolation=TF.InterpolationMode.NEAREST, fill=0.0,
    )
    return image, mask


def bsm_color_augment(image: torch.Tensor) -> torch.Tensor:
    """论文 CA：亮度/色彩平衡/对比度/锐度调整 + 随机高斯模糊。
    论文未给出具体范围，这里用较常见的区间实现。"""
    image = TF.adjust_brightness(image, random.uniform(0.7, 1.4))
    image = TF.adjust_contrast(image, random.uniform(0.7, 1.4))
    image = TF.adjust_saturation(image, random.uniform(0.7, 1.4))
    image = TF.adjust_sharpness(image, random.uniform(0.0, 3.0))
    if random.random() < 0.5:
        image = TF.gaussian_blur(
            image, kernel_size=(5, 5), sigma=random.uniform(0.1, 1.0)
        )
    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=Path("checkpoints/adain_weights"))
    args = parser.parse_args()

    net = BSMStyleTransfer(args.weights_dir, device="cuda")
    dummy = torch.rand(4, 3, 1024, 1024, device="cuda")
    mixed = net.mix(dummy, alpha=0.5, size=512)
    print("输入:", tuple(dummy.shape), "范围", float(dummy.min()), float(dummy.max()))
    print("输出:", tuple(mixed.shape), "范围", float(mixed.min()), float(mixed.max()))
    print("BSMStyleTransfer 自检通过")
