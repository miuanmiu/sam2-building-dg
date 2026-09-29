from __future__ import annotations

import torch


def channel_mean_std(
    image: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 BCHW/CHW 图像逐样本、逐通道的空间均值和标准差。"""
    if image.ndim not in (3, 4):
        raise ValueError(
            f"image 必须是 CHW 或 BCHW 张量，当前形状为 {tuple(image.shape)}"
        )
    if not image.is_floating_point():
        raise TypeError("AdaIN 输入必须是浮点张量。")

    spatial_dims = (-2, -1)
    mean = image.mean(dim=spatial_dims, keepdim=True)
    variance = image.var(dim=spatial_dims, keepdim=True, unbiased=False)
    std = (variance + eps).sqrt()
    return mean, std


def adaptive_instance_normalization(
    content: torch.Tensor,
    style: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """把 style 的逐通道统计量迁移到 content，并保持空间布局不变。"""
    if content.shape != style.shape:
        raise ValueError(
            "content 与 style 形状必须完全一致："
            f"{tuple(content.shape)} != {tuple(style.shape)}"
        )
    content_mean, content_std = channel_mean_std(content, eps=eps)
    style_mean, style_std = channel_mean_std(style, eps=eps)
    normalized = (content - content_mean) / content_std
    return normalized * style_std + style_mean
