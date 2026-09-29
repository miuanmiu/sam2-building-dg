"""FOSMix 核心模块（与论文/官方代码一致，从项目 train_fosmix_s0.py 提取）

论文: Frequency-Based Optimal Style Mix for Domain Generalization in
Semantic Segmentation of Remote Sensing Images (IEEE TGRS 2024)
官方代码: https://github.com/Reo-I/FOSMix

组件:
- Full Mix: mask 全 0，所有 DCT 系数替换为参考图排序匹配后的值
- Optimal Mix: 图像条件 mask 网络决定每个 DCT 系数保留原图还是换成参考图
- 一致性/正则: 由训练脚本使用 mask 的 L1 均值、方差和 logits 一致性损失
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def _dct_1d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Differentiable DCT-II over the final dimension."""
    shape = x.shape
    length = shape[-1]
    flat = x.contiguous().reshape(-1, length)
    reordered = torch.cat([flat[:, ::2], flat[:, 1::2].flip(1)], dim=1)
    spectrum = torch.view_as_real(torch.fft.fft(reordered, dim=1))
    k = -torch.arange(length, device=x.device, dtype=x.dtype)[None] * math.pi / (2 * length)
    values = spectrum[..., 0] * torch.cos(k) - spectrum[..., 1] * torch.sin(k)
    if norm == "ortho":
        values[:, 0] /= math.sqrt(length) * 2
        values[:, 1:] /= math.sqrt(length / 2) * 2
    return (2 * values).reshape(shape)


def _idct_1d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Differentiable inverse DCT-II over the final dimension."""
    shape = x.shape
    length = shape[-1]
    values = x.contiguous().reshape(-1, length) / 2
    if norm == "ortho":
        values[:, 0] *= math.sqrt(length) * 2
        values[:, 1:] *= math.sqrt(length / 2) * 2
    k = torch.arange(length, device=x.device, dtype=x.dtype)[None] * math.pi / (2 * length)
    mirrored_imaginary = torch.cat(
        [values[:, :1] * 0, -values.flip(1)[:, :-1]], dim=1
    )
    complex_spectrum = torch.complex(
        values * torch.cos(k) - mirrored_imaginary * torch.sin(k),
        values * torch.sin(k) + mirrored_imaginary * torch.cos(k),
    )
    reordered = torch.fft.irfft(complex_spectrum, n=length, dim=1)
    result = reordered.new_zeros(reordered.shape)
    result[:, ::2] += reordered[:, : length - (length // 2)]
    result[:, 1::2] += reordered.flip(1)[:, : length // 2]
    return result.reshape(shape)


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    return _dct_1d(_dct_1d(x).transpose(-1, -2)).transpose(-1, -2)


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    return _idct_1d(_idct_1d(x).transpose(-1, -2)).transpose(-1, -2)


class FOSMixAugment(nn.Module):
    """Image-conditioned optimal frequency style mixing used only in training."""

    def __init__(self, use_mask_net: bool = True) -> None:
        super().__init__()
        self.use_mask_net = use_mask_net
        self.mask_generator = (
            smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=3,
                classes=3,
                activation="sigmoid",
                decoder_attention_type="scse",
            )
            if use_mask_net
            else None
        )

    @staticmethod
    def _transfer(
        content: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        content_frequency = dct_2d(content)
        reference_frequency = dct_2d(reference)
        batch, channels, height, width = content_frequency.shape
        content_flat = content_frequency.reshape(batch, channels, -1)
        reference_flat = reference_frequency.reshape(batch, channels, -1)
        _, content_order = torch.sort(content_flat, dim=-1)
        sorted_reference, _ = torch.sort(reference_flat, dim=-1)
        inverse_order = content_order.argsort(dim=-1)
        transferred = sorted_reference.gather(-1, inverse_order).reshape_as(content_frequency)
        # The detached subtraction preserves the FOSMix forward mixture while
        # allowing stable identity gradients for the content image.
        mixed_frequency = content_frequency + (transferred - content_frequency.detach()) * (1 - mask)
        return idct_2d(mixed_frequency)

    def forward(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mask_generator is None:
            full_mask = torch.zeros(
                image.shape[0], 3, image.shape[-2], image.shape[-1],
                device=image.device, dtype=image.dtype,
            )
            full_image = self._transfer(image, reference, full_mask)
            return full_image, full_image, full_mask
        optimal_mask = self.mask_generator(image)
        full_mask = torch.zeros_like(optimal_mask)
        optimal_image = self._transfer(image, reference, optimal_mask)
        full_image = self._transfer(image, reference, full_mask)
        return optimal_image, full_image, optimal_mask
