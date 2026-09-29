from __future__ import annotations

import torch
from torch import nn

from src.models.adain import adaptive_instance_normalization


class BatchStyleMixing(nn.Module):
    """在一个 batch 内用另一幅影像的 RGB 统计量进行输入级风格混合。"""

    def __init__(
        self,
        probability: float = 0.5,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        eps: float = 1e-6,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability 必须在 [0, 1] 内。")
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError("必须满足 0 <= alpha_min <= alpha_max <= 1。")
        self.probability = probability
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.eps = eps
        self.clamp = clamp

    @staticmethod
    def style_indices(batch_size: int, device: torch.device) -> torch.Tensor:
        """生成无自配对的随机循环置换；batch_size=1 时保持原样。"""
        indices = torch.arange(batch_size, device=device)
        if batch_size < 2:
            return indices
        order = torch.randperm(batch_size, device=device)
        shift = int(torch.randint(1, batch_size, (1,), device=device).item())
        return order.roll(shifts=shift)[torch.argsort(order)]

    def forward(
        self,
        images: torch.Tensor,
        *,
        force: bool = False,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images 必须是 BCHW 张量，当前为 {tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError("BSM 输入必须是浮点张量。")
        batch_size = images.shape[0]
        if batch_size < 2 or (not self.training and not force):
            return images

        style = images[self.style_indices(batch_size, images.device)]
        stylized = adaptive_instance_normalization(images, style, eps=self.eps)
        alpha = torch.empty(
            (batch_size, 1, 1, 1), device=images.device, dtype=images.dtype
        ).uniform_(self.alpha_min, self.alpha_max)
        mixed = images + alpha * (stylized - images)

        if not force and self.probability < 1.0:
            apply_mask = torch.rand(
                (batch_size, 1, 1, 1), device=images.device
            ) < self.probability
            mixed = torch.where(apply_mask, mixed, images)
        return mixed.clamp(0.0, 1.0) if self.clamp else mixed
