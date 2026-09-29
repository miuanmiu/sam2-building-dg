# -*- coding: utf-8 -*-
"""DSU / FACT 模块（按论文思路自实现，未用官方代码）。

- DSU:  Uncertainty modeling for out-of-distribution generalization (ICLR 2022)
        特征统计量不确定性扰动，作为 MixStyle 的直接扩展。
- FACT: A Fourier-based Framework for Domain Generalization (CVPR 2023)
        这里实现其核心的输入级傅里叶振幅混合（保留源图相位）。

引用前需按原文核对超参与公式细节。
"""

import math
import random

import torch
import torch.nn as nn


class DSU(nn.Module):
    """特征级不确定性扰动。

    对 batch 的 channel-wise mean/std 加不确定性噪声：
        mu_unc = std / sqrt(N), sigma_unc = std / sqrt(2*(N-1))
        mu' = mu + alpha * eps_mu * mu_unc
        sigma' = sigma + alpha * eps_sigma * sigma_unc
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p or x.size(0) < 2:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + self.eps
        n = x.size(0)
        mean_unc = std / math.sqrt(n)
        std_unc = std / math.sqrt(max(2 * (n - 1), 1))
        mean_new = mean + self.alpha * torch.randn_like(mean) * mean_unc
        std_new = std + self.alpha * torch.randn_like(std) * std_unc
        x_norm = (x - mean) / std
        return x_norm * std_new + mean_new


class FACT(nn.Module):
    """输入级傅里叶振幅混合（FACT 风格自实现）。

    batch 内 roll 一位取参考图，混合振幅谱、保留源图相位，逆 FFT 回空间域。
    训练时以概率 p 启用；推理恒等。
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p or x.size(0) < 2:
            return x
        ref = torch.roll(x, shifts=1, dims=0)
        fx = torch.fft.fft2(x, norm="ortho")
        fr = torch.fft.fft2(ref, norm="ortho")
        # 更贴近"振幅混合"的写法：amp_mix = (1-lam)*amp_x + lam*amp_r，lam 逐样本随机
        lam = torch.rand(x.size(0), device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        amp_mix = (1.0 - lam) * torch.abs(fx) + lam * torch.abs(fr)
        out = amp_mix * torch.exp(1j * torch.angle(fx))
        return torch.fft.ifft2(out, norm="ortho").real.clamp_(0.0, 1.0)
