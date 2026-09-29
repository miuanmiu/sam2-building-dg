import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """
    连续执行两次卷积，用于提取和组合图像特征。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    编码器下采样模块：
    先把宽高缩小一半，再提取特征。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    解码器上采样模块：
    放大深层特征，并与编码器保存的细节进行拼接。
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        # 把深层特征图放大一倍
        x = self.up(x)

        # 保险处理：保证上采样结果与跳跃连接特征尺寸一致
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # 沿通道方向拼接深层语义和浅层边缘细节
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    建筑物二元语义分割U-Net。

    输入：
        [批量大小, 3, 高度, 宽度]

    输出：
        [批量大小, 1, 高度, 宽度]

    输出为未经Sigmoid转换的logits。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        # 左侧编码器：逐渐缩小影像，提取高级特征
        self.input_block = DoubleConv(
            in_channels=in_channels,
            out_channels=base_channels,
        )

        self.down1 = DownBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
        )

        self.down2 = DownBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
        )

        self.down3 = DownBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
        )

        self.down4 = DownBlock(
            in_channels=base_channels * 8,
            out_channels=base_channels * 16,
        )

        # 右侧解码器：逐渐放大，并取回编码器细节
        self.up1 = UpBlock(
            in_channels=base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
        )

        self.up2 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )

        self.up3 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        self.up4 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        # 把最后32个特征通道合成为1个建筑物输出通道
        self.output_layer = nn.Conv2d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码阶段，同时保存各个尺度的细节
        encoder1 = self.input_block(x)
        encoder2 = self.down1(encoder1)
        encoder3 = self.down2(encoder2)
        encoder4 = self.down3(encoder3)
        bottleneck = self.down4(encoder4)

        # 解码阶段，把对应尺度的编码器特征接回来
        decoder1 = self.up1(bottleneck, encoder4)
        decoder2 = self.up2(decoder1, encoder3)
        decoder3 = self.up3(decoder2, encoder2)
        decoder4 = self.up4(decoder3, encoder1)

        logits = self.output_layer(decoder4)

        return logits