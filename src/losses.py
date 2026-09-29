import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    Dice重叠损失。

    形象理解：
    比较模型涂白的建筑区域，
    与真实标签中的白色建筑区域重叠了多少。
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "预测与标签形状不一致："
                f"{logits.shape} 和 {targets.shape}"
            )

        # 模型输出是logits，先转换成0～1概率
        probabilities = torch.sigmoid(logits.float())
        targets = targets.float()

        # 除批量维度外，对通道、高度和宽度求和
        sum_dimensions = tuple(
            range(1, probabilities.ndim)
        )

        intersection = (
            probabilities * targets
        ).sum(dim=sum_dimensions)

        denominator = (
            probabilities.sum(dim=sum_dimensions)
            + targets.sum(dim=sum_dimensions)
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


class BCEDiceLoss(nn.Module):
    """
    二元交叉熵损失与Dice损失的组合。

    BCE负责逐像素纠错；
    Dice负责建筑区域整体重叠。
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("损失权重不能为负数。")

        if bce_weight + dice_weight == 0:
            raise ValueError("两个损失权重不能同时为0。")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "预测与标签形状不一致："
                f"{logits.shape} 和 {targets.shape}"
            )

        # 转成float32计算，保证数值稳定
        bce = self.bce_loss(
            logits.float(),
            targets.float(),
        )

        dice = self.dice_loss(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        return total_loss