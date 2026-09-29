import torch


def metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int = 0,
) -> dict[str, float]:
    """
    根据二元分割的像素统计量计算评价指标。

    true_positive（TP）：
        正确预测为建筑物的像素。

    false_positive（FP）：
        实际不是建筑物，却被误判为建筑物的像素。

    false_negative（FN）：
        实际是建筑物，却被模型漏掉的像素。

    true_negative（TN）：
        正确预测为非建筑物的像素。
    """

    predicted_positive = (
        true_positive + false_positive
    )

    actual_positive = (
        true_positive + false_negative
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    # 模型没有预测任何建筑物时，
    # Precision定义为0，而不是错误地显示为1。
    if predicted_positive > 0:
        precision = (
            true_positive / predicted_positive
        )
    else:
        precision = 0.0

    # 标签中没有建筑物时，Recall没有正常分母。
    # 当前统一记为0，正式评价时会在整个数据集上累计，
    # 一般不会出现整个数据集都没有建筑物的情况。
    if actual_positive > 0:
        recall = (
            true_positive / actual_positive
        )
    else:
        recall = 0.0

    # IoU的分母是预测区域与真实区域的并集。
    # 如果预测和标签都完全为空，则认为两者完全一致。
    if union > 0:
        iou = true_positive / union
    else:
        iou = 1.0

    # 直接使用TP、FP、FN计算F1，
    # 避免Precision和Recall均为0时出现除零问题。
    f1_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if f1_denominator > 0:
        f1_score = (
            2 * true_positive
            / f1_denominator
        )
    else:
        f1_score = 1.0

    return {
        "iou": float(iou),
        "f1": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
    }


@torch.no_grad()
def binary_segmentation_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    """
    统计一个批次中的TP、FP、FN和TN。

    logits：
        模型未经Sigmoid处理的原始输出。

    targets：
        真实的0/1建筑物标签。

    threshold：
        将建筑物概率转换成二值结果的阈值。
    """

    if logits.shape != targets.shape:
        raise ValueError(
            "预测与标签形状不一致："
            f"{logits.shape} 和 {targets.shape}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold必须在0到1之间，当前为：{threshold}"
        )

    probabilities = torch.sigmoid(
        logits.float()
    )

    predictions = probabilities >= threshold
    target_binary = targets >= 0.5

    true_positive = int(
        (
            predictions
            & target_binary
        ).sum().item()
    )

    false_positive = int(
        (
            predictions
            & ~target_binary
        ).sum().item()
    )

    false_negative = int(
        (
            ~predictions
            & target_binary
        ).sum().item()
    )

    true_negative = int(
        (
            ~predictions
            & ~target_binary
        ).sum().item()
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


@torch.no_grad()
def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    直接根据一个批次的模型输出和真实标签计算指标。

    这个函数保持原来的调用方式不变，
    因此smoke_train_unet.py不需要修改。
    """

    counts = binary_segmentation_counts(
        logits=logits,
        targets=targets,
        threshold=threshold,
    )

    return metrics_from_counts(
        true_positive=counts["true_positive"],
        false_positive=counts["false_positive"],
        false_negative=counts["false_negative"],
        true_negative=counts["true_negative"],
    )


class BinarySegmentationMeter:
    """
    用于累计整个验证集的二元分割指标。

    形象理解：
    不单独平均每一批的成绩，
    而是先把整个验证集的正确、误判和漏检像素全部加起来，
    最后统一计算一次指标。

    正式训练和测试阶段会使用这个类。
    """

    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold必须在0到1之间。"
            )

        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """清空之前累计的所有像素统计。"""

        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """加入一个批次的预测结果和真实标签。"""

        counts = binary_segmentation_counts(
            logits=logits,
            targets=targets,
            threshold=self.threshold,
        )

        self.true_positive += counts[
            "true_positive"
        ]

        self.false_positive += counts[
            "false_positive"
        ]

        self.false_negative += counts[
            "false_negative"
        ]

        self.true_negative += counts[
            "true_negative"
        ]

    def compute(self) -> dict[str, float]:
        """根据累计结果计算整个数据集的指标。"""

        return metrics_from_counts(
            true_positive=self.true_positive,
            false_positive=self.false_positive,
            false_negative=self.false_negative,
            true_negative=self.true_negative,
        )