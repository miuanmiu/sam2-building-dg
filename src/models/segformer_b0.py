import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
)


DEFAULT_PRETRAINED_MODEL = "nvidia/mit-b0"

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


class SegFormerB0(nn.Module):
    """
    用于二分类建筑物提取的SegFormer-B0。

    输入：
        [B, 3, H, W]，数值范围应为0～1。

    输出：
        [B, 1, H, W]，输出为未经Sigmoid的logits。

    本包装类会：
    1. 按ImageNet均值和标准差归一化输入；
    2. 调用SegFormer-B0；
    3. 将约1/4分辨率的输出恢复到输入尺寸。
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.pretrained_model_name = pretrained_model_name

        config = SegformerConfig.from_pretrained(
            pretrained_model_name,
        )

        config.num_labels = 1
        config.id2label = {
            0: "building",
        }
        config.label2id = {
            "building": 0,
        }

        if pretrained:
            self.model = (
                SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_model_name,
                    config=config,
                    ignore_mismatched_sizes=True,
                )
            )
        else:
            self.model = (
                SegformerForSemanticSegmentation(
                    config
                )
            )

        image_mean = torch.tensor(
            IMAGENET_MEAN,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        image_std = torch.tensor(
            IMAGENET_STD,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "image_mean",
            image_mean,
            persistent=False,
        )

        self.register_buffer(
            "image_std",
            image_std,
            persistent=False,
        )

    def normalize_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        将0～1 RGB影像转换为预训练编码器使用的尺度。
        """

        return (
            images - self.image_mean
        ) / self.image_std

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行归一化、前向传播和输出尺寸恢复。
        """

        if images.ndim != 4:
            raise ValueError(
                "输入影像必须是[B, C, H, W]四维张量。"
            )

        if images.shape[1] != 3:
            raise ValueError(
                "SegFormer-B0要求输入3通道RGB影像。"
            )

        input_height = images.shape[-2]
        input_width = images.shape[-1]

        normalized_images = self.normalize_images(
            images
        )

        outputs = self.model(
            pixel_values=normalized_images,
        )

        logits = outputs.logits

        if logits.shape[-2:] != (
            input_height,
            input_width,
        ):
            logits = F.interpolate(
                logits,
                size=(
                    input_height,
                    input_width,
                ),
                mode="bilinear",
                align_corners=False,
            )

        return logits


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    统计需要训练的模型参数数量。
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    """
    单独运行本文件时进行一次快速前向测试。
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("========== SegFormer-B0快速测试 ==========")
    print("计算设备：", device)

    model = SegFormerB0(
        pretrained=True,
    ).to(device)

    model.eval()

    test_images = torch.rand(
        1,
        3,
        512,
        512,
        device=device,
    )

    with torch.no_grad():
        normalized_images = (
            model.normalize_images(test_images)
        )

        test_logits = model(test_images)

    print(
        "可训练参数量：",
        count_trainable_parameters(model),
    )

    print(
        "输入尺寸：",
        tuple(test_images.shape),
    )

    print(
        "归一化后尺寸：",
        tuple(normalized_images.shape),
    )

    print(
        "输出尺寸：",
        tuple(test_logits.shape),
    )

    expected_shape = (
        1,
        1,
        512,
        512,
    )

    if tuple(test_logits.shape) != expected_shape:
        raise RuntimeError(
            "SegFormer-B0输出尺寸不正确。"
        )

    if not torch.isfinite(test_logits).all():
        raise RuntimeError(
            "SegFormer-B0输出中出现非有限数值。"
        )

    print("ImageNet归一化：已启用")
    print("SegFormer-B0包装模型测试通过。")


if __name__ == "__main__":
    main()