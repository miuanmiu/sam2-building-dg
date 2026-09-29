import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DeepLabV3ResNet50(nn.Module):
    """用于二分类建筑物提取的 DeepLabV3+（ResNet-50 骨干，ImageNet/COCO 预训练）。

    输入：[B, 3, H, W]，数值范围 0～1。
    输出：[B, 1, H, W]，未经 Sigmoid 的 logits（与输入同分辨率）。
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)
        # 分类头改为 1 类建筑物 logits
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        out = self.model(x)["out"]
        return out
