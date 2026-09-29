import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    """
    Inria建筑物提取切片数据集。

    支持两种读取方式：

    1. 原有文件夹模式
       processed_root/split/images/*.png
       processed_root/split/masks/*.png

    2. 清单模式
       根据tiles_manifest.csv中的split、image_path和mask_path读取。
       这种方式可以复用原有切片，不需要复制或重新切图。

    输出：
        image:
            [3, H, W]，torch.float32，范围0～1

        mask:
            [1, H, W]，torch.float32，值为0或1

        name:
            当前切片名称，不含扩展名

        city:
            城市名称。文件夹模式无法可靠获取时返回空字符串

        source_name:
            对应的原始5000×5000大图名称。
            文件夹模式无法可靠获取时返回空字符串
    """

    def __init__(
        self,
        processed_root: str | Path | None,
        split: str,
        transform: Callable | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        """
        初始化数据集。

        processed_root:
            原有切片数据集根目录。
            使用清单模式时可以传None。

        split:
            train、val或test。

        transform:
            可选的数据增强函数。

        manifest_path:
            可选的切片清单路径。
            提供后启用清单模式，并按照清单中的split筛选样本。
        """
        self.processed_root = (
            Path(processed_root)
            if processed_root is not None
            else None
        )
        self.split = split
        self.transform = transform
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else None
        )

        valid_splits = {"train", "val", "test"}

        if split not in valid_splits:
            raise ValueError(
                f"split必须是train、val或test，当前收到：{split}"
            )

        if self.manifest_path is not None:
            self.samples = self._load_samples_from_manifest(
                self.manifest_path
            )
            self.read_mode = "manifest"
        else:
            if self.processed_root is None:
                raise ValueError(
                    "未提供manifest_path时，processed_root不能为None。"
                )

            self.samples = self._load_samples_from_folders(
                self.processed_root
            )
            self.read_mode = "folder"

        if not self.samples:
            raise RuntimeError(
                f"数据集中没有找到split={self.split}的样本。"
            )

    def _load_samples_from_folders(
        self,
        processed_root: Path,
    ) -> list[dict[str, str | Path]]:
        """按照原有train/val/test文件夹结构读取样本。"""
        images_dir = (
            processed_root / self.split / "images"
        )
        masks_dir = (
            processed_root / self.split / "masks"
        )

        if not images_dir.exists():
            raise FileNotFoundError(
                f"影像目录不存在：{images_dir}"
            )

        if not masks_dir.exists():
            raise FileNotFoundError(
                f"标签目录不存在：{masks_dir}"
            )

        image_paths = sorted(
            images_dir.glob("*.png")
        )
        mask_paths = sorted(
            masks_dir.glob("*.png")
        )

        if not image_paths:
            raise RuntimeError(
                f"影像目录为空：{images_dir}"
            )

        if not mask_paths:
            raise RuntimeError(
                f"标签目录为空：{masks_dir}"
            )

        image_names = {
            path.name for path in image_paths
        }
        mask_names = {
            path.name for path in mask_paths
        }

        missing_masks = sorted(
            image_names - mask_names
        )
        missing_images = sorted(
            mask_names - image_names
        )

        if missing_masks:
            raise RuntimeError(
                "以下影像缺少同名标签："
                f"{missing_masks[:10]}"
            )

        if missing_images:
            raise RuntimeError(
                "以下标签缺少同名影像："
                f"{missing_images[:10]}"
            )

        mask_map = {
            path.name: path
            for path in mask_paths
        }

        samples: list[dict[str, str | Path]] = []

        for image_path in image_paths:
            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_map[image_path.name],
                    "name": image_path.stem,
                    "city": "",
                    "source_name": "",
                }
            )

        return samples

    def _load_samples_from_manifest(
        self,
        manifest_path: Path,
    ) -> list[dict[str, str | Path]]:
        """按照跨城市tiles_manifest.csv读取样本。"""
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"切片清单不存在：{manifest_path}"
            )

        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise RuntimeError(
                    f"切片清单没有表头：{manifest_path}"
                )

            required_fields = {
                "split",
                "city",
                "source_name",
                "tile_name",
                "image_path",
                "mask_path",
            }

            missing_fields = (
                required_fields - set(reader.fieldnames)
            )

            if missing_fields:
                raise RuntimeError(
                    "切片清单缺少字段："
                    f"{sorted(missing_fields)}"
                )

            selected_records = [
                record
                for record in reader
                if record["split"].strip().lower()
                == self.split
            ]

        samples: list[dict[str, str | Path]] = []
        missing_files: list[str] = []

        for record in selected_records:
            image_path = Path(
                record["image_path"].strip()
            )
            mask_path = Path(
                record["mask_path"].strip()
            )

            if not image_path.exists():
                missing_files.append(
                    f"影像：{image_path}"
                )

            if not mask_path.exists():
                missing_files.append(
                    f"标签：{mask_path}"
                )

            if len(missing_files) >= 10:
                break

            tile_name = record["tile_name"].strip()

            samples.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "name": Path(tile_name).stem,
                    "city": record["city"].strip().lower(),
                    "source_name": (
                        record["source_name"].strip()
                    ),
                }
            )

        if missing_files:
            raise FileNotFoundError(
                "清单中的部分文件不存在，示例：\n"
                + "\n".join(missing_files)
            )

        samples.sort(
            key=lambda sample: (
                str(sample["city"]),
                str(sample["source_name"]),
                str(sample["name"]),
            )
        )

        return samples

    def __len__(self) -> int:
        """返回影像与标签的组数。"""
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:
        """根据索引读取一组影像与标签。"""
        sample = self.samples[index]

        image_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])

        with Image.open(image_path) as image_file:
            image = np.asarray(
                image_file.convert("RGB"),
                dtype=np.uint8,
            ).copy()

        with Image.open(mask_path) as mask_file:
            mask = np.asarray(
                mask_file.convert("L"),
                dtype=np.uint8,
            ).copy()

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"影像和标签尺寸不一致：{image_path.name}，"
                f"影像尺寸 {image.shape[:2]}，"
                f"标签尺寸 {mask.shape}"
            )

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # 影像：[H, W, C] → [C, H, W]
        # uint8 0～255 → float32 0～1
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(
                image.transpose(2, 0, 1)
            ).float() / 255.0
        elif isinstance(image, torch.Tensor):
            image_tensor = image.float()

            # 兼容某些增强库输出0～255张量的情况。
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
        else:
            raise TypeError(
                f"不支持的影像类型：{type(image)}"
            )

        # 标签：[H, W] → [1, H, W]
        # 任意正值统一变成1
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(
                (mask > 0).astype(np.float32)
            ).unsqueeze(0)
        elif isinstance(mask, torch.Tensor):
            mask_tensor = mask.float()

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

            mask_tensor = (
                mask_tensor > 0
            ).float()
        else:
            raise TypeError(
                f"不支持的标签类型：{type(mask)}"
            )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": str(sample["name"]),
            "city": str(sample["city"]),
            "source_name": str(sample["source_name"]),
        }
