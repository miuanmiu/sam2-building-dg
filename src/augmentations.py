import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class RemoteSensingStyleAugmentation:
    """
    面向跨城市遥感建筑提取的风格增强。

    只改变影像外观，不改变空间位置，因此标签保持不变。

    可随机执行：
        1. 亮度变化
        2. 对比度变化
        3. 饱和度变化
        4. 色调变化
        5. Gamma变化
        6. 轻微高斯模糊
        7. 轻微高斯噪声

    输入：
        image：H×W×3的uint8 RGB数组
        mask：H×W的标签数组

    输出：
        {
            "image": 增强后的uint8 RGB数组,
            "mask": 原样复制的标签数组,
        }
    """

    def __init__(
        self,
        overall_probability: float = 0.90,
        brightness_probability: float = 0.70,
        contrast_probability: float = 0.70,
        saturation_probability: float = 0.70,
        hue_probability: float = 0.40,
        gamma_probability: float = 0.40,
        blur_probability: float = 0.20,
        noise_probability: float = 0.25,
        brightness_range: tuple[float, float] = (0.75, 1.25),
        contrast_range: tuple[float, float] = (0.75, 1.25),
        saturation_range: tuple[float, float] = (0.70, 1.30),
        hue_shift_range: tuple[float, float] = (-0.04, 0.04),
        gamma_range: tuple[float, float] = (0.80, 1.20),
        blur_radius_range: tuple[float, float] = (0.10, 1.00),
        noise_std_range: tuple[float, float] = (2.0, 10.0),
    ) -> None:
        self.overall_probability = self._validate_probability(
            overall_probability,
            "overall_probability",
        )
        self.brightness_probability = self._validate_probability(
            brightness_probability,
            "brightness_probability",
        )
        self.contrast_probability = self._validate_probability(
            contrast_probability,
            "contrast_probability",
        )
        self.saturation_probability = self._validate_probability(
            saturation_probability,
            "saturation_probability",
        )
        self.hue_probability = self._validate_probability(
            hue_probability,
            "hue_probability",
        )
        self.gamma_probability = self._validate_probability(
            gamma_probability,
            "gamma_probability",
        )
        self.blur_probability = self._validate_probability(
            blur_probability,
            "blur_probability",
        )
        self.noise_probability = self._validate_probability(
            noise_probability,
            "noise_probability",
        )

        self.brightness_range = self._validate_range(
            brightness_range,
            "brightness_range",
            minimum=0.0,
        )
        self.contrast_range = self._validate_range(
            contrast_range,
            "contrast_range",
            minimum=0.0,
        )
        self.saturation_range = self._validate_range(
            saturation_range,
            "saturation_range",
            minimum=0.0,
        )
        self.hue_shift_range = self._validate_range(
            hue_shift_range,
            "hue_shift_range",
        )
        self.gamma_range = self._validate_range(
            gamma_range,
            "gamma_range",
            minimum=0.0,
        )
        self.blur_radius_range = self._validate_range(
            blur_radius_range,
            "blur_radius_range",
            minimum=0.0,
        )
        self.noise_std_range = self._validate_range(
            noise_std_range,
            "noise_std_range",
            minimum=0.0,
        )

    @staticmethod
    def _validate_probability(
        value: float,
        name: str,
    ) -> float:
        value = float(value)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name}必须在0到1之间，当前为：{value}"
            )

        return value

    @staticmethod
    def _validate_range(
        value: tuple[float, float],
        name: str,
        minimum: float | None = None,
    ) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(
                f"{name}必须包含两个数值。"
            )

        lower = float(value[0])
        upper = float(value[1])

        if lower > upper:
            raise ValueError(
                f"{name}的下限不能大于上限。"
            )

        if minimum is not None and lower < minimum:
            raise ValueError(
                f"{name}不能小于{minimum}。"
            )

        return lower, upper

    @staticmethod
    def _validate_inputs(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image必须是numpy数组，当前为：{type(image)}"
            )

        if not isinstance(mask, np.ndarray):
            raise TypeError(
                f"mask必须是numpy数组，当前为：{type(mask)}"
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "image必须是H×W×3的RGB数组。"
            )

        if mask.ndim != 2:
            raise ValueError(
                "mask必须是H×W的单通道数组。"
            )

        if image.shape[:2] != mask.shape:
            raise ValueError(
                "image和mask的空间尺寸不一致。"
            )

    @staticmethod
    def _apply_hue_shift(
        image: Image.Image,
        hue_shift: float,
    ) -> Image.Image:
        hsv_image = np.asarray(
            image.convert("HSV"),
            dtype=np.uint8,
        ).copy()

        shift_value = int(
            round(hue_shift * 255.0)
        )

        hue_channel = hsv_image[:, :, 0].astype(
            np.int16
        )

        hsv_image[:, :, 0] = (
            (hue_channel + shift_value) % 256
        ).astype(np.uint8)

        return Image.fromarray(
            hsv_image,
            mode="HSV",
        ).convert("RGB")

    @staticmethod
    def _apply_gamma(
        image: Image.Image,
        gamma: float,
    ) -> Image.Image:
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        normalized = image_array / 255.0

        corrected = np.power(
            normalized,
            gamma,
        )

        corrected = np.clip(
            corrected * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        return Image.fromarray(
            corrected,
            mode="RGB",
        )

    @staticmethod
    def _apply_noise(
        image: Image.Image,
        standard_deviation: float,
    ) -> Image.Image:
        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        noise = np.random.normal(
            loc=0.0,
            scale=standard_deviation,
            size=image_array.shape,
        ).astype(np.float32)

        noisy_image = np.clip(
            image_array + noise,
            0.0,
            255.0,
        ).astype(np.uint8)

        return Image.fromarray(
            noisy_image,
            mode="RGB",
        )

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        """
        对一组影像和标签执行随机风格增强。
        """

        self._validate_inputs(
            image=image,
            mask=mask,
        )

        output_image = image.copy()
        output_mask = mask.copy()

        if random.random() > self.overall_probability:
            return {
                "image": output_image,
                "mask": output_mask,
            }

        pil_image = Image.fromarray(
            output_image,
            mode="RGB",
        )

        operations: list[str] = []

        if random.random() < self.brightness_probability:
            operations.append("brightness")

        if random.random() < self.contrast_probability:
            operations.append("contrast")

        if random.random() < self.saturation_probability:
            operations.append("saturation")

        if random.random() < self.hue_probability:
            operations.append("hue")

        if random.random() < self.gamma_probability:
            operations.append("gamma")

        if random.random() < self.blur_probability:
            operations.append("blur")

        if random.random() < self.noise_probability:
            operations.append("noise")

        random.shuffle(operations)

        for operation in operations:
            if operation == "brightness":
                factor = random.uniform(
                    *self.brightness_range
                )
                pil_image = ImageEnhance.Brightness(
                    pil_image
                ).enhance(factor)

            elif operation == "contrast":
                factor = random.uniform(
                    *self.contrast_range
                )
                pil_image = ImageEnhance.Contrast(
                    pil_image
                ).enhance(factor)

            elif operation == "saturation":
                factor = random.uniform(
                    *self.saturation_range
                )
                pil_image = ImageEnhance.Color(
                    pil_image
                ).enhance(factor)

            elif operation == "hue":
                hue_shift = random.uniform(
                    *self.hue_shift_range
                )
                pil_image = self._apply_hue_shift(
                    pil_image,
                    hue_shift,
                )

            elif operation == "gamma":
                gamma = random.uniform(
                    *self.gamma_range
                )
                pil_image = self._apply_gamma(
                    pil_image,
                    gamma,
                )

            elif operation == "blur":
                radius = random.uniform(
                    *self.blur_radius_range
                )
                pil_image = pil_image.filter(
                    ImageFilter.GaussianBlur(
                        radius=radius
                    )
                )

            elif operation == "noise":
                standard_deviation = random.uniform(
                    *self.noise_std_range
                )
                pil_image = self._apply_noise(
                    pil_image,
                    standard_deviation,
                )

        output_image = np.asarray(
            pil_image,
            dtype=np.uint8,
        ).copy()

        return {
            "image": output_image,
            "mask": output_mask,
        }
