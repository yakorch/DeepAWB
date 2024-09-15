import pathlib
from dataclasses import dataclass
from typing import Callable, Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger as console_logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..__init__ import ORIGINAL_IMAGE_DIMS, TEST_SET_FOLDER, TRAIN_SET_FOLDER


def parse_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    modified_annotations = annotations.drop(columns=["mean_r", "mean_g", "mean_b"], inplace=False)

    modified_annotations["R/G"] = annotations["mean_r"] / annotations["mean_g"]
    modified_annotations["B/G"] = annotations["mean_b"] / annotations["mean_g"]
    return modified_annotations


@console_logger.catch
def load_processable_image(image_path: pathlib.Path, transform: A.Compose) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.frombuffer(image.tobytes(), dtype=np.uint8)
    image_np = (image_np.reshape((image.height, image.width, 3))).astype(np.float32) / 255.0

    return transform(image=image_np)["image"]


class RawAWBDataset(Dataset):
    def __init__(self, csv_annotations: pathlib.Path, images_dir: pathlib.Path, transform: A.Compose, cache_data: bool = False):
        """
        Args:
            csv_file (pathlib.Path): Path to the csv file with annotations.
            images_dir (pathlib.Path): Directory with all the images.
            transform (callable, optional): Transform to be applied on a sample.
            cache_data (bool): Whether to keep the image-label pairs in memory or not.
        """
        assert csv_annotations.exists(), f"{csv_annotations} does not exist"
        assert images_dir.is_dir(), f"{images_dir} does not exist"

        self.annotations = parse_annotations(pd.read_csv(csv_annotations))
        self.images_dir = images_dir
        self.transform = transform

        self._getitem: Callable = self._actual_getitem

        self.to_cache_data = cache_data
        self.cached_pairs = {} if cache_data else None

    def __len__(self):
        return len(self.annotations["image"])

    def _actual_getitem(self, idx: int):
        img_name = self.images_dir / (self.annotations["image"].iloc[idx] + ".png")
        image = load_processable_image(img_name, self.transform)
        label = torch.tensor(self.annotations.iloc[idx][["R/G", "B/G"]].values.astype(np.float32))

        return image, label

    def __getitem__(self, idx: int):
        if not self.to_cache_data:
            return self._actual_getitem(idx)

        if (cached_pair := self.cached_pairs.get(idx)) is not None:
            return cached_pair
        pair = self._getitem(idx)
        self.cached_pairs[idx] = pair
        return pair


_preprocessed_strategy_to_image_normalization: dict[str, A.Normalize] = {
    # these are precomputed on the training set only
    "PROCESSED_MLE": A.Normalize(mean=(0.4373, 0.3857, 0.3163), std=(0.2495, 0.2281, 0.2024), max_pixel_value=1),
    "PROCESSED_UNIFORM": A.Normalize(mean=(0.0954, 0.4912, 0.2624), std=(0.0917, 0.2781, 0.1800), max_pixel_value=1),
}

_preprocess_strategy = "PROCESSED_MLE"
assert _preprocess_strategy in _preprocessed_strategy_to_image_normalization


@dataclass(frozen=False, order=False)
class DatasetInfo:
    _original_image_dims: tuple[int, int]
    _image_scale: Optional[float]
    _resize_transform: Optional[A.Compose]
    _normalize_transform: A.Normalize
    train_augmentations: Optional[A.Compose]

    @property
    def image_dims(self):
        assert self._image_scale is not None
        return tuple(int(dim / self._image_scale) for dim in self._original_image_dims)

    def setup(self, image_scale: float = 1) -> None:
        self._image_scale = image_scale
        image_height, image_width = self.image_dims
        self._resize_transform = A.Resize(height=image_height, width=image_width)

    @property
    def train_transform(self):
        transformations = [self._resize_transform]

        if self.train_augmentations is not None:
            transformations.append(self.train_augmentations)

        transformations.extend([self._normalize_transform, ToTensorV2()])

        return A.Compose(transformations)

    @property
    def test_transform(self):
        return A.Compose([self._resize_transform, self._normalize_transform, ToTensorV2()])


SimpleCubePPDatasetInfo = DatasetInfo(
    ORIGINAL_IMAGE_DIMS,
    None,
    None,
    _preprocessed_strategy_to_image_normalization[_preprocess_strategy],
    A.Compose(
        [
            A.Affine(rotate=15, translate_percent=(0.1, 0.1), scale=(0.9, 1.1), p=0.75),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.GaussianBlur(blur_limit=(3, 7), p=0.8),
            A.MotionBlur(blur_limit=(3, 7), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.75),
            A.GaussNoise(var_limit=0.001, p=0.5, per_channel=False),
        ]
    ),
)


def get_train_dataset(preprocess_strategy: str = _preprocess_strategy):
    return RawAWBDataset(
        csv_annotations=TRAIN_SET_FOLDER / "gt.csv", images_dir=TRAIN_SET_FOLDER / preprocess_strategy, transform=SimpleCubePPDatasetInfo.train_transform, cache_data=False
    )


def get_test_dataset(preprocess_strategy: str = _preprocess_strategy):
    return RawAWBDataset(
        csv_annotations=TEST_SET_FOLDER / "gt.csv",
        images_dir=TEST_SET_FOLDER / preprocess_strategy,
        transform=SimpleCubePPDatasetInfo.test_transform,
        cache_data=True,
    )


def get_train_data_loader():
    return DataLoader(get_train_dataset(), batch_size=32, shuffle=True, num_workers=7, persistent_workers=True, pin_memory=True, drop_last=True)


def get_test_data_loader():
    return DataLoader(get_test_dataset(), batch_size=32, shuffle=False, num_workers=4, persistent_workers=True, drop_last=False)


if __name__ == "__main__":
    raise NotImplementedError()
