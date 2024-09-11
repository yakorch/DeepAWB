import pathlib
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..__init__ import TEST_SET_FOLDER, TRAIN_SET_FOLDER


def parse_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    modified_annotations = annotations.drop(columns=["mean_r", "mean_g", "mean_b"], inplace=False)

    modified_annotations["R/G"] = annotations["mean_r"] / annotations["mean_g"]
    modified_annotations["B/G"] = annotations["mean_b"] / annotations["mean_g"]
    return modified_annotations


class RawAWBDataset(Dataset):
    def __init__(self, csv_annotations: pathlib.Path, images_dir: pathlib.Path, transform: Optional[transforms.Compose] = None, cache_data: bool = False):
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
        image = Image.open(img_name).convert("RGB")
        label = torch.tensor(self.annotations.iloc[idx][["R/G", "B/G"]].values.astype(np.float32))

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __getitem__(self, idx: int):
        if not self.to_cache_data:
            return self._actual_getitem(idx)

        if (cached_pair := self.cached_pairs.get(idx)) is not None:
            return cached_pair
        pair = self._getitem(idx)
        self.cached_pairs[idx] = pair
        return pair


IMAGE_HEIGHT, IMAGE_WIDTH = 432, 648

_COMMON_TRANSFORM = ...
_TRAIN_AUGMENTATIONS = transforms.Compose(
    [
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ]
)


def setup_common_transform(image_scale: float = 1):
    global _COMMON_TRANSFORM

    if image_scale == 1:
        _COMMON_TRANSFORM = transforms.Compose([transforms.ToTensor()])
    else:
        _COMMON_TRANSFORM = transforms.Compose([transforms.Resize((int(IMAGE_HEIGHT / image_scale), int(IMAGE_WIDTH / image_scale))), transforms.ToTensor()])


def get_train_dataset():
    train_transform = transforms.Compose([_COMMON_TRANSFORM, _TRAIN_AUGMENTATIONS])
    return RawAWBDataset(csv_annotations=TRAIN_SET_FOLDER / "gt.csv", images_dir=TRAIN_SET_FOLDER / "PNG", transform=train_transform, cache_data=False)


def get_test_dataset():
    test_transform = _COMMON_TRANSFORM
    return RawAWBDataset(csv_annotations=TEST_SET_FOLDER / "gt.csv", images_dir=TEST_SET_FOLDER / "PNG", transform=test_transform, cache_data=True)


def get_train_data_loader():
    return DataLoader(get_train_dataset(), batch_size=32, shuffle=True, num_workers=7, persistent_workers=True, pin_memory=True)


def get_test_data_loader():
    return DataLoader(get_test_dataset(), batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)


if __name__ == "__main__":
    raise NotImplementedError()
