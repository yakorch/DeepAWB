import argparse
import pathlib
from typing import Optional

import cv2
import numpy as np
from loguru import logger

_CCM = np.array([1.8795, -1.0326, 0.1531, -0.2198, 1.7153, -0.4955, 0.0069, -0.5150, 1.5081]).reshape((3, 3))


def parse_args():
    parser = argparse.ArgumentParser(description="Processing RAW images to RGB.")
    parser.add_argument("--raw_image", type=pathlib.Path, required=True, help="Path to either the directory with RAW images.")
    parser.add_argument("--wb_gains", type=float, nargs=2, required=False, help="White balance gains.")
    parser.add_argument("--processed_path", type=pathlib.Path, required=True, help="Path to the output directory.")

    return parser.parse_args()


def linearize(img, black_lvl=2048, saturation_lvl=2**14 - 1) -> np.ndarray:
    """
    :param saturation_lvl: 2**14-1 is a common value. Not all images
                           have the same value.
    """
    return np.clip((img - black_lvl) / (saturation_lvl - black_lvl), 0, 1)


def compute_unbalanced_image(raw_image_path: pathlib.Path) -> np.ndarray:
    assert raw_image_path.suffix == ".png"

    cam = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
    return linearize(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB).astype(np.float64))


def perform_color_correction_and_gamma(balanced_image: np.ndarray) -> np.ndarray:
    corrected = np.clip(np.dot(balanced_image, _CCM.T), 0, 1)
    corrected = np.clip(corrected ** (1 / 2.2), 0, 1)
    corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)

    return corrected


def apply_white_balance(image: np.ndarray, wb_gains: tuple[float, float]) -> np.ndarray:
    channel_wise_gains = np.array([wb_gains[0], 1, wb_gains[1]], dtype=np.float64)
    channel_wise_gains /= channel_wise_gains.sum()
    return image / (channel_wise_gains)


def save_image(path: pathlib.Path, image: np.ndarray):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@logger.catch
def image_processing_pipeline(image_path: pathlib.Path, processed_path: Optional[pathlib.Path], wb_gains: Optional[tuple[float, float]] = None):
    if wb_gains is None:
        wb_gains = (1, 1)

    image = compute_unbalanced_image(image_path)
    image = apply_white_balance(image, wb_gains)
    image = perform_color_correction_and_gamma(image)
    if processed_path is not None:
        save_image(processed_path, image)
    return image


if __name__ == "__main__":
    args = parse_args()
    image_processing_pipeline(args.raw_image, args.processed_path, args.wb_gains)
