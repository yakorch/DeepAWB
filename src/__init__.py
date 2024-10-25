import pathlib
import sys

from loguru import logger as console_logger

console_logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "SimpleCube++"

TRAIN_SET_FOLDER = DATASET_DIR / "train"
RAW_TRAIN_IMAGES_FOLDER = TRAIN_SET_FOLDER / "PNG"

TEST_SET_FOLDER = DATASET_DIR / "test"
RAW_TEST_IMAGES_FOLDER = TEST_SET_FOLDER / "PNG"

ORIGINAL_IMAGE_DIMS = (432, 648)
