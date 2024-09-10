import pathlib

from loguru import logger

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "SimpleCube++"
assert DATASET_DIR.is_dir()

TRAIN_SET_FOLDER = DATASET_DIR / "train"
TEST_SET_FOLDER = DATASET_DIR / "test"
