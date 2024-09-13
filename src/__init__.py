import pathlib
import sys

from loguru import logger as console_logger

console_logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset" / "SimpleCube++"
assert DATASET_DIR.is_dir()

TRAIN_SET_FOLDER = DATASET_DIR / "train"
TEST_SET_FOLDER = DATASET_DIR / "test"
