import pathlib

DEEP_AWB_DIR = pathlib.Path(__file__).absolute().parent

LOG_DIR = DEEP_AWB_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
