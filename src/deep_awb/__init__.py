import pathlib

DEEP_AWB_DIR = pathlib.Path(__file__).absolute().parent

LOG_DIR = DEEP_AWB_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEEP_MODELS = DEEP_AWB_DIR / "models"
assert DEEP_MODELS.exists()

FINAL_DEEPAWB_MODEL = DEEP_MODELS / "trained_traced_model.pt"
EXPERIMENTAL_DEEPAWB_MODEL = DEEP_MODELS / "experimental_traced_model.pt"