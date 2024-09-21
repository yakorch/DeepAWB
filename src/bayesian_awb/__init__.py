import pathlib

BAYESIAN_AWB_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BAYESIAN_AWB_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

GP_MODEL_PATH = MODELS_DIR / "gp_model.pth"
GP_LIKELIHOOD_PATH = MODELS_DIR / "gp_likelihood.pth"
