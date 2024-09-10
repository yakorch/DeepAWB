from pathlib import Path

import torchx
from ax.runners.torchx import TorchXRunner
from torchx import specs
from torchx.components import utils

from . import LOG_DIR
from .search_space import _FEATURE_EXTRACTOR_WIDTH


def trainer(log_path: str, total_hidden_neurons: int, MLP_depth: int, learning_rate: float, epochs: int, image_scale: float, trial_idx: int = -1, **kwargs) -> specs.AppDef:
    # define the log path so we can pass it to the TorchX ``AppDef``
    if trial_idx >= 0:
        log_path = Path(log_path).joinpath(str(trial_idx)).absolute().as_posix()

    n_kernels, kernel_sizes, strides = ([str(kwargs[f"{attribute}_{i}"]) for i in range(_FEATURE_EXTRACTOR_WIDTH)] for attribute in ["n_kernels", "kernel_size", "stride"])

    return utils.python(
        "--log_path",
        log_path,
        "--total_hidden_neurons",
        str(total_hidden_neurons),
        "--MLP_depth",
        str(MLP_depth),
        "--learning_rate",
        str(learning_rate),
        "--epochs",
        str(epochs),
        "--image_scale",
        str(image_scale),
        "--n_kernels",
        *n_kernels,
        "--kernel_size",
        *kernel_sizes,
        "--stride",
        *strides,
        name="trainer",
        m="src.deep_awb.train_model",
        image=torchx.version.TORCHX_IMAGE,
    )


ax_runner = TorchXRunner(
    tracker_base="/tmp/",
    component=trainer,
    scheduler="local_cwd",
    component_const_params={"log_path": LOG_DIR},
    cfg={},
)


if __name__ == "__main__":
    raise NotImplementedError()
