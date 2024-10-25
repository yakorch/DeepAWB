from pathlib import Path

import torchx
from ax.runners.torchx import TorchXRunner
from torchx import specs
from torchx.components import utils

from . import EXPERIMENTAL_DEEPAWB_MODEL, LOG_DIR
from .search_space import _FEATURE_EXTRACTOR_DEPTH


def trainer(
    log_path: str,
    total_hidden_neurons: int,
    decay_rate: float,
    MLP_depth: int,
    initial_learning_rate: float,
    final_learning_rate: float,
    image_scale: float,
    trial_idx: int = -1,
    **kwargs,
) -> specs.AppDef:
    # define the log path so we can pass it to the TorchX ``AppDef``
    if trial_idx >= 0:
        log_path = Path(log_path).joinpath(str(trial_idx)).absolute().as_posix()
    n_kernels, kernel_sizes, strides = ([str(kwargs[f"{attribute}_{i}"]) for i in range(_FEATURE_EXTRACTOR_DEPTH)] for attribute in ["n_kernels", "kernel_size", "stride"])

    epochs = 40

    return utils.python(
        "--log_path",
        log_path,
        "--total_hidden_neurons",
        str(total_hidden_neurons),
        "--decay_rate",
        str(decay_rate),
        "--MLP_depth",
        str(MLP_depth),
        "--learning_rates",
        str(initial_learning_rate),
        str(final_learning_rate),
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
        "--redirect",
        "--val_every_n_epochs",
        str(epochs),
        "--script_module_path",
        str(EXPERIMENTAL_DEEPAWB_MODEL),
        "--measure_locally",
        name="trainer",
        m="src.deep_awb.model_training",
        image=torchx.version.TORCHX_IMAGE,
    )


ax_runner = TorchXRunner(
    tracker_base="/tmp/",
    component=trainer,
    scheduler="local_cwd",
    component_const_params={"log_path": str(LOG_DIR)},
    cfg={},
)

if __name__ == "__main__":
    raise NotImplementedError()

    # example_params = {
    #     "total_hidden_neurons": 250,
    #     "decay_rate": 0.5,
    #     "MLP_depth": 3,
    #     "initial_learning_rate": 0.001,
    #     "final_learning_rate": 0.0001,
    #     "epochs": 1,
    #     "image_scale": 1.0,
    #     "trial_idx": 0,
    #     **{f"n_kernels_{i}": 16 * (i + 1) for i in range(_FEATURE_EXTRACTOR_DEPTH)},
    #     **{f"kernel_size_{i}": 3 + i for i in range(_FEATURE_EXTRACTOR_DEPTH)},
    #     **{f"stride_{i}": 2 for i in range(_FEATURE_EXTRACTOR_DEPTH)},
    # }

    # from torchx.runner import get_runner

    # runner = get_runner()

    # app = trainer(log_path="src/deep_awb/logs", **example_params)
    # app_handle = runner.run(app, scheduler="local_cwd")
    # status = runner.wait(app_handle)
    # print(status)
