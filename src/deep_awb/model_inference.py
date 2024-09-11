import argparse
import os
import pathlib
import platform
import resource
import time
from dataclasses import dataclass

import numpy as np
import psutil
import torch
import torch.nn as nn
from loguru import logger as console_logger

from .data_loaders import IMAGE_HEIGHT, IMAGE_WIDTH
from .model_architecture import DeepAWBModel

_CURRENT_PLATFORM = platform.system()
_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the inference measuring experiment.")
    parser.add_argument("--image_scale", type=float, required=True, help="Scale factor for images.")
    parser.add_argument("--checkpoint_path", type=pathlib.Path, required=True, help="Path to `pytorch_lightning` checkpoint.")

    return parser.parse_args()


def get_max_memory_usage_bytes():
    max_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if _CURRENT_PLATFORM == "Darwin":
        max_mem_usage_bytes = max_mem_usage
    elif _CURRENT_PLATFORM == "Linux":
        max_mem_usage_bytes = max_mem_usage * 1024
    else:
        raise NotImplementedError(f"Unsupported platform: {_CURRENT_PLATFORM}")
    return max_mem_usage_bytes


def optimize_model(model: nn.Module, sample_input: torch.Tensor) -> None:
    torch.jit.enable_onednn_fusion(True)

    # import torch.fx.experimental.optimization as optimization
    # optimization.fuse(model, inplace=True)
    model = torch.jit.trace(model, sample_input)
    model = torch.jit.freeze(model)
    model = torch.compile(model)

    # model = torch.jit.optimize_for_inference(model)


@dataclass(frozen=True, order=False)
class InferenceStats:
    cpu_time: float
    wall_time: float
    max_memory_usage_MB: float


def measure_inference(image_scale: float, checkpoint_path: pathlib.Path) -> InferenceStats:
    TOTAL_RUNS = 200

    this_process = psutil.Process(os.getpid())
    current_mem_usage_bytes = this_process.memory_info().rss
    max_mem_usage_bytes = get_max_memory_usage_bytes()
    assert current_mem_usage_bytes == max_mem_usage_bytes, f"The current ({current_mem_usage_bytes}) and max ({max_mem_usage_bytes}) memory usage are not the same."

    model = DeepAWBModel.load_from_checkpoint(checkpoint_path).model.to(_TORCH_DEVICE)
    model.eval()
    random_image_input = torch.randn(size=(1, 3, int(IMAGE_HEIGHT / image_scale), int(IMAGE_WIDTH / image_scale)), device=_TORCH_DEVICE)
    optimize_model(model, random_image_input)

    cpu_time = np.inf
    wall_time = np.inf

    with torch.no_grad():
        for _ in range(TOTAL_RUNS):
            start_cpu = time.process_time()
            start_wall = time.perf_counter_ns()

            model(random_image_input)

            end_cpu = time.process_time()
            end_wall = time.perf_counter_ns()

            cpu_time = min(cpu_time, (end_cpu - start_cpu))
            wall_time = min(wall_time, (end_wall - start_wall))

    max_mem_usage_bytes = get_max_memory_usage_bytes()

    return InferenceStats(cpu_time, wall_time / 1e9, (max_mem_usage_bytes - current_mem_usage_bytes) / 1024**2)


def submit_measuring_job(): ...


if __name__ == "__main__":
    args = parse_args()
    console_logger.info(measure_inference(args.image_scale, args.checkpoint_path))
