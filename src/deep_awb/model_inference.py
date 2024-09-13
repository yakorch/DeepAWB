import argparse
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import yaml
from loguru import logger as console_logger
from memory_profiler import memory_usage

from .data_loaders import SimpleCubePPDatasetInfo

_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True, order=False)
class InferenceStats:
    cpu_time: float
    wall_time: float
    peak_memory_usage_MiB: float


def parse_args():
    parser = argparse.ArgumentParser(description="Run the inference measuring experiment.")
    parser.add_argument("--image_scale", type=int, required=True, help="Image dimensions.")
    parser.add_argument("--optimize", type=bool, required=True, help="Whether to optimize the model for inference.")
    parser.add_argument("--n_runs", type=int, required=True, help="Number of runs to measure the inference time.")
    parser.add_argument("--script_module_path", type=pathlib.Path, required=True, help="Path to the `ScriptModule` model.")
    parser.add_argument("--yaml_path", type=pathlib.Path, required=True, help="YAML inference results path.")

    return parser.parse_args()


def get_model_size(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()

    for buffer in model.buffers():
        total_params += buffer.numel() * buffer.element_size()

    return total_params / (1024**2)


def optimize_model(model: torch.jit.ScriptModule, sample_input: torch.Tensor) -> Callable:
    """
    Optimizes the model for inference.
    """
    model = torch.jit.freeze(model)
    model = torch.compile(model)
    # model = torch.jit.optimize_for_inference(model)
    return model


def min_inference_time(model: Callable, sample_input: torch.Tensor, n_runs: int) -> tuple[int, int]:
    cpu_time = np.inf
    wall_time = np.inf

    with torch.no_grad():
        for _ in range(n_runs):
            start_cpu = time.process_time()
            start_wall = time.perf_counter()

            model(sample_input)

            end_cpu = time.process_time()
            end_wall = time.perf_counter()

            cpu_time = min(cpu_time, (end_cpu - start_cpu))
            wall_time = min(wall_time, (end_wall - start_wall))
    return cpu_time, wall_time


if __name__ == "__main__":
    args = parse_args()

    SimpleCubePPDatasetInfo.setup(args.image_scale)
    sample_input = torch.rand(1, 3, *SimpleCubePPDatasetInfo.image_dims)

    script_module_path = args.script_module_path
    assert script_module_path.exists()

    model = torch.jit.load(script_module_path)
    model.to(_TORCH_DEVICE)
    console_logger.info(f"Current device: {_TORCH_DEVICE}")
    model.eval()

    if args.optimize:
        model = optimize_model(model, sample_input)

    (mem_usage_MiB_samples, (cpu_time, wall_time)) = memory_usage((min_inference_time, (model, sample_input, args.n_runs), {}), interval=0.005, timeout=1, retval=True)
    peak_memory_increment = max(mem_usage_MiB_samples) - mem_usage_MiB_samples[0]

    inference_stats = InferenceStats(cpu_time, wall_time, peak_memory_increment)
    console_logger.info(f"{inference_stats=}")

    with open(args.yaml_path, "w") as yaml_file:
        yaml.dump(asdict(inference_stats), yaml_file)
