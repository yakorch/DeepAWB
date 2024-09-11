import time

import numpy as np
import torch

from .data_loaders import get_test_dataset, setup_common_transform
from .model_training import console_logger, create_DeepAWB_model, parse_args

console_logger.level("WARNING")


def measure_inference():
    args = parse_args()
    setup_common_transform(args.image_scale)

    AWB_model = create_DeepAWB_model(args)
    TOTAL_RUNS = 50
    random_image_input = torch.tensor(np.array([get_test_dataset()[0][0]]))

    start = time.process_time()
    for _ in range(TOTAL_RUNS):
        AWB_model(random_image_input)
    end = time.process_time()

    total_time = end - start
    average_time = total_time / TOTAL_RUNS

    return average_time


def submit_measuring_job():
    ...


if __name__ == "__main__":
    print(measure_inference())
