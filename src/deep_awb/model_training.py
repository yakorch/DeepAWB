import sys

if __name__ == "__main__":
    # Since this job may be run with the `torchx` CLI, we may redirect the stdout and stderr to log files for development purposes.
    sys.stdout = open("./stdout-log.txt", "w")
    sys.stderr = open("./stderr-log.txt", "w")

import argparse

import numpy as np
from loguru import logger as console_logger
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from .data_loaders import setup_common_transform
from .model_architecture import _N_CLASSES, ConvReLUMaxPoolBlockConfig, DeepAWBModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run the training experiment.")

    parser.add_argument("--image_scale", type=float, default=1, help="Scale factor for images.")

    parser.add_argument("--total_hidden_neurons", type=int, required=True, help="Total number of hidden neurons.")
    parser.add_argument("--MLP_depth", type=int, required=True, help="Depth of the MLP.")

    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")

    parser.add_argument("--n_kernels", type=int, nargs="+", required=True, help="Number of kernels.")
    parser.add_argument("--kernel_size", type=int, nargs="+", required=True, help="Kernel size.")
    parser.add_argument("--stride", type=int, nargs="+", required=True, help="Kernel stride.")

    parser.add_argument("--log_path", type=str, required=True, help="Path to the log file.")

    args = parser.parse_args()
    assert len(args.n_kernels) == len(args.kernel_size) == len(args.stride), "All kernel parameters must have the same length."
    return args


def create_DeepAWB_model(args) -> DeepAWBModel:
    conv_block_configs = [ConvReLUMaxPoolBlockConfig(n_kernels=args.n_kernels[i], kernel_size=args.kernel_size[i], stride=args.stride[i]) for i in range(len(args.n_kernels))]

    hidden_neurons = []
    n_hidden_layers = args.MLP_depth - 1
    decay_rate = 0.5
    distribution = []
    for i in range(n_hidden_layers):
        proportion = np.exp(-decay_rate * i)
        distribution.append(proportion)
    distribution = np.array(distribution) / np.sum(distribution)
    hidden_neurons = [int(args.total_hidden_neurons * d) for d in distribution]
    hidden_neurons[0] -= sum(hidden_neurons) - args.total_hidden_neurons

    hidden_neurons.append(_N_CLASSES)

    console_logger.debug(f"{hidden_neurons=}")

    return DeepAWBModel(
        block_configs=conv_block_configs,
        hidden_neurons=hidden_neurons,
        learning_rate=args.learning_rate,
    )


@console_logger.catch
def fit_model_and_log():
    args = parse_args()
    console_logger.debug(f"{args=}")

    setup_common_transform(args.image_scale)
    AWB_model = create_DeepAWB_model(args)

    trainer = Trainer(
        logger=False,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.epochs,
        enable_progress_bar=False,
        deterministic=True,
        default_root_dir=args.log_path,
        callbacks=[],
        enable_checkpointing=False,
    )

    logger = pl_loggers.TensorBoardLogger(args.log_path)

    print(f"Logging to path: {args.log_path}.")
    import time

    start = time.time()
    trainer.fit(model=AWB_model)
    end = time.time()
    train_time = end - start
    logger.log_metrics({"train_time": train_time})

    val_loss = trainer.callback_metrics["val_loss"]

    logger.log_metrics({"val_loss": val_loss})

    import random

    logger.log_metrics({"inference_time": random.random()})
    logger.log_metrics({"peak_RAM_usage": random.random()})

    logger.save()


if __name__ == "__main__":
    fit_model_and_log()
