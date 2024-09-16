from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger as console_logger
from torch.nn import functional as F

from .data_loaders import get_test_data_loader, get_train_data_loader, get_train_dataset
from .density_estimation import estimate_wb_gains_density

_N_CLASSES = 2
_KDE = estimate_wb_gains_density(subset="train", visualize=False)
_KDE_Epsilon = 1e-6


class ConvReLUMaxPoolBlock(nn.Module):
    def __init__(self, in_channels: int, kernels: int, kernel_size: int, stride: int) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            console_logger.warning(f"{kernel_size=} is not odd.")

        padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=kernels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(kernels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(F.relu(self.bn(self.conv_layer(x))))


@dataclass(frozen=True, order=False)
class ConvReLUMaxPoolBlockConfig:
    n_kernels: int
    kernel_size: int
    stride: int


def FeatureExtractorBuilder(block_configs: list[ConvReLUMaxPoolBlockConfig]) -> nn.Module:
    """
    Takes in the block configs.
    Returns:
    - a feature extractor that is a sequence of `ConvReLUMaxPoolBlock`s, followed by an average pooling and flattening.
    - a dimension of feature extractor output.
    """
    assert len(block_configs)

    input_image_channels = 3
    first_block = ConvReLUMaxPoolBlock(
        in_channels=input_image_channels, kernels=block_configs[0].n_kernels, kernel_size=block_configs[0].kernel_size, stride=block_configs[0].stride
    )
    other_blocks = []
    for i, block_config in enumerate(block_configs[1:]):
        other_blocks.append(
            ConvReLUMaxPoolBlock(in_channels=block_configs[i].n_kernels, kernels=block_config.n_kernels, kernel_size=block_config.kernel_size, stride=block_config.stride)
        )
    return nn.Sequential(first_block, *other_blocks, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()), block_configs[-1].n_kernels


def RegressionBuilder(embedding_space_dim: int, hidden_neurons: list[int]) -> nn.Module:
    all_neurons = [embedding_space_dim] + hidden_neurons
    hidden_layers = []
    for i in range(len(hidden_neurons)):
        hidden_layers.append(nn.Linear(all_neurons[i], all_neurons[i + 1]))
        hidden_layers.append(nn.ReLU())
    hidden_layers.pop()
    return nn.Sequential(*hidden_layers)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def ScaledSigmoid(scale):
    return LambdaLayer(lambda x: F.sigmoid(x) * scale)


def ScaledTanh(requested_min: float, requested_max: float):
    tanh_min, tanh_max = -1, 1

    scale = (requested_max - requested_min) / (tanh_max - tanh_min)
    bias = requested_min - scale * tanh_min

    return LambdaLayer(lambda x: F.tanh(x) * scale + bias)


class DeepAWBModel(nn.Module):
    MAX_POSSIBLE_GAIN = 1.2

    def __init__(self, block_configs: list[ConvReLUMaxPoolBlockConfig], hidden_neurons: list[int]) -> None:
        super().__init__()

        self.feature_extractor, embedding_space_dim = FeatureExtractorBuilder(block_configs)
        self.regression_layer = RegressionBuilder(embedding_space_dim, hidden_neurons)

        # final_activation = ScaledSigmoid(self.MAX_POSSIBLE_GAIN)
        final_activation = ScaledTanh(requested_min=0, requested_max=self.MAX_POSSIBLE_GAIN)

        self.model = nn.Sequential(self.feature_extractor, self.regression_layer, final_activation)
        console_logger.debug(f"{self.model=}")

    def forward(self, x):
        return self.model(x)


class DeepAWBLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: DeepAWBModel,
        epochs: int,
        learning_rates: Iterable[float],
    ) -> None:
        super().__init__()

        self.model = model

        assert isinstance(learning_rates, Iterable)
        self.learning_rates = learning_rates
        self.epochs = epochs

        train_densities = _KDE(get_train_dataset().annotations[["R/G", "B/G"]].T)
        train_weights = 1 / (train_densities + _KDE_Epsilon)
        self.train_weights_mean = train_weights.mean()
        self.weight_boundaries = (0.1, 15)

    def forward(self, x):
        return self.model(x)

    def _process_batch(self, batch, batch_idx, is_train: bool):
        x, y = batch
        predictions = self(x)
        if not is_train:
            return F.mse_loss(predictions, y)

        loss = F.mse_loss(predictions, y, reduction="none")
        weights = (1 / self.train_weights_mean) / (_KDE(y.cpu().T) + _KDE_Epsilon)
        np.clip(weights, *self.weight_boundaries, out=weights)
        weights = torch.tensor(weights, device=loss.device, dtype=torch.float32).unsqueeze(1)

        weighted_loss = loss * weights

        return weighted_loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self._process_batch(batch, batch_idx, is_train=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._process_batch(batch, batch_idx, is_train=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if len(self.learning_rates) == 1:
            initial_lr = self.learning_rates[0]
            final_lr = 1e-6
        elif len(self.learning_rates) == 2:
            initial_lr, final_lr = self.learning_rates
        else:
            raise ValueError(f"Learning rate type is not supported. {type(self.learning_rates)=}")

        optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=final_lr)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage=None): ...

    def train_dataloader(self):
        return get_train_data_loader()

    def val_dataloader(self):
        return get_test_data_loader()


if __name__ == "__main__":
    raise NotImplementedError()
