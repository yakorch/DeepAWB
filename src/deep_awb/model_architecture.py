from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger as console_logger
from torch.nn import functional as F

from .data_loaders import get_test_data_loader, get_train_data_loader
from .fit_loss_function import estimate_wb_gains_density

_N_CLASSES = 2


_KDE = estimate_wb_gains_density(visualize=False)


class ConvReLUMaxPoolBlock(nn.Module):
    def __init__(self, in_channels: int, kernels: int, kernel_size: int, stride: int) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            console_logger.warning(f"{kernel_size=} is not odd.")

        padding = (kernel_size - 1) // 2

        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=kernels, kernel_size=kernel_size, stride=stride, padding=padding)
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


class DeepAWBModel(pl.LightningModule):
    MAX_POSSIBLE_GAIN = 1.2

    def __init__(
        self,
        block_configs: list[ConvReLUMaxPoolBlockConfig],
        hidden_neurons: list[int],
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()

        self.feature_extractor, embedding_space_dim = FeatureExtractorBuilder(block_configs)
        self.regression_layer = RegressionBuilder(embedding_space_dim, hidden_neurons)
        self.model = nn.Sequential(self.feature_extractor, self.regression_layer)

        self.learning_rate = learning_rate

    def forward(self, x):
        return F.sigmoid(self.model(x)) * self.MAX_POSSIBLE_GAIN

    def _process_batch(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.mse_loss(predictions, y, reduction="none")

        weights = 1 / np.sqrt(_KDE(y.cpu().T))
        weights = torch.tensor(weights, device=loss.device, dtype=torch.float32).unsqueeze(1)
        weighted_loss = loss * weights

        return weighted_loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._process_batch(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage=None): ...

    def train_dataloader(self):
        return get_train_data_loader()

    def val_dataloader(self):
        return get_test_data_loader()


if __name__ == "__main__":
    raise NotImplementedError()
