import pathlib
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import torch
from loguru import logger as console_logger

from src.visualization.image_visualization import compute_unbalanced_image

from .estimate_CT_curve import load_models


def compute_delta_squared_sum(gains: tuple[float, float], red_ratios: np.array, blue_ratios: np.array, max_delta_squared_value) -> float:
    red_gain, blue_gain = (1 / x for x in gains)
    return np.clip((red_ratios * red_gain - 1) ** 2 + (blue_ratios * blue_gain - 1) ** 2, None, max_delta_squared_value).sum()


_GP_model, _GP_likelihood = load_models()
_red_upper_bound = 1.2
_red_lower_bound = 0.2
_red_coarse_points = 100
_CT_red_g_step = (_red_upper_bound - _red_lower_bound) / _red_coarse_points


def get_gain_pairs():
    r_gains = torch.linspace(_red_lower_bound, _red_upper_bound, _red_coarse_points)

    with torch.no_grad():
        predictions = _GP_likelihood(_GP_model(r_gains))
        b_gains = predictions.mean
    b_gains = b_gains.numpy()
    return np.vstack([r_gains.numpy(), b_gains]).T


_coarse_gain_pairs = get_gain_pairs()


def coarse_search(red_ratios, blue_ratios, max_delta_squared: float) -> int:
    best_gain_pair = None
    best_delta_squared_sum = np.inf

    for i in range(_coarse_gain_pairs.shape[0]):
        delta_squared_sum = compute_delta_squared_sum(_coarse_gain_pairs[i], red_ratios, blue_ratios, max_delta_squared_value=max_delta_squared)
        if delta_squared_sum < best_delta_squared_sum:
            best_delta_squared_sum = delta_squared_sum
            best_gain_pair = _coarse_gain_pairs[i]
    return best_gain_pair


def fine_search(gain, metric_by_gain: Callable, visualize=False):
    r_gain, _ = gain

    CT_steps = 10
    CT_max_step = _CT_red_g_step

    Transverse_steps = 10
    Transverse_step_size = _CT_red_g_step / 10

    close_r_gains = np.linspace(r_gain - CT_max_step, r_gain + CT_max_step, CT_steps)
    with torch.no_grad():
        close_b_gains = _GP_likelihood(_GP_model(torch.tensor(close_r_gains))).mean.numpy()

    if visualize:
        r_gain_points = []
        b_gain_points = []
        delta_squared_sums = []

    best_gain_pair = None
    best_metric_value = np.inf

    for i in range(1, CT_steps - 1):
        current_r_gain, current_b_gain = close_r_gains[i], close_b_gains[i]

        previous_r_gain, previous_b_gain = close_r_gains[i - 1], close_b_gains[i - 1]
        next_r_gain, next_b_gain = close_r_gains[i + 1], close_b_gains[i + 1]

        gradient_slope = (next_b_gain - previous_b_gain) / (next_r_gain - previous_r_gain)
        orthogonal_slope = -1 / gradient_slope

        for j in range(Transverse_steps):
            transverse_r_gain = current_r_gain + (j - Transverse_steps // 2) * Transverse_step_size
            transverse_b_gain = current_b_gain + orthogonal_slope * (transverse_r_gain - current_r_gain)
            delta_squared_sum = metric_by_gain((transverse_r_gain, transverse_b_gain))
            if delta_squared_sum < best_metric_value:
                best_metric_value = delta_squared_sum
                best_gain_pair = (transverse_r_gain, transverse_b_gain)

            if visualize:
                r_gain_points.append(transverse_r_gain)
                b_gain_points.append(transverse_b_gain)
                delta_squared_sums.append(delta_squared_sum)

    if not visualize:
        return best_gain_pair

    fig = go.Figure(
        data=go.Scatter(
            x=r_gain_points,
            y=b_gain_points,
            mode="markers",
            marker=dict(
                size=8,
                color=delta_squared_sums,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Delta Squared Sum"),
            ),
            text=[f"Delta2: {val:.3f}" for val in delta_squared_sums],
            hoverinfo="text",
        )
    )

    smooth_r_gains = np.linspace(r_gain - 2 * CT_max_step, r_gain + 2 * CT_max_step, 100)
    with torch.no_grad():
        smooth_b_gains = _GP_likelihood(_GP_model(torch.tensor(smooth_r_gains))).mean.numpy()
    fig.add_trace(go.Scatter(x=smooth_r_gains, y=smooth_b_gains, mode="lines", line=dict(color="blue", width=2), name="GP"))

    fig.update_layout(title="Fine Search of Gains", xaxis_title="R Gain", yaxis_title="B Gain", hovermode="closest", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    fig.show()

    return best_gain_pair


def get_image_patches(image):
    H, W, C = image.shape
    assert C == 3

    height, width = 12, 18
    assert (H % height == 0) and (W % width == 0)

    block_H = H // height
    block_W = W // width

    blocks = image.reshape(height, block_H, width, block_W, C)
    blocks = blocks.swapaxes(1, 2)
    return blocks


def compute_channel_patches_averages(blocks, min_avg_G):
    output = []
    for h in range(blocks.shape[0]):
        for w in range(blocks.shape[1]):
            block = blocks[h, w]
            mean = block.mean(axis=(0, 1))
            if mean[1] < min_avg_G:
                continue
            output.append(mean)
    return output


_MLE_gains_estimate = (0.425, 0.7)


def perform_bayes_AWB(image: pathlib.Path | np.ndarray, min_avg_G=0.05, max_delta_squared=0.35, visualize=False):
    """
    Takes either an image path, or an image array.
    """
    if isinstance(image, pathlib.Path):
        image = compute_unbalanced_image(image)

    blocks = get_image_patches(image)
    averages = np.array(compute_channel_patches_averages(blocks, min_avg_G=min_avg_G))
    if len(averages) == 0:
        console_logger.warning("No patches with enough green average found.")
        return _MLE_gains_estimate
    else:
        red_ratios = averages[:, 0] / averages[:, 1]
        blue_ratios = averages[:, 2] / averages[:, 1]
        gain_pair = coarse_search(red_ratios, blue_ratios, max_delta_squared)

    gain_pair = fine_search(gain_pair, lambda gains: compute_delta_squared_sum(gains, red_ratios, blue_ratios, max_delta_squared), visualize=visualize)
    return gain_pair
