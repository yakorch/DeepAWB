from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .. import console_logger
from .data_loaders import SimpleCubePPDatasetInfo, get_test_dataset, get_train_dataset


@console_logger.catch
def estimate_wb_gains_density(subset: Literal["train", "test", "all"], visualize: bool) -> Callable:
    SimpleCubePPDatasetInfo.setup()

    if subset == "train":
        dataset = get_train_dataset()
        annotations = dataset.annotations
    elif subset == "test":
        dataset = get_test_dataset()
        annotations = dataset.annotations
    elif subset == "all":
        train_dataset = get_train_dataset()
        test_dataset = get_test_dataset()
        annotations = pd.concat([train_dataset.annotations, test_dataset.annotations])
    else:
        raise ValueError(f"Invalid subset: {subset}")

    wb_gains = np.vstack([annotations["R/G"], annotations["B/G"]])

    kde = gaussian_kde(wb_gains, bw_method=0.5)

    if not visualize:
        return kde

    import plotly.graph_objects as go

    x, y = np.mgrid[0.2:1.2:200j, 0.2:1.2:200j]
    positions = np.vstack([x.ravel(), y.ravel()])
    density = kde(positions).T
    density = np.reshape(density, x.shape)

    z_min, z_max = density.min(), density.max()

    surface = go.Surface(
        z=density,
        x=x,
        y=y,
        colorscale="Viridis",
        opacity=0.7,
        cmin=z_min,
        cmax=z_max,
        showscale=True,
    )
    z_vals = kde(wb_gains)

    scatter = go.Scatter3d(
        x=wb_gains[0],
        y=wb_gains[1],
        z=z_vals,
        mode="markers",
        marker=dict(size=2, color="pink"),
        name="WB Gains",
    )

    fig = go.Figure(data=[surface, scatter])

    fig.update_layout(
        scene=dict(xaxis_title="R/G Gain", yaxis_title="B/G Gain", zaxis_title="Density"),
        title="3D KDE of WB Gains",
    )
    fig.show()

    return kde


if __name__ == "__main__":
    _ = estimate_wb_gains_density(subset="all", visualize=True)
