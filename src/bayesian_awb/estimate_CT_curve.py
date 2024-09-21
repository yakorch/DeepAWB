import gpytorch
import plotly.graph_objects as go
import torch
import tqdm

from ..deep_awb.data_loaders import SimpleCubePPDatasetInfo, get_train_dataset
from . import GP_LIKELIHOOD_PATH, GP_MODEL_PATH


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@(lambda f: f())
def _get_train_data():
    SimpleCubePPDatasetInfo.setup()
    train_annotations = get_train_dataset().annotations
    x, y = train_annotations["R/G"].values, train_annotations["B/G"].values
    x, y = (torch.tensor(item) for item in (x, y))
    return x, y


def fit_GP(iters: int = 888, save_models: bool = True):
    x, y = _get_train_data

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x, y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    progress_bar = tqdm.trange(iters)
    for _ in progress_bar:
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        progress_bar.set_description(f"Loss={loss.item():.4f}. `lengthscale`={model.covar_module.base_kernel.lengthscale.item():.4f}. `noise`={model.likelihood.noise.item():.4f}.")
        optimizer.step()
    model.eval()
    likelihood.eval()

    if save_models:
        torch.save(model.state_dict(), GP_MODEL_PATH)
        torch.save(likelihood.state_dict(), GP_LIKELIHOOD_PATH)

    return model, likelihood


def load_models(gp_model_path=GP_MODEL_PATH, likelihood_path=GP_LIKELIHOOD_PATH):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    x, y = _get_train_data

    model = ExactGPModel(x, y, likelihood)

    model.load_state_dict(torch.load(gp_model_path, weights_only=True))
    likelihood.load_state_dict(torch.load(likelihood_path, weights_only=True))

    model.eval()
    likelihood.eval()

    return model, likelihood


def show_GP_CT_curve(gp_model, likelihood):
    r_gains = torch.linspace(0.2, 1.2, 2500)

    with torch.no_grad():
        predictions = likelihood(gp_model(r_gains.unsqueeze(-1)))

        lower, upper = predictions.confidence_region()
        x, y = _get_train_data

        fig = go.Figure(
            [
                go.Scatter(x=r_gains, y=predictions.mean.flatten(), mode="markers", line=dict(color="blue"), name="Prediction"),
                go.Scatter(x=r_gains, y=upper, fill=None, mode="lines", line=dict(color="lightblue"), showlegend=False),
                go.Scatter(x=r_gains, y=lower, fill="tonexty", mode="lines", line=dict(color="lightblue"), fillcolor="rgba(0,100,255,0.2)", name="Confidence Interval"),
                go.Scatter(x=x, y=y, mode="markers", marker=dict(color="black", size=1.5), name="Training Data"),
            ]
        )
        fig.update_layout(
            title="Gaussian Process Regression of CT curve with Confidence Interval",
            xaxis_title="R/G gain",
            yaxis_title="B/G gain",
        )

        fig.update_xaxes(
            range=[0, 1.2],
        )
        fig.update_yaxes(
            range=[0, 1.2],
        )

        fig.show()


if __name__ == "__main__":
    # fit_GP()
    show_GP_CT_curve(*load_models())
