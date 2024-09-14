import pathlib

from ax.core import MultiObjective, Objective, ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.metrics.tensorboard import TensorboardMetric
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer

from .__init__ import LOG_DIR


class DeepAWBTensorboardMetric(TensorboardMetric):
    # NOTE: We need to tell the new TensorBoard metric how to get the id /
    # file handle for the TensorBoard logs from a trial. In this case
    # our convention is to just save a separate file per trial in
    # the prespecified log dir.
    def _get_event_multiplexer_for_trial(self, trial):
        mul = event_multiplexer.EventMultiplexer(max_reload_threads=20)
        mul.AddRunsFromDirectory(pathlib.Path(LOG_DIR).joinpath(str(trial.index)).as_posix(), None)
        mul.Reload()
        return mul

    # This indicates whether the metric is queryable while the trial is
    # still running. We don't use this in the current tutorial, but Ax
    # utilizes this to implement trial-level early-stopping functionality.
    @classmethod
    def is_available_while_running(cls):
        return False


final_val_loss = DeepAWBTensorboardMetric(
    name="final_val_loss",
    tag="final_val_loss",
    lower_is_better=True,
)

inference_time = DeepAWBTensorboardMetric(
    name="inference_time",
    tag="inference_time",
    lower_is_better=True,
)

peak_RAM_usage = DeepAWBTensorboardMetric(
    name="peak_RAM_usage",
    tag="peak_RAM_usage",
    lower_is_better=True,
)


opt_config = MultiObjectiveOptimizationConfig(
    objective=MultiObjective(
        objectives=[
            Objective(metric=final_val_loss, minimize=True),
            Objective(metric=inference_time, minimize=True),
            Objective(metric=peak_RAM_usage, minimize=True),
        ],
    ),
    objective_thresholds=[
        ObjectiveThreshold(metric=final_val_loss, bound=0.0005, relative=False),
        ObjectiveThreshold(metric=inference_time, bound=2e-2, relative=False),
        ObjectiveThreshold(metric=peak_RAM_usage, bound=5 * 10**6, relative=False),
    ],
)
