import pickle

from ax.core import Experiment
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import Scheduler, SchedulerOptions

from .experiment_metrics import opt_config
from .search_space import search_space
from .torchx_ax_runner import ax_runner

experiment = Experiment(
    name="torchx_awb",
    search_space=search_space,
    optimization_config=opt_config,
    runner=ax_runner,
)


total_trials = 10  # total evaluation budget


gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=total_trials,
)


scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(total_trials=total_trials, max_pending_trials=1, tolerated_trial_failure_rate=0.9),
)


if __name__ == "__main__":
    scheduler.run_all_trials()

    with open("experiment.pkl", "wb") as f:
        pickle.dump(experiment, f)
