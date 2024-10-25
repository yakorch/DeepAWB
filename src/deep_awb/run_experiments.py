from ax.core import Experiment
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import Scheduler, SchedulerOptions

from src.deep_awb.experiment_metrics import opt_config
from src.deep_awb.search_space import search_space
from src.deep_awb.torchx_ax_runner import ax_runner

experiment = Experiment(
    name="torchx_awb",
    search_space=search_space,
    optimization_config=opt_config,
    runner=ax_runner,
)


total_trials = 10


gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=total_trials,
    num_initialization_trials=3,
)


scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(total_trials=total_trials, max_pending_trials=1, tolerated_trial_failure_rate=0.9),
)

if __name__ == "__main__":
    scheduler.run_all_trials()

    from ax.service.utils.report_utils import exp_to_df

    df = exp_to_df(experiment)
    print(df.head(10))

    df.to_csv("torchx_Ax_Deep_AWB.csv")
