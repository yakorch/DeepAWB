from ax.core import ChoiceParameter, OrderConstraint, Parameter, ParameterType, RangeParameter, SearchSpace

_FEATURE_EXTRACTOR_DEPTH = 3

parameters = [
    RangeParameter(name="total_hidden_neurons", lower=40, upper=250, parameter_type=ParameterType.INT),
    RangeParameter(name="decay_rate", lower=0, upper=1.5, parameter_type=ParameterType.FLOAT),
    RangeParameter(
        name="MLP_depth",
        lower=2,
        upper=5,
        parameter_type=ParameterType.INT,
    ),
    RangeParameter(
        name="initial_learning_rate",
        lower=1e-4,
        upper=1e-2,
        parameter_type=ParameterType.FLOAT,
        log_scale=True,
    ),
    RangeParameter(
        name="final_learning_rate",
        lower=1e-6,
        upper=1e-4,
        parameter_type=ParameterType.FLOAT,
        log_scale=True,
    ),
    RangeParameter(
        name="epochs",
        lower=10,
        upper=50,
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(name="image_scale", values=[1, 1.5, 2, 2.5, 3, 3.5], parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
    *(RangeParameter(name=f"n_kernels_{i}", lower=16, upper=256, parameter_type=ParameterType.INT) for i in range(_FEATURE_EXTRACTOR_DEPTH)),
    *(RangeParameter(name=f"kernel_size_{i}", lower=3, upper=7, parameter_type=ParameterType.INT) for i in range(_FEATURE_EXTRACTOR_DEPTH)),
    *(RangeParameter(name=f"stride_{i}", lower=1, upper=5, parameter_type=ParameterType.INT) for i in range(_FEATURE_EXTRACTOR_DEPTH)),
]

param_name_map: dict[str, Parameter] = {param.name: param for param in parameters}

assert param_name_map["initial_learning_rate"].lower >= param_name_map["final_learning_rate"].upper

search_space = SearchSpace(
    parameters=parameters,
    parameter_constraints=[
        *(OrderConstraint(lower_parameter=param_name_map[f"stride_{i}"], upper_parameter=param_name_map[f"kernel_size_{i}"]) for i in range(_FEATURE_EXTRACTOR_DEPTH)),
        *(OrderConstraint(lower_parameter=param_name_map[f"n_kernels_{i}"], upper_parameter=param_name_map[f"n_kernels_{i+1}"]) for i in range(_FEATURE_EXTRACTOR_DEPTH - 1)),
    ],
)


if __name__ == "__main__":
    raise NotImplementedError()
