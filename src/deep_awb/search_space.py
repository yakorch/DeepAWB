from ax.core import ChoiceParameter, OrderConstraint, Parameter, ParameterType, RangeParameter, SearchSpace

_FEATURE_EXTRACTOR_WIDTH = 3

parameters = [
    ChoiceParameter(name="total_hidden_neurons", values=[40, 80, 120, 180, 250], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True),
    RangeParameter(
        name="MLP_depth",
        lower=2,
        upper=5,
        parameter_type=ParameterType.INT,
    ),
    RangeParameter(
        name="learning_rate",
        lower=1e-4,
        upper=1e-2,
        parameter_type=ParameterType.FLOAT,
        log_scale=True,
    ),
    RangeParameter(
        name="epochs",
        lower=3,
        upper=10,
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(name="image_scale", values=[1, 1.5, 2], parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True),
    *(ChoiceParameter(name=f"n_kernels_{i}", values=[8, 16, 32], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True) for i in range(_FEATURE_EXTRACTOR_WIDTH)),
    *(RangeParameter(name=f"kernel_size_{i}", lower=3, upper=7, parameter_type=ParameterType.INT) for i in range(_FEATURE_EXTRACTOR_WIDTH)),
    *(RangeParameter(name=f"stride_{i}", lower=1, upper=5, parameter_type=ParameterType.INT) for i in range(_FEATURE_EXTRACTOR_WIDTH)),
]

param_name_map: dict[str, Parameter] = {param.name: param for param in parameters}

search_space = SearchSpace(
    parameters=parameters,
    parameter_constraints=[
        *(OrderConstraint(lower_parameter=param_name_map[f"stride_{i}"], upper_parameter=param_name_map[f"kernel_size_{i}"]) for i in range(_FEATURE_EXTRACTOR_WIDTH)),
    ],
)


if __name__ == "__main__":
    raise NotImplementedError()
