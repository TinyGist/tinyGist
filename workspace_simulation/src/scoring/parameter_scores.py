import numpy as np
import torch

from .definitions import canonical_parameter_score_method


def parameter_score_vector(
        model_parameters: torch.Tensor,
        parameter_score_method: str,
        parameter_score_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    parameter_score_method = canonical_parameter_score_method(parameter_score_method)
    if parameter_score_method == "weight_abs":
        return model_parameters.detach().abs()

    if parameter_score_weights is None:
        raise ValueError(
            f"parameter_score_method={parameter_score_method} requires buffered parameter scores"
        )
    if parameter_score_weights.numel() != model_parameters.numel():
        raise ValueError(
            "Buffered parameter score length does not match selected model parameter length"
        )
    return parameter_score_weights.detach().to(device=model_parameters.device, dtype=model_parameters.dtype).abs()


def create_parameter_score_thresholds(parameter_scores: torch.Tensor, seg_divided_number):
    scores = parameter_scores.detach().cpu().numpy()
    percents_list = np.linspace(start=0, stop=100, num=seg_divided_number + 1).tolist()[1:-1]
    return [np.percentile(scores, percent) for percent in percents_list]


def create_bitmapidx_based_parameter_scores(parameter_scores: torch.Tensor, threshold_list: list):
    scores = parameter_scores.detach().cpu().numpy()
    parameter_length = parameter_scores.numel()
    if not threshold_list:
        return [np.arange(parameter_length).tolist()]
    params_boolean_table_after_threshold = []
    for i in range(len(threshold_list) + 1):
        if i == 0:
            params_boolean_table = scores < threshold_list[i]
        elif i == len(threshold_list):
            params_boolean_table = scores >= threshold_list[i - 1]
        else:
            params_boolean_table = (
                scores >= threshold_list[i - 1]
            ) & (
                scores < threshold_list[i]
            )
        params_boolean_table_after_threshold.append(params_boolean_table)

    total_idx = np.arange(parameter_length)
    return [total_idx[boolean_table].tolist() for boolean_table in params_boolean_table_after_threshold]
