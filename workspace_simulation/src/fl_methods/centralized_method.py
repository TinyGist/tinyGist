import logging

import torch

from src.utils.communication_recorder import PacketPayload

from .base import FLMethods
from .definitions import FISHER_AGGREGATION_SCORE_SOURCES, canonical_aggregation_score_source
from .segment_ops import aggregate_dense_weighted_values


log = logging.getLogger(__name__)


class Centralized(FLMethods):
    def __init__(
            self,
            total_models_dict: dict,
            parameter_scope="all",
            aggregation_score_source="uniform",
            bn_mode="affine",
            bn_process_as_base_unit=True,
            bn_aggregation_source="score",
    ):
        super().__init__(
            total_models_dict,
            parameter_scope=parameter_scope,
            bn_mode=bn_mode,
            bn_process_as_base_unit=bn_process_as_base_unit,
            bn_aggregation_source=bn_aggregation_source,
        )
        self.__model_parameters_before_aggregation = None
        self.__model_parameters_after_aggregation = None
        self.__aggregation_score_source = canonical_aggregation_score_source(aggregation_score_source)

    def simulate_method(
            self,
            current_target_model_idx_list: list | None = None,
            current_scores_dict: dict | None = None,
            current_fisher_weights_dict: dict | None = None,
            current_model_idx_list: list | None = None,
            current_global_round: int | None = None,
            current_local_round_dict: dict | None = None,
            aggregation_score_source=None,
            **kwargs,
    ):
        previous_score_source = self.__aggregation_score_source
        if aggregation_score_source is not None:
            self.__aggregation_score_source = canonical_aggregation_score_source(
                aggregation_score_source
            )
        try:
            return self._simulate_method(
                current_target_model_idx_list=current_target_model_idx_list,
                current_scores_dict=current_scores_dict,
                current_fisher_weights_dict=current_fisher_weights_dict,
                current_model_idx_list=current_model_idx_list,
                current_global_round=current_global_round,
                current_local_round_dict=current_local_round_dict,
                **kwargs,
            )
        finally:
            self.__aggregation_score_source = previous_score_source

    def _simulate_method(
            self,
            current_target_model_idx_list: list | None = None,
            current_scores_dict: dict | None = None,
            current_fisher_weights_dict: dict | None = None,
            current_model_idx_list: list | None = None,
            current_global_round: int | None = None,
            current_local_round_dict: dict | None = None,
            **_,
    ):
        if current_target_model_idx_list is None:
            current_target_model_idx_list = current_model_idx_list
        assert current_target_model_idx_list is not None, "Please provide target model indexes"

        num_current_models = len(current_target_model_idx_list)
        if num_current_models <= 1:
            log.info("The number of current models is %s. So there is no aggregation", num_current_models)
            return

        current_target_models_dict = {
            model_idx: self._total_model_dict[model_idx]
            for model_idx in current_target_model_idx_list
        }
        current_outgoing_models_dict = {
            model_idx: self._outgoing_model_dict[model_idx]
            for model_idx in current_target_model_idx_list
        }
        self.__model_parameters_before_aggregation = self._get_parameters_from_models(
            current_outgoing_models_dict
        )
        assert isinstance(self.__model_parameters_before_aggregation, dict), "Not get parameters yet"
        averaged_bn_parameters = None
        if self._uses_uniform_bn_aggregation():
            bn_vectors = [
                self._get_bn_from_model(model).detach().clone()
                for model in current_outgoing_models_dict.values()
            ]
            if bn_vectors and bn_vectors[0].numel() > 0:
                for model_idx, bn_vector in zip(current_outgoing_models_dict, bn_vectors):
                    if not torch.all(torch.isfinite(bn_vector)):
                        raise ValueError(
                            f"Centralized BatchNorm input for model {model_idx} "
                            "contains non-finite values"
                        )
                averaged_bn_parameters = torch.stack(bn_vectors).mean(dim=0)
                if not torch.all(torch.isfinite(averaged_bn_parameters)):
                    raise ValueError(
                        "Centralized BatchNorm average contains non-finite values"
                    )
        if self.__aggregation_score_source in FISHER_AGGREGATION_SCORE_SOURCES:
            assert isinstance(current_fisher_weights_dict, dict), "Invalid Fisher weights"
        elif self.__aggregation_score_source == "val_acc":
            assert isinstance(current_scores_dict, dict), "Invalid scores"
        separate_bn_elements = 0 if averaged_bn_parameters is None else averaged_bn_parameters.numel()
        self._record_centralized_uploads(
            current_target_model_idx_list,
            current_global_round,
            current_local_round_dict,
            current_fisher_weights_dict,
            separate_bn_elements,
        )

        if self.__aggregation_score_source in FISHER_AGGREGATION_SCORE_SOURCES:
            averaged_parameters = self.__aggregate_weighted(
                self.__model_parameters_before_aggregation,
                current_fisher_weights_dict,
            )
        elif self.__aggregation_score_source == "val_acc":
            averaged_parameters = self.__aggregate_weighted(
                self.__model_parameters_before_aggregation,
                current_scores_dict,
            )
        else:
            averaged_parameters = self.__aggregate_uniform(self.__model_parameters_before_aggregation)

        self._record_centralized_downloads(
            current_target_model_idx_list,
            current_global_round,
            current_local_round_dict,
            averaged_parameters,
            separate_bn_elements,
        )

        self.__model_parameters_after_aggregation = {
            model_idx: averaged_parameters
            for model_idx in current_target_models_dict.keys()
        }
        self._load_parameters_to_models(current_target_models_dict, self.__model_parameters_after_aggregation)
        if averaged_bn_parameters is not None:
            for model in current_target_models_dict.values():
                self._load_bn_to_model(model, averaged_bn_parameters)

    def __aggregate_uniform(self, parameter_dict: dict):
        if not parameter_dict:
            raise ValueError("Centralized aggregation requires at least one model")
        first_parameter = next(iter(parameter_dict.values()))
        parameter_sum = torch.zeros_like(first_parameter)
        for model_idx, parameter in parameter_dict.items():
            if parameter.shape != first_parameter.shape:
                raise ValueError("Centralized model parameter lengths differ")
            if not torch.all(torch.isfinite(parameter)):
                raise ValueError(
                    f"Centralized parameters for model {model_idx} "
                    "contain non-finite values"
                )
            parameter_sum += parameter
        averaged_parameters = parameter_sum / len(parameter_dict)
        if not torch.all(torch.isfinite(averaged_parameters)):
            raise ValueError("Centralized parameter average contains non-finite values")
        return averaged_parameters

    def __aggregate_weighted(self, parameter_dict, weight_dict):
        model_indices = list(parameter_dict)
        if self.__aggregation_score_source in FISHER_AGGREGATION_SCORE_SOURCES:
            for model_idx in model_indices:
                fisher_numel = torch.as_tensor(weight_dict[model_idx]).numel()
                parameter_numel = parameter_dict[model_idx].numel()
                if fisher_numel != parameter_numel:
                    raise ValueError(
                        "Fisher weight length does not match parameters for "
                        f"model {model_idx}: got {fisher_numel}, "
                        f"expected {parameter_numel}"
                    )
        return aggregate_dense_weighted_values(
            [parameter_dict[idx] for idx in model_indices],
            [weight_dict[idx] for idx in model_indices],
            exponential_base=self._aggregation_weight_exp_base,
        )
    def _record_centralized_uploads(
            self,
            current_target_model_idx_list,
            current_global_round,
            current_local_round_dict,
            current_fisher_weights_dict,
            separate_bn_elements,
    ):
        if self._communication_recorder is None:
            return

        current_global_round = 0 if current_global_round is None else current_global_round
        current_local_round_dict = current_local_round_dict or {}

        for model_idx in current_target_model_idx_list:
            if self.__aggregation_score_source in FISHER_AGGREGATION_SCORE_SOURCES:
                aggregation_weight_elements = current_fisher_weights_dict[model_idx].numel()
            elif self.__aggregation_score_source == "val_acc":
                aggregation_weight_elements = 1
            else:
                aggregation_weight_elements = 0

            self._record_centralized_packet(
                global_round=current_global_round,
                packet_kind="model_upload",
                source_device=model_idx,
                destination_device="coordinator",
                source_local_round=current_local_round_dict.get(model_idx),
                destination_local_round=None,
                payload=PacketPayload(
                    model_parameter_elements=self.__model_parameters_before_aggregation[model_idx].numel(),
                    aggregation_weight_elements=aggregation_weight_elements,
                    batch_norm_elements=separate_bn_elements,
                ),
            )

    def _record_centralized_downloads(
            self,
            current_target_model_idx_list,
            current_global_round,
            current_local_round_dict,
            averaged_parameters,
            separate_bn_elements,
    ):
        if self._communication_recorder is None:
            return

        current_global_round = 0 if current_global_round is None else current_global_round
        current_local_round_dict = current_local_round_dict or {}
        for model_idx in current_target_model_idx_list:
            self._record_centralized_packet(
                global_round=current_global_round,
                packet_kind="model_download",
                source_device="coordinator",
                destination_device=model_idx,
                source_local_round=None,
                destination_local_round=current_local_round_dict.get(model_idx),
                payload=PacketPayload(
                    model_parameter_elements=averaged_parameters.numel(),
                    batch_norm_elements=separate_bn_elements,
                ),
            )

    def _record_centralized_packet(self, **packet):
        self._record_communication_packet(
            selection_mode="centralized",
            status="delivered",
            **packet,
        )

    def get_models(self) -> dict:
        return self._total_model_dict
