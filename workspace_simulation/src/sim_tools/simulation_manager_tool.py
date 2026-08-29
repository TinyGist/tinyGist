import csv
import logging
import os
from contextlib import contextmanager
from datetime import datetime

import torch
import yaml

from src.differential_privacy import (
    estimate_privacy_from_config,
    LocalDPSGDController,
    ModelUpdateDPController,
)
from src.data_getter import DATASETS
from src.models import NETWORKS
from src.models import Criteria
from src.models.model_registry import isolated_model_initialization_rng
from src.fl_methods import METHODS
from src.fl_methods.definitions import (
    BLOCK_SEGMENT_UNITS,
    FISHER_AGGREGATION_SCORE_SOURCES,
    LIPSCHITZ_BLOCK_SCORE_METHODS,
)
from src.fl_methods.method_factory import (
    build_fl_method,
    run_method,
)
from src.fl_methods.segment_ops import compute_parameter_block_score_tensor
from src.experiment_probes import (
    ImportanceCorrelationProbe,
    load_experiment_probe_config,
    preserve_random_state,
)
from src.utils.adaptive_loss_stair_tool import AdaptiveDFLLearningRate
from src.utils.communication_recorder import CommunicationRecorder
from src.utils.communication_simulator_tool import CommunicationSimulator
from src.utils.stale_training_tool import StaleTrainingSimulator
from src.sim_tools.config_abbreviation import config_abbreviation
from src.sim_tools.device import cuda_status, get_default_device
from src.utils.early_stop import EarlyStopController
from src.scoring.gradient_buffer_tool import GradientBuffer
from src.scoring.block_score_ema_buffer import BlockScoreEmaBuffer
from src.scoring.definitions import (
    DIRECT_PARAMETER_SCORE_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_POST_METHOD,
    FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD,
    GRADIENT_ABS_POST_METHOD,
    GRADIENT_SIGNAL_PRESERVATION_POST_METHOD,
    GRADIENT_WEIGHT_ABS_POST_METHOD,
    HESSIAN_EMA_PARAMETER_SCORE_METHODS,
    HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD,
    HUTCHINSON_DIAGONAL_POST_METHOD,
    HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD,
    canonical_parameter_score_method,
    POST_TRAINING_PARAMETER_SCORE_METHODS,
    parameter_score_record_dependencies,
)
from src.sim_tools.logging_utils import round_filter, setup_round_logging
from src.utils.object_detection_tools import (
    build_detection_metrics,
    infer_object_detection_task,
    validate_object_detection_configuration,
)
from src.sim_tools.simulation_config import (
    load_simulation_config,
    resolve_method_abbreviation,
    resolve_training_abbreviation,
)
from src.sim_tools.simulation_metrics import MetricValues, SimulationMetricsRecorder
from src.sim_tools.run_paths import (
    compact_run_label,
    path_component_hash,
    sanitize_path_part,
    shorten_path_component,
    summarize_run_config_stem,
)
from src.sim_tools.wall_clock import measure_wall_clock_stage, synchronized_perf_counter

log = logging.getLogger(__name__)
device = get_default_device()
RUN_LABEL_ENV_VAR = "WORKSPACE_SIM_RUN_LABEL"
SOURCE_CONFIG_ENV_VAR = "WORKSPACE_SIM_SOURCE_CONFIG"
AGGREGATION_DEVICE_POLICIES = {"auto", "cpu", "cuda"}
SAFE_RUN_DIR_NAME_MAX_BYTES = 96


def resolve_aggregation_device(policy, runtime_device=None) -> tuple[str, torch.device]:
    """Resolve the runner policy for temporary FL-method computations."""
    canonical_policy = str(policy).strip().lower()
    if canonical_policy not in AGGREGATION_DEVICE_POLICIES:
        raise ValueError(
            "aggregation_device must be one of auto, cpu, or cuda"
        )
    runtime_device = torch.device(runtime_device if runtime_device is not None else device)
    if canonical_policy == "cpu":
        return canonical_policy, torch.device("cpu")
    if canonical_policy == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "aggregation_device=cuda was requested, but CUDA is unavailable"
            )
        if runtime_device.type != "cuda":
            raise ValueError(
                "aggregation_device=cuda requires the simulation models to use CUDA"
            )
    return canonical_policy, runtime_device


class FederatedLearningSim:
    METRICS_CHECKPOINT_INTERVAL = 10

    WALL_CLOCK_COLUMNS = [
        "round",
        "task",
        "trainable_device_count",
        "total_s",
        "begin_round_s",
        "train_s",
        "post_score_s",
        "privacy_prepare_s",
        "training_metrics_s",
        "validation_before_s",
        "importance_probe_s",
        "topology_refresh_s",
        "fl_method_s",
        "test_after_s",
        "adaptive_lr_s",
        "finish_round_s",
        "bookkeeping_s",
    ]

    ROUND_RUNNERS = {
        "Classification": "run_simulation_classification_multi",
        "Object_Detection": "run_simulation_object_detection_multi",
    }

    def __init__(
            self,
            config_file,
            replicate_experiment: bool = False,
            pre_data_path=None,
            dataloader_workers: int = 0,
            aggregation_device: str = "auto",
    ):
        if isinstance(dataloader_workers, bool) or not isinstance(dataloader_workers, int):
            raise ValueError("dataloader_workers must be a non-negative integer")
        if dataloader_workers < 0:
            raise ValueError("dataloader_workers must be a non-negative integer")
        self.__dataloader_workers = dataloader_workers
        (
            self.__aggregation_device_policy,
            self.__aggregation_device,
        ) = resolve_aggregation_device(aggregation_device)
        self.__loaded_config_file = str(config_file)
        self.__source_config_file = os.environ.get(SOURCE_CONFIG_ENV_VAR, self.__loaded_config_file)
        self.settings = load_simulation_config(config_file)
        self.__planned_privacy_estimate = (
            estimate_privacy_from_config(self.settings)
            if self.settings.differential_privacy.enabled
            else None
        )
        self.__experiment_probe_config = load_experiment_probe_config(self.settings)
        config = self.settings.parser
        self.config = config
        self.supported_loss_function_dict = {'ce_loss': torch.nn.CrossEntropyLoss}
        self.supported_loss_function_dict.update({
            name: criterion for name, criterion in Criteria.items()
            if name.endswith("_loss")
        })
        self.supported_optimizer_dict = {
            'adam_cross_round': torch.optim.Adam,
            'adam_round': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        self.__early_stop_config = None
        self.__early_stop_controller = None
        self.__apply_config_settings()
        used_method_str = (
            resolve_method_abbreviation(self.settings.fl_method, self.settings.utils)
            + "+"
            + resolve_training_abbreviation(self.settings.training)
        )
        self.__used_method_log_abbreviation = used_method_str
        run_label = sanitize_path_part(os.environ.get(RUN_LABEL_ENV_VAR, ""))
        self.__run_label = run_label
        directory_run_label = compact_run_label(run_label) if run_label else ""
        run_label_str = f"{directory_run_label}_" if directory_run_label else ""

        full_log_dir_name = (
            f'{datetime.now().strftime("%y%m%d_%H%M%S")}_'
            f'{run_label_str}'
            f'{config_abbreviation[self.__model_name]}_'
            f'{config_abbreviation[self.__dataset_name]}_'
            f'{used_method_str}_'
            f'{config_abbreviation[self.__topology_shape]}_'
            f'device{self.__device_count}_'
            f'label{self.__labels_per_device}_'
            f'data{self.__training_data_per_device}'
        )
        self.__full_log_dir_name = full_log_dir_name
        canonical_config = yaml.safe_dump(self.config.source_data, sort_keys=True)
        source_config_filename = os.path.basename(self.__source_config_file)
        config_hash = path_component_hash(
            f"{source_config_filename}\0{run_label}\0{canonical_config}"
        )
        source_config_stem = os.path.splitext(source_config_filename)[0]
        safe_config_stem = summarize_run_config_stem(
            sanitize_path_part(source_config_stem) or "config"
        )
        optimizer_slug = {
            "adam_cross_round": "adamx",
            "adam_round": "adamr",
            "sgd": "sgd",
        }.get(self.__optimizer_name, self.__optimizer_name)
        batch_norm_slug = (
            "vbn" if self.__refresh_batch_norm_from_validation else "novbn"
        )
        privacy_slug = {
            "model_update": "dpupd",
            "local_dp_sgd": "dpsgd",
        }.get(
            self.__differential_privacy_config.mode,
            "nodp",
        ) if self.__differential_privacy_config.enabled else "nodp"
        concise_log_dir_name = (
            f'{datetime.now().strftime("%y%m%d_%H%M%S")}_'
            f'{run_label_str}{safe_config_stem}_'
            f'{self.__fl_method_name.lower()}_{batch_norm_slug}_{privacy_slug}_{optimizer_slug}_'
            f'h{config_hash}'
        )
        self.__log_dir_name = shorten_path_component(
            concise_log_dir_name,
            min(self.__log_run_dir_name_max_length, SAFE_RUN_DIR_NAME_MAX_BYTES),
        )
        self.__log_dir_name_shortened = self.__log_dir_name != concise_log_dir_name
        self.__log_dir_name_hash = config_hash

        # below parameters are used to store running parameters
        self.log_file_path = f'./log/{self.__log_dir_name}/'

        self.pre_data_path = None
        self.repli_exp = replicate_experiment
        if self.repli_exp:
            if pre_data_path is None:
                raise RuntimeError("Please enter the data path of previous data first")
            else:
                self.pre_data_path = pre_data_path

        self.__training_dataloader_dict = None
        self.__test_dataloader =None
        self.__valid_dataloader =None

        self.__model_dict = None
        self.__model_has_batchnorm_dict = None
        self.__loss_function_dict = None

        self.__optimizer_dict = None
        self.__lr_dict = None
        self.__total_rounds_dict = None
        self.__current_round_dict = None
        self.__topology_connectivity_dict = None
        self.__current_trainable_list = None
        self.__current_scores_dict = None
        self.__current_fisher_weights_dict = None
        self.__current_selection_lipschitz_score_weights_dict = None
        self.__current_parameter_score_weights_dict = None
        self.__current_block_parameter_score_weights_dict = None
        self.__current_block_interaction_score_weights_dict = None

        self.__lr_strategy = None
        self.__topology_manager_tool = None
        self.__stale_training_tool = None
        self.__gradient_buffer_tool = None
        self.__selection_lipschitz_score_ema_buffer = None
        self.__aggregation_lipschitz_score_ema_buffer = None
        self.__importance_correlation_probe = None
        self.__communication_recorder = None
        self.__differential_privacy_controller = None
        self.__privacy_summary_logged = False

        self.__train_acc_value_dict = None
        self.__train_recall_value_dict = None
        self.__train_precision_value_dict = None
        self.__train_loss_value_dict = None
        self.__train_batch_count_value_dict = None
        self.__test_acc_value_dict = None
        self.__test_recall_value_dict = None
        self.__test_precision_value_dict = None
        self.__test_loss_value_dict = None

        self.__fl_method_instance = None
        self.__metrics_recorder = SimulationMetricsRecorder()
        self.__object_detection_task = None
        self.__detection_metrics_tool = None
        self.__early_stop_unavailable_warned = False
        self.__wall_clock_rows = []
        self.__wall_clock_run_start_perf = None
        self.__wall_clock_run_started_at = None
        self.__wall_clock_run_finished_at = None
        self.__wall_clock_run_total_s = None

        self.__ready_to_simulate = False


    @staticmethod
    def setup_logging(log_path="logs/fl_simulation.log"):
        setup_round_logging(log_path)

    @staticmethod
    def __wall_clock_now() -> float:
        return synchronized_perf_counter(device)

    @contextmanager
    def __measure_wall_clock_stage(self, timings, stage_name):
        with measure_wall_clock_stage(timings, stage_name, device):
            yield

    @staticmethod
    def __model_has_batchnorm(model):
        batchnorm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
        )
        return any(isinstance(module, batchnorm_types) for module in model.modules())

    def __refresh_batchnorm_stats_if_needed(self, device_idx, dataloader, model=None):
        if not self.__model_has_batchnorm_dict or not self.__model_has_batchnorm_dict.get(device_idx, False):
            return

        model = self.__model_dict[device_idx] if model is None else model
        was_training = model.training
        model.train()
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device, non_blocking=True)
                _ = model(data)
        model.train(was_training)

    def __refresh_batchnorm_before_test_if_needed(self, device_idx):
        if (
                self.__uses_validation_weight()
                and self.__refresh_batch_norm_from_validation
        ):
            self.__refresh_batchnorm_stats_if_needed(
                device_idx,
                self.__valid_dataloader,
            )


    @staticmethod
    def __unique_parameter_score_methods(methods):
        unique_methods = []
        for method in methods:
            if method is None or method == DIRECT_PARAMETER_SCORE_METHOD:
                continue
            if method not in unique_methods:
                unique_methods.append(method)
        return unique_methods

    @staticmethod
    def __parameter_metric_or_none(metric):
        if metric is None:
            return None
        try:
            return canonical_parameter_score_method(metric)
        except ValueError:
            return None

    @staticmethod
    def __block_selection_score_metrics_for_config(fl_method):
        if (
                fl_method.segment_create_method != "importance"
                or fl_method.segment_unit not in BLOCK_SEGMENT_UNITS
        ):
            return []

        metrics = [fl_method.segment_importance_metric]
        if fl_method.group_enabled:
            metrics.append(fl_method.group_criterion_metric)
        return list(dict.fromkeys(metrics))

    @staticmethod
    def __parameter_score_record_methods(methods):
        record_methods = []
        for method in methods:
            record_methods.extend(parameter_score_record_dependencies(method))
        return FederatedLearningSim.__unique_parameter_score_methods(record_methods)

    @staticmethod
    def __block_parameter_score_methods_for_config(fl_method):
        methods = []
        for metric in FederatedLearningSim.__block_selection_score_metrics_for_config(fl_method):
            parameter_metric = FederatedLearningSim.__parameter_metric_or_none(metric)
            if parameter_metric is not None:
                methods.append(parameter_metric)
        return FederatedLearningSim.__unique_parameter_score_methods(methods)

    @staticmethod
    def __parameter_vector_score_methods_for_config(fl_method):
        if (
            fl_method.segment_create_method == "importance"
            and fl_method.segment_unit == "parameter"
        ):
            parameter_metric = FederatedLearningSim.__parameter_metric_or_none(
                fl_method.segment_importance_metric
            )
            return FederatedLearningSim.__unique_parameter_score_methods(
                [parameter_metric] if parameter_metric is not None else []
            )
        return []

    @staticmethod
    def __uses_block_interaction_scores(fl_method):
        metrics = set(FederatedLearningSim.__block_selection_score_metrics_for_config(fl_method))
        return (
                fl_method.segment_create_method == "importance"
                and fl_method.segment_unit in BLOCK_SEGMENT_UNITS
                and "fisher_lipschitz_cooperation" in metrics
        )

    def __apply_config_settings(self):
        training = self.settings.training
        dataset = self.settings.dataset
        model = self.settings.model
        fl_method = self.settings.fl_method
        differential_privacy = self.settings.differential_privacy
        utils = self.settings.utils
        experiment = self.settings.experiment

        self.__experiment_task = experiment.task

        self.__rounds = training.rounds
        self.__device_count = training.device_count
        self.__device_indicator_prefix = training.device_indicator_prefix
        self.__optimizer_name = training.optimizer_name
        self.__initial_lr = training.initial_lr
        self.__weight_decay = training.weight_decay
        self.__epoch_per_round = training.epoch_per_round
        self.__max_batch_per_epoch = training.max_batch_per_epoch
        self.__early_stop_config = training.early_stop
        self.__early_stop_controller = EarlyStopController(self.__early_stop_config)
        self.__loss_function_name = training.loss_function_name

        self.__dataset_name = dataset.dataset_name
        self.__training_data_per_device = dataset.training_data_per_device
        self.__labels_per_device = dataset.labels_per_device
        self.__label_allocating_method = dataset.label_allocating_method
        self.__label_allocating_loop_step = dataset.label_allocating_loop_step
        self.__data_allocating_method = dataset.data_allocating_method
        self.__data_allocating_alpha = dataset.data_allocating_alpha
        self.__test_data_size_total = dataset.test_data_size_total
        self.__valid_data_size_total = dataset.valid_data_size_total
        self.__train_batch_size = dataset.train_batch_size
        self.__test_batch_size = dataset.test_batch_size
        self.__valid_batch_size = dataset.valid_batch_size

        self.__model_name = model.model_name
        self.__input_size = model.input_size
        self.__torch_random_seed = model.torch_random_seed
        self.__output_class_number = model.output_class_number

        self.__fl_method_name = fl_method.fl_method_name
        self.__parameter_scope = fl_method.parameter_scope
        self.__segment_unit = fl_method.segment_unit
        importance_probe_config = self.__experiment_probe_config.importance_correlation
        probe_reset_methods = (
            ()
            if importance_probe_config is None
            else importance_probe_config.round_reset_parameter_score_methods
        )
        self.__parameter_score_reset_methods_each_round = tuple(dict.fromkeys(
            (*fl_method.round_reset_parameter_score_methods, *probe_reset_methods)
        ))
        self.__reset_block_interaction_each_round = fl_method.reset_block_interaction_each_round
        self.__hutchinson_z_time = fl_method.hutchinson_z_time
        self.__hutchinson_batch_limit = fl_method.hutchinson_batch_limit
        self.__selection_lipschitz_score_ema_enabled = (
            fl_method.selection_lipschitz_score_ema_enabled
            or fl_method.group_criterion_lipschitz_score_ema_enabled
        )
        self.__channel_length = fl_method.channel_length
        self.__block_refinement = fl_method.block_refinement
        self.__base_bn_mode = (
            fl_method.bn_mode if fl_method.bn_process_as_base_unit else "none"
        )
        self.__segment_importance_metric = fl_method.segment_importance_metric
        self.__importance_parameter_to_unit = fl_method.importance_parameter_to_unit
        self.__segment_importance_start_round = fl_method.segment_importance_start_round
        self.__aggregation_score_source = fl_method.aggregation_score_source
        self.__aggregation_weight_start_round = fl_method.aggregation_weight_start_round
        self.__refresh_batch_norm_from_validation = fl_method.refresh_batch_norm_from_validation
        self.__aggregation_lipschitz_score_ema_enabled = fl_method.aggregation_lipschitz_score_ema_enabled
        self.__block_selection_score_metrics_to_use = self.__block_selection_score_metrics_for_config(fl_method)
        self.__use_fisher = (
            fl_method.aggregation_score_source in FISHER_AGGREGATION_SCORE_SOURCES
        )
        self.__use_selection_lipschitz_score_ema = (
            self.__selection_lipschitz_score_ema_enabled
            and fl_method.segment_create_method == "importance"
            and fl_method.segment_unit in BLOCK_SEGMENT_UNITS
            and bool(
                set(self.__block_selection_score_metrics_to_use)
                & LIPSCHITZ_BLOCK_SCORE_METHODS
            )
        )
        self.__use_aggregation_lipschitz_score_ema = (
            self.__aggregation_lipschitz_score_ema_enabled
            and fl_method.aggregation_score_source == "fisher_lipschitz"
        )
        self.__parameter_vector_score_methods_to_use = self.__parameter_vector_score_methods_for_config(fl_method)
        self.__block_parameter_score_methods_to_use = self.__block_parameter_score_methods_for_config(fl_method)
        self.__use_block_interaction_score_buffer = self.__uses_block_interaction_scores(fl_method)
        self.__method_parameter_score_methods_to_record = self.__parameter_score_record_methods(
            self.__parameter_vector_score_methods_to_use
            + self.__block_parameter_score_methods_to_use
        )
        probe_score_methods = (
            []
            if importance_probe_config is None
            else list(importance_probe_config.required_internal_score_methods)
        )
        self.__parameter_score_methods_to_record = self.__parameter_score_record_methods(
            self.__method_parameter_score_methods_to_record + probe_score_methods
        )
        self.__use_parameter_score_buffer = len(self.__parameter_score_methods_to_record) > 0
        self.__fisher_cal = fl_method.fisher_cal
        self.__fisher_granularity = fl_method.fisher_granularity
        self.__fisher_block_reduce_method = fl_method.fisher_block_reduce_method
        self.__aggregation_weight_normalization = fl_method.aggregation_weight_normalization
        self.__fisher_reset_buffer_each_round = fl_method.fisher_reset_buffer_each_round
        self.__local_update_unit_l2_mode = fl_method.local_update_unit_l2_mode
        self.__local_update_l2_start_round = fl_method.local_update_l2_start_round
        self.__current_global_round = 0

        self.__differential_privacy_config = differential_privacy

        self.__use_adaptive_lr = utils.use_adaptive_lr
        self.__use_AdaLoss = utils.use_AdaLoss
        self.__use_AdaStair = utils.use_AdaStair
        self.__topology_shape = utils.topology_shape
        self.__topology_position_change = utils.topology_position_change
        self.__com_stability_mean = utils.com_stability_mean
        self.__com_stability_std = utils.com_stability_std
        self.__com_highest_stability = utils.com_highest_stability
        self.__com_lowest_stability = utils.com_lowest_stability
        self.__stale_sim_method = utils.stale_sim_method
        self.__stale_sim_distribution = utils.stale_sim_distribution
        self.__stale_gauss_mean = utils.stale_gauss_mean
        self.__stale_gauss_std = utils.stale_gauss_std
        self.__stale_chi_square_k = utils.stale_chi_square_k
        self.__stale_uniform_multiplier = utils.stale_uniform_multiplier
        self.__stale_highest_probability = utils.stale_highest_probability
        self.__stale_lowest_probability = utils.stale_lowest_probability
        self.__log_run_dir_name_max_length = utils.log_run_dir_name_max_length

    def __get_dataset_dict(self):
        log.info(f'Used dataset is {self.__dataset_name}')
        data_getter_object = DATASETS[self.__dataset_name]
        data_getter_instance = data_getter_object(
            num_devices=self.__device_count,
            label_per_device=self.__labels_per_device,
            data_per_device=self.__training_data_per_device,
            prefix_name=self.__device_indicator_prefix
        )
        if self.repli_exp:
            data_getter_instance.get_data_from_store(
                self.pre_data_path,
                self.__train_batch_size,
                self.__test_batch_size,
                self.__valid_batch_size,
                num_workers=self.__dataloader_workers,
            )
            self.__training_dataloader_dict = data_getter_instance.get_training_dataloader_dict()
            self.__test_dataloader = data_getter_instance.get_test_dataloader()
            self.__valid_dataloader = data_getter_instance.get_valid_dataloader()
        else:
            data_store_path = (
                f'./dataset/'
                f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_'
                f'{self.__dataset_name}_'
                f'device{self.__device_count}_'
                f'labelsPerDevice{self.__labels_per_device}_'
                f'trainDataPerDevice{self.__training_data_per_device}/'
            )
            data_getter_instance.get_data_from_new(
                self.__label_allocating_method, self.__label_allocating_loop_step,
                self.__data_allocating_method, self.__data_allocating_alpha,
                self.__test_data_size_total, self.__valid_data_size_total,
                data_store_path,
                self.__train_batch_size, self.__test_batch_size, self.__valid_batch_size,
                num_workers=self.__dataloader_workers,
            )
            self.__training_dataloader_dict = data_getter_instance.get_training_dataloader_dict()
            self.__test_dataloader = data_getter_instance.get_test_dataloader()
            self.__valid_dataloader = data_getter_instance.get_valid_dataloader()

            with open(data_store_path+'used_config.yml', 'w') as f:
                self.config.write(f)

        if self.__experiment_task == "Classification":
            dataset_class_count = data_getter_instance.get_total_classes()
            if self.__output_class_number != dataset_class_count:
                raise ValueError(
                    f"Classification dataset {self.__dataset_name} contains "
                    f"{dataset_class_count} classes, but model.num_classes is "
                    f"{self.__output_class_number}"
                )

    def __get_models_dict(self):
        log.info(f'Used model is {self.__model_name}')
        model_object = NETWORKS[self.__model_name]
        self.__model_dict = dict()
        for idx in range(self.__device_count):
            with isolated_model_initialization_rng():
                model = model_object(
                    self.__input_size,
                    device_num=idx,
                    random_seed=self.__torch_random_seed,
                    num_class=self.__output_class_number,
                )
            self.__model_dict[f'{self.__device_indicator_prefix}{idx}'] = model.to(device)
        self.__model_has_batchnorm_dict = {
            model_idx: self.__model_has_batchnorm(model)
            for model_idx, model in self.__model_dict.items()
        }
        first_model = next(iter(self.__model_dict.values()))
        first_parameter = next(first_model.parameters(), None)
        if first_parameter is not None:
            log.info("First model parameter device is [%s]", first_parameter.device)

    def __get_differential_privacy_controller(self):
        if not self.__differential_privacy_config.enabled:
            log.info("Differential privacy is disabled")
            return
        controller_type = {
            "model_update": ModelUpdateDPController,
            "local_dp_sgd": LocalDPSGDController,
        }[self.__differential_privacy_config.mode]
        controller_kwargs = {
            "config": self.__differential_privacy_config,
            "model_dict": self.__model_dict,
            "log_dir": self.log_file_path,
        }
        if self.__differential_privacy_config.mode == "local_dp_sgd":
            controller_kwargs["steps_per_epoch"] = self.__max_batch_per_epoch
        self.__differential_privacy_controller = controller_type(**controller_kwargs)
        estimate = self.__planned_privacy_estimate
        cost = estimate.privacy_cost
        log.info(
            "Differential privacy enabled\n"
            "  mode: %s\n"
            "  adjacency: %s\n"
            "  clipping_norm: %.10g\n"
            "  noise_multiplier: %.10g\n"
            "  expected_batch_size: %s\n"
            "  dataset_size_per_device: %s\n"
            "  sample_rate: %s\n"
            "  mechanisms_per_participation: %s\n"
            "  planned_participations_per_device: %s\n"
            "  planned_total_releases_or_steps: %s\n"
            "  delta: %.10g\n"
            "  planned_epsilon: %.10g\n"
            "  planned_optimal_alpha: %.10g\n"
            "  participation_assumption: configured_round_upper_bound",
            self.__differential_privacy_config.mode,
            estimate.adjacency,
            self.__differential_privacy_config.clipping_norm,
            self.__differential_privacy_config.noise_multiplier,
            estimate.expected_batch_size,
            estimate.dataset_size,
            (
                format(estimate.sample_rate, ".10g")
                if estimate.sample_rate is not None
                else None
            ),
            estimate.mechanisms_per_participation,
            estimate.participations,
            estimate.total_mechanisms,
            cost.delta,
            cost.epsilon,
            cost.optimal_alpha,
        )

    def __get_loss_function_dict(self):
        log.info(f'Used loss function is {self.__loss_function_name}')
        self.__loss_function_dict = dict()
        loss_function_object = self.supported_loss_function_dict[self.__loss_function_name]
        for idx in range(self.__device_count):
            loss_function = loss_function_object()
            if isinstance(loss_function, torch.nn.Module):
                loss_function = loss_function.to(device)
            self.__loss_function_dict[
                f'{self.__device_indicator_prefix}{idx}'
            ] = loss_function

    def __get_optimizer_dict(self):
        optimizer_object = self.supported_optimizer_dict[self.__optimizer_name]
        self.__optimizer_dict = dict()
        self.__lr_dict = dict()
        self.__total_rounds_dict = dict()
        self.__current_round_dict = dict()
        for device_idx, device_model in self.__model_dict.items():
            self.__optimizer_dict[device_idx] = optimizer_object(
                lr=self.__initial_lr,
                params=device_model.parameters(),
                weight_decay=self.__weight_decay,
            )
            self.__lr_dict[device_idx] = self.__initial_lr
            self.__total_rounds_dict[device_idx] = self.__rounds
            self.__current_round_dict[device_idx] = 0

    def __get_acc_recall_dict(self):
        self.__train_acc_value_dict = dict()
        self.__train_recall_value_dict = dict()
        self.__train_precision_value_dict = dict()
        self.__train_loss_value_dict = dict()
        self.__train_batch_count_value_dict = dict()
        self.__test_acc_value_dict = dict()
        self.__test_recall_value_dict = dict()
        self.__test_precision_value_dict = dict()
        self.__test_loss_value_dict = dict()
        self.__current_scores_dict = dict()

        for device_idx in self.__model_dict.keys():
            self.__train_acc_value_dict[device_idx] = -1
            self.__train_recall_value_dict[device_idx] = -1
            self.__train_precision_value_dict[device_idx] = -1
            self.__train_loss_value_dict[device_idx] = -1
            self.__train_batch_count_value_dict[device_idx] = -1
            self.__test_acc_value_dict[device_idx] = -1
            self.__test_recall_value_dict[device_idx] = -1
            self.__test_precision_value_dict[device_idx] = -1
            self.__test_loss_value_dict[device_idx] = -1
            self.__current_scores_dict[device_idx] = 100

    def __reset_training_metrics(self):
        for device_idx in self.__model_dict.keys():
            self.__train_acc_value_dict[device_idx] = -1
            self.__train_recall_value_dict[device_idx] = -1
            self.__train_precision_value_dict[device_idx] = -1
            self.__train_loss_value_dict[device_idx] = -1
            self.__train_batch_count_value_dict[device_idx] = -1

    def __reset_test_metrics(self):
        for device_idx in self.__model_dict.keys():
            self.__test_acc_value_dict[device_idx] = -1
            self.__test_recall_value_dict[device_idx] = -1
            self.__test_precision_value_dict[device_idx] = -1
            self.__test_loss_value_dict[device_idx] = -1
            self.__current_scores_dict[device_idx] = 100

    def __get_fl_method(self):
        log.info(
            "Used fl method is %s; aggregation device policy=%s, resolved device=%s",
            self.__fl_method_name,
            self.__aggregation_device_policy,
            self.__aggregation_device,
        )
        self.__fl_method_instance = build_fl_method(
            method_config=self.settings.fl_method,
            total_models_dict=self.__model_dict,
            connectivity_dict=self.__topology_connectivity_dict,
        )
        self.__fl_method_instance.set_working_device(self.__aggregation_device)
        self.__fl_method_instance.set_communication_recorder(self.__communication_recorder)

    def __get_communication_recorder(self):
        bn_distribution = {
            "base_unit": "as_base_unit",
            "separate_per_segment": "with_each_segment",
            "separate_per_recipient": "once_per_recipient",
        }[self.settings.fl_method.bn_process_mode]
        self.__communication_recorder = CommunicationRecorder(
            self.log_file_path + "communication_packets.csv",
            method=self.__fl_method_name,
            aggregation_weight_metric=self.settings.fl_method.aggregation_weight_name,
            parameter_scope=self.__parameter_scope,
            segment_unit=self.__segment_unit,
            bn_distribution=bn_distribution,
        )

    def __get_gradient_buffer_tool(self):
        importance_probe_config = self.__experiment_probe_config.importance_correlation
        if (
                self.__use_fisher
                or self.__use_parameter_score_buffer
                or self.__use_block_interaction_score_buffer
        ):
            reasons = []
            if self.__use_fisher:
                reasons.append("method")
            if importance_probe_config is not None:
                reasons.append("importance_correlation_probe")
            if self.__use_parameter_score_buffer:
                reasons.append("parameter_score")
            if self.__use_block_interaction_score_buffer:
                reasons.append("block_interaction_score")
            log.info(
                "Gradient/score buffer enabled for %s: fisher_calculation=%s, fisher_granularity=%s, "
                "block_reduce_method=%s, aggregation_weight=%s, aggregation_normalization=%s, "
                "segment_importance_metric=%s, "
                "parameter_to_unit=%s, recorded_parameter_score_methods=%s, "
                "hutchinson_z_time=%s, hutchinson_batch_limit=%s, bn_mode=%s",
                ",".join(reasons),
                self.__fisher_cal,
                self.__fisher_granularity,
                self.__fisher_block_reduce_method,
                self.__aggregation_score_source,
                self.__aggregation_weight_normalization,
                self.__segment_importance_metric,
                self.__importance_parameter_to_unit,
                self.__parameter_score_methods_to_record,
                self.__hutchinson_z_time,
                self.__hutchinson_batch_limit,
                self.__base_bn_mode,
            )
            self.__gradient_buffer_tool = GradientBuffer(self.__model_dict)

    def __records_step_fisher_buffer(self):
        return self.__use_fisher

    def __get_selection_lipschitz_score_ema_buffer(self):
        if not self.__selection_lipschitz_score_ema_enabled:
            return
        if not self.__use_selection_lipschitz_score_ema:
            log.info(
                "Selection Lipschitz-score EMA requested but inactive for seg_create_method=%s, "
                "partition.unit=%s and segment_importance.metric=%s",
                self.settings.fl_method.segment_create_method,
                self.__segment_unit,
                self.__segment_importance_metric,
            )
            return
        log.info(
            "Selection Lipschitz-score EMA enabled: segment_unit=%s, channel_length=%s, bn_mode=%s",
            self.__segment_unit,
            self.__channel_length,
            self.__base_bn_mode,
        )
        self.__selection_lipschitz_score_ema_buffer = BlockScoreEmaBuffer(self.__model_dict)

    def __get_aggregation_lipschitz_score_ema_buffer(self):
        if not self.__aggregation_lipschitz_score_ema_enabled:
            return
        if not self.__use_aggregation_lipschitz_score_ema:
            log.info(
                "Aggregation Lipschitz-score EMA requested but inactive for aggregation.score.metric=%s",
                self.__aggregation_score_source,
            )
            return
        log.info(
            "Aggregation Lipschitz-score EMA enabled: segment_unit=%s, channel_length=%s, bn_mode=%s",
            self.__segment_unit,
            self.__channel_length,
            self.__base_bn_mode,
        )
        self.__aggregation_lipschitz_score_ema_buffer = BlockScoreEmaBuffer(self.__model_dict)

    def __get_experiment_probes(self):
        importance_probe_config = self.__experiment_probe_config.importance_correlation
        if importance_probe_config is None:
            return
        self.__importance_correlation_probe = ImportanceCorrelationProbe(
            importance_probe_config,
            self.log_file_path,
        )
        log.info(
            "Importance correlation probe enabled: output=%s, stage=%s, baseline=%s, "
            "reference_scores=%s, measurements=%s, top_k=%s, evaluation_rounds=%s",
            self.__importance_correlation_probe.output_path,
            importance_probe_config.stage,
            importance_probe_config.comparison_baseline.name,
            [metric.name for metric in importance_probe_config.reference_scores],
            importance_probe_config.measurements,
            importance_probe_config.top_k,
            importance_probe_config.evaluation_rounds or "all",
        )

    def __record_gradient_if_needed(self, device_idx):
        if self.__gradient_buffer_tool is None:
            return
        self.__gradient_buffer_tool.update(
            [device_idx],
            fisher_cal=self.__fisher_cal,
            parameter_score_methods=self.__parameter_score_methods_to_record,
            block_interaction_score_configs=self.__block_interaction_score_configs(),
            record_fisher=self.__records_step_fisher_buffer(),
        )

    def __backward_and_record_scores(self, device_idx, loss):
        if (
                self.__use_parameter_score_buffer
                and HESSIAN_EMA_PARAMETER_SCORE_METHODS.intersection(self.__parameter_score_methods_to_record)
        ):
            if self.__gradient_buffer_tool is None:
                raise RuntimeError("Hessian EMA scores require gradient_buffer_tool")
            self.__gradient_buffer_tool.update_from_loss(
                device_idx,
                loss,
                fisher_cal=self.__fisher_cal,
                parameter_score_methods=self.__parameter_score_methods_to_record,
                block_interaction_score_configs=self.__block_interaction_score_configs(),
                record_fisher=self.__records_step_fisher_buffer(),
            )
            return
        loss.backward()
        self.__record_gradient_if_needed(device_idx)


    @staticmethod
    def __accumulate_score_sums(target, current):
        for name, score_value in current.items():
            if isinstance(score_value, dict):
                nested_target = target.setdefault(name, {})
                FederatedLearningSim.__accumulate_score_sums(
                    nested_target,
                    score_value,
                )
            elif name not in target:
                target[name] = score_value.detach().clone()
            else:
                target[name].add_(
                    score_value.detach().to(
                        device=target[name].device,
                        dtype=target[name].dtype,
                    )
                )

    def __compute_local_train_sample_score_sums_for_device(
            self,
            device_idx,
            batch_score_sum_fn,
            max_batches=0,
    ):
        model = self.__model_dict[device_idx]
        train_dataloader = self.__training_dataloader_dict[device_idx]
        optimizer = self.__optimizer_dict[device_idx]
        loss_function = self.__loss_function_dict[device_idx]
        was_training = model.training
        loss_was_training = getattr(loss_function, "training", None)
        score_sums = {}
        sample_count = 0

        model.eval()
        if hasattr(loss_function, "eval"):
            loss_function.eval()
        optimizer.zero_grad(set_to_none=True)
        try:
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                batch_score_sums, batch_sample_count = batch_score_sum_fn(
                    data,
                    labels,
                    loss_function,
                )
                self.__accumulate_score_sums(score_sums, batch_score_sums)
                sample_count += batch_sample_count
        finally:
            model.train(was_training)
            if loss_was_training is not None and hasattr(loss_function, "train"):
                loss_function.train(loss_was_training)
            optimizer.zero_grad(set_to_none=True)

        return score_sums, sample_count

    def __compute_full_local_train_loss_score_for_device(
            self,
            device_idx,
            score_fn,
    ):
        model = self.__model_dict[device_idx]
        train_dataloader = self.__training_dataloader_dict[device_idx]
        optimizer = self.__optimizer_dict[device_idx]
        loss_function = self.__loss_function_dict[device_idx]
        was_training = model.training
        loss_was_training = getattr(loss_function, "training", None)
        loss_sum = None
        sample_count = 0

        model.eval()
        if hasattr(loss_function, "eval"):
            loss_function.eval()
        optimizer.zero_grad(set_to_none=True)
        try:
            for data, labels in train_dataloader:
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(data)
                loss = loss_function(outputs, labels)
                if loss.ndim != 0:
                    loss = loss.mean()
                batch_sample_count = int(data.shape[0])
                if batch_sample_count == 0:
                    continue
                weighted_loss = loss * batch_sample_count
                loss_sum = weighted_loss if loss_sum is None else loss_sum + weighted_loss
                sample_count += batch_sample_count

            if loss_sum is None or sample_count == 0:
                return {}, 0

            full_dataset_loss = loss_sum / sample_count
            score_tensors = score_fn(full_dataset_loss) or {}
            return score_tensors, sample_count
        finally:
            model.train(was_training)
            if loss_was_training is not None and hasattr(loss_function, "train"):
                loss_function.train(loss_was_training)
            optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def __average_score_sums(score_sums, denominator):
        return {
            name: score_sum / denominator
            for name, score_sum in score_sums.items()
        }

    def __compute_post_training_hessian_scores_if_needed(self, round_idx, active_score_methods):
        if HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD not in active_score_methods:
            return
        if self.__gradient_buffer_tool is None:
            raise RuntimeError("hessian_taylor_exact.round requires the gradient/score buffer")

        log.info("Computing post-training Hessian parameter scores for round %s from the full local train loss", round_idx)
        for device_idx in self.__current_trainable_list:
            self.__compute_post_training_hessian_score_for_device(device_idx)

    def __compute_post_training_hessian_score_for_device(self, device_idx):
        hessian_scores, sample_count = self.__compute_full_local_train_loss_score_for_device(
            device_idx,
            lambda loss: self.__gradient_buffer_tool.compute_hessian_score_tensors_from_loss(device_idx, loss),
        )

        if sample_count == 0:
            self.__gradient_buffer_tool.clear_parameter_scores([HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD], [device_idx])
            log.warning("Skip post-training Hessian score for %s because no local train sample was available", device_idx)
            return

        weighted_scores = self.__gradient_buffer_tool.weight_score_tensors_by_parameter_square(
            device_idx,
            hessian_scores,
        )
        self.__gradient_buffer_tool.set_parameter_score_tensors(
            device_idx,
            HESSIAN_TAYLOR_SECOND_EXACT_POST_METHOD,
            weighted_scores,
        )

    @staticmethod
    def __post_training_gradient_fisher_score_methods(active_score_methods):
        gradient_methods = [
            method
            for method in (
                GRADIENT_ABS_POST_METHOD,
                GRADIENT_WEIGHT_ABS_POST_METHOD,
            )
            if method in active_score_methods
        ]
        fisher_methods = [
            method
            for method in (
                FISHER_EMPIRICAL_DIAGONAL_POST_METHOD,
                FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD,
            )
            if method in active_score_methods
        ]
        return gradient_methods, fisher_methods

    def __compute_post_training_gradient_fisher_scores_if_needed(
            self,
            round_idx,
            active_score_methods,
    ):
        (
            gradient_score_methods,
            empirical_fisher_score_methods,
        ) = self.__post_training_gradient_fisher_score_methods(
            active_score_methods
        )
        score_methods = gradient_score_methods + empirical_fisher_score_methods
        if not score_methods:
            return
        if self.__gradient_buffer_tool is None:
            raise RuntimeError(
                "Post-training gradient/Fisher importance requires the "
                "gradient/score buffer"
            )

        log.info(
            "Computing post-training gradient and empirical Fisher scores for "
            "round %s in one per-sample gradient pass using methods %s",
            round_idx,
            score_methods,
        )
        for device_idx in self.__current_trainable_list:
            self.__compute_post_training_gradient_fisher_scores_for_device(
                device_idx,
                gradient_score_methods,
                empirical_fisher_score_methods,
            )

    def __compute_post_training_gradient_fisher_scores_for_device(
            self,
            device_idx,
            gradient_score_methods,
            empirical_fisher_score_methods,
    ):
        score_sums, sample_count = (
            self.__compute_local_train_sample_score_sums_for_device(
                device_idx,
                lambda data, labels, loss_function:
                    self.__gradient_buffer_tool.compute_per_sample_gradient_abs_and_square_score_sums(
                        device_idx,
                        data,
                        labels,
                        loss_function,
                    ),
            )
        )
        score_methods = gradient_score_methods + empirical_fisher_score_methods
        if sample_count == 0:
            self.__gradient_buffer_tool.clear_parameter_scores(
                score_methods,
                [device_idx],
            )
            log.warning(
                "Skip post-training gradient/Fisher scores for %s because no "
                "local train sample was available",
                device_idx,
            )
            return

        if gradient_score_methods:
            averaged_gradient_scores = self.__average_score_sums(
                score_sums["abs"],
                sample_count,
            )
            if GRADIENT_ABS_POST_METHOD in gradient_score_methods:
                self.__gradient_buffer_tool.set_parameter_score_tensors(
                    device_idx,
                    GRADIENT_ABS_POST_METHOD,
                    averaged_gradient_scores,
                )
            if GRADIENT_WEIGHT_ABS_POST_METHOD in gradient_score_methods:
                weighted_scores = (
                    self.__gradient_buffer_tool.weight_score_tensors_by_parameter_abs(
                        device_idx,
                        averaged_gradient_scores,
                    )
                )
                self.__gradient_buffer_tool.set_parameter_score_tensors(
                    device_idx,
                    GRADIENT_WEIGHT_ABS_POST_METHOD,
                    weighted_scores,
                )

        if empirical_fisher_score_methods:
            averaged_fisher_scores = self.__average_score_sums(
                score_sums["square"],
                sample_count,
            )
            if FISHER_EMPIRICAL_DIAGONAL_POST_METHOD in empirical_fisher_score_methods:
                self.__gradient_buffer_tool.set_parameter_score_tensors(
                    device_idx,
                    FISHER_EMPIRICAL_DIAGONAL_POST_METHOD,
                    averaged_fisher_scores,
                )
            if (
                    FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD
                    in empirical_fisher_score_methods
            ):
                weighted_scores = (
                    self.__gradient_buffer_tool.weight_score_tensors_by_parameter_square(
                        device_idx,
                        averaged_fisher_scores,
                    )
                )
                self.__gradient_buffer_tool.set_parameter_score_tensors(
                    device_idx,
                    FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD,
                    weighted_scores,
                )

    def __compute_post_training_gradient_signal_preservation_scores_if_needed(
            self,
            round_idx,
            active_score_methods,
    ):
        if GRADIENT_SIGNAL_PRESERVATION_POST_METHOD not in active_score_methods:
            return
        if self.__gradient_buffer_tool is None:
            raise RuntimeError("gradient_signal_preservation.round requires the gradient/score buffer")

        log.info(
            "Computing post-training gradient-signal-preservation scores for round %s with batch HVPs over the full local train dataset",
            round_idx,
        )
        for device_idx in self.__current_trainable_list:
            self.__compute_post_training_gradient_signal_preservation_score_for_device(device_idx)

    def __compute_post_training_gradient_signal_preservation_score_for_device(self, device_idx):
        gradient_sums, gradient_sample_count = self.__compute_local_train_sample_score_sums_for_device(
            device_idx,
            lambda data, labels, loss_function: self.__gradient_buffer_tool.compute_gradient_sum_tensors(
                device_idx,
                data,
                labels,
                loss_function,
            ),
        )

        if gradient_sample_count == 0:
            self.__gradient_buffer_tool.clear_parameter_scores(
                [GRADIENT_SIGNAL_PRESERVATION_POST_METHOD],
                [device_idx],
            )
            log.warning(
                "Skip gradient-signal-preservation score for %s because no local train sample was available",
                device_idx,
            )
            return

        full_gradient = self.__average_score_sums(gradient_sums, gradient_sample_count)
        signal_score_sums, signal_sample_count = self.__compute_local_train_sample_score_sums_for_device(
            device_idx,
            lambda data, labels, loss_function: self.__gradient_buffer_tool.compute_gradient_signal_preservation_score_sums(
                device_idx,
                data,
                labels,
                loss_function,
                signal_tensors=full_gradient,
            ),
        )

        if signal_sample_count == 0:
            self.__gradient_buffer_tool.clear_parameter_scores(
                [GRADIENT_SIGNAL_PRESERVATION_POST_METHOD],
                [device_idx],
            )
            log.warning(
                "Skip gradient-signal-preservation score for %s because no local train sample was available",
                device_idx,
            )
            return

        signal_scores = {
            name: (score_sum / signal_sample_count).abs()
            for name, score_sum in signal_score_sums.items()
        }
        weighted_scores = self.__gradient_buffer_tool.weight_score_tensors_by_parameter_abs(
            device_idx,
            signal_scores,
        )
        self.__gradient_buffer_tool.set_parameter_score_tensors(
            device_idx,
            GRADIENT_SIGNAL_PRESERVATION_POST_METHOD,
            weighted_scores,
        )

    @staticmethod
    def __post_training_hutchinson_score_methods(active_score_methods):
        return [
            method
            for method in (HUTCHINSON_DIAGONAL_POST_METHOD, HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD)
            if method in active_score_methods
        ]

    def __compute_post_training_hutchinson_scores_if_needed(
            self,
            round_idx,
            active_score_methods,
            hutchinson_z_time,
            hutchinson_batch_limit,
    ):
        hutchinson_score_methods = self.__post_training_hutchinson_score_methods(
            active_score_methods
        )
        if not hutchinson_score_methods:
            return
        if self.__gradient_buffer_tool is None:
            raise RuntimeError(
                "Hutchinson segment importance requires the gradient/score buffer"
            )

        log.info(
            "Computing post-training Hutchinson scores for round %s using methods %s, %s probes, and batch_limit=%s",
            round_idx,
            hutchinson_score_methods,
            hutchinson_z_time,
            hutchinson_batch_limit or "all",
        )
        for device_idx in self.__current_trainable_list:
            self.__compute_post_training_hutchinson_score_for_device(
                device_idx,
                hutchinson_score_methods,
                hutchinson_z_time,
                hutchinson_batch_limit,
            )

    def __compute_post_training_hutchinson_score_for_device(
            self,
            device_idx,
            hutchinson_score_methods,
            hutchinson_z_time,
            hutchinson_batch_limit,
    ):
        hutchinson_score_sums, sample_count = self.__compute_local_train_sample_score_sums_for_device(
            device_idx,
            lambda data, labels, loss_function: self.__gradient_buffer_tool.compute_hutchinson_diagonal_score_sums(
                device_idx,
                data,
                labels,
                loss_function,
                hutchinson_z_time=hutchinson_z_time,
            ),
            max_batches=hutchinson_batch_limit,
        )

        if sample_count == 0:
            self.__gradient_buffer_tool.clear_parameter_scores(hutchinson_score_methods, [device_idx])
            log.warning("Skip Hutchinson scores for %s because no local train sample was available", device_idx)
            return

        hutchinson_scores = {
            name: (score_sum / sample_count).abs()
            for name, score_sum in hutchinson_score_sums.items()
        }
        if HUTCHINSON_DIAGONAL_POST_METHOD in hutchinson_score_methods:
            self.__gradient_buffer_tool.set_parameter_score_tensors(
                device_idx,
                HUTCHINSON_DIAGONAL_POST_METHOD,
                hutchinson_scores,
            )
        if HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD in hutchinson_score_methods:
            weighted_scores = self.__gradient_buffer_tool.weight_score_tensors_by_parameter_square(
                device_idx,
                hutchinson_scores,
            )
            self.__gradient_buffer_tool.set_parameter_score_tensors(
                device_idx,
                HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD,
                weighted_scores,
            )

    def __block_interaction_score_configs(self):
        if not self.__use_block_interaction_score_buffer:
            return []
        include_other_blocks = (
            False
            if self.__fl_method_instance is None
            else getattr(self.__fl_method_instance, "include_other_blocks", False)
        )
        return [{
            "parameter_scope": self.__parameter_scope,
            "conv_mode": self.__segment_unit,
            "channel_length": self.__channel_length,
            "include_other_blocks": include_other_blocks,
            "bn_mode": self.__base_bn_mode,
            "block_refinement": self.__block_refinement,
        }]

    def __early_stop_controller_instance(self):
        if self.__early_stop_controller is None:
            if self.__early_stop_config is None:
                raise RuntimeError("training.local.early_stop config is not initialized.")
            self.__early_stop_controller = EarlyStopController(self.__early_stop_config)
        return self.__early_stop_controller

    def __early_stop_state(self, device_idx):
        return self.__early_stop_controller_instance().create_state(device_idx)

    def __update_early_stop(self, device_idx, state, completed_epoch_count, metrics):
        return self.__early_stop_controller_instance().update(
            device_idx,
            state,
            completed_epoch_count,
            metrics,
        )

    def __warn_early_stop_unavailable_once(self, reason):
        if self.__early_stop_config is None or not self.__early_stop_config.enabled:
            return
        if self.__early_stop_unavailable_warned:
            return
        log.warning("training.early_stop is ignored: %s", reason)
        self.__early_stop_unavailable_warned = True

    def __reset_fisher_buffer_if_needed(self, round_idx):
        if not self.__fisher_reset_buffer_each_round:
            return
        if self.__gradient_buffer_tool is None:
            return
        self.__gradient_buffer_tool.clear()
        log.info("Reset Fisher gradient buffer before round %s training", round_idx)

    def __reset_parameter_score_buffer_if_needed(self, round_idx):
        parameter_score_methods_to_reset = self.__parameter_score_reset_methods_each_round
        reset_block_interaction_scores = (
            self.__reset_block_interaction_each_round
            and self.__use_block_interaction_score_buffer
        )
        if not parameter_score_methods_to_reset and not reset_block_interaction_scores:
            return
        if self.__gradient_buffer_tool is None:
            return
        if parameter_score_methods_to_reset:
            self.__gradient_buffer_tool.clear_parameter_scores(parameter_score_methods_to_reset)
        if reset_block_interaction_scores:
            self.__gradient_buffer_tool.clear_block_interaction_scores()
        log.info(
            "Reset stage score buffer before round %s training: stage_methods=%s, "
            "block_interaction_score=%s",
            round_idx,
            parameter_score_methods_to_reset,
            reset_block_interaction_scores,
        )

    def __build_fisher_weights_if_needed(self, aggregation_score_source=None):
        self.__current_fisher_weights_dict = None
        effective_score_source = (
            self.__aggregation_score_source
            if aggregation_score_source is None
            else aggregation_score_source
        )
        if effective_score_source not in FISHER_AGGREGATION_SCORE_SOURCES:
            return None
        if self.__gradient_buffer_tool is None:
            raise RuntimeError("Fisher-based aggregation or block scoring requires gradient_buffer_tool")
        include_other_blocks = getattr(self.__fl_method_instance, "include_other_blocks", False)

        if effective_score_source == "fisher":
            if self.__segment_unit in BLOCK_SEGMENT_UNITS:
                parameter_fisher = {
                    model_idx: self.__gradient_buffer_tool.get_parameter_fisher_vector(
                        model_idx,
                        parameter_scope=self.__parameter_scope,
                        bn_mode=self.__base_bn_mode,
                        target_device=self.__aggregation_device,
                    )
                    for model_idx in self.__model_dict
                }
                if self.__fisher_granularity == "block":
                    self.__current_fisher_weights_dict = {
                        model_idx: self.__fl_method_instance.reduce_parameter_vector_to_block_scores(
                            fisher_vector,
                            self.__fisher_block_reduce_method,
                        )
                        for model_idx, fisher_vector in parameter_fisher.items()
                    }
                else:
                    self.__current_fisher_weights_dict = parameter_fisher
            else:
                fisher_payload_unit = (
                    "parameter"
                    if self.__fl_method_name == "Centralized"
                    else self.__segment_unit
                )
                self.__current_fisher_weights_dict = self.__gradient_buffer_tool.get_fisher_weights(
                    parameter_scope=self.__parameter_scope,
                    segment_unit=fisher_payload_unit,
                    channel_length=self.__channel_length,
                    fisher_granularity=self.__fisher_granularity,
                    fisher_block_reduce_method=self.__fisher_block_reduce_method,
                    include_other_blocks=include_other_blocks,
                    bn_mode=self.__base_bn_mode,
                    block_refinement=self.__block_refinement,
                    target_device=self.__aggregation_device,
                )
        elif effective_score_source == "fisher_lipschitz":
            self.__current_fisher_weights_dict = self.__build_fisher_lipschitz_weights(include_other_blocks)
        if self.__aggregation_weight_normalization == "device_layer_l2":
            model_indices = tuple(self.__current_fisher_weights_dict)
            normalized_weights = self.__fl_method_instance.normalize_block_scores_by_layer_l2(
                torch.stack([
                    self.__current_fisher_weights_dict[model_idx]
                    for model_idx in model_indices
                ])
            )
            self.__current_fisher_weights_dict = dict(
                zip(model_indices, normalized_weights.unbind(0))
            )
        return self.__current_fisher_weights_dict

    def __build_fisher_lipschitz_weights(self, include_other_blocks):
        if self.__segment_unit not in BLOCK_SEGMENT_UNITS:
            raise ValueError("Fisher-Lipschitz aggregation requires block partitioning")

        fisher_weights_dict = self.__gradient_buffer_tool.get_fisher_weights(
            parameter_scope=self.__parameter_scope,
            segment_unit=self.__segment_unit,
            channel_length=self.__channel_length,
            fisher_granularity="block",
            fisher_block_reduce_method=self.__fisher_block_reduce_method,
            include_other_blocks=include_other_blocks,
            bn_mode=self.__base_bn_mode,
            block_refinement=self.__block_refinement,
            target_device=self.__aggregation_device,
        )
        lipschitz_weights_dict = self.__block_lipschitz_weights_dict(include_other_blocks)
        fisher_lipschitz_weights_dict = {}
        for model_idx, fisher_weights in fisher_weights_dict.items():
            lipschitz_weights = lipschitz_weights_dict[model_idx]
            assert fisher_weights.numel() == lipschitz_weights.numel(), \
                "Fisher-Lipschitz block score length does not match block layout"
            fisher_weights = fisher_weights.detach().to(dtype=torch.float32)
            fisher_lipschitz_weights_dict[model_idx] = (
                fisher_weights
                * lipschitz_weights.detach().to(device=fisher_weights.device, dtype=torch.float32)
            )
        return fisher_lipschitz_weights_dict

    def __block_lipschitz_weights_dict(self, include_other_blocks):
        if self.__use_aggregation_lipschitz_score_ema:
            if self.__aggregation_lipschitz_score_ema_buffer is None:
                raise RuntimeError("Aggregation Lipschitz-score EMA requires aggregation_lipschitz_score_ema_buffer")
            return self.__aggregation_lipschitz_score_ema_buffer.get_block_scores(
                target_device=self.__aggregation_device,
            )

        block_score_dict = {}
        for model_idx, model in self.__model_dict.items():
            if not hasattr(model, "get_parameter_blocks"):
                raise ValueError(
                    "Fisher-Lipschitz aggregation requires models to implement get_parameter_blocks(...)."
                )
            blocks = self.__fl_method_instance.get_parameter_blocks_for_working_device(model)
            block_score_dict[model_idx] = torch.stack([
                compute_parameter_block_score_tensor(block, "lipschitz")
                for block in blocks
            ])
        return block_score_dict

    def __update_aggregation_lipschitz_score_ema_if_needed(self):
        if not self.__use_aggregation_lipschitz_score_ema:
            return None
        if self.__aggregation_lipschitz_score_ema_buffer is None:
            raise RuntimeError("Aggregation Lipschitz-score EMA requires aggregation_lipschitz_score_ema_buffer")

        include_other_blocks = getattr(self.__fl_method_instance, "include_other_blocks", False)
        return self.__aggregation_lipschitz_score_ema_buffer.update(
            parameter_scope=self.__parameter_scope,
            conv_mode=self.__segment_unit,
            channel_length=self.__channel_length,
            block_score_method="lipschitz",
            include_other_blocks=include_other_blocks,
            bn_mode=self.__base_bn_mode,
            block_refinement=self.__block_refinement,
            target_device=self.__aggregation_device,
        )

    def __build_selection_lipschitz_scores_if_needed(self):
        self.__current_selection_lipschitz_score_weights_dict = None
        if not self.__use_selection_lipschitz_score_ema:
            return None
        if self.__selection_lipschitz_score_ema_buffer is None:
            raise RuntimeError("Selection Lipschitz-score EMA requires selection_lipschitz_score_ema_buffer")

        include_other_blocks = getattr(self.__fl_method_instance, "include_other_blocks", False)
        self.__current_selection_lipschitz_score_weights_dict = self.__selection_lipschitz_score_ema_buffer.update(
            parameter_scope=self.__parameter_scope,
            conv_mode=self.__segment_unit,
            channel_length=self.__channel_length,
            block_score_method="lipschitz",
            include_other_blocks=include_other_blocks,
            bn_mode=self.__base_bn_mode,
            block_refinement=self.__block_refinement,
            target_device=self.__aggregation_device,
        )
        return self.__current_selection_lipschitz_score_weights_dict

    def __build_parameter_scores_if_needed(self):
        self.__current_parameter_score_weights_dict = None
        if not self.__parameter_vector_score_methods_to_use:
            return None
        if self.__gradient_buffer_tool is None:
            raise RuntimeError("Buffered parameter scoring requires gradient_buffer_tool")

        self.__current_parameter_score_weights_dict = self.__gradient_buffer_tool.get_parameter_score_weights(
            self.__parameter_vector_score_methods_to_use[0],
            parameter_scope=self.__parameter_scope,
            bn_mode=self.__base_bn_mode,
            target_device=self.__aggregation_device,
        )
        return self.__current_parameter_score_weights_dict

    def __build_block_parameter_scores_if_needed(self):
        self.__current_block_parameter_score_weights_dict = None
        if not self.__block_parameter_score_methods_to_use:
            return None
        if self.__segment_unit not in BLOCK_SEGMENT_UNITS:
            return None
        if self.__gradient_buffer_tool is None:
            raise RuntimeError("Buffered block parameter scoring requires gradient_buffer_tool")

        self.__current_block_parameter_score_weights_dict = {
            parameter_score_method: self.__gradient_buffer_tool.get_parameter_score_weights(
                parameter_score_method,
                parameter_scope=self.__parameter_scope,
                bn_mode=self.__base_bn_mode,
                target_device=self.__aggregation_device,
            )
            for parameter_score_method in self.__block_parameter_score_methods_to_use
        }
        return self.__current_block_parameter_score_weights_dict

    def __build_block_interaction_scores_if_needed(self):
        self.__current_block_interaction_score_weights_dict = None
        if not self.__use_block_interaction_score_buffer:
            return None
        if self.__gradient_buffer_tool is None:
            raise RuntimeError("Block interaction scoring requires gradient_buffer_tool")

        include_other_blocks = getattr(self.__fl_method_instance, "include_other_blocks", False)
        self.__current_block_interaction_score_weights_dict = (
            self.__gradient_buffer_tool.get_block_interaction_scores_dict(
                parameter_scope=self.__parameter_scope,
                conv_mode=self.__segment_unit,
                channel_length=self.__channel_length,
                include_other_blocks=include_other_blocks,
                bn_mode=self.__base_bn_mode,
                block_refinement=self.__block_refinement,
            )
        )
        return self.__current_block_interaction_score_weights_dict

    @staticmethod
    def __schedule_active(round_idx, start_round):
        return round_idx >= start_round

    def __effective_aggregation_score_source(self, round_idx):
        if self.__schedule_active(round_idx, self.__aggregation_weight_start_round):
            return self.__aggregation_score_source
        return "uniform"

    def __segment_importance_active(self, round_idx):
        return self.__schedule_active(
            round_idx,
            self.__segment_importance_start_round,
        )

    def __local_update_l2_active(self, round_idx):
        return (
            self.__local_update_unit_l2_mode != "none"
            and self.__schedule_active(round_idx, self.__local_update_l2_start_round)
        )

    def __uses_validation_weight(self, round_idx=None):
        if round_idx is None:
            round_idx = self.__current_global_round
        return self.__effective_aggregation_score_source(round_idx) == "val_acc"

    def __get_topology_connectivity_dict(self, visualize=True):
        if self.__topology_manager_tool is None:
            self.__topology_manager_tool = CommunicationSimulator(
                num_nodes=self.__device_count,
                shape=self.__topology_shape,
                com_stability_mean=self.__com_stability_mean,
                com_stability_std=self.__com_stability_std,
                highest_stability=self.__com_highest_stability,
                lowest_stability=self.__com_lowest_stability,
            )
        self.__topology_connectivity_dict = self.__topology_manager_tool.get_connectivity_dict(
            prefix_name=self.__device_indicator_prefix,
            pos_regenerate=self.__topology_position_change,
            random_mapping=self.__topology_position_change,
        )
        if visualize:
            self.__topology_manager_tool.visualize_graph(
                store_dir=self.log_file_path
            )

        log.info("Topology connectivity dict is:\n")
        log.info(self.__topology_connectivity_dict)

    def __get_stale_sim_tool(self):
        self.__stale_training_tool = StaleTrainingSimulator(
            device_dict=self.__model_dict,
            sim_method=self.__stale_sim_method,
            sim_distribution=self.__stale_sim_distribution,
            gauss_mean=self.__stale_gauss_mean, gauss_std=self.__stale_gauss_std,
            chi_square_k=self.__stale_chi_square_k,
            uniform_multiplier=self.__stale_uniform_multiplier,
            lowest_probability=self.__stale_lowest_probability, highest_probability=self.__stale_highest_probability,
        )

    def __train_models_classification_multi(self):
        self.__reset_training_metrics()
        uses_local_dp_sgd = (
            self.__differential_privacy_controller is not None
            and self.__differential_privacy_config.mode == "local_dp_sgd"
        )
        for device_idx in self.__current_trainable_list:
            log.info(f"Training model {device_idx}")
            running_correct = 0
            running_loss_sum = 0.0
            sample_count = 0
            batch_count = 0
            base_train_dataloader = self.__training_dataloader_dict[device_idx]
            train_dataloader = (
                self.__differential_privacy_controller.training_dataloader(
                    device_idx,
                    base_train_dataloader,
                )
                if uses_local_dp_sgd
                else base_train_dataloader
            )
            early_stop_state = self.__early_stop_state(device_idx)
            stop_current_round_training = False
            for epoch in range(self.__epoch_per_round):
                if uses_local_dp_sgd:
                    self.__differential_privacy_controller.prepare_model_for_training(
                        self.__model_dict[device_idx]
                    )
                else:
                    self.__model_dict[device_idx].train()
                epoch_correct = 0
                epoch_loss_sum = 0.0
                epoch_sample_count = 0
                for idx, batch in enumerate(train_dataloader):
                    if (
                            not uses_local_dp_sgd
                            and (idx + 1) > self.__max_batch_per_epoch
                    ):
                        break
                    if uses_local_dp_sgd and batch is None:
                        self.__differential_privacy_controller.private_step(
                            device_idx,
                            self.__model_dict[device_idx],
                            self.__optimizer_dict[device_idx],
                            self.__loss_function_dict[device_idx],
                            None,
                            None,
                        )
                        batch_count += 1
                        continue
                    data, labels = batch
                    data = data.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    if uses_local_dp_sgd:
                        outputs, loss = (
                            self.__differential_privacy_controller.private_step(
                                device_idx,
                                self.__model_dict[device_idx],
                                self.__optimizer_dict[device_idx],
                                self.__loss_function_dict[device_idx],
                                data,
                                labels,
                            )
                        )
                    else:
                        self.__optimizer_dict[device_idx].zero_grad(set_to_none=True)
                        outputs = self.__model_dict[device_idx](data)
                        loss = self.__loss_function_dict[device_idx](outputs, labels)
                        self.__backward_and_record_scores(device_idx, loss)
                        self.__optimizer_dict[device_idx].step()
                    current_batch_size = int(labels.shape[0])
                    batch_loss_sum = loss.item() * current_batch_size
                    running_loss_sum += batch_loss_sum
                    epoch_loss_sum += batch_loss_sum
                    sample_count += current_batch_size
                    epoch_sample_count += current_batch_size
                    batch_count += 1
                    pred = outputs.argmax(dim=1)
                    correct = int((pred == labels).sum().item())
                    running_correct += correct
                    epoch_correct += correct
                if epoch_sample_count == 0:
                    continue
                epoch_acc = epoch_correct / epoch_sample_count
                epoch_loss = epoch_loss_sum / epoch_sample_count
                early_stop_decision = self.__update_early_stop(
                    device_idx,
                    early_stop_state,
                    epoch + 1,
                    {
                        "acc": epoch_acc,
                        "loss": epoch_loss,
                    },
                )
                if early_stop_decision.should_stop:
                    log.info(
                        "Stop local training for model %s after local epoch %s: "
                        "early_stop reason=%s, metric=%s, value=%.6f, reference=%.6f, plateau_count=%s",
                        device_idx,
                        epoch + 1,
                        early_stop_decision.reason,
                        early_stop_decision.metric,
                        early_stop_decision.metric_value,
                        early_stop_decision.best_value,
                        early_stop_decision.plateau_count,
                    )
                    stop_current_round_training = True
                if stop_current_round_training:
                    break
            self.__train_acc_value_dict[device_idx] = (
                running_correct / sample_count if sample_count else 0.0
            )
            self.__train_loss_value_dict[device_idx] = (
                running_loss_sum / sample_count if sample_count else 0.0
            )
            self.__train_batch_count_value_dict[device_idx] = batch_count

    def __evaluate_classification_model(self, device_idx, evaluation_model, dataloader):
        running_correct = 0
        running_loss_sum = 0.0
        sample_count = 0
        evaluation_model.eval()
        with torch.inference_mode():
            for data, labels in dataloader:
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = evaluation_model(data)
                loss = self.__loss_function_dict[device_idx](outputs, labels)
                current_batch_size = int(labels.shape[0])
                running_loss_sum += loss.item() * current_batch_size
                sample_count += current_batch_size
                running_correct += int((outputs.argmax(dim=1) == labels).sum().item())
        return (
            running_correct / sample_count if sample_count else 0.0,
            running_loss_sum / sample_count if sample_count else 0.0,
        )

    def __test_models_classification_multi(self, val=False):
        self.__reset_test_metrics()
        split_name = "Validation" if val else "Test"
        dataloader = self.__valid_dataloader if val else self.__test_dataloader
        for device_idx in self.__current_trainable_list:
            log.info("%s model %s", split_name, device_idx)
            if not val:
                self.__refresh_batchnorm_before_test_if_needed(device_idx)

            if val and isinstance(
                    self.__differential_privacy_controller,
                    ModelUpdateDPController,
            ):
                with self.__differential_privacy_controller.private_validation_model(
                        device_idx,
                        device,
                ) as evaluation_model:
                    accuracy, loss = self.__evaluate_classification_model(
                        device_idx,
                        evaluation_model,
                        dataloader,
                    )
            else:
                accuracy, loss = self.__evaluate_classification_model(
                    device_idx,
                    self.__model_dict[device_idx],
                    dataloader,
                )
            self.__test_acc_value_dict[device_idx] = accuracy
            self.__test_loss_value_dict[device_idx] = loss
            self.__current_scores_dict[device_idx] = max(1, round(accuracy * 60000))

    def __use_fl_method(self, current_round):
        effective_aggregation_score_source = self.__effective_aggregation_score_source(
            current_round
        )
        segment_importance_active = self.__segment_importance_active(current_round)
        if effective_aggregation_score_source != self.__aggregation_score_source:
            log.info(
                "Aggregation weight is inactive until round %s; use uniform aggregation in round %s",
                self.__aggregation_weight_start_round,
                current_round,
            )
        if not segment_importance_active:
            log.info(
                "Segment importance is inactive until round %s; use random-per-device construction and uniform selection in round %s",
                self.__segment_importance_start_round,
                current_round,
            )

        # Keep cross-round score state warm before the score starts affecting decisions.
        self.__update_aggregation_lipschitz_score_ema_if_needed()
        self.__build_selection_lipschitz_scores_if_needed()
        if segment_importance_active:
            self.__build_parameter_scores_if_needed()
            self.__build_block_parameter_scores_if_needed()
            self.__build_block_interaction_scores_if_needed()
        else:
            self.__current_selection_lipschitz_score_weights_dict = None
            self.__current_parameter_score_weights_dict = None
            self.__current_block_parameter_score_weights_dict = None
            self.__current_block_interaction_score_weights_dict = None
        self.__build_fisher_weights_if_needed(effective_aggregation_score_source)
        try:
            self.__model_dict = run_method(
                method_instance=self.__fl_method_instance,
                current_trainable_list=self.__current_trainable_list,
                current_round=current_round,
                current_round_dict=self.__current_round_dict,
                current_scores_dict=self.__current_scores_dict,
                current_fisher_weights_dict=self.__current_fisher_weights_dict,
                current_selection_lipschitz_score_weights_dict=self.__current_selection_lipschitz_score_weights_dict,
                current_parameter_score_weights_dict=self.__current_parameter_score_weights_dict,
                current_block_parameter_score_weights_dict=self.__current_block_parameter_score_weights_dict,
                current_block_interaction_score_weights_dict=self.__current_block_interaction_score_weights_dict,
                aggregation_score_source=effective_aggregation_score_source,
                segment_importance_active=segment_importance_active,
            )
        except BaseException:
            try:
                self.__flush_communication_round(current_round)
            except Exception:
                log.exception(
                    "Failed to flush communication packets while handling a round %s error",
                    current_round,
                )
            raise
        else:
            self.__flush_communication_round(current_round)

    def __flush_communication_round(self, current_round):
        summary = self.__communication_recorder.flush_round(current_round)
        log.info(
            "Communication round %s: packets=%s, delivered=%s, dropped=%s, "
            "sent=%s bytes, delivered=%s bytes, dropped=%s bytes, "
            "model_parameters=%s bytes, aggregation_weight=%s bytes, "
            "batch_norm=%s bytes, bitmap=%s bytes",
            summary.global_round,
            summary.packet_count,
            summary.delivered_packet_count,
            summary.dropped_packet_count,
            summary.sent_bytes,
            summary.delivered_bytes,
            summary.dropped_bytes,
            summary.model_parameter_bytes,
            summary.aggregation_weight_bytes,
            summary.batch_norm_bytes,
            summary.bitmap_bytes,
        )

    def __record_importance_correlation_if_needed(self, round_idx, stage):
        if self.__importance_correlation_probe is None:
            return
        if self.__gradient_buffer_tool is None:
            raise RuntimeError(
                "Importance correlation measurement requires the gradient/score buffer"
            )
        self.__importance_correlation_probe.record(
            round_idx=round_idx,
            stage=stage,
            gradient_buffer=self.__gradient_buffer_tool,
            model_idx_list=self.__current_trainable_list,
        )

    def __close_experiment_probes_if_needed(self):
        if self.__importance_correlation_probe is not None:
            self.__importance_correlation_probe.close()

    def __close_communication_recorder_if_needed(self):
        if self.__communication_recorder is not None:
            self.__communication_recorder.close()

    def __close_differential_privacy_controller_if_needed(self):
        if self.__differential_privacy_controller is not None:
            self.__log_actual_privacy_summary_if_needed()
            self.__differential_privacy_controller.close()
    def __actual_privacy_summary(self):
        if self.__differential_privacy_controller is None:
            return None
        costs = self.__differential_privacy_controller.privacy_costs()
        if not costs:
            return None
        worst = max(costs, key=lambda cost: cost.epsilon)
        return {
            "devices_accounted": len(costs),
            "minimum_epsilon": min(cost.epsilon for cost in costs),
            "maximum_epsilon": worst.epsilon,
            "worst_device": worst.device_id,
            "worst_device_release_count": worst.release_count,
            "worst_device_optimal_alpha": worst.optimal_alpha,
            "delta": worst.delta,
        }

    def __log_actual_privacy_summary_if_needed(self):
        if self.__privacy_summary_logged:
            return
        summary = self.__actual_privacy_summary()
        if summary is None:
            return
        log.info(
            "DP actual privacy summary\n"
            "  devices_accounted: %s\n"
            "  minimum_epsilon: %.10g\n"
            "  maximum_epsilon: %.10g\n"
            "  worst_device: %s\n"
            "  worst_device_release_count: %s\n"
            "  worst_device_optimal_alpha: %.10g\n"
            "  delta: %.10g",
            summary["devices_accounted"],
            summary["minimum_epsilon"],
            summary["maximum_epsilon"],
            summary["worst_device"],
            summary["worst_device_release_count"],
            summary["worst_device_optimal_alpha"],
            summary["delta"],
        )
        self.__privacy_summary_logged = True

    def __finalize_outputs_after_simulation_error(self):
        cleanup_actions = (
            ("metrics output", self.__store_metrics_to_files),
            ("wall-clock output", self.__finish_wall_clock_run),
            ("communication output", self.__close_communication_recorder_if_needed),
            ("privacy accounting", self.__close_differential_privacy_controller_if_needed),
            ("experiment probes", self.__close_experiment_probes_if_needed),
        )
        for output_name, cleanup in cleanup_actions:
            try:
                cleanup()
            except Exception:
                log.exception("Failed to finalize %s while handling a simulation error", output_name)

    def __training_metrics(self):
        return MetricValues(
            acc=self.__train_acc_value_dict,
            loss=self.__train_loss_value_dict,
            recall=self.__train_recall_value_dict,
            precision=self.__train_precision_value_dict,
        )

    def __test_metrics(self):
        return MetricValues(
            acc=self.__test_acc_value_dict,
            loss=self.__test_loss_value_dict,
            recall=self.__test_recall_value_dict,
            precision=self.__test_precision_value_dict,
        )

    def __store_metrics_to_files(self, round_idx=None):
        self.__metrics_recorder.save_excel(self.log_file_path + "metrics.xlsx")
        if round_idx is not None:
            self.__last_metrics_output_round = round_idx

    def __store_final_metrics_if_needed(self):
        if self.__last_metrics_output_round != self.__rounds:
            self.__store_metrics_to_files(self.__rounds)

    def __store_wall_clock_to_file(self):
        os.makedirs(self.log_file_path, exist_ok=True)
        with open(
            self.log_file_path + "wall_clock.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=self.WALL_CLOCK_COLUMNS)
            writer.writeheader()
            writer.writerows(self.__wall_clock_rows)
    def __append_wall_clock_row(self, row):
        os.makedirs(self.log_file_path, exist_ok=True)
        file_path = self.log_file_path + "wall_clock.csv"
        write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.WALL_CLOCK_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


    def __start_wall_clock_run(self):
        self.__last_metrics_output_round = None
        self.__wall_clock_rows = []
        self.__store_wall_clock_to_file()
        self.__wall_clock_run_started_at = datetime.now().isoformat(timespec="seconds")
        self.__wall_clock_run_finished_at = None
        self.__wall_clock_run_total_s = None
        self.__wall_clock_run_start_perf = self.__wall_clock_now()
        self.__store_run_metadata()

    def __finish_wall_clock_run(self):
        if self.__wall_clock_run_start_perf is None:
            return
        self.__wall_clock_run_total_s = round(
            self.__wall_clock_now() - self.__wall_clock_run_start_perf,
            6,
        )
        self.__wall_clock_run_finished_at = datetime.now().isoformat(timespec="seconds")
        if self.__topology_position_change and self.__topology_manager_tool is not None:
            self.__topology_manager_tool.visualize_graph(
                store_dir=self.log_file_path
            )
        self.__store_wall_clock_to_file()
        self.__store_run_metadata()

    def __record_wall_clock_round(self, round_idx, timings, round_start_time):
        total_s = self.__wall_clock_now() - round_start_time
        measured_stage_s = sum(timings.values())
        row = {
            "round": round_idx,
            "task": self.__experiment_task,
            "trainable_device_count": len(self.__current_trainable_list or []),
            "total_s": round(total_s, 6),
            "bookkeeping_s": round(max(0.0, total_s - measured_stage_s), 6),
        }
        for column in self.WALL_CLOCK_COLUMNS:
            if column.endswith("_s") and column not in row:
                row[column] = round(timings.get(column, 0.0), 6)
        self.__wall_clock_rows.append(row)
        self.__append_wall_clock_row(row)
        log.info(
            "Round %s wall-clock total %.3fs: train %.3fs, post_score %.3fs, "
            "privacy %.3fs, validation %.3fs, fl_method %.3fs, test %.3fs",
            round_idx,
            row["total_s"],
            row["train_s"],
            row["post_score_s"],
            row["privacy_prepare_s"],
            row["validation_before_s"],
            row["fl_method_s"],
            row["test_after_s"],
        )

    def __store_run_metadata(self):
        dp_config = self.__differential_privacy_config
        dp_enabled = dp_config.enabled
        dp_mode = dp_config.mode if dp_enabled else None
        metadata = {
            "source_config_file": self.__source_config_file,
            "loaded_config_file": self.__loaded_config_file,
            "used_config_snapshot": "used_config.yml",
            "log_directory": {
                "path": self.log_file_path,
                "name": self.__log_dir_name,
                "full_name": self.__full_log_dir_name,
                "shortened": self.__log_dir_name_shortened,
                "hash": self.__log_dir_name_hash,
                "configured_max_name_length": self.__log_run_dir_name_max_length,
                "max_name_length": min(
                    self.__log_run_dir_name_max_length,
                    SAFE_RUN_DIR_NAME_MAX_BYTES,
                ),
            },
            "run_label": self.__run_label,
            "model": self.__model_name,
            "dataset": self.__dataset_name,
            "method": self.__fl_method_name,
            "method_training_abbreviation": self.__used_method_log_abbreviation,
            "model_initialization_seed": self.__torch_random_seed,
            "dataloader_workers": self.__dataloader_workers,
            "aggregation_device": {
                "policy": self.__aggregation_device_policy,
                "resolved": str(self.__aggregation_device),
            },
            "refresh_batch_norm_from_validation": self.__refresh_batch_norm_from_validation,
            "activation_schedule": {
                "segment_importance_start_round": self.__segment_importance_start_round,
                "aggregation_weight_start_round": self.__aggregation_weight_start_round,
                "local_update_l2_start_round": self.__local_update_l2_start_round,
            },
            "wall_clock": {
                "round_file": "wall_clock.csv",
                "started_at": self.__wall_clock_run_started_at,
                "finished_at": self.__wall_clock_run_finished_at,
                "total_s": self.__wall_clock_run_total_s,
                "rounds_recorded": len(self.__wall_clock_rows),
            },
            "communication": {
                "packet_file": "communication_packets.csv",
                "numeric_payload_dtype": "float32",
                "bitmap_storage": "ceil(bits / 8) bytes",
            },
            "differential_privacy": {
                "enabled": dp_enabled,
                "mode": dp_mode,
                "clipping_norm": dp_config.clipping_norm if dp_enabled else None,
                "noise_multiplier": dp_config.noise_multiplier if dp_enabled else None,
                "delta": dp_config.delta if dp_enabled else None,
                "accounting_file": (
                    "privacy_accounting.csv"
                    if self.__differential_privacy_config.enabled
                    else None
                ),
                "privacy_unit": (
                    "sample_within_device"
                    if dp_mode == "local_dp_sgd"
                    else "device" if dp_mode == "model_update"
                    else None
                ),
                "protected_release": (
                    "dp_sgd_trained_outgoing_model"
                    if dp_mode == "local_dp_sgd"
                    else "outgoing_trainable_model_parameters" if dp_enabled
                    else None
                ),
                "sampling": (
                    "poisson_with_configured_batches_per_epoch"
                    if dp_mode == "local_dp_sgd"
                    else None
                ),
                "adjacency": dp_config.adjacency if dp_enabled else None,
                "l2_sensitivity": dp_config.l2_sensitivity if dp_enabled else None,
                "private_batch_norm_buffers": (
                    "initial_or_public_validation_only" if dp_enabled else None
                ),
                "local_diagnostic_outputs_dp_protected": False,
                "planned_privacy": self.__planned_privacy_metadata(),
                "actual_privacy": self.__actual_privacy_summary(),
            },
        }
        with open(self.log_file_path + "run_metadata.yml", "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)
    def __planned_privacy_metadata(self):
        estimate = self.__planned_privacy_estimate
        if estimate is None:
            return None
        cost = estimate.privacy_cost
        return {
            "accountant": (
                "poisson_sampled_gaussian_rdp"
                if estimate.mode == "local_dp_sgd"
                else "gaussian_rdp"
            ),
            "participation_assumption": "configured_round_upper_bound",
            "participations_per_device": estimate.participations,
            "mechanisms_per_participation": estimate.mechanisms_per_participation,
            "total_releases_or_steps": estimate.total_mechanisms,
            "dataset_size_per_device": estimate.dataset_size,
            "expected_batch_size": estimate.expected_batch_size,
            "sample_rate": estimate.sample_rate,
            "epsilon": cost.epsilon,
            "delta": cost.delta,
            "optimal_alpha": cost.optimal_alpha,
        }

    def __begin_round(self, round_idx):
        self.__current_global_round = round_idx
        round_filter.set_round(round_idx)
        log.info(f"\nRound {round_idx} Begin\n")
        self.__current_trainable_list = self.__stale_training_tool.get_current_trainable_devices(round_idx)
        self.__reset_round_optimizer_state_if_needed()
        for device_idx in self.__current_round_dict.keys():
            if device_idx in self.__current_trainable_list:
                self.__current_round_dict[device_idx] += 1
        log.info(f"Each device is in its local round of {self.__current_round_dict}")
        self.__reset_fisher_buffer_if_needed(round_idx)
        self.__reset_parameter_score_buffer_if_needed(round_idx)
        if self.__differential_privacy_controller is not None:
            self.__differential_privacy_controller.begin_round(
                self.__model_dict,
                self.__current_trainable_list,
            )

    def __snapshot_local_update_unit_l2_if_needed(self, round_idx):
        if not self.__local_update_l2_active(round_idx):
            return
        snapshot = getattr(
            self.__fl_method_instance,
            "snapshot_local_update_unit_l2",
            None,
        )
        if snapshot is not None:
            snapshot(self.__current_trainable_list)

    def __apply_local_update_unit_l2_if_needed(self, round_idx):
        if not self.__local_update_l2_active(round_idx):
            return
        apply_constraint = getattr(
            self.__fl_method_instance,
            "apply_local_update_unit_l2",
            None,
        )
        if apply_constraint is not None:
            apply_constraint(self.__current_trainable_list)

    def __discard_local_update_unit_l2_snapshot_if_needed(self):
        discard = getattr(
            self.__fl_method_instance,
            "discard_local_update_unit_l2_snapshot",
            None,
        )
        if discard is not None:
            discard()
    def __prepare_private_models_if_needed(self, round_idx):
        if self.__differential_privacy_controller is None:
            return
        private_models = (
            self.__differential_privacy_controller.prepare_private_models(
                self.__model_dict,
                self.__current_trainable_list,
                round_idx,
            )
        )
        self.__fl_method_instance.set_outgoing_models(private_models)
        log.info(
            "Prepared DP outgoing models for %s devices using mode=%s",
            len(self.__current_trainable_list),
            self.__differential_privacy_config.mode,
        )

    def __reset_round_optimizer_state_if_needed(self):
        if self.__optimizer_name != "adam_round":
            return
        for device_idx in self.__current_trainable_list:
            self.__optimizer_dict[device_idx].state.clear()

    def __log_training_metrics(self, round_idx):
        log.info(
            f'After round {round_idx} training\n'
            f'Training accuracy is {self.__train_acc_value_dict}\n'
            f'Training loss is {self.__train_loss_value_dict}\n'
            f'Training batch count is {self.__train_batch_count_value_dict}\n'
            f'Training recall is {self.__train_recall_value_dict}\n'
            f'Training precision is {self.__train_precision_value_dict}'
        )

    def __log_evaluation_metrics(self, round_idx, split_name, stage):
        log.info(
            f'After round {round_idx} training, {stage}\n'
            f'{split_name} accuracy is {self.__test_acc_value_dict}\n'
            f'{split_name} loss is {self.__test_loss_value_dict}\n'
            f'{split_name} recall is {self.__test_recall_value_dict}\n'
            f'{split_name} precision is {self.__test_precision_value_dict}'
        )

    def __refresh_topology_if_needed(self):
        if self.__topology_position_change:
            self.__get_topology_connectivity_dict(visualize=False)
            self.__fl_method_instance.set_connectivity_dict(
                self.__topology_connectivity_dict
            )

    def __update_adaptive_lr_if_needed(self):
        if self.__use_adaptive_lr:
            self.__lr_strategy.get_new_optimizer_dict(
                optimizer_dict=self.__optimizer_dict,
                current_round_dict=self.__current_round_dict,
                current_loss_dict=self.__train_loss_value_dict,
                current_model_idx_list=self.__current_trainable_list
            )

    def __compute_post_training_score_methods(
            self,
            round_idx,
            active_score_methods,
            hutchinson_z_time,
            hutchinson_batch_limit,
    ):
        self.__compute_post_training_hessian_scores_if_needed(
            round_idx,
            active_score_methods,
        )
        self.__compute_post_training_gradient_fisher_scores_if_needed(
            round_idx,
            active_score_methods,
        )
        self.__compute_post_training_gradient_signal_preservation_scores_if_needed(
            round_idx,
            active_score_methods,
        )
        self.__compute_post_training_hutchinson_scores_if_needed(
            round_idx,
            active_score_methods,
            hutchinson_z_time,
            hutchinson_batch_limit,
        )

    def __importance_probe_post_score_methods(self, round_idx, stage):
        probe_config = self.__experiment_probe_config.importance_correlation
        if (
                probe_config is None
                or not probe_config.should_evaluate(round_idx, stage)
        ):
            return None, set()
        return probe_config, (
            set(probe_config.required_internal_score_methods)
            & POST_TRAINING_PARAMETER_SCORE_METHODS
        )

    def __compute_probe_post_score_methods(
            self,
            round_idx,
            probe_config,
            score_methods,
    ):
        if not score_methods:
            return
        probe_dataloaders = (
            self.__training_dataloader_dict[device_idx]
            for device_idx in self.__current_trainable_list
        )
        with preserve_random_state(probe_dataloaders):
            self.__compute_post_training_score_methods(
                round_idx,
                score_methods,
                probe_config.hutchinson.probe_vectors,
                probe_config.hutchinson.batch_limit,
            )

    def __compute_post_training_scores_if_needed(self, round_idx):
        method_score_methods = (
            set(self.__method_parameter_score_methods_to_record)
            & POST_TRAINING_PARAMETER_SCORE_METHODS
        )
        (
            probe_config,
            probe_score_methods,
        ) = self.__importance_probe_post_score_methods(
            round_idx,
            "after_training",
        )

        # Abs-gradient and empirical-Fisher values come from the same
        # per-sample gradients. Weighted forms reuse those same base tensors.
        method_compute_methods = set(method_score_methods)
        shared_score_families = (
            {
                GRADIENT_ABS_POST_METHOD,
                GRADIENT_WEIGHT_ABS_POST_METHOD,
                FISHER_EMPIRICAL_DIAGONAL_POST_METHOD,
                FISHER_EMPIRICAL_DIAGONAL_WEIGHT_POST_METHOD,
            },
            {
                HUTCHINSON_DIAGONAL_POST_METHOD,
                HUTCHINSON_DIAGONAL_WEIGHT_POST_METHOD,
            },
        )
        for family in shared_score_families:
            if method_score_methods & family:
                method_compute_methods.update(probe_score_methods & family)

        self.__compute_post_training_score_methods(
            round_idx,
            method_compute_methods,
            self.__hutchinson_z_time,
            self.__hutchinson_batch_limit,
        )
        if probe_config is None:
            return
        self.__compute_probe_post_score_methods(
            round_idx,
            probe_config,
            probe_score_methods - method_compute_methods,
        )

    def __compute_post_aggregation_probe_scores_if_needed(self, round_idx):
        (
            probe_config,
            probe_score_methods,
        ) = self.__importance_probe_post_score_methods(
            round_idx,
            "after_aggregation",
        )
        if probe_config is None:
            return
        # The model changed during aggregation, so post-training metrics used
        # by the method before aggregation must be recomputed at this stage.
        self.__compute_probe_post_score_methods(
            round_idx,
            probe_config,
            probe_score_methods,
        )

    def __finish_round(self, round_idx):
        log.info(
            f"Finish Round {round_idx}\n"
            f"**********************************************************************\n"
            f"**********************************************************************\n"
            f"**********************************************************************\n"
        )
        if round_idx % self.METRICS_CHECKPOINT_INTERVAL == 0:
            self.__store_metrics_to_files(round_idx)

    def compile_all_settings(self):
        self.__ready_to_simulate = False
        if self.__model_name not in NETWORKS.keys():
            raise NotImplementedError(f"Model {self.__model_name} is not implemented yet, only support {NETWORKS.keys()}")
        if self.__dataset_name not in DATASETS.keys():
            raise NotImplementedError(f"Dataset {self.__dataset_name} is not implemented yet, only support {DATASETS.keys()}")
        if self.__fl_method_name not in METHODS.keys():
            raise NotImplementedError(f"FL method {self.__fl_method_name} is not implemented yet, only support {METHODS.keys()}")
        if self.__loss_function_name not in self.supported_loss_function_dict.keys():
            raise NotImplementedError(f"Loss function {self.__loss_function_name} is not implemented yet, only support {self.supported_loss_function_dict.keys()}")
        if self.__optimizer_name not in self.supported_optimizer_dict.keys():
            raise NotImplementedError(
                f"optimizer {self.__optimizer_name} is not implemented yet, only support {self.supported_optimizer_dict.keys()}")
        if self.__experiment_task == "Object_Detection":
            validate_object_detection_configuration(
                self.__dataset_name,
                self.__model_name,
                self.__loss_function_name,
                self.__output_class_number,
            )
        # set log
        self.setup_logging(self.log_file_path+"fl_simulation.log")
        if self.__log_dir_name_shortened:
            log.info(
                "Log directory name shortened to %s bytes with hash suffix: %s",
                len(self.__log_dir_name.encode("utf-8")),
                self.__log_dir_name,
            )
        log.info(
            "Runtime device is [%s], CUDA status is %s",
            device,
            cuda_status(),
        )
        with open(self.log_file_path+'used_config.yml', 'w') as f:
            self.config.write(f)
        self.__store_run_metadata()
        # Ready required buffers and instance
        self.__get_dataset_dict()
        self.__get_models_dict()
        self.__get_gradient_buffer_tool()
        self.__get_selection_lipschitz_score_ema_buffer()
        self.__get_aggregation_lipschitz_score_ema_buffer()
        self.__get_experiment_probes()
        self.__get_optimizer_dict()
        self.__get_loss_function_dict()
        self.__get_acc_recall_dict()
        self.__get_topology_connectivity_dict()
        self.__get_communication_recorder()
        self.__get_fl_method()
        self.__get_stale_sim_tool()
        self.__get_differential_privacy_controller()

        if self.__use_adaptive_lr:
            self.__lr_strategy = AdaptiveDFLLearningRate(
                lr_dict=self.__lr_dict, total_round_dict=self.__total_rounds_dict,
                if_ada_loss=self.__use_AdaLoss, if_ada_stair=self.__use_AdaStair
            )
        self.__ready_to_simulate = True
        self.__metrics_recorder.initialize(
            training=self.__training_metrics(),
            test=self.__test_metrics(),
        )

        if self.__experiment_task == "Object_Detection":
            self.__object_detection_task = infer_object_detection_task(self.__model_name)
            self.__detection_metrics_tool = build_detection_metrics(
                self.__object_detection_task,
                self.__loss_function_name,
            )

    def __run_round_with_wall_clock(self, round_idx, train_fn, test_fn):
        round_timings = {}
        round_start_time = self.__wall_clock_now()
        with self.__measure_wall_clock_stage(round_timings, "begin_round_s"):
            self.__begin_round(round_idx)

        self.__snapshot_local_update_unit_l2_if_needed(round_idx)
        try:
            with self.__measure_wall_clock_stage(round_timings, "train_s"):
                train_fn()
        except BaseException:
            self.__discard_local_update_unit_l2_snapshot_if_needed()
            raise
        with self.__measure_wall_clock_stage(round_timings, "local_update_l2_s"):
            self.__apply_local_update_unit_l2_if_needed(round_idx)
        with self.__measure_wall_clock_stage(round_timings, "post_score_s"):
            self.__compute_post_training_scores_if_needed(round_idx)
        with self.__measure_wall_clock_stage(round_timings, "importance_probe_s"):
            self.__record_importance_correlation_if_needed(
                round_idx,
                "after_training",
            )
        with self.__measure_wall_clock_stage(round_timings, "privacy_prepare_s"):
            self.__prepare_private_models_if_needed(round_idx)
        with self.__measure_wall_clock_stage(round_timings, "training_metrics_s"):
            self.__log_training_metrics(round_idx)
            self.__metrics_recorder.append_training(round_idx, self.__training_metrics())

        if self.__uses_validation_weight(round_idx):
            with self.__measure_wall_clock_stage(round_timings, "validation_before_s"):
                test_fn(val=True)
                self.__log_evaluation_metrics(round_idx, "Validation", "before aggregation")
                self.__metrics_recorder.append_validation_pre_aggregation(
                    round_idx,
                    self.__test_metrics(),
                )
        else:
            log.info(
                "Skip shared validation before aggregation; configured metric=[%s], effective metric=[%s]",
                self.__aggregation_score_source,
                self.__effective_aggregation_score_source(round_idx),
            )

        with self.__measure_wall_clock_stage(round_timings, "topology_refresh_s"):
            self.__refresh_topology_if_needed()
        with self.__measure_wall_clock_stage(round_timings, "fl_method_s"):
            self.__use_fl_method(round_idx)
        with self.__measure_wall_clock_stage(round_timings, "importance_probe_s"):
            self.__compute_post_aggregation_probe_scores_if_needed(round_idx)
            self.__record_importance_correlation_if_needed(
                round_idx,
                "after_aggregation",
            )

        with self.__measure_wall_clock_stage(round_timings, "test_after_s"):
            test_fn(val=False)
            self.__log_evaluation_metrics(round_idx, "Test", "after aggregation")
            self.__metrics_recorder.append_test_post_aggregation(
                round_idx,
                self.__test_metrics(),
            )

        with self.__measure_wall_clock_stage(round_timings, "adaptive_lr_s"):
            self.__update_adaptive_lr_if_needed()
        with self.__measure_wall_clock_stage(round_timings, "finish_round_s"):
            self.__finish_round(round_idx)
        self.__record_wall_clock_round(round_idx, round_timings, round_start_time)

    def run_simulation_classification_multi(self, round_idx):
        self.__run_round_with_wall_clock(
            round_idx,
            self.__train_models_classification_multi,
            self.__test_models_classification_multi,
        )

    @staticmethod
    def __accumulate_fomo_counts(total_counts, batch_counts):
        detached_counts = tuple(count.detach() for count in batch_counts)
        if total_counts is None:
            return tuple(count.clone() for count in detached_counts)
        if len(total_counts) != len(detached_counts):
            raise ValueError("FOMO metric count tuples must have the same length")
        return tuple(total + batch for total, batch in zip(total_counts, detached_counts))

    def __fomo_metric_values(self, counts):
        if counts is None:
            return 0.0, 0.0, 0.0
        precision, recall, f1 = self.__detection_metrics_tool.metrics_from_counts(*counts)
        return (
            float(precision.item()),
            float(recall.item()),
            float(f1.item()),
        )

    def __train_models_object_detection_multi(self, task):
        self.__reset_training_metrics()
        if task == "fomo":
            for device_idx in self.__current_trainable_list:
                log.info(f"Training model {device_idx}")
                running_loss = 0
                running_counts = None
                sample_count = 0
                batch_count = 0
                train_dataloader = self.__training_dataloader_dict[device_idx]
                early_stop_state = self.__early_stop_state(device_idx)
                stop_current_round_training = False
                for epoch in range(self.__epoch_per_round):
                    self.__model_dict[device_idx].train()
                    epoch_counts = None
                    epoch_running_loss = 0
                    epoch_sample_count = 0
                    for idx, (data, labels) in enumerate(train_dataloader):
                        if (idx + 1) > self.__max_batch_per_epoch:
                            break
                        data = data.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        self.__optimizer_dict[device_idx].zero_grad(set_to_none=True)
                        outputs = self.__model_dict[device_idx](data)
                        loss = self.__loss_function_dict[device_idx](outputs, labels)
                        self.__backward_and_record_scores(device_idx, loss)
                        self.__optimizer_dict[device_idx].step()
                        current_batch_size = int(labels.shape[0])
                        batch_loss_sum = loss.item() * current_batch_size
                        running_loss += batch_loss_sum
                        epoch_running_loss += batch_loss_sum
                        sample_count += current_batch_size
                        epoch_sample_count += current_batch_size
                        batch_count += 1

                        batch_counts = self.__detection_metrics_tool.counts(outputs, labels)
                        running_counts = self.__accumulate_fomo_counts(
                            running_counts, batch_counts
                        )
                        epoch_counts = self.__accumulate_fomo_counts(
                            epoch_counts, batch_counts
                        )
                    if epoch_sample_count == 0:
                        continue
                    _, _, epoch_acc = self.__fomo_metric_values(epoch_counts)
                    epoch_loss = epoch_running_loss / epoch_sample_count
                    early_stop_decision = self.__update_early_stop(
                        device_idx,
                        early_stop_state,
                        epoch + 1,
                        {
                            "acc": epoch_acc,
                            "loss": epoch_loss,
                        },
                    )
                    if early_stop_decision.should_stop:
                        log.info(
                            "Stop local training for model %s after local epoch %s: "
                            "early_stop reason=%s, metric=%s, value=%.6f, reference=%.6f, plateau_count=%s",
                            device_idx,
                            epoch + 1,
                            early_stop_decision.reason,
                            early_stop_decision.metric,
                            early_stop_decision.metric_value,
                            early_stop_decision.best_value,
                            early_stop_decision.plateau_count,
                        )
                        stop_current_round_training = True
                    if stop_current_round_training:
                        break

                denominator = sample_count if sample_count else 1
                precision, recall, f1 = self.__fomo_metric_values(running_counts)
                self.__train_acc_value_dict[device_idx] = f1
                self.__train_loss_value_dict[device_idx] = running_loss / denominator
                self.__train_precision_value_dict[device_idx] = precision
                self.__train_recall_value_dict[device_idx] = recall
                self.__train_batch_count_value_dict[device_idx] = batch_count
        elif task == "yolo":
            self.__warn_early_stop_unavailable_once(
                "YOLO training does not expose local epoch metrics for early stop."
            )
            for device_idx in self.__current_trainable_list:
                log.info(f"Training model {device_idx}")
                running_loss = 0
                sample_count = 0
                batch_count = 0
                train_dataloader = self.__training_dataloader_dict[device_idx]
                self.__detection_metrics_tool.clear()
                for epoch in range(self.__epoch_per_round):
                    self.__model_dict[device_idx].train()
                    for idx, (data, labels) in enumerate(train_dataloader):
                        if (idx + 1) > self.__max_batch_per_epoch:
                            break
                        data = data.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        self.__optimizer_dict[device_idx].zero_grad(set_to_none=True)
                        outputs = self.__model_dict[device_idx](data)
                        loss = self.__loss_function_dict[device_idx](outputs, labels)
                        self.__backward_and_record_scores(device_idx, loss)
                        self.__optimizer_dict[device_idx].step()
                        current_batch_size = int(labels.shape[0])
                        running_loss += loss.item() * current_batch_size
                        sample_count += current_batch_size
                        batch_count += 1
                        self.__detection_metrics_tool(outputs, labels)
                mAP = self.__detection_metrics_tool.mAP_calculate()

                self.__train_acc_value_dict[device_idx] = float(mAP["map_50"])
                self.__train_loss_value_dict[device_idx] = (
                    running_loss / sample_count if sample_count else 0.0
                )
                self.__train_batch_count_value_dict[device_idx] = batch_count
        else:
            raise NotImplementedError(f"Task {task} not implemented.")

    def __evaluate_object_detection_model(
            self,
            task,
            device_idx,
            evaluation_model,
            dataloader,
    ):
        if task not in {"fomo", "yolo"}:
            raise NotImplementedError(f"Task {task} not implemented.")
        running_counts = None
        running_loss = 0.0
        sample_count = 0
        if task == "yolo":
            self.__detection_metrics_tool.clear()

        evaluation_model.eval()
        with torch.inference_mode():
            for data, labels in dataloader:
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = evaluation_model(data)
                loss = self.__loss_function_dict[device_idx](outputs, labels)
                current_batch_size = int(labels.shape[0])
                running_loss += loss.item() * current_batch_size
                sample_count += current_batch_size
                if task == "fomo":
                    running_counts = self.__accumulate_fomo_counts(
                        running_counts,
                        self.__detection_metrics_tool.counts(outputs, labels),
                    )
                else:
                    self.__detection_metrics_tool(outputs, labels)

        if task == "fomo":
            precision, recall, accuracy = self.__fomo_metric_values(running_counts)
            self.__test_precision_value_dict[device_idx] = precision
            self.__test_recall_value_dict[device_idx] = recall
        else:
            accuracy = float(self.__detection_metrics_tool.mAP_calculate()["map_50"])
        self.__test_acc_value_dict[device_idx] = accuracy
        self.__test_loss_value_dict[device_idx] = (
            running_loss / sample_count if sample_count else 0.0
        )
        self.__current_scores_dict[device_idx] = max(1, round(accuracy * 60000))

    def __test_models_object_detection_multi(self, task, val=False):
        self.__reset_test_metrics()
        split_name = "Validation" if val else "Test"
        dataloader = self.__valid_dataloader if val else self.__test_dataloader
        for device_idx in self.__current_trainable_list:
            log.info("%s model %s", split_name, device_idx)
            if not val:
                self.__refresh_batchnorm_before_test_if_needed(device_idx)

            if val and isinstance(
                    self.__differential_privacy_controller,
                    ModelUpdateDPController,
            ):
                with self.__differential_privacy_controller.private_validation_model(
                        device_idx,
                        device,
                ) as evaluation_model:
                    self.__evaluate_object_detection_model(
                        task,
                        device_idx,
                        evaluation_model,
                        dataloader,
                    )
            else:
                self.__evaluate_object_detection_model(
                    task,
                    device_idx,
                    self.__model_dict[device_idx],
                    dataloader,
                )

    def run_simulation_object_detection_multi(self, round_idx):
        self.__run_round_with_wall_clock(
            round_idx,
            lambda: self.__train_models_object_detection_multi(task=self.__object_detection_task),
            lambda val: self.__test_models_object_detection_multi(
                task=self.__object_detection_task,
                val=val,
            ),
        )


    def run_simulation_multi(self):
        if not self.__ready_to_simulate:
            raise RuntimeError("Please execute [compile_all_settings] first")
        runner_name = self.ROUND_RUNNERS.get(self.__experiment_task)
        if runner_name is None:
            raise NotImplementedError(f"Experiment task {self.__experiment_task} is not implemented.")
        runner = getattr(self, runner_name)
        try:
            self.__start_wall_clock_run()
            for round_idx in range(1, self.__rounds + 1):
                runner(round_idx)
        except BaseException:
            self.__finalize_outputs_after_simulation_error()
            raise
        else:
            try:
                self.__store_final_metrics_if_needed()
                self.__finish_wall_clock_run()
            finally:
                try:
                    self.__close_communication_recorder_if_needed()
                finally:
                    try:
                        self.__close_differential_privacy_controller_if_needed()
                    finally:
                        self.__close_experiment_probes_if_needed()
