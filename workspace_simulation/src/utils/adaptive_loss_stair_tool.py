import numpy as np
import logging

log = logging.getLogger(__name__)

class AdaptiveDFLLearningRate:
    def __init__(self, lr_dict:dict, total_round_dict:dict,
                 if_ada_stair=False, if_ada_loss=False,
                 ada_stair_list=None, ada_loss_list=None):
        if ada_stair_list is None:
            ada_stair_list = [0.4, 0.65, 0.75]
        if ada_loss_list is None:
            ada_loss_list = [0.10, 0.20, 0.20, 0.30]

        ada_stair_list = sorted(ada_stair_list)
        ada_loss_list = sorted(ada_loss_list, reverse=True)

        ada_stair_array = np.array(ada_stair_list)
        ada_loss_array = np.array(ada_loss_list)

        self.__lr_dict = lr_dict
        self.__total_round_dict = total_round_dict
        self.__if_ada_stair = if_ada_stair
        self.__if_ada_loss = if_ada_loss
        self.__ada_stair_dict = dict()
        self.__ada_loss_dict = dict()
        self.__pre_loss_dict = dict()
        self.__loss_count_dict = dict()
        self.__ada_loss_index_dict = dict()
        self.__applied_stair_rounds = dict()
        self.__min_lr = 1e-5 # to avoid too small lr

        for device_idx, device_total_round in total_round_dict.items():
            self.__ada_stair_dict[device_idx] = np.ceil(ada_stair_array*device_total_round).tolist()
            self.__ada_loss_dict[device_idx] = np.ceil(ada_loss_array*device_total_round).tolist()
            self.__pre_loss_dict[device_idx] = -1 # use -1 to indicate the null loss
            self.__loss_count_dict[device_idx] = 0
            self.__ada_loss_index_dict[device_idx] = 0
            self.__applied_stair_rounds[device_idx] = set()

        log.info(
            f'\nAdaptive Learning Rate Strategy is used\n'
            f'AdaStair is used [{self.__if_ada_stair}]\n'
            f'AdaLoss is used [{self.__if_ada_loss}]\n'
        )



    def __apply_ada_stair(self, current_round_dict:dict, current_model_idx_list:list):
        for device_idx in current_model_idx_list:
            device_current_round = current_round_dict[device_idx]
            if (
                    device_current_round in self.__ada_stair_dict[device_idx]
                    and device_current_round not in self.__applied_stair_rounds[device_idx]
            ):
                self.__lr_dict[device_idx] /= 8.0
                self.__lr_dict[device_idx] = max(self.__lr_dict[device_idx], self.__min_lr)
                self.__applied_stair_rounds[device_idx].add(device_current_round)
                log.info(
                    f'AdaStair is applied successfully.\n'
                    f'Current round in [{device_current_round}]\n'
                    f'the [{device_idx}] (device) has a new learning rate of [{self.__lr_dict[device_idx]}]\n'
                    f'The minimum lr can be set is [{self.__min_lr}]'
                )

    def __apply_ada_loss(self, current_loss_dict: dict, current_round_dict: dict,
                         current_model_idx_list: list):
        for device_idx, device_current_loss in current_loss_dict.items():
            if device_idx not in current_model_idx_list:
                continue
            if not np.isfinite(device_current_loss) or device_current_loss < 0:
                raise ValueError(
                    f'loss must be finite and non-negative, but the current loss is '
                    f'[{device_current_loss}]'
                )
            device_current_round = current_round_dict[device_idx]
            if device_current_round <= 0:
                raise ValueError(
                    f'current round must be positive, got [{device_current_round}] '
                    f'for [{device_idx}]'
                )
            if self.__pre_loss_dict[device_idx] == -1:
                self.__pre_loss_dict[device_idx] = device_current_loss
            else:
                historical_mean_loss = self.__pre_loss_dict[device_idx]
                if device_current_loss >= historical_mean_loss:
                    self.__loss_count_dict[device_idx] += 1
                else:
                    self.__loss_count_dict[device_idx] = 0
                self.__pre_loss_dict[device_idx] = (
                    historical_mean_loss * (device_current_round - 1)
                    + device_current_loss
                ) / device_current_round
                if self.__loss_count_dict[device_idx] > self.__ada_loss_dict[device_idx][self.__ada_loss_index_dict[device_idx]]:
                    self.__lr_dict[device_idx] /= 2.0
                    self.__lr_dict[device_idx] = max(self.__lr_dict[device_idx], self.__min_lr)
                    log.info(
                        f'AdaLoss is applied successfully.\n'
                        f'Current round in [{device_current_round}]\n'
                        f'The loss of [{device_idx}] (device) does not decrease for [{self.__loss_count_dict[device_idx]}] rounds.\n'
                        f'Thus, it gets a new learning rate of [{self.__lr_dict[device_idx]}]\n'
                        f'The minimum lr can be set is [{self.__min_lr}]'
                     )
                    self.__loss_count_dict[device_idx] = 0
                    if self.__ada_loss_index_dict[device_idx] < len(self.__ada_loss_dict[device_idx]) - 1:
                        self.__ada_loss_index_dict[device_idx] += 1

    def __update_learning_rates(self, current_round_dict: dict,
                                current_loss_dict: dict,
                                current_model_idx_list: list):
        if self.__if_ada_stair:
            self.__apply_ada_stair(current_round_dict, current_model_idx_list)
        if self.__if_ada_loss:
            self.__apply_ada_loss(current_loss_dict, current_round_dict, current_model_idx_list)

    def get_new_optimizer_dict(self, optimizer_dict: dict, current_round_dict: dict,
                               current_loss_dict: dict,
                               current_model_idx_list: list) -> dict:
        self.__update_learning_rates(
            current_round_dict,
            current_loss_dict,
            current_model_idx_list,
        )

        for device_idx, device_optimizer in optimizer_dict.items():
            for param_group in device_optimizer.param_groups:
                param_group['lr'] = self.__lr_dict[device_idx]

        return optimizer_dict

    def get_new_lr_dict(self, current_round_dict: dict, current_loss_dict: dict,
                        current_model_idx_list: list) -> dict:
        self.__update_learning_rates(
            current_round_dict,
            current_loss_dict,
            current_model_idx_list,
        )
        return self.__lr_dict

