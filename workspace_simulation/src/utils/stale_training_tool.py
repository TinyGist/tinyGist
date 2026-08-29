import numpy as np
import logging

log = logging.getLogger(__name__)

class StaleTrainingSimulator:
    def __init__(self, device_dict: dict, sim_method="probabilistic", sim_distribution="gaussian",
                    gauss_mean=0.8, gauss_std=0.1,
                    chi_square_k = 2,
                    uniform_multiplier = 2,
                    lowest_probability=0.5,
                    highest_probability=1.0
                 ):
        self.__device_dict = device_dict
        if sim_method not in {"probabilistic", "fixed_round"}:
            raise ValueError(f"{sim_method} is not a valid sim_method")
        self.__sim_method = sim_method
        if sim_distribution not in {"gaussian", "uniform", "chi_square"}:
            raise ValueError(f"{sim_distribution} is not a valid sim_distribution")
        self.__sim_distribution = sim_distribution

        self.__gauss_mean = gauss_mean
        if gauss_std < 0:
            raise ValueError("gauss_std must be non-negative")
        self.__gauss_std = gauss_std
        if chi_square_k <= 0:
            raise ValueError("chi_square_k must be greater than 0")
        self.__chi_square_k = chi_square_k
        if uniform_multiplier <= 0:
            raise ValueError("uniform_multiplier must be greater than 0")
        self.__uniform_multiplier = uniform_multiplier

        if not 0 <= lowest_probability <= highest_probability <= 1:
            raise ValueError(
                "probability bounds must satisfy "
                "0 <= lowest_probability <= highest_probability <= 1"
            )
        self.__lowest_probability = lowest_probability
        self.__highest_probability = highest_probability

        self.__num_devices = len(self.__device_dict)
        permuted_idx_list = np.random.permutation(self.__num_devices).tolist()
        self.__idx_to_indicator_mapping = {idx:indicator for idx, indicator in zip(permuted_idx_list, list(self.__device_dict.keys()))}

        self.__to_train_probabilities = None
        self.__to_train_rounds = None


    def __get_chosen_probabilities(self):
        if self.__sim_distribution == "gaussian":
            to_train_probabilities_array = np.random.randn(self.__num_devices)
            to_train_probabilities_array = to_train_probabilities_array * self.__gauss_std
            to_train_probabilities_array = to_train_probabilities_array + self.__gauss_mean
        elif self.__sim_distribution == "uniform":
            to_train_probabilities_array = np.random.rand(self.__num_devices)
        elif self.__sim_distribution == "chi_square":
            to_train_probabilities_array = np.random.chisquare(
                self.__chi_square_k, self.__num_devices
            ) / self.__chi_square_k
        else:
            raise NotImplementedError

        to_train_probabilities_array = np.clip(
            to_train_probabilities_array,
            self.__lowest_probability,
            self.__highest_probability,
        )

        self.__to_train_probabilities = to_train_probabilities_array.tolist()

    def __get_chosen_rounds(self):
        if self.__sim_distribution == "gaussian":
            to_train_rounds_array = np.random.randn(self.__num_devices)
            to_train_rounds_array = to_train_rounds_array * self.__gauss_std
            to_train_rounds_array = to_train_rounds_array + self.__gauss_mean
        elif self.__sim_distribution == "uniform":
            to_train_rounds_array = np.random.rand(self.__num_devices) * self.__uniform_multiplier
        elif self.__sim_distribution == "chi_square":
            to_train_rounds_array = np.random.chisquare(self.__chi_square_k, self.__num_devices)
        else:
            raise NotImplementedError

        to_train_rounds_array = np.round(to_train_rounds_array)
        to_train_rounds_array[to_train_rounds_array < 1] = 1
        to_train_rounds_array = to_train_rounds_array.astype(int)

        self.__to_train_rounds = to_train_rounds_array.tolist()


    def get_current_trainable_devices(self, current_global_round):
        current_train_device_list = []
        if self.__sim_method == "probabilistic":
            if self.__to_train_probabilities is None:
                self.__get_chosen_probabilities()
            for idx in range(self.__num_devices):
                if self.__to_train_probabilities[idx] < np.random.rand():
                    continue
                current_train_device_list.append(self.__idx_to_indicator_mapping[idx])
            log.info(f"Probabilistic stale training method is used")
            log.info(
                f"Probabilities and mappings are \n {self.__to_train_probabilities}\n {self.__idx_to_indicator_mapping}")
            log.info(f"current global round is {current_global_round}")
            log.info(f"current_train_device_list: {current_train_device_list}")

        elif self.__sim_method == "fixed_round":
            if self.__to_train_rounds is None:
                self.__get_chosen_rounds()
            to_train_rounds_array = np.array(self.__to_train_rounds)
            to_train_rounds_array = current_global_round % to_train_rounds_array
            to_train_rounds_list = to_train_rounds_array.tolist()
            for idx in range(self.__num_devices):
                if to_train_rounds_list[idx] != 0:
                    continue
                current_train_device_list.append(self.__idx_to_indicator_mapping[idx])
            log.info(f"Fixed-round stale training method is used")
            log.info(
                f"Fixed-round and mappings are \n {self.__to_train_rounds}\n {self.__idx_to_indicator_mapping}")
            log.info(f"current global round is {current_global_round}")
            log.info(f"current_train_device_list: {current_train_device_list}")

        return current_train_device_list

