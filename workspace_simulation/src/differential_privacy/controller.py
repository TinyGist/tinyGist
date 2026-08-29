"""Lifecycle controller for private outgoing model views."""

import copy
from contextlib import contextmanager
from pathlib import Path
import secrets

import torch

from .accountant import PerDeviceRDPAccountant
from .model_update import privatize_model_update
from .recorder import PrivacyRecorder


class ModelUpdateDPController:
    def __init__(self, config, model_dict, log_dir):
        if not config.enabled or config.mode != "model_update":
            raise ValueError("ModelUpdateDPController requires enabled model_update config")
        self.config = config
        self._private_models = {
            device_id: copy.deepcopy(model).to("cpu")
            for device_id, model in model_dict.items()
        }
        self._accountant = PerDeviceRDPAccountant(
            config.noise_multiplier,
            config.delta,
        )
        self._recorder = PrivacyRecorder(
            output_path=Path(log_dir) / "privacy_accounting.csv",
            orders=self._accountant.orders,
        )
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(secrets.randbits(63))
        self._snapshotted_devices = set()
        self._prepared_devices = set()

    @property
    def private_models(self):
        return self._private_models

    def begin_round(self, model_dict, trainable_devices):
        trainable_devices = tuple(trainable_devices)
        self._validate_device_ids(model_dict, trainable_devices)
        for device_id in trainable_devices:
            source_model = model_dict[device_id]
            private_model = self._private_models[device_id]
            private_model.to("cpu")
            source_parameters = dict(source_model.named_parameters())
            private_parameters = dict(private_model.named_parameters())
            if source_parameters.keys() != private_parameters.keys():
                raise ValueError(
                    "Training and private models have different named parameters"
                )
            with torch.no_grad():
                for name, source_parameter in source_parameters.items():
                    if not source_parameter.requires_grad:
                        continue
                    private_parameters[name].copy_(
                        source_parameter.detach().to(
                            device="cpu",
                            dtype=private_parameters[name].dtype,
                        )
                    )
            private_model.train(source_model.training)
        self._snapshotted_devices = set(trainable_devices)
        self._prepared_devices.clear()

    def prepare_private_models(self, model_dict, trainable_devices, global_round):
        trainable_devices = tuple(trainable_devices)
        self._validate_device_ids(model_dict, trainable_devices)
        if set(trainable_devices) != self._snapshotted_devices:
            raise RuntimeError("DP round snapshot does not match current trainable devices")
        for device_id in trainable_devices:
            if device_id in self._prepared_devices:
                raise RuntimeError(f"DP update for {device_id} was already prepared this round")
            privatize_model_update(
                trained_model=model_dict[device_id],
                baseline_model=self._private_models[device_id],
                clipping_norm=self.config.clipping_norm,
                noise_multiplier=self.config.noise_multiplier,
                generator=self._generator,
            )
            privacy_cost = self._accountant.step(device_id)
            self._recorder.record(global_round, self.config, privacy_cost)
            self._prepared_devices.add(device_id)
        self._recorder.flush()
        return self._private_models

    @contextmanager
    def private_validation_model(self, device_id, runtime_device):
        if device_id not in self._prepared_devices:
            raise RuntimeError(f"Private model for {device_id} is not prepared")
        model = self._private_models[device_id]
        runtime_device = torch.device(runtime_device)
        model.to(runtime_device)
        try:
            yield model
        finally:
            model.to("cpu")

    def close(self):
        self._recorder.close()

    def privacy_costs(self):
        return tuple(
            cost
            for device_id in self._private_models
            if (cost := self._accountant.privacy_cost(device_id)).release_count > 0
        )

    def _validate_device_ids(self, model_dict, device_ids):
        missing = [
            device_id
            for device_id in device_ids
            if device_id not in model_dict or device_id not in self._private_models
        ]
        if missing:
            raise ValueError(f"Unknown differential-privacy device(s): {missing}")
