"""Lifecycle controller for local, sample-level DP-SGD."""

from pathlib import Path
import logging
import secrets

import torch

from .accountant import PerDeviceSampledGaussianAccountant
from .dp_sgd import freeze_batch_norm_statistics, private_optimizer_step
from .poisson import build_poisson_dataloader
from .recorder import PrivacyRecorder


log = logging.getLogger(__name__)


class LocalDPSGDController:
    def __init__(self, config, model_dict, log_dir, *, steps_per_epoch):
        if not config.enabled or config.mode != "local_dp_sgd":
            raise ValueError("LocalDPSGDController requires enabled local_dp_sgd config")
        if (
                isinstance(steps_per_epoch, bool)
                or not isinstance(steps_per_epoch, int)
                or steps_per_epoch <= 0
        ):
            raise ValueError("DP-SGD steps_per_epoch must be a positive integer")
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        self._models = model_dict
        self._accountant = PerDeviceSampledGaussianAccountant(
            config.noise_multiplier,
            config.delta,
        )
        self._recorder = PrivacyRecorder(
            output_path=Path(log_dir) / "privacy_accounting.csv",
            orders=self._accountant.orders,
        )
        self._loaders = {}
        self._loader_sources = {}
        self._samplers = {}
        self._sampling_generators = {}
        self._noise_generators = {}
        self._active_devices = set()
        self._round_steps = {}
        self._round_prepared = False
        self._fallback_warned = False
        self._fallback_devices = set()

    def _sampling_generator(self, device_id):
        device_id = str(device_id)
        generator = self._sampling_generators.get(device_id)
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(secrets.randbits(63))
            self._sampling_generators[device_id] = generator
        return generator

    def _noise_generator(self, runtime_device):
        runtime_device = torch.device(runtime_device)
        key = str(runtime_device)
        generator = self._noise_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=runtime_device)
            generator.manual_seed(secrets.randbits(63))
            self._noise_generators[key] = generator
        return generator

    def training_dataloader(self, device_id, base_loader):
        if (
                device_id not in self._loaders
                or self._loader_sources.get(device_id) is not base_loader
        ):
            loader, sampler = build_poisson_dataloader(
                base_loader,
                steps_per_epoch=self.steps_per_epoch,
                generator=self._sampling_generator(device_id),
            )
            self._loaders[device_id] = loader
            self._loader_sources[device_id] = base_loader
            self._samplers[device_id] = sampler
        return self._loaders[device_id]

    def begin_round(self, model_dict, trainable_devices):
        self._validate_device_ids(model_dict, trainable_devices)
        self._active_devices = set(trainable_devices)
        self._round_steps = {
            device_id: 0
            for device_id in trainable_devices
        }
        self._round_prepared = False

    @staticmethod
    def prepare_model_for_training(model):
        model.train()
        freeze_batch_norm_statistics(model)

    def private_step(
            self,
            device_id,
            model,
            optimizer,
            loss_function,
            data,
            labels,
    ):
        if device_id not in self._active_devices:
            raise RuntimeError(f"Device {device_id} is not active in this DP round")
        sampler = self._samplers.get(device_id)
        if sampler is None:
            raise RuntimeError(f"DP-SGD DataLoader for {device_id} is not prepared")
        if data is None:
            runtime_device = next(model.parameters()).device
        else:
            runtime_device = data.device
        outputs, loss, used_fallback, fallback_reason = private_optimizer_step(
            model,
            optimizer,
            loss_function,
            data,
            labels,
            clipping_norm=self.config.clipping_norm,
            noise_multiplier=self.config.noise_multiplier,
            expected_batch_size=sampler.expected_batch_size,
            generator=self._noise_generator(runtime_device),
            force_fallback=device_id in self._fallback_devices,
        )
        if used_fallback:
            self._fallback_devices.add(device_id)
        self._round_steps[device_id] += 1
        if used_fallback and not self._fallback_warned:
            log.warning(
                "DP-SGD torch.func vectorization is unsupported by this model; "
                "using the slower per-sample fallback. First reason: %s",
                fallback_reason,
            )
            self._fallback_warned = True
        return outputs, loss

    def prepare_private_models(self, model_dict, trainable_devices, global_round):
        self._validate_device_ids(model_dict, trainable_devices)
        if set(trainable_devices) != self._active_devices:
            raise RuntimeError("DP-SGD round devices do not match trainable devices")
        if self._round_prepared:
            raise RuntimeError("DP-SGD round was already finalized")
        for device_id in trainable_devices:
            sampler = self._samplers.get(device_id)
            if sampler is None:
                raise RuntimeError(f"DP-SGD DataLoader for {device_id} is not prepared")
            privacy_cost = self._accountant.step(
                device_id,
                sampler.sample_rate,
                steps=self._round_steps[device_id],
            )
            self._recorder.record(
                global_round,
                self.config,
                privacy_cost,
                steps_this_round=self._round_steps[device_id],
                sample_rate=sampler.sample_rate,
                expected_batch_size=sampler.expected_batch_size,
                dataset_size=sampler.dataset_size,
            )
        self._recorder.flush()
        self._round_prepared = True
        return model_dict


    def close(self):
        self._recorder.close()

    def privacy_costs(self):
        return tuple(
            cost
            for device_id in self._models
            if (cost := self._accountant.privacy_cost(device_id)).release_count > 0
        )

    def _validate_device_ids(self, model_dict, device_ids):
        missing = [
            device_id
            for device_id in device_ids
            if device_id not in model_dict or device_id not in self._models
        ]
        if missing:
            raise ValueError(f"Unknown differential-privacy device(s): {missing}")
