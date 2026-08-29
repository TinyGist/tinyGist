"""Whole-model update clipping and Gaussian perturbation."""

import torch

from .config import MODEL_UPDATE_REPLACE_ONE_SENSITIVITY_FACTOR


def privatize_model_update(
        trained_model,
        baseline_model,
        clipping_norm,
        noise_multiplier,
        generator,
):
    """Overwrite ``baseline_model`` with baseline + clipped_delta + noise.

    Only trainable named parameters participate. This includes BatchNorm affine
    parameters and intentionally excludes running-stat buffers. The mechanism
    uses replace-one device-update adjacency, whose L2 sensitivity is ``2C``
    after clipping each possible update to the radius-``C`` ball.
    """
    trained_parameters = dict(trained_model.named_parameters())
    baseline_parameters = dict(baseline_model.named_parameters())
    if trained_parameters.keys() != baseline_parameters.keys():
        raise ValueError("Trained and baseline models have different named parameters")

    deltas = []
    parameter_count = 0
    squared_norm = torch.zeros((), dtype=torch.float64, device="cpu")
    for name, trained_parameter in trained_parameters.items():
        baseline_parameter = baseline_parameters[name]
        if not trained_parameter.requires_grad:
            continue
        if trained_parameter.shape != baseline_parameter.shape:
            raise ValueError(f"Parameter shape changed for {name}")
        delta = (
            trained_parameter.detach().to(device="cpu", dtype=torch.float32)
            - baseline_parameter.detach().to(device="cpu", dtype=torch.float32)
        )
        deltas.append((baseline_parameter, delta))
        parameter_count += delta.numel()
        squared_norm += delta.square().sum(dtype=torch.float64)

    if parameter_count == 0:
        raise ValueError("Differential privacy requires at least one trainable parameter")
    update_norm = float(squared_norm.sqrt().item())
    clip_factor = min(1.0, float(clipping_norm) / max(update_norm, 1e-12))
    noise_std = (
        MODEL_UPDATE_REPLACE_ONE_SENSITIVITY_FACTOR
        * float(clipping_norm)
        * float(noise_multiplier)
    )

    with torch.no_grad():
        for baseline_parameter, delta in deltas:
            noise = torch.randn(
                delta.shape,
                dtype=torch.float32,
                device="cpu",
                generator=generator,
            ).mul_(noise_std)
            private_value = (
                baseline_parameter.detach().to(device="cpu", dtype=torch.float32)
                + delta.mul(clip_factor)
                + noise
            )
            baseline_parameter.copy_(
                private_value.to(
                    device=baseline_parameter.device,
                    dtype=baseline_parameter.dtype,
                )
            )
