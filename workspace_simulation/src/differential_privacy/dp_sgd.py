"""Per-sample, whole-model DP-SGD gradient processing."""

from collections import OrderedDict

import torch


PER_SAMPLE_GRAD_CHUNK_SIZE = 8
_BATCH_NORM = torch.nn.modules.batchnorm._BatchNorm


def freeze_batch_norm_statistics(model):
    """Keep BN affine parameters trainable while freezing private batch statistics."""
    for module in model.modules():
        if isinstance(module, _BATCH_NORM):
            module.eval()


def _trainable_parameters(model):
    parameters = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("DP-SGD requires at least one trainable parameter")
    devices = {parameter.device for parameter in parameters.values()}
    if len(devices) != 1:
        raise ValueError("DP-SGD requires all trainable parameters on one device")
    return parameters


def _accumulate_clipped_grads(
        clipped_sums,
        per_sample_grads,
        clipping_norm,
):
    batch_size = next(iter(per_sample_grads.values())).shape[0]
    squared_norms = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=next(iter(per_sample_grads.values())).device,
    )
    for gradient in per_sample_grads.values():
        squared_norms.add_(
            gradient.detach().to(dtype=torch.float32)
            .reshape(batch_size, -1)
            .square()
            .sum(dim=1)
        )
    factors = (
        float(clipping_norm)
        / squared_norms.sqrt().clamp_min(1e-12)
    ).clamp(max=1.0)
    for name, gradient in per_sample_grads.items():
        factor_shape = (batch_size,) + (1,) * (gradient.ndim - 1)
        clipped_sums[name].add_(
            (
                gradient.detach().to(dtype=torch.float32)
                * factors.reshape(factor_shape)
            ).sum(dim=0)
        )


def _vectorized_per_sample_grads(
        model,
        loss_function,
        parameters,
        data,
        labels,
        clipped_sums,
        clipping_norm,
):
    static_state = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    static_state.update(dict(model.named_buffers()))

    def single_sample_loss(trainable, sample, label):
        state = dict(static_state)
        state.update(trainable)
        output = torch.func.functional_call(
            model,
            state,
            (sample.unsqueeze(0),),
        )
        loss = loss_function(output, label.unsqueeze(0)).mean()
        return loss, output.squeeze(0)

    grad_and_value = torch.func.grad_and_value(
        single_sample_loss,
        has_aux=True,
    )
    batched_grad_and_value = torch.func.vmap(
        grad_and_value,
        in_dims=(None, 0, 0),
        randomness="different",
    )
    losses = []
    outputs = []
    for beginning in range(0, int(labels.shape[0]), PER_SAMPLE_GRAD_CHUNK_SIZE):
        ending = min(beginning + PER_SAMPLE_GRAD_CHUNK_SIZE, int(labels.shape[0]))
        gradients, (chunk_losses, chunk_outputs) = batched_grad_and_value(
            parameters,
            data[beginning:ending],
            labels[beginning:ending],
        )
        _accumulate_clipped_grads(
            clipped_sums,
            gradients,
            clipping_norm,
        )
        losses.append(chunk_losses.detach())
        outputs.append(chunk_outputs.detach())
    return torch.cat(outputs, dim=0), torch.cat(losses, dim=0).mean()


def _loop_per_sample_grads(
        model,
        loss_function,
        parameters,
        data,
        labels,
        clipped_sums,
        clipping_norm,
):
    losses = []
    outputs = []
    parameter_values = tuple(parameters.values())
    for index in range(int(labels.shape[0])):
        output = model(data[index:index + 1])
        loss = loss_function(
            output,
            labels[index:index + 1],
        ).mean()
        gradients = torch.autograd.grad(
            loss,
            parameter_values,
            allow_unused=True,
        )
        gradient_dict = OrderedDict(
            (
                name,
                torch.zeros_like(parameter).unsqueeze(0)
                if gradient is None
                else gradient.unsqueeze(0)
            )
            for (name, parameter), gradient in zip(parameters.items(), gradients)
        )
        _accumulate_clipped_grads(
            clipped_sums,
            gradient_dict,
            clipping_norm,
        )
        outputs.append(output.detach())
        losses.append(loss.detach())
    return torch.cat(outputs, dim=0), torch.stack(losses).mean()


def private_optimizer_step(
        model,
        optimizer,
        loss_function,
        data,
        labels,
        *,
        clipping_norm,
        noise_multiplier,
        expected_batch_size,
        generator,
        force_fallback=False,
):
    parameters = _trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    clipped_sums = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in parameters.items()
    }
    used_fallback = False
    fallback_reason = None
    outputs = None
    loss = None
    if data is not None and force_fallback:
        outputs, loss = _loop_per_sample_grads(
            model,
            loss_function,
            parameters,
            data,
            labels,
            clipped_sums,
            clipping_norm,
        )
        used_fallback = True
    elif data is not None:
        try:
            outputs, loss = _vectorized_per_sample_grads(
                model,
                loss_function,
                parameters,
                data,
                labels,
                clipped_sums,
                clipping_norm,
            )
        except (RuntimeError, NotImplementedError) as exc:
            clipped_sums = {
                name: torch.zeros_like(parameter, dtype=torch.float32)
                for name, parameter in parameters.items()
            }
            outputs, loss = _loop_per_sample_grads(
                model,
                loss_function,
                parameters,
                data,
                labels,
                clipped_sums,
                clipping_norm,
            )
            used_fallback = True
            fallback_reason = str(exc)

    noise_std = float(clipping_norm) * float(noise_multiplier)
    denominator = float(expected_batch_size)
    for name, parameter in parameters.items():
        noise = torch.randn(
            parameter.shape,
            dtype=torch.float32,
            device=parameter.device,
            generator=generator,
        ).mul_(noise_std)
        parameter.grad = (
            (clipped_sums[name] + noise)
            / denominator
        ).to(dtype=parameter.dtype)
    optimizer.step()
    return outputs, loss, used_fallback, fallback_reason
