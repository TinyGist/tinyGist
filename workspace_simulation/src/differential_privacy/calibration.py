"""Whole-model norm measurements used for non-private clipping calibration."""

from collections import OrderedDict

import torch


PER_SAMPLE_GRAD_CHUNK_SIZE = 8


def _trainable_parameters(model):
    parameters = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("Clipping calibration requires trainable parameters")
    devices = {parameter.device for parameter in parameters.values()}
    if len(devices) != 1:
        raise ValueError("All trainable parameters must be on one device")
    return parameters


def _whole_model_norms(per_sample_grads):
    first = next(iter(per_sample_grads.values()))
    batch_size = int(first.shape[0])
    squared_norms = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=first.device,
    )
    for gradient in per_sample_grads.values():
        squared_norms.add_(
            gradient.detach().to(dtype=torch.float32)
            .reshape(batch_size, -1)
            .square()
            .sum(dim=1)
        )
    return squared_norms.sqrt()


def _vectorized_per_sample_gradient_norms(
        model,
        loss_function,
        parameters,
        data,
        labels,
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
        return loss_function(output, label.unsqueeze(0)).mean()

    batched_grad = torch.func.vmap(
        torch.func.grad(single_sample_loss),
        in_dims=(None, 0, 0),
        randomness="different",
    )
    norm_parts = []
    for beginning in range(0, int(labels.shape[0]), PER_SAMPLE_GRAD_CHUNK_SIZE):
        ending = min(beginning + PER_SAMPLE_GRAD_CHUNK_SIZE, int(labels.shape[0]))
        gradients = batched_grad(
            parameters,
            data[beginning:ending],
            labels[beginning:ending],
        )
        norm_parts.append(_whole_model_norms(gradients))
    return torch.cat(norm_parts)


def _loop_per_sample_gradient_norms(
        model,
        loss_function,
        parameters,
        data,
        labels,
):
    parameter_items = tuple(parameters.items())
    parameter_values = tuple(parameters.values())
    norms = []
    for index in range(int(labels.shape[0])):
        output = model(data[index:index + 1])
        loss = loss_function(output, labels[index:index + 1]).mean()
        gradients = torch.autograd.grad(
            loss,
            parameter_values,
            allow_unused=True,
        )
        gradient_dict = OrderedDict(
            (
                name,
                (
                    torch.zeros_like(parameter).unsqueeze(0)
                    if gradient is None
                    else gradient.unsqueeze(0)
                ),
            )
            for (name, parameter), gradient in zip(parameter_items, gradients)
        )
        norms.append(_whole_model_norms(gradient_dict)[0])
    return torch.stack(norms)


def per_sample_gradient_l2_norms(
        model,
        loss_function,
        data,
        labels,
        *,
        force_fallback=False,
):
    """Return one whole-trainable-model gradient norm per input sample."""
    if data is None or labels is None or int(labels.shape[0]) == 0:
        return torch.empty(0, dtype=torch.float32), False, None
    parameters = _trainable_parameters(model)
    if force_fallback:
        return (
            _loop_per_sample_gradient_norms(
                model,
                loss_function,
                parameters,
                data,
                labels,
            ).detach(),
            True,
            None,
        )
    try:
        norms = _vectorized_per_sample_gradient_norms(
            model,
            loss_function,
            parameters,
            data,
            labels,
        )
        return norms.detach(), False, None
    except (RuntimeError, NotImplementedError) as exc:
        norms = _loop_per_sample_gradient_norms(
            model,
            loss_function,
            parameters,
            data,
            labels,
        )
        return norms.detach(), True, str(exc)


def snapshot_trainable_parameters(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def model_update_l2_norm(model, baseline_parameters):
    """Measure one whole-model trainable update without including buffers."""
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if current.keys() != baseline_parameters.keys():
        raise ValueError("Model and baseline contain different trainable parameters")
    squared_norm = torch.zeros(
        (),
        dtype=torch.float32,
        device=next(iter(current.values())).device,
    )
    for name, parameter in current.items():
        baseline = baseline_parameters[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        squared_norm.add_(
            parameter.detach().to(dtype=torch.float32)
            .sub(baseline.to(dtype=torch.float32))
            .square()
            .sum()
        )
    return squared_norm.sqrt()


def trainable_parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
