"""Per-sample gradient score kernels shared by score-buffer orchestration."""

import torch


def gradient_score_tensor(grad_tensor, score_transform):
    if score_transform == "abs":
        return grad_tensor.abs()
    if score_transform == "square":
        return grad_tensor.square()
    raise ValueError(
        f"Unsupported per-sample gradient score transform: {score_transform}"
    )


def per_sample_gradient_score_sum(grad_tensor, score_transform):
    return (
        gradient_score_tensor(grad_tensor.detach(), score_transform)
        .sum(dim=0)
        .to(dtype=torch.float32)
    )


def _trainable_named_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _empty_score_sums(named_params, score_transforms):
    return {
        score_transform: {
            name: torch.zeros_like(parameter.detach(), dtype=torch.float32)
            for name, parameter in named_params
        }
        for score_transform in score_transforms
    }


def compute_per_sample_gradient_score_sums_vmap(
        model,
        data,
        labels,
        loss_function,
        score_transform,
):
    return compute_per_sample_gradient_multi_score_sums_vmap(
        model,
        data,
        labels,
        loss_function,
        (score_transform,),
    )[score_transform]


def compute_per_sample_gradient_multi_score_sums_vmap(
        model,
        data,
        labels,
        loss_function,
        score_transforms,
):
    score_transforms = tuple(dict.fromkeys(score_transforms))
    named_params = _trainable_named_parameters(model)
    if not named_params:
        return {score_transform: {} for score_transform in score_transforms}

    params = {name: parameter for name, parameter in named_params}
    buffers = dict(model.named_buffers())

    def loss_one(functional_params, functional_buffers, sample, target):
        outputs = torch.func.functional_call(
            model,
            (functional_params, functional_buffers),
            (sample.unsqueeze(0),),
        )
        loss = loss_function(outputs, target.unsqueeze(0))
        return loss.mean() if loss.ndim != 0 else loss

    grad_one = torch.func.grad(loss_one, argnums=0)
    parameter_bytes = sum(
        parameter.numel() * max(parameter.element_size(), 4)
        for parameter in params.values()
    )
    chunk_size = min(int(data.shape[0]), 32)
    if data.device.type == "cuda" and parameter_bytes > 0:
        free_bytes, _ = torch.cuda.mem_get_info(data.device)
        memory_limited_chunk = max(
            1,
            int(free_bytes * 0.1) // max(parameter_bytes * 3, 1),
        )
        chunk_size = min(chunk_size, memory_limited_chunk)

    score_sums = _empty_score_sums(named_params, score_transforms)
    chunk_start = 0
    while chunk_start < int(data.shape[0]):
        chunk_end = min(chunk_start + chunk_size, int(data.shape[0]))
        try:
            per_sample_grads = torch.func.vmap(
                grad_one,
                in_dims=(None, None, 0, 0),
            )(
                params,
                buffers,
                data[chunk_start:chunk_end],
                labels[chunk_start:chunk_end],
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            if data.device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        for name, grad_tensor in per_sample_grads.items():
            for score_transform in score_transforms:
                score_sums[score_transform][name].add_(
                    per_sample_gradient_score_sum(
                        grad_tensor,
                        score_transform,
                    )
                )
        chunk_start = chunk_end
    return score_sums


def compute_per_sample_gradient_score_sums_loop(
        model,
        data,
        labels,
        loss_function,
        score_transform,
):
    return compute_per_sample_gradient_multi_score_sums_loop(
        model,
        data,
        labels,
        loss_function,
        (score_transform,),
    )[score_transform]


def compute_per_sample_gradient_multi_score_sums_loop(
        model,
        data,
        labels,
        loss_function,
        score_transforms,
):
    score_transforms = tuple(dict.fromkeys(score_transforms))
    named_params = _trainable_named_parameters(model)
    if not named_params:
        return {score_transform: {} for score_transform in score_transforms}

    names = [name for name, _ in named_params]
    params = [parameter for _, parameter in named_params]
    score_sums = _empty_score_sums(named_params, score_transforms)
    for sample_idx in range(int(data.shape[0])):
        outputs = model(data[sample_idx:sample_idx + 1])
        loss = loss_function(outputs, labels[sample_idx:sample_idx + 1])
        if loss.ndim != 0:
            loss = loss.mean()
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            allow_unused=True,
        )
        for name, grad in zip(names, grads):
            if grad is None:
                continue
            for score_transform in score_transforms:
                score_sums[score_transform][name].add_(
                    gradient_score_tensor(
                        grad.detach(),
                        score_transform,
                    ).to(
                        device=score_sums[score_transform][name].device,
                        dtype=torch.float32,
                    )
                )
    return score_sums
