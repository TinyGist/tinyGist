import torch


def cuda_status() -> dict:
    status = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": None,
        "device_name": None,
    }
    if status["available"] and status["device_count"] > 0:
        current_device = torch.cuda.current_device()
        status["current_device"] = current_device
        status["device_name"] = torch.cuda.get_device_name(current_device)
    return status


def pick_gpu_with_most_free_mem() -> int:
    n = torch.cuda.device_count()
    assert n > 0, "Keine CUDA-GPU gefunden."
    free = []
    for i in range(n):
        free_mem, _ = torch.cuda.mem_get_info(i)
        free.append((i, free_mem))
    return max(free, key=lambda x: x[1])[0]


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        gpu = pick_gpu_with_most_free_mem()
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")
