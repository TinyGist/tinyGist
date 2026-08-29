"""CUDA-aware wall-clock measurement helpers."""

import time
from contextlib import contextmanager

import torch


def synchronized_perf_counter(runtime_device) -> float:
    runtime_device = torch.device(runtime_device)
    if runtime_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(runtime_device)
    return time.perf_counter()


@contextmanager
def measure_wall_clock_stage(timings, stage_name, runtime_device):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        timings[stage_name] = timings.get(stage_name, 0.0) + (
            synchronized_perf_counter(runtime_device) - start_time
        )
