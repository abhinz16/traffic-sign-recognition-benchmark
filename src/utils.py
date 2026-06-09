import os
import random
from pathlib import Path

import numpy as np
import torch


def project_path(*parts):
    root = Path(__file__).resolve().parents[1]
    return root.joinpath(*parts)


def make_dirs():
    for folder in ["data", "figures", "outputs", "outputs/failure_cases", "models"]:
        project_path(folder).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cpu_workers(leave_free=3):
    total = os.cpu_count() or 1
    return max(0, total - int(leave_free))


def worker_count(requested="auto", leave_free=3):
    available = cpu_workers(leave_free)
    if requested == "auto" or requested is None:
        return available
    return max(0, min(int(requested), available))


def configure_cpu_threads(leave_free=3):
    n_threads = max(1, cpu_workers(leave_free))
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, n_threads)))
    except RuntimeError:
        pass
    return n_threads
