import random
import time
import numpy as np
import torch
from .model import build_model


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    b = sum(p.numel() * p.element_size() for p in model.parameters())
    b += sum(x.numel() * x.element_size() for x in model.buffers())
    return b / 1024 ** 2


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg, classes = ckpt["config"], ckpt["classes"]
    model = build_model(cfg["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), classes, cfg


@torch.no_grad()
def measure_latency(model, device, img_size=224, runs=50, warmup=10):
    """Average ms per single image (batch size 1)."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000
