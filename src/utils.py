# აერთიანებს პითონის და პაიტორჩის დამხმარებელ ფუნქციებს: მოდელის ჩატვირთვას,
#რეპროდუცირებადობისთვის seed-ის დაფიქსირებას,
#პარამეტრების დათვლას, ზომის გამოთვლას მეგაბაიტებში 
import random
import time
import numpy as np
import torch
from .model import build_model


def set_seed(seed=42):
    """რანდომიზაციის საწყისი წერტილის (seed) დაფიქსირება ექსპერიმენტების რეპროდუცირებადობისთვის."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    """ავტომატურად ადგენს და აბრუნებს ხელმისაწვდომ გამოთვლით მოწყობილობას (GPU ან CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    """ითვლის და აბრუნებს მოდელის მწვრთნელი პარამეტრების (წონების) ჯამურ რაოდენობას."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """გამოთვლის მოდელის მიერ დაკავებული მეხსიერების ზომას მეგაბაიტებში (MB) პარამეტრებისა და ბუფერების გათვალისწინებით."""
    b = sum(p.numel() * p.element_size() for p in model.parameters())
    b += sum(x.numel() * x.element_size() for x in model.buffers())
    return b / 1024 ** 2


def load_model(ckpt_path, device):
    """ტვირთავს მოდელის შენახულ ჩექპოინტს (checkpoint), აწყობს არქიტექტურას და აბრუნებს მოდელს, კლასებსა და კონფიგურაციას."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg, classes = ckpt["config"], ckpt["classes"]
    model = build_model(cfg["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), classes, cfg


@torch.no_grad()
def measure_latency(model, device, img_size=224, runs=50, warmup=10):
    """ზომავს მოდელის ინფერენსის საშუალო სიჩქარეს (ლატენტობას) მილიწამებში ერთ სურათზე (batch size 1)."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    # მოდელის "გახურება" (Warmup) ქეშირებისა და ოპტიმიზაციის სტაბილიზაციისთვის
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # დროის ათვლა ბენჩმარკისთვის
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # აბრუნებს საშუალო დროს მილიწამებში
    return (time.perf_counter() - t) / runs * 1000
