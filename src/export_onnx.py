"""Export to ONNX and compare PyTorch vs ONNX Runtime (CPU).
python -m src.export_onnx --name resnet18_finetune_basic
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .utils import load_model, measure_latency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    args = p.parse_args()
    
    # ექსპერიმენტის დირექტორიისა და CPU მოწყობილობის განსაზღვრა
    run_dir = Path("runs") / args.name
    device = torch.device("cpu")
    # საუკეთესო მოდელისა და კლასების ჩატვირთვა
    model, classes, _ = load_model(run_dir / "best.pt", device)

    # დამხმარე (dummy) ტენზორი მოდელის არქიტექტურის გამოსაწერად
    dummy = torch.randn(1, 3, 224, 224)
    path = run_dir / "model.onnx"
    kw = dict(input_names=["image"], output_names=["logits"],
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        # მოდელის ექსპორტი ONNX ფორმატში (თანამედროვე PyTorch ვერსიებისთვის)
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
    except TypeError:  # ძველ PyTorch ვერსიებს არ აქვთ `dynamo` არგუმენტი
        torch.onnx.export(model, dummy, str(path), **kw)

    # ONNX Runtime ინფერენსის სესიის შექმნა CPU-სთვის
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": dummy.numpy()})[0]
    with torch.no_grad():
        # PyTorch-სა და ONNX-ს პროგნოზებს შორის მაქსიმალური აბსოლუტური სხვაობის გამოთვლა
        diff = float(np.abs(model(dummy).numpy() - ort_out).max())

    # ONNX Runtime-ის ლატენტობის (სიჩქარის) გაზომვა
    x = dummy.numpy()
    for _ in range(10): sess.run(None, {"image": x})  # მოდელის "გახურება" (Warmup)
    t = time.perf_counter()
    for _ in range(50): sess.run(None, {"image": x})
    ort_ms = (time.perf_counter() - t) / 50 * 1000

    # შედეგების ლექსიკონის შექმნა და JSON ფაილში ჩაწერა
    res = dict(max_abs_diff=diff, torch_cpu_ms=measure_latency(model, device),
               onnxruntime_cpu_ms=ort_ms, onnx_size_mb=os.path.getsize(path) / 1024 ** 2)
    (run_dir / "onnx_benchmark.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
