"""Evaluate a checkpoint on the official TEST split.
python -m src.evaluate --name resnet18_finetune_basic
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from .data import get_test_loader, MEAN, STD
from .utils import get_device, load_model, count_params, model_size_mb, measure_latency


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data-root", default="data")
    args = p.parse_args()

    device = get_device()
    run_dir = Path("runs") / args.name
    model, classes, cfg = load_model(run_dir / "best.pt", device)
    loader, _ = get_test_loader(args.data_root)

    y_true, y_pred, top5_hits, worst = [], [], 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = model(x).softmax(1)
        conf, pred = probs.max(1)
        top5 = probs.topk(5, dim=1).indices
        top5_hits += (top5 == y[:, None]).any(1).sum().item()
        y_true += y.tolist(); y_pred += pred.tolist()
        # keep the 12 most confident mistakes for error analysis
        for i in (pred != y).nonzero().flatten().tolist():
            worst.append((conf[i].item(), x[i].cpu(), y[i].item(), pred[i].item()))
        worst = sorted(worst, key=lambda t: -t[0])[:12]

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    top5_acc = top5_hits / len(y_true)

    # confusion matrix + most confused pairs
    cm = confusion_matrix(y_true, y_pred)
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = [(int(off[i, j]), classes[i], classes[j])
             for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None)[:10], off.shape))]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized confusion matrix")
    fig.tight_layout(); fig.savefig(run_dir / "confusion_matrix.png", dpi=130); plt.close(fig)

    # confident mistakes grid
    if worst:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax, (conf, x, t, pr) in zip(axes.flat, worst):
            img = (x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)
            ax.imshow(img.permute(1, 2, 0)); ax.axis("off")
            ax.set_title(f"true: {classes[t]}\npred: {classes[pr]} ({conf:.2f})", fontsize=8)
        fig.tight_layout(); fig.savefig(run_dir / "worst_errors.png", dpi=110); plt.close(fig)

    per_class = {classes[i]: float(cm[i, i] / cm[i].sum()) for i in range(len(classes))}
    hardest = sorted(per_class.items(), key=lambda kv: kv[1])[:5]

    metrics = dict(
        name=args.name, config=cfg, test_acc=acc, test_top5_acc=top5_acc, test_macro_f1=f1,
        best_val_acc=None, params_m=count_params(model) / 1e6, size_mb=model_size_mb(model),
        latency_ms_bs1=measure_latency(model, device), device=str(device),
        most_confused_pairs=pairs, hardest_classes=hardest)
    hist = run_dir / "history.json"
    if hist.exists():
        metrics["best_val_acc"] = max(h["val_acc"] for h in json.loads(hist.read_text()))
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=3))

    print(f"[{args.name}] test acc {acc:.4f} | top-5 {top5_acc:.4f} | macro-F1 {f1:.4f}")
    print("most confused (count, true, predicted):", pairs[:5])
    print("hardest classes:", hardest)


if __name__ == "__main__":
    main()
