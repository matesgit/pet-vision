"""Train a classifier.  Example:
python -m src.train --name resnet18_finetune_basic --model resnet18 --augment basic --epochs 10 --lr 3e-4
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import get_loaders
from .model import build_model
from .utils import set_seed, get_device, count_params


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)
                loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="resnet18", choices=["simple_cnn", "resnet18", "mobilenet_v3_small"])
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--augment", default="basic", choices=["none", "basic", "strong"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    amp = args.amp and device.type == "cuda"
    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, classes = get_loaders(
        args.data_root, args.batch_size, args.augment, seed=args.seed)
    model = build_model(args.model, len(classes), args.pretrained, args.freeze_backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device} amp={amp} params={count_params(model):,} trainable={trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history, best_val = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, amp=amp)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        print(f"epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s")
        if va_acc > best_val:  # keep the best epoch according to VALIDATION accuracy
            best_val = va_acc
            torch.save(dict(model_state=model.state_dict(), config=vars(args),
                            classes=classes, val_acc=va_acc, epoch=epoch), out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val acc: {best_val:.4f} -> {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
