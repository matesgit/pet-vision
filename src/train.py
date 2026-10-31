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


# ერთი ფუნქცია ორივე შემთხვევისთვის: თუ optimizer გადაეცა, ვსწავლობთ (train), თუ არა, მხოლოდ ვზომავთ (validation)
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    # train რეჟიმში Dropout და BatchNorm სწავლობენ, eval რეჟიმში გამორთულია და შედეგი სტაბილურია
    model.train(training)
    loss_sum, correct, n = 0.0, 0, 0
    # validation-ზე გრადიენტები არ გვჭირდება: უფრო სწრაფია, მეხსიერებას ზოგავს და წონები არ იცვლება
    with torch.set_grad_enabled(training):
        for x, y in loader:
            # non_blocking: ფოტოების GPU-ზე გადატანა სწრაფდება
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # autocast = mixed precision: GPU-ზე გამოთვლა უფრო სწრაფია (float16)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)                   # პროგნოზი: ქულა თითო ჯიშზე
                loss = criterion(out, y)         # რამდენად ცდება მოდელი
            if training:
                # გრადიენტები ნაგულისხმევად ჯამდება, ამიტომ ყოველ ბიჯზე ვასუფთავებთ
                optimizer.zero_grad(set_to_none=True)
                # float16-ში პატარა გრადიენტები ნულად შეიძლება იქცეს, ამიტომ loss-ს ჯერ ვზრდით (scale)
                scaler.scale(loss).backward()    # backward: რომელმა წონამ რა წილად შეცდომა დაუშვა
                scaler.step(optimizer)           # წონების გასწორება
                scaler.update()
            # loss-ს ბეჩის ზომაზე ვამრავლებთ, რომ ბოლო (პატარა) ბეჩიც სწორად ჩაითვალოს
            loss_sum += loss.item() * x.size(0)
            # პროგნოზი = ის ჯიში, რომლის ქულაც ყველაზე მაღალია
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


def main():
    # პარამეტრებს ბრძანებიდან ვიღებთ, რომ ექსპერიმენტის შესაცვლელად კოდის შეცვლა არ დაგვჭირდეს
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

    set_seed(args.seed)                          # ფიქსირებული seed: ექსპერიმენტი ზუსტად მეორდება
    device = get_device()
    amp = args.amp and device.type == "cuda"     # mixed precision მხოლოდ GPU-ზე გამოგვადგება
    out_dir = Path("runs") / args.name           # ყოველ ექსპერიმენტს თავისი საქაღალდე აქვს
    out_dir.mkdir(parents=True, exist_ok=True)

    # test ნაკრებს აქ არ ვიღებთ (_): ტრენინგისას მას არ უნდა შევეხოთ
    train_loader, val_loader, _, classes = get_loaders(
        args.data_root, args.batch_size, args.augment, seed=args.seed)
    model = build_model(args.model, len(classes), args.pretrained, args.freeze_backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device} amp={amp} params={count_params(model):,} trainable={trainable:,}")

    criterion = nn.CrossEntropyLoss()
    # optimizer-ს მხოლოდ სასწავლ პარამეტრებს ვაძლევთ (linear probe-ში გაყინულები გამოირიცხება);
    # weight_decay დიდი წონების დაჯარიმებით overfitting-ს ამცირებს
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    # learning rate ეტაპობრივად მცირდება ნულთან ახლოს, რომ ბოლოს წონები წვრილად დაიხვეწოს
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history, best_val = [], 0.0                  # history: გრაფიკებისთვის (curves.png)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, amp=amp)
        scheduler.step()                         # learning rate-ის შემცირება ერთხელ ყოველ epoch-ზე
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        print(f"epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s")
        if va_acc > best_val:  # keep the best epoch according to VALIDATION accuracy
            best_val = va_acc
            # ვინახავთ წონებს და კონფიგურაციას და ჯიშების სიას, რომ მოდელი მოგვიანებით ზუსტად აღვადგინოთ
            torch.save(dict(model_state=model.state_dict(), config=vars(args),
                            classes=classes, val_acc=va_acc, epoch=epoch), out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val acc: {best_val:.4f} -> {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
