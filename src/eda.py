"""Exploratory data analysis for Oxford-IIIT Pet.

Prints class balance and image-size statistics and saves three figures:
class_distribution.png, image_sizes.png, samples.png

Usage:
    python -m src.eda --data-root data --out eda
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets

# კატების ჯიშების სია
CATS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
        "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}
# ჯიშები, რომლების მონაცემებსაც განსაკუთრებულად ვაკვირდებით
WATCH = ["American Pit Bull Terrier", "Staffordshire Bull Terrier", "Ragdoll", "Birman"]

def load_stats(ds):
    """აბრუნებს torchvision მონაცემთა ნაკრების თითოეული სურათის ეტიკეტებს და (სიგანე, სიმაღლე) ზომებს."""
    sizes = []
    for path in ds._images:
        with Image.open(path) as im:
            sizes.append(im.size)
    return np.array(ds._labels), np.array(sizes)

def print_summary(name, labels, sizes, n_classes, cat_idx):
    # ითვლის თითოეული კლასის სურათების რაოდენობას
    counts = np.bincount(labels, minlength=n_classes)
    w, h = sizes[:, 0], sizes[:, 1]
    cats = counts[cat_idx].sum()
    print(f"[{name}] images={len(labels)} | per class min/median/max = "
          f"{counts.min()}/{int(np.median(counts))}/{counts.max()}")
    print(f"   cats={cats} dogs={counts.sum() - cats}")
    print(f"   width  min/median/max = {w.min()}/{int(np.median(w))}/{w.max()}")
    print(f"   height min/median/max = {h.min()}/{int(np.median(h))}/{h.max()}")
    print(f"   median aspect ratio w/h = {np.median(w / h):.2f} | "
          f"short side < 224 px: {(sizes.min(1) < 224).mean() * 100:.1f}%")
    return counts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="eda")
    args = p.parse_args()

    # შედეგების შესანახი საქაღალდის შექმნა
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Oxford-IIIT Pet მონაცემთა ნაკრების ჩატვირთვა trainval და test სექციებისთვის
    kw = dict(root=args.data_root, target_types="category", download=True)
    train = datasets.OxfordIIITPet(split="trainval", **kw)
    test = datasets.OxfordIIITPet(split="test", **kw)
    classes = train.classes
    cat_idx = [i for i, c in enumerate(classes) if c in CATS]

    # სტატისტიკის ამოღება
    y_tr, s_tr = load_stats(train)
    y_te, s_te = load_stats(test)
    c_tr = print_summary("trainval", y_tr, s_tr, len(classes), cat_idx)
    c_te = print_summary("test", y_te, s_te, len(classes), cat_idx)
    for name in WATCH:
        i = classes.index(name)
        print(f"{name}: trainval={c_tr[i]}, test={c_te[i]}")

    # 1) სურათების რაოდენობა თითოეულ კლასში (სვეტოვანი დიაგრამა)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(classes))
    ax.bar(x - 0.2, c_tr, 0.4, label="trainval")
    ax.bar(x + 0.2, c_te, 0.4, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_ylabel("images")
    ax.set_title("Images per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=130)
    plt.close(fig)

    # 2) სურათების ზომების განაწილება (სკატერპლოტი)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(s_tr[:, 0], s_tr[:, 1], s=4, alpha=0.4)
    ax.axhline(224, color="r", lw=0.8)
    ax.axvline(224, color="r", lw=0.8)
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image sizes (red lines = 224 px model input)")
    fig.tight_layout()
    fig.savefig(out / "image_sizes.png", dpi=130)
    plt.close(fig)

    # 3) შემთხვევითი სურათების ნიმუშები (ბადე)
    random.seed(0)
    idx = random.sample(range(len(train)), 12)
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for ax, i in zip(axes.flat, idx):
        img, y = train[i]
        ax.imshow(img)
        ax.set_title(classes[y], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=130)
    plt.close(fig)

    print(f"saved figures to {out}/")

if __name__ == "__main__":
    main()
