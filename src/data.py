"""Data loading: Oxford-IIIT Pet (37 cat/dog breeds)."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

# ImageNet statistics: required because pretrained models were trained with them.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(augment: str = "basic", img_size: int = 224):
    norm = T.Normalize(MEAN, STD)
    eval_tf = T.Compose([
        T.Resize(int(round(img_size / 0.875))),  # 224 -> 256
        T.CenterCrop(img_size),
        T.ToTensor(),
        norm,
    ])
    if augment == "none":
        train_tf = eval_tf
    elif augment == "basic":
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ])
    elif augment == "strong":
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.25),
        ])
    else:
        raise ValueError(f"unknown augment: {augment}")
    return train_tf, eval_tf


def get_loaders(root="data", batch_size=32, augment="basic", img_size=224,
                val_frac=0.15, seed=42, num_workers=2):
    """Returns train, val, test loaders and class names.

    The official 'trainval' split is divided into train/val (fixed seed).
    The official 'test' split is touched ONLY for the final evaluation.
    """
    train_tf, eval_tf = get_transforms(augment, img_size)
    kw = dict(root=root, target_types="category", download=True)
    full_train = datasets.OxfordIIITPet(split="trainval", transform=train_tf, **kw)
    full_val = datasets.OxfordIIITPet(split="trainval", transform=eval_tf, **kw)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(len(perm) * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(full_train, train_idx)   # uses train transforms
    val_ds = Subset(full_val, val_idx)         # same images split, eval transforms
    test_ds = datasets.OxfordIIITPet(split="test", transform=eval_tf, **kw)

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=num_workers, pin_memory=True)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), full_train.classes


def get_test_loader(root="data", batch_size=64, img_size=224, num_workers=2):
    _, eval_tf = get_transforms("none", img_size)
    ds = datasets.OxfordIIITPet(root=root, split="test", target_types="category",
                                transform=eval_tf, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers), ds.classes
