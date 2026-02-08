import os
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image


DATA_ROOT = Path(__file__).resolve().parent 


CLASS_NAMES = ["A", "B", "C"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CFG:
    out_dir: str = "./resnet50_abc_out"
    epochs: int = 2
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    img_size: int = 224
    seed: int = 42

    val_ratio: float = 0.2

    best_name: str = "best_resnet50_abc.pt"
    last_name: str = "last_resnet50_abc.pt"
    ckpt_name: str = "checkpoint_latest.pt"


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def scan_abc_samples(root: Path):

    samples = []
    for cname in CLASS_NAMES:
        cdir = root / cname
        if not cdir.exists():
            continue
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append((p, CLASS_TO_IDX[cname]))
    return samples


class PathLabelDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  
        self.transform = transform
        self.classes = CLASS_NAMES
        self.targets = [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_transforms(cfg: CFG):
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        #transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3), value='random'),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


def build_loaders(cfg: CFG):
 
    train_tf, val_tf = build_transforms(cfg)

    train_root = DATA_ROOT / "train"
    val_root = DATA_ROOT / "val"

    if train_root.exists() and val_root.exists():
        train_samples = scan_abc_samples(train_root)
        val_samples = scan_abc_samples(val_root)

        if len(train_samples) == 0:
            raise FileNotFoundError(f"No images found in: {train_root}/A,B,C")
        if len(val_samples) == 0:
            raise FileNotFoundError(f"No images found in: {val_root}/A,B,C")

        train_ds = PathLabelDataset(train_samples, transform=train_tf)
        val_ds = PathLabelDataset(val_samples, transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True
        )
        return train_ds, val_ds, train_loader, val_loader

    all_samples = scan_abc_samples(DATA_ROOT)
    if len(all_samples) == 0:
        raise FileNotFoundError(
            f"No images found.\n"
            f"Expected either:\n"
            f"  {DATA_ROOT}/train/A,B,C and {DATA_ROOT}/val/A,B,C\n"
            f"or:\n"
            f"  {DATA_ROOT}/A,B,C\n"
        )

    full_ds = PathLabelDataset(all_samples, transform=None)  

    total_len = len(full_ds)
    val_len = max(1, int(total_len * cfg.val_ratio))
    train_len = total_len - val_len
    if train_len <= 0:
        raise ValueError(f"Too few images ({total_len}) for val_ratio={cfg.val_ratio}")

    generator = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(full_ds, [train_len, val_len], generator=generator)

    train_samples = [all_samples[i] for i in train_subset.indices]
    val_samples = [all_samples[i] for i in val_subset.indices]

    train_ds = PathLabelDataset(train_samples, transform=train_tf)
    val_ds = PathLabelDataset(val_samples, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_ds, val_ds, train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return loss_sum / max(total, 1), correct / max(total, 1)


def save_best(model, cfg, best_path, best_acc, val_acc):
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "classes": CLASS_NAMES,
            "img_size": cfg.img_size
        }, best_path)
        print(f"[SAVE:BEST] {best_path} (acc={best_acc:.4f})")
    return best_acc


def save_last(model, cfg, last_path, best_acc):
    torch.save({
        "model_state": model.state_dict(),
        "classes": CLASS_NAMES,
        "img_size": cfg.img_size,
        "best_acc": float(best_acc),
    }, last_path)
    print(f"[SAVE:LAST] {last_path}")


def save_checkpoint(model, opt, scheduler, epoch, best_acc, cfg, ckpt_path):
    torch.save({
        "epoch": int(epoch),
        "best_acc": float(best_acc),
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "classes": CLASS_NAMES,
        "img_size": cfg.img_size,
        "cfg": cfg.__dict__,
    }, ckpt_path)
    print(f"[SAVE:CKPT] {ckpt_path} (epoch={epoch})")


def main():
    print("RUNNING FILE:", Path(__file__).resolve()) 
    print("DATA_ROOT:", DATA_ROOT)

    cfg = CFG()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    out_dir = Path(cfg.out_dir)
    best_path = out_dir / cfg.best_name
    last_path = out_dir / cfg.last_name
    ckpt_path = out_dir / cfg.ckpt_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg)

    print("classes (MUST be ['A','B','C']):", train_ds.classes)
    print("train_len:", len(train_ds), "val_len:", len(val_ds))

    num_classes = 3

    targets = torch.tensor(train_ds.targets, dtype=torch.long)
    class_counts = torch.bincount(targets, minlength=num_classes)
    class_weights = (1.0 / class_counts.float().clamp(min=1)).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    print("class_counts(train):", class_counts.tolist())     
    print("class_weights:", class_weights.detach().cpu().tolist())
    print("unique targets:", sorted(set(train_ds.targets)))   

    # ResNet50 pretrained
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()), lr=float(opt.param_groups[0]["lr"]))

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[VAL] loss={val_loss:.4f} acc={val_acc:.4f}")

        best_acc = save_best(model, cfg, best_path, best_acc, val_acc)
        save_checkpoint(model, opt, scheduler, epoch, best_acc, cfg, ckpt_path)

    save_last(model, cfg, last_path, best_acc)
    print("DONE best_acc:", best_acc)


if __name__ == "__main__":
    main()
