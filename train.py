
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS, CLASS_NAMES_PATH, DATA_DIR, IMAGE_SIZE, MODEL_PATH  # noqa: E402
from modeling import build_model  # noqa: E402


class IntFolderImageDataset(torch.utils.data.Dataset):
    """One class per subfolder named with digits; class index follows sorted(int(names))."""

    def __init__(self, root: Path, transform):
        self.root = Path(root)
        self.transform = transform
        ids = sorted(int(d.name) for d in self.root.iterdir() if d.is_dir() and d.name.isdigit())
        if not ids:
            raise ValueError(f"No class folders in {root}")
        self.folder_ids = ids
        self.idx_to_folder = {j: i for j, i in enumerate(ids)}
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples: list[tuple[str, int]] = []
        for j, fid in enumerate(ids):
            folder = self.root / str(fid)
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in exts:
                    self.samples.append((str(f), j))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        from PIL import Image

        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    if n_val >= n:
        n_val = max(0, n // 5)
    return idx[n_val:], idx[:n_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-names", type=Path, default=CLASS_NAMES_PATH)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tf_train = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full = IntFolderImageDataset(args.data_dir, tf_train)
    n = len(full)
    if n < 2:
        raise SystemExit("Not enough images to train.")

    train_idx, val_idx = split_indices(n, args.val_frac, args.seed)
    train_set = Subset(full, train_idx)
    val_full = IntFolderImageDataset(args.data_dir, tf_val)
    val_set = Subset(val_full, val_idx)

    num_classes = len(full.folder_ids)
    model = build_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    labels: list[str] = []
    if args.class_names.is_file():
        meta = json.loads(args.class_names.read_text(encoding="utf-8"))
        by_folder = {c["folder_id"]: c["label"] for c in meta["classes"]}
        labels = [by_folder.get(fid, str(fid)) for fid in full.folder_ids]
    else:
        labels = [str(i) for i in full.folder_ids]

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.avgpool.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        list(model.classifier.parameters()) + list(model.avgpool.parameters()),
        lr=args.lr,
        weight_decay=0.02,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    loss_fn = nn.CrossEntropyLoss()

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    best_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        sched.step()

        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)
        va = v_correct / max(1, v_total)
        ta = correct / max(1, total)
        print(f"epoch {epoch+1}: train_acc={ta:.3f} val_acc={va:.3f} train_loss={running/total:.4f}")
        if va >= best_val:
            best_val = va
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "folder_ids": full.folder_ids,
                    "labels": labels,
                    "image_size": IMAGE_SIZE,
                    "best_val_acc": best_val,
                },
                args.out,
            )
            print(f"  saved checkpoint -> {args.out}")

    print("Done. Best val acc:", best_val)


if __name__ == "__main__":
    main()
