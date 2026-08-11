"""
Train the deepfake detector on your own dataset.

Expected folder layout (Kaggle-style, same as the notebook):

    <data_root>/
        train/
            real/   *.jpg *.png ...
            fake/   *.jpg *.png ...
        val/
            real/
            fake/
        test/            (optional, only used by evaluate.py-style checks)
            real/
            fake/

Usage:
    python train.py --data_root /path/to/dataset --epochs 10 --batch_size 32

Output:
    checkpoints/best_model.pt        <- load this in app.py
    checkpoints/checkpoint_epochNNN.pt
    checkpoints/history.json
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from tqdm import tqdm

from model import DeepfakeDetector


# ─────────────────────────────── Data ────────────────────────────────────
def get_train_transforms(size=224):
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomGrayscale(0.05),
        transforms.GaussianBlur(3, (0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DeepfakeDataset(Dataset):
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, root, transform=None, max_samples=None):
        self.transform = transform
        self.samples = self._scan(Path(root))
        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        counts = {0: 0, 1: 0}
        for _, l in self.samples:
            counts[l] += 1
        print(f"  {Path(root).name:8s}: {counts[0]:,} real | {counts[1]:,} fake | {len(self.samples):,} total")
        self.counts = counts

    def _scan(self, root):
        samples, lmap = [], {'real': 0, 'fake': 1, '0': 0, '1': 1, 'original': 0, 'manipulated': 1}
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {root}\n"
                f"Make sure --data_root points to the parent of train/val/test, "
                f"and each split has real/ and fake/ subfolders."
            )
        for d in root.iterdir():
            if not d.is_dir():
                continue
            label = lmap.get(d.name.lower())
            if label is None:
                continue
            for f in d.rglob('*'):
                if f.suffix.lower() in self.IMG_EXTS:
                    samples.append((str(f), label))
        return samples

    def get_weights(self):
        total = len(self.samples)
        cw = {k: total / v for k, v in self.counts.items() if v > 0}
        return [cw[l] for _, l in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────── Loss ─────────────────────────────────────
class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n = logits.size(1)
        lp = torch.log_softmax(logits, 1)
        smooth = torch.full_like(lp, self.smoothing / (n - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth * lp).sum(1).mean()


# ───────────────────────────── Epoch loop ──────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, scaler=None, device='cuda', desc=''):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits.detach(), 1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.argmax(1))
            all_probs.append(probs)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    labels_arr = np.array(all_labels)
    probs_arr = np.vstack(all_probs)
    preds_arr = np.array(all_preds)

    return {
        'loss': total_loss / len(loader),
        'acc': accuracy_score(labels_arr, preds_arr),
        'f1': f1_score(labels_arr, preds_arr, average='binary', zero_division=0),
        'auc': roc_auc_score(labels_arr, probs_arr[:, 1]) if len(np.unique(labels_arr)) > 1 else 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='Folder containing train/ and val/ (each with real/ and fake/)')
    ap.add_argument('--epochs', type=int, default=10,
                    help='Total target epoch number. When resuming, training continues from the checkpoint '
                         'epoch + 1 up to this value (e.g. resume at epoch 3 with --epochs 20 trains 4..20)')
    ap.add_argument('--resume', default=None,
                    help='Path to a checkpoint (e.g. checkpoints/checkpoint_epoch003.pt) to continue training '
                         'from. Restores model, optimizer, AMP scaler, history, best AUC and early-stopping state.')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--warmup_epochs', type=int, default=2)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--max_train', type=int, default=None, help='Cap on training samples, for a quick smoke test')
    ap.add_argument('--save_dir', default='checkpoints')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Building datasets...")
    train_ds = DeepfakeDataset(f'{args.data_root}/train', get_train_transforms(args.img_size), args.max_train)
    # Support both "val" and "valid" folder names (Kaggle dataset uses "valid")
    val_path = f'{args.data_root}/val'
    if not Path(val_path).exists():
        val_path = f'{args.data_root}/valid'
    val_ds = DeepfakeDataset(val_path, get_val_transforms(args.img_size))

    sampler = WeightedRandomSampler(train_ds.get_weights(), len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"DataLoaders ready | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = DeepfakeDetector(pretrained=True, freeze_layers=3).to(device)
    criterion = LabelSmoothCE(0.1)
    scaler = GradScaler(device='cuda', enabled=(device.type == 'cuda'))

    backbone_params = list(model.spatial_backbone.parameters())
    head_params = list(model.attention.parameters()) + list(model.freq.parameters()) + list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    history = {'train': [], 'val': []}
    best_auc = 0.0
    patience_c = 0
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        history = ckpt.get('history', history)
        best_auc = float(ckpt.get('best_auc', 0.0))
        patience_c = int(ckpt.get('patience_counter', 0))
        if start_epoch >= args.warmup_epochs + 1:
            for p in model.spatial_backbone.parameters():
                p.requires_grad = True
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch - 1} but --epochs is {args.epochs}. "
                f"Raise --epochs to continue."
            )
        print(f"Resumed from epoch {start_epoch - 1} | continuing to epoch {args.epochs}")
        print(f"  Resumed best AUC: {best_auc:.4f} | patience: {patience_c}/{args.patience}")

    print(f"Starting training on {device}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            for p in model.spatial_backbone.parameters():
                p.requires_grad = True
            print(f"  [Epoch {epoch}] Backbone unfrozen")

        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler, device, f'Train {epoch}')
        val_m = run_epoch(model, val_loader, criterion, device=device, desc=f'Val   {epoch}')

        history['train'].append(train_m)
        history['val'].append(val_m)

        elapsed = time.time() - t0
        is_best = val_m['auc'] > best_auc
        if is_best:
            best_auc = val_m['auc']
            patience_c = 0
        else:
            patience_c += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.0f}s) | "
            f"Train loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} | "
            f"Val loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} "
            f"AUC={val_m['auc']:.4f} F1={val_m['f1']:.4f}"
            + ("  BEST" if is_best else "")
        )

        ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'history': history,
                'best_auc': best_auc,
                'patience_counter': patience_c,
                'warmup_epochs': args.warmup_epochs,
                'metrics': val_m}
        torch.save(ckpt, f'{args.save_dir}/checkpoint_epoch{epoch:03d}.pt')
        if is_best:
            torch.save(ckpt, f'{args.save_dir}/best_model.pt')

        if patience_c >= args.patience:
            print(f"\nEarly stopping after epoch {epoch}")
            break

    print(f"\nTraining complete! Best Val AUC: {best_auc:.4f}")
    with open(f'{args.save_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Best checkpoint saved to: {args.save_dir}/best_model.pt")


if __name__ == '__main__':
    main()
