"""
Score a trained checkpoint on a held-out test set.

Expected folder layout (same as train.py):

    <data_root>/
        test/            (falls back to val/ or valid/ if no test/)
            real/
            fake/

Usage:
    python evaluate.py --data_root data/real_vs_fake/real-vs-fake \
                       --checkpoint checkpoints/best_model.pt

Output:
    Accuracy, ROC-AUC, F1, per-class precision/recall, and a handful of
    example predictions (filename -> verdict + confidence) to eyeball.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DeepfakeDetector
from train import DeepfakeDataset, get_val_transforms

LABELS = {0: 'REAL', 1: 'FAKE'}


def main():
    ap = argparse.ArgumentParser(description='Evaluate a trained deepfake detector on a held-out test set')
    ap.add_argument('--data_root', default='data/real_vs_fake/real-vs-fake',
                    help='Folder containing test/ (or val/valid/) with real/ and fake/ subfolders')
    ap.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--max_samples', type=int, default=None,
                    help='Cap on samples to evaluate, for a quick check')
    ap.add_argument('--examples', type=int, default=10,
                    help='Number of example predictions to print')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    split = f'{args.data_root}/test'
    if not os.path.exists(split):
        split = f'{args.data_root}/val'
    if not os.path.exists(split):
        split = f'{args.data_root}/valid'

    print("Building dataset...")
    ds = DeepfakeDataset(split, get_val_transforms(args.img_size), args.max_samples)
    loader = DataLoader(ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"DataLoader ready | Batches: {len(loader)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepfakeDetector(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    if isinstance(ckpt, dict) and 'metrics' in ckpt:
        print(f"Checkpoint val metrics: {ckpt['metrics']}")

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluating', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs)
            probs = torch.softmax(logits.detach().float(), 1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(probs.argmax(1))
            all_probs.append(probs)

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.vstack(all_probs)

    acc = accuracy_score(labels_arr, preds_arr)
    auc = roc_auc_score(labels_arr, probs_arr[:, 1]) if len(np.unique(labels_arr)) > 1 else 0.5
    f1 = f1_score(labels_arr, preds_arr, average='binary', zero_division=0)

    print()
    print("=" * 60)
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test ROC-AUC  : {auc:.4f}")
    print(f"  Test F1       : {f1:.4f}")
    print("=" * 60)
    print(classification_report(labels_arr, preds_arr,
                                target_names=[LABELS[0], LABELS[1]], digits=4))

    paths = [p for p, _ in ds.samples]
    print("\nExample predictions (first %d in scan order):" % args.examples)
    print(f"{'true':>6} {'pred':>6} {'conf':>7}  file")
    print("-" * 60)
    for i in range(min(args.examples, len(labels_arr))):
        truth = LABELS[int(labels_arr[i])]
        pred = LABELS[int(preds_arr[i])]
        conf = probs_arr[i, 1] if pred == 'FAKE' else probs_arr[i, 0]
        print(f"{truth:>6} {pred:>6} {conf:7.2%}  {os.path.basename(paths[i])}")

    worst = np.argsort(np.abs(probs_arr[:, 1] - labels_arr))[::-1][:args.examples]
    print("\nWorst misclassified examples:")
    print(f"{'true':>6} {'pred':>6} {'conf':>7}  file")
    print("-" * 60)
    for i in worst:
        truth = LABELS[int(labels_arr[i])]
        pred = LABELS[int(preds_arr[i])]
        conf = probs_arr[i, 1] if pred == 'FAKE' else probs_arr[i, 0]
        print(f"{truth:>6} {pred:>6} {conf:7.2%}  {os.path.basename(paths[i])}")

    summary = {'acc': round(float(acc), 4), 'auc': round(float(auc), 4),
               'f1': round(float(f1), 4), 'n': int(len(labels_arr))}
    print("\nsummary:", summary)


if __name__ == '__main__':
    main()
