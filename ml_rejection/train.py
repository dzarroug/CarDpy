"""
train.py
Small CNN that classifies each image accept (0) / reject (1).

  python ml_rejection/train.py              # train, pick best epoch + threshold on VAL
  python ml_rejection/train.py --crop 64    # same, on a centered 64x64 window around the heart
  python ml_rejection/train.py --test       # evaluate the saved model on TEST (run once, at the end)
Each run saves to models/<run name>/ (default: full or crop64), so runs don't overwrite each other.

Design (kept deliberately simple and defensible):
  - input: C x 128 x 128 from dataset.py (C read from the cache; currently 3:
    image, direction median, b-value-shell median)
  - 4 conv blocks (conv-BN-ReLU-maxpool), 16->32->64->128 channels, global avg pool, 1 logit
  - BCEWithLogitsLoss with pos_weight = neg/pos on TRAIN (handles the ~7% imbalance)
  - model selection on VAL average precision (PR-AUC), threshold chosen on VAL to maximize F1
  - TEST is never touched during training or threshold selection
Outputs (ml_rejection/models/, gitignored): best.pt (weights + threshold), history.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve

from dataset import get_loaders

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


class RejectionCNN(nn.Module):
    def __init__(self, in_ch=3, widths=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple Silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def metrics_at(p, y, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def best_threshold(p, y):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i])


def run_dir(args):
    name = args.name or (f"crop{args.crop}" if args.crop else "full")
    return os.path.join(MODELS, name)


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = device()
    train_dl, val_dl, _, pos_weight = get_loaders(batch_size=args.batch_size, crop=args.crop)
    model = RejectionCNN(in_ch=train_dl.dataset.X.shape[1]).to(dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = run_dir(args)
    os.makedirs(out, exist_ok=True)
    print(f"run {out}")
    print(f"device {dev} | train {len(train_dl.dataset)} | val {len(val_dl.dataset)} | "
          f"pos_weight {pos_weight.item():.1f} | params {sum(p.numel() for p in model.parameters()):,}")

    history, best_ap, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); total = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            total += loss.item() * len(y)
        sched.step()

        p, y = predict(model, val_dl, dev)
        ap = average_precision_score(y, p)
        thr = best_threshold(p, y)
        m = metrics_at(p, y, thr)
        row = dict(epoch=epoch, train_loss=total / len(train_dl.dataset), val_ap=ap, thr=thr, **m,
                   secs=time.time() - t0)
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out, "history.csv"), index=False)
        print(f"epoch {epoch:2d} | loss {row['train_loss']:.4f} | val AP {ap:.3f} | "
              f"F1 {m['f1']:.3f} (P {m['precision']:.3f} R {m['recall']:.3f} @ {thr:.2f}) | "
              f"{row['secs']:.0f}s")

        if ap > best_ap:
            best_ap, stale = ap, 0
            torch.save(dict(state_dict=model.state_dict(), threshold=thr, epoch=epoch,
                            val_ap=ap, val_metrics=m, args=vars(args)),
                       os.path.join(out, "best.pt"))
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no val AP gain for {args.patience} epochs")
                break

    ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    print(f"\nbest epoch {ck['epoch']} | val AP {ck['val_ap']:.3f} | threshold {ck['threshold']:.2f} | "
          f"val F1 {ck['val_metrics']['f1']:.3f} P {ck['val_metrics']['precision']:.3f} "
          f"R {ck['val_metrics']['recall']:.3f}")


def test(args):
    dev = device()
    out = run_dir(args)
    ck = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    crop = ck["args"].get("crop")
    _, _, test_dl, _ = get_loaders(batch_size=args.batch_size, crop=crop)
    model = RejectionCNN(in_ch=test_dl.dataset.X.shape[1]).to(dev); model.load_state_dict(ck["state_dict"])
    p, y = predict(model, test_dl, dev)
    m = metrics_at(p, y, ck["threshold"])
    print(f"TEST ({len(y)} imgs, {int(y.sum())} rejected) | AP {average_precision_score(y, p):.3f} | "
          f"threshold {ck['threshold']:.2f} (from val)")
    print(f"F1 {m['f1']:.3f} | precision {m['precision']:.3f} | recall {m['recall']:.3f}")
    print(f"confusion: TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")
    meta = test_dl.dataset.meta.copy(); meta["prob"] = p; meta["pred"] = (p >= ck["threshold"]).astype(int)
    meta.to_csv(os.path.join(out, "test_predictions.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=None, help="centered crop size, e.g. 64")
    ap.add_argument("--name", default=None, help="run folder name under models/")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    test(args) if args.test else train(args)