"""
dataset.py
PyTorch loaders over the cache written by build_cache.py.

    from dataset import get_loaders
    train_dl, val_dl, test_dl, pos_weight = get_loaders()

Each batch: x float32 [B, 2, 128, 128], y float32 [B].
Training batches get light augmentation (random flips / 90-degree rotations,
applied identically to both channels). Val/test are never augmented.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class RejectionDataset(Dataset):
    def __init__(self, split, cache=CACHE, augment=False):
        meta = pd.read_csv(os.path.join(cache, "meta.csv"))
        self.idx = np.flatnonzero(meta["split"].to_numpy() == split)
        self.meta = meta.iloc[self.idx].reset_index(drop=True)
        self.y = self.meta["y"].to_numpy(dtype=np.float32)
        self.X = np.load(os.path.join(cache, "X.npy"), mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = np.asarray(self.X[self.idx[i]], dtype=np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
            x = np.rot90(x, k=np.random.randint(4), axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(self.y[i])


def get_loaders(batch_size=64, cache=CACHE, num_workers=0):
    train = RejectionDataset("train", cache, augment=True)
    val = RejectionDataset("val", cache)
    test = RejectionDataset("test", cache)
    pos = train.y.sum()
    pos_weight = torch.tensor((len(train) - pos) / max(pos, 1), dtype=torch.float32)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val, batch_size=batch_size, num_workers=num_workers),
            DataLoader(test, batch_size=batch_size, num_workers=num_workers),
            pos_weight)


if __name__ == "__main__":
    train_dl, val_dl, test_dl, pw = get_loaders()
    x, y = next(iter(train_dl))
    print(f"train/val/test: {len(train_dl.dataset)}/{len(val_dl.dataset)}/{len(test_dl.dataset)}")
    print(f"batch x {tuple(x.shape)} {x.dtype} | y {tuple(y.shape)} | positives in batch: {int(y.sum())}")
    print(f"pos_weight: {pw.item():.1f}")