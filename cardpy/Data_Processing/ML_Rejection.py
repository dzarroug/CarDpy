"""
ML_Rejection.py -- rejection_mode = 'ml'

Scores every image with the CNN trained in ml_rejection/train.py and returns a
stacked keep matrix [slices, volumes] (1 = keep, 0 = reject), the same format
shot_rejection's algorithm produces, so _rejection_result handles the rest.

Input construction MUST match ml_rejection/build_cache.py exactly:
  ch0 = image, ch1 = median of its slice/direction, ch2 = median of its
  b-value shell in that slice; all scaled by the 99th percentile of ch1,
  fit to 128x128, stored as float16, then center-cropped (crop from checkpoint).

torch is an optional dependency: pip install "cardpy-cmr[ml]".
"""
import os

import numpy as np

SIZE = 128
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'rejection_cnn.pt')


def _fit(img, size=SIZE):
    """Center-crop or zero-pad a 2D image to size x size (same as build_cache.fit)."""
    out = np.zeros((size, size), dtype=np.float32)
    h, w = img.shape
    ys, xs = max(0, (h - size) // 2), max(0, (w - size) // 2)
    crop = img[ys:ys + size, xs:xs + size]
    oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def _direction_ids(bvals, bvecs, decimals=4):
    """Group stacked volumes by unique (bval, bvec), in order of first appearance
    (same as extract_rejection_labels.direction_average_index)."""
    keys, ids = {}, []
    for b, g in zip(np.round(bvals, 1), np.round(bvecs, decimals)):
        k = (b, *g)
        if k not in keys:
            keys[k] = len(keys)
        ids.append(keys[k])
    return np.array(ids)


def prepare_inputs(stacked, bvals, bvecs):
    """Magnitude stacked data [rows, cols, slices, vols] -> float32 [slices, vols, 3, 128, 128]."""
    U = np.abs(np.asarray(stacked)).astype(np.float32)
    bvals = np.asarray(bvals, dtype=float).ravel()
    bvecs = np.asarray(bvecs, dtype=float)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    direction = _direction_ids(bvals, bvecs)
    _, _, n_slices, n_vols = U.shape
    out = np.zeros((n_slices, n_vols, 3, SIZE, SIZE), dtype=np.float32)
    for s in range(n_slices):
        shell_med = {}
        for d in np.unique(direction):
            vols = np.flatnonzero(direction == d)
            b = bvals[vols[0]]
            if b not in shell_med:
                shell_vols = np.flatnonzero(np.isclose(bvals, b))
                shell_med[b] = np.median(U[:, :, s, shell_vols], axis=2)
            med = np.median(U[:, :, s, vols], axis=2)
            scale = max(float(np.percentile(med, 99)), 1e-6)
            med_f, shell_f = _fit(med / scale), _fit(shell_med[b] / scale)
            for v in vols:
                out[s, v, 0] = _fit(U[:, :, s, v] / scale)
                out[s, v, 1] = med_f
                out[s, v, 2] = shell_f
    return out.astype(np.float16).astype(np.float32)   # training data was stored as float16


def _build_model(in_ch, widths=(16, 32, 64, 128), dropout=0.3):
    """Architecture must match RejectionCNN in ml_rejection/train.py."""
    import torch.nn as nn

    class RejectionCNN(nn.Module):
        def __init__(self):
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

    return RejectionCNN()


def ml_keep_matrix(stacked, bvals, bvecs, model_path=None, threshold=None,
                   batch_size=64, save_dir=None):
    """
    Returns (keep [slices, vols] uint8, probs [slices, vols] float32).
    threshold=None uses the threshold saved in the checkpoint (chosen on validation).
    If save_dir is given, writes ML_Rejection_Probabilities.npy there.
    """
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError as e:
        raise ImportError("rejection_mode='ml' needs PyTorch: pip install \"cardpy-cmr[ml]\"") from e

    model_path = model_path or DEFAULT_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ML rejection model not found at {model_path}. "
                                "Set config['ml_model_path'] to your best.pt.")
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    in_ch = ck['state_dict']['features.0.weight'].shape[1]
    model = _build_model(in_ch)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    thr = float(ck['threshold'] if threshold is None else threshold)
    crop = (ck.get('args') or {}).get('crop')

    X = prepare_inputs(stacked, bvals, bvecs)
    n_slices, n_vols = X.shape[:2]
    X = X.reshape(n_slices * n_vols, *X.shape[2:])[:, :in_ch]
    if crop:
        c0 = (SIZE - crop) // 2
        X = X[:, :, c0:c0 + crop, c0:c0 + crop]

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            probs.append(torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])))).numpy())
    probs = np.concatenate(probs).reshape(n_slices, n_vols).astype(np.float32)
    keep = (probs < thr).astype(np.uint8)

    print('ML rejection: model %s | threshold %.2f | crop %s | rejected %i / %i images'
          % (os.path.basename(model_path), thr, crop, int((keep == 0).sum()), keep.size))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'ML_Rejection_Probabilities.npy'), probs)
    return keep, probs