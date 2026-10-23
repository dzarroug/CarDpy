"""Display saved uncertainty maps with Matplotlib."""
import numpy as np


def plot_maps(path, *, slice_index=0, mask=None):
    import matplotlib.pyplot as plt
    with np.load(path) as saved:
        maps = {key: saved[key] for key in ('mMD', 'mFA', 'dMD', 'dFA', 'dE1', 'dE2', 'dE3')}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout='constrained')
    for ax, key in zip(axes.flat, maps):
        image = maps[key][:, :, slice_index]
        if mask is not None:
            image = np.where(np.asarray(mask)[:, :, slice_index] > 0, image, np.nan)
        display = ax.imshow(image, cmap='magma' if key.startswith('d') else 'viridis',
                            vmin=0, vmax=({'dMD': 1, 'dFA': .5}.get(key,
                                         90 if key.startswith('dE') else None)))
        ax.set_title(key + (' (degrees)' if 'E' in key else (' (µm²/ms)' if 'MD' in key else '')))
        ax.axis('off')
        fig.colorbar(display, ax=ax, shrink=.8)
    axes.flat[-1].axis('off')
    return fig
