import glob, numpy as np, nibabel as nib, pandas as pd, matplotlib.pyplot as plt

ROOT = "/Users/dahniazarroug/Downloads/Healthy_Volunteer_006"
df = pd.read_csv("rejection_labels_test.csv")
amb = df[df["ambiguous"]]

for (ds, s, src), _ in amb.groupby(["dataset", "slice", "source_volume"]):
    U = np.asarray(nib.load(glob.glob(f"{ROOT}/{ds}/01_Unrung/*.nii*")[0]).dataobj)
    members = df[(df.dataset == ds) & (df.slice == s) & (df.source_volume == src)]
    fig, axes = plt.subplots(1, len(members), figsize=(3 * len(members), 3))
    for ax, (_, r) in zip(np.atleast_1d(axes), members.iterrows()):
        ax.imshow(U[:, :, s, r.volume].T, cmap="gray", origin="lower")
        ax.set_title(f"vol {r.volume} — {r.label.upper()}", color="red" if r.label == "rejected" else "green")
        ax.axis("off")
    fig.suptitle(f"{ds}, slice {s}, margin {members.margin.iloc[0]:.3f}")
    plt.tight_layout(); plt.show()