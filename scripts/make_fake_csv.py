import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import os

OUT_DIR = "embeddings/fake"
os.makedirs(OUT_DIR, exist_ok=True)

existing = [f for f in os.listdir(OUT_DIR) if f.startswith("fake_embeddings")]
idx = len(existing) + 1

X, y = make_blobs(
    n_samples=800,
    centers=4,
    n_features=32,
    cluster_std=0.8,
    random_state=None
)

is_amp = y >= 2

np.save(f"{OUT_DIR}/fake_embeddings{idx}.npy", X.astype("float32"))

ids = [f"pep_{i:04d}" for i in range(X.shape[0])]
seqs = ["".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 12)) for _ in range(X.shape[0])]

meta = pd.DataFrame({
    "id": ids,
    "sequence": seqs,
    "is_amp": is_amp
})

meta.to_csv(f"{OUT_DIR}/fake_metadata{idx}.csv", index=False)

print(f"Saved dataset #{idx}")