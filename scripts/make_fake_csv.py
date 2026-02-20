import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=800,
    centers=6,
    n_features=32,
    cluster_std=1.3,
    random_state=42
)

ids = [f"pep_{i:04d}" for i in range(X.shape[0])]

df = pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])
df.insert(0, "label", y)
df.insert(0, "id", ids)

df.to_csv("fake_embeddings.csv", index=False)
print("Wrote fake_embeddings.csv with shape:", df.shape)
