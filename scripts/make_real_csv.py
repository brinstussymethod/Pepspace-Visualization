import numpy as np
import pandas as pd

embeddings = np.load("embeddings/peptides_embeddings_sol_v0.npy")
meta = pd.read_csv("embeddings/peptides_metadata_sol_v0.csv")

meta = meta.rename(columns={
    "peptide_id": "id",
    "source": "label"   # this becomes the color groups
})

# Create embedding column names
embed_cols = [f"e{i}" for i in range(embeddings.shape[1])]

embed_df = pd.DataFrame(embeddings, columns=embed_cols)

final_df = pd.concat([meta, embed_df], axis=1)

final_df.to_csv("peptide_umap_ready.csv", index=False)

print("Saved peptide_umap_ready.csv")
