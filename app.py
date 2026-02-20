import time
import streamlit as st
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
import numpy as np
import os

st.set_page_config(page_title="UMAP Peptide Embeddings", layout="wide")
st.title("UMAP — ProteoGPT - Veltri Embeddings")

EMB_PATH = "embeddings/veltri/veltri_embeddings.npy"
META_PATH = "embeddings/veltri/veltri_metadata.csv"
VELTRI_PATH = "data/veltri/all_veltri.csv"

# allows for which CSV to use
st.sidebar.header("Data Source")
meta_choice = st.sidebar.radio(
    "Metadata to use for labels/hover",
    ["Veltri metadata (recommended)", "Raw Veltri CSV (all_veltri.csv)"],
    index=0,
)


selected_meta_path = META_PATH if meta_choice == "Veltri metadata (recommended)" else VELTRI_PATH

# load embeddings
try:
    X = np.load(EMB_PATH).astype("float32")  # (N, dim)
except Exception as e:
    st.error(f"Could not read {EMB_PATH}: {e}")
    st.info(f"Make sure the file exists at: {EMB_PATH}")
    st.stop()

# load the SELECTED metadata (was always META_PATH before)
try:
    meta = pd.read_csv(selected_meta_path)
except Exception as e:
    st.error(f"Could not read {selected_meta_path}: {e}")
    if selected_meta_path == VELTRI_PATH:
        st.info("If you haven't downloaded it yet, run: python scripts/veltri_dataset.py")
        st.info("Also make sure it exists at: umap_project/data/veltri/all_veltri.csv")
    else:
        st.info(f"Make sure the file exists at: {META_PATH}")
    st.stop()

# helper to map unknown column names
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# detect likely columns
seq_col = pick_col(meta, ["sequence", "aa_seq", "seq", "peptide", "peptide_sequence", "amp_sequence"])
id_col = pick_col(meta, ["peptide_id", "id", "identifier", "name", "entry", "accession"])
label_col = pick_col(meta, ["source", "label", "class", "family", "dataset", "type"])
npz_col = pick_col(meta, ["npz_path", "path", "file", "filepath"])

# build a plotting df with safe fallbacks
df = pd.DataFrame({
    "id": meta[id_col].astype(str) if id_col else meta.index.astype(str),
    "label": meta[label_col].astype(str) if label_col else (
        "veltri" if selected_meta_path == VELTRI_PATH else "unknown"
    ),
    "sequence": meta[seq_col].astype(str) if seq_col else "",
    "npz_path": meta[npz_col].astype(str) if npz_col else "",
})


if selected_meta_path == META_PATH:
    # veltri_metadata.csv expected columns: sequence, is_amp
    if "is_amp" in meta.columns:
        df["is_amp"] = meta["is_amp"].astype(bool)
    else:
        st.error("Expected 'is_amp' column in embeddings/veltri/veltri_metadata.csv")
        st.stop()
else:
    # raw all_veltri.csv expected column: AMP (0/1)
    if "AMP" in meta.columns:
        df["is_amp"] = meta["AMP"].astype(int).eq(1)
    else:
        # fallback if no label exists
        df["is_amp"] = False

# show what columns were detected
with st.expander("Debug: detected columns"):
    st.write("Using metadata file:", selected_meta_path)
    st.write("Detected id_col:", id_col)
    st.write("Detected label_col:", label_col)
    st.write("Detected seq_col:", seq_col)
    st.write("Detected npz_col:", npz_col)
    st.write("All columns:", list(meta.columns))
    st.dataframe(meta.head(5), width="stretch")  # ✅ CHANGED (new Streamlit API)

# safety check so the app doesn't crash if metadata rows != embeddings rows
nX = X.shape[0]
nM = df.shape[0]
if nX != nM:
    st.warning(
        f"Row mismatch: embeddings have {nX} rows but metadata has {nM} rows.\n\n"
        "UMAP will still run, but labels/hover may not match correctly.\n"
        "I will truncate both to the smaller size to keep the app running."
    )
    n = min(nX, nM)
    X = X[:n]
    df = df.iloc[:n].reset_index(drop=True)

# baseline metrics
st.subheader("Baseline metrics")
st.write("Rows (samples):", X.shape[0])
st.write("Embedding dim:", X.shape[1])

# UMAP settings
st.sidebar.header("UMAP Settings")
n_neighbors = st.sidebar.slider("n_neighbors", 5, 100, 15, 1)
min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, 0.1, 0.01)

reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)

start = time.time()
X_umap = reducer.fit_transform(X)
runtime = time.time() - start

st.write(f"UMAP runtime: {runtime:.3f} seconds")
st.write("UMAP output shape:", X_umap.shape)


plot_df = pd.DataFrame({
    "UMAP1": X_umap[:, 0],
    "UMAP2": X_umap[:, 1],
    "id": df["id"],
    "sequence": df["sequence"],
    "is_amp": df["is_amp"],
})

# Debug check: confirms all embeddings are used
st.write("Points being plotted:", len(plot_df))

# WebGL rendering for >1000 points
fig = px.scatter(
    plot_df,
    x="UMAP1",
    y="UMAP2",
    color="is_amp", "red" 
    hover_data=["id", "is_amp", "sequence"],  
    title="UMAP projection (Veltri embeddings)",
    render_mode="webgl"
)

# Makes large datasets clearer + faster
fig.update_traces(marker=dict(size=4, opacity=0.6))

# Streamlit new API (replaces use_container_width=True)
st.plotly_chart(fig, width="stretch")

with st.expander("Show first 10 peptides"):
    st.dataframe(df.head(10), width="stretch")