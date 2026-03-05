import time
import streamlit as st
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
import numpy as np
import os

#cfg class: allows for reproducibility with the same results
class CFG: 
    #path files 
    EMB_PATH = "embeddings/veltri/veltri_embeddings.npy"
    META_PATH = "embeddings/veltri/veltri_metadata.csv"
    VELTRI_PATH = "data/veltri/all_veltri.csv"
    #parameters
    n_neighbors = 21
    n_components = 2
    min_dist = 0.0289
    metric = "cosine"
    seed =42

st.set_page_config(page_title="UMAP Peptide Embeddings", layout="wide")
st.title("UMAP/DensMAP — ProteoGPT - Veltri Embeddings")

st.sidebar.header("Upload Local Files")

#allows users to locally import files
uploaded_embeddings = st.sidebar.file_uploader(
    "Upload embeddings (.npy)",
    type=["npy"]
)

uploaded_metadata = st.sidebar.file_uploader(
    "Upload metadata (.csv)",
    type=["csv"]
)




# allows for which CSV to use
st.sidebar.header("Data Source")
meta_choice = st.sidebar.radio(
    "Metadata to use for labels/hover",
    ["Veltri metadata (recommended)", "Raw Veltri CSV (all_veltri.csv)"],
    index=0,
)

#use cfg paths
selected_meta_path = (
    CFG.META_PATH
    if meta_choice == "Veltri metadata (recommended)"
    else CFG.VELTRI_PATH
)

# load embeddings (uploaded overrides default)
try:
    if uploaded_embeddings is not None:
        st.sidebar.success("Using uploaded embeddings")
        X = np.load(uploaded_embeddings).astype("float32")  # (N, dim)
        emb_source = "Uploaded .npy"
    else:
        X = np.load(CFG.EMB_PATH).astype("float32")  # (N, dim)
        emb_source = CFG.EMB_PATH
except Exception as e:
    st.error(f"Could not read embeddings: {e}")
    st.info(f"Default expected at: {CFG.EMB_PATH}")
    st.stop()

st.caption(f"Embeddings source: {emb_source}")

# load metadata (uploaded overrides radio selection)
try:
    if uploaded_metadata is not None:
        st.sidebar.success("Using uploaded metadata")
        meta = pd.read_csv(uploaded_metadata)
        meta_source = "Uploaded .csv"
    else:
        meta = pd.read_csv(selected_meta_path)
        meta_source = selected_meta_path
except Exception as e:
    st.error(f"Could not read metadata: {e}")
    if uploaded_metadata is None and selected_meta_path == CFG.VELTRI_PATH:
        st.info("If you haven't downloaded it yet, run: python scripts/veltri_dataset.py")
        st.info("Also make sure it exists at: umap_project/data/veltri/all_veltri.csv")
    else:
        st.info(f"Default expected at: {CFG.META_PATH}")
    st.stop()

st.caption(f"Metadata source: {meta_source}")

# helper to map unknown column names
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# Auto detect columns
seq_col = pick_col(meta, ["sequence", "aa_seq", "seq", "peptide", "peptide_sequence", "amp_sequence"])
id_col = pick_col(meta, ["peptide_id", "id", "identifier", "name", "entry", "accession"])
label_col = pick_col(meta, ["source", "label", "class", "family", "dataset", "type"])
npz_col = pick_col(meta, ["npz_path", "path", "file", "filepath"])

#MANUAL COLUMN OVERRIDE
st.sidebar.header("Column Override (if auto detection fails)")

cols = list(meta.columns)

# sequence column
if seq_col is None:
    st.sidebar.warning("Sequence column not detected automatically.")
    seq_col = st.sidebar.selectbox("Select sequence column", cols)

# id column (optional)
if id_col is None:
    id_choice = st.sidebar.selectbox("Select ID column (optional)", ["None"] + cols)
    if id_choice != "None":
        id_col = id_choice

# label column (optional)
if label_col is None:
    label_choice = st.sidebar.selectbox("Select label column (optional)", ["None"] + cols)
    if label_choice != "None":
        label_col = label_choice

# build a plotting df with safe fallbacks
df = pd.DataFrame({
    "id": meta[id_col].astype(str) if id_col else meta.index.astype(str),
    "label": meta[label_col].astype(str) if label_col else (
        "veltri" if selected_meta_path == CFG.VELTRI_PATH else "unknown"
    ),
    "sequence": meta[seq_col].astype(str) if seq_col else "",
    "npz_path": meta[npz_col].astype(str) if npz_col else "",
})


# detect AMP label automatically
if "is_amp" in meta.columns:
    df["is_amp"] = meta["is_amp"].astype(bool)

elif "AMP" in meta.columns:
    df["is_amp"] = meta["AMP"].astype(int).eq(1)

elif label_col is not None:
    lab = meta[label_col]

    if pd.api.types.is_numeric_dtype(lab):
        df["is_amp"] = lab.astype(int).eq(1)
    else:
        positives = {"1","true","yes","y","amp","positive"}
        df["is_amp"] = lab.astype(str).str.lower().isin(positives)

else:
    st.warning("No label column detected. All peptides set to False.")
    df["is_amp"] = False
    

# EXPANDER; show what columns were detected
with st.expander("Debug: detected columns"):
    st.write("Using metadata source:", selected_meta_path)
    st.write("Detected id_col:", id_col)
    st.write("Detected label_col:", label_col)
    st.write("Detected seq_col:", seq_col)
    st.write("Detected npz_col:", npz_col)
    st.write("All columns:", list(meta.columns))
    st.dataframe(meta.head(5), width="stretch")

# safety check so the app doesn't crash if metadata rows != embeddings rows
nX = X.shape[0]
nM = df.shape[0]
if nX != nM:
    st.warning(
        f"Row mismatch: embeddings have {nX} rows but metadata has {nM} rows.\n\n"
        "UMAP will still run, but labels/hover may not match correctly."
    )
    n = min(nX, nM)
    X = X[:n]
    df = df.iloc[:n].reset_index(drop=True)

# baseline metrics
st.subheader("Baseline metrics")
st.write("Rows (samples):", X.shape[0])
st.write("Embedding dim:", X.shape[1])

# UMAP settings ; sidebars were not previously functional
st.sidebar.header("UMAP Settings")
CFG.n_neighbors = st.sidebar.slider("n_neighbors", 5, 100, CFG.n_neighbors, 1) 
CFG.min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, CFG.min_dist, 0.01)

#UMAP/DensMAP Switch
st.sidebar.header("Embedding Method")

method = st.sidebar.radio(
    "Projection Type",
    ["UMAP", "densMAP"],
    index=0
)

# Only show densMAP controls if selected
if method == "densMAP":
    dens_lambda = st.sidebar.slider("dens_lambda", 0.0, 10.0, 2.0, 0.1)
    dens_frac = st.sidebar.slider("dens_frac", 0.0, 1.0, 0.3, 0.05)
    dens_var_shift = st.sidebar.slider("dens_var_shift", 0.0, 1.0, 0.1, 0.01)
                                       
# OLD REDUCER 
#reducer = umap.UMAP(
    #n_neighbors=CFG.n_neighbors,
   # min_dist=CFG.min_dist,
   # n_components=CFG.n_components,
   # metric=CFG.metric,
   # random_state=CFG.seed)

#NEW REDUCER FOR DENSMAP AND UMAP
reducer_kwargs = dict(
    n_neighbors=CFG.n_neighbors,
    min_dist=CFG.min_dist,
    n_components=CFG.n_components,
    metric=CFG.metric,
    random_state=CFG.seed,
)

if method == "densMAP":
    reducer_kwargs.update(
        densmap=True,
        dens_lambda=dens_lambda,
        dens_frac=dens_frac,
        dens_var_shift=dens_var_shift,
    )

reducer = umap.UMAP(**reducer_kwargs)

start = time.time()
X_umap = reducer.fit_transform(X)
runtime = time.time() - start

st.write(f"{method} runtime: {runtime:.3f} seconds")
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
    color="is_amp",
    color_discrete_map={False:"blue", True:"red"},
    hover_data=["id", "is_amp", "sequence"],  
    title=f"{method} projection (Veltri embeddings)",
    render_mode="webgl"
)

# Makes large datasets clearer + faster
fig.update_traces(marker=dict(size=4, opacity=0.6))

# Streamlit new API (replaces use_container_width=True)
st.plotly_chart(fig, width="stretch")

with st.expander("Show first 10 peptides"):
    st.dataframe(df.head(10), width="stretch")