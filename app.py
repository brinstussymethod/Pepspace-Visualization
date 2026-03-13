import time
import streamlit as st
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
import numpy as np
import os

# NEIGHBORHOOD PRESERVATION
from sklearn.neighbors import NearestNeighbors

def neighborhood_preservation(X_high, X_low, k=10):

    nn_high = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(X_high)
    high_neighbors = nn_high.kneighbors(return_distance=False)[:, 1:]

    nn_low = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X_low)
    low_neighbors = nn_low.kneighbors(return_distance=False)[:, 1:]

    overlaps = []

    for i in range(len(X_high)):
        overlap = len(np.intersect1d(high_neighbors[i], low_neighbors[i]))
        overlaps.append(overlap / k)

    return np.mean(overlaps)


#cfg class: allows for reproducibility with the same results
class CFG: 
    #path files 
    EMB_PATH = "embeddings/veltri/veltri_embeddings.npy"
    META_PATH = "embeddings/veltri/veltri_metadata.csv"
    VELTRI_PATH = "data/veltri/all_veltri.csv"
    #parameters
    n_neighbors = 5 #from optimizer
    n_components = 2
    min_dist = 0.005527128631029834 #from optimizer
    metric = "cosine"
    seed =42


# Cached functions
@st.cache_data
def load_embeddings(file):
    return np.load(file).astype("float32")


@st.cache_data
def load_metadata(file):
    return pd.read_csv(file)


@st.cache_data
def run_umap(
    X,
    n_neighbors,
    min_dist,
    n_components,
    metric,
    seed,
    method,
    dens_lambda=None,
    dens_frac=None,
    dens_var_shift=None,
):

    reducer_kwargs = dict(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
    )

    if method == "densMAP":
        reducer_kwargs.update(
            densmap=True,
            dens_lambda=dens_lambda,
            dens_frac=dens_frac,
            dens_var_shift=dens_var_shift,
        )

    reducer = umap.UMAP(**reducer_kwargs)

    return reducer.fit_transform(X)


#streamlit page setup
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


# Data Source Selection
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

# Load Emebeddings
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


# Load Metdata (uploaded overrides radio selection)
try:
    if uploaded_metadata is not None:
        st.sidebar.success("Using uploaded metadata")
        meta = load_metadata(uploaded_metadata)
        meta_source = "Uploaded .csv"
    else:
        meta = load_metadata(selected_meta_path)
        meta_source = selected_meta_path

except Exception as e:
    st.error(f"Could not read metadata: {e}")
    st.stop()

st.caption(f"Metadata source: {meta_source}")


# Column Detection Helper
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


# Building Plotting Datafram
df = pd.DataFrame({
    "id": meta[id_col].astype(str) if id_col else meta.index.astype(str),
    "label": meta[label_col].astype(str) if label_col else "unknown",
    "sequence": meta[seq_col].astype(str) if seq_col else "",
})


# AMP Detection
if "is_amp" in meta.columns:
    df["is_amp"] = meta["is_amp"].astype(bool)

elif "AMP" in meta.columns:
    df["is_amp"] = meta["AMP"].astype(int).eq(1)

else:
    df["is_amp"] = False



# Row Safety Check
nX = X.shape[0]
nM = df.shape[0]

if nX != nM:
    st.warning(f"Row mismatch: embeddings {nX} vs metadata {nM}")
    n = min(nX, nM)
    X = X[:n]
    df = df.iloc[:n]


#Baseline Metrics

st.subheader("Baseline metrics")

st.write("Rows:", X.shape[0])
st.write("Embedding dim:", X.shape[1])



# UMAP Controls
st.sidebar.header("UMAP Settings")

CFG.n_neighbors = st.sidebar.slider("n_neighbors", 5, 100, CFG.n_neighbors)

CFG.min_dist = st.sidebar.slider("min_dist",0.0,1.0,CFG.min_dist,0.001,format="%.4f")

# UMAP / densMAP Switch
st.sidebar.header("Embedding Method")

method = st.sidebar.radio(
    "Projection Type",
    ["UMAP", "densMAP"],
)

dens_lambda = None
dens_frac = None
dens_var_shift = None

if method == "densMAP":
    dens_lambda = st.sidebar.slider("dens_lambda", 0.0, 10.0, 2.0)
    dens_frac = st.sidebar.slider("dens_frac", 0.0, 1.0, 0.3)
    dens_var_shift = st.sidebar.slider("dens_var_shift", 0.0, 1.0, 0.1)



# Run UMAP (cached)
start = time.time()

X_umap = run_umap(
    X,
    CFG.n_neighbors,
    CFG.min_dist,
    CFG.n_components,
    CFG.metric,
    CFG.seed,
    method,
    dens_lambda,
    dens_frac,
    dens_var_shift,
)

runtime = time.time() - start


#Baseline Metrics
st.subheader("Baseline metrics")

st.write("Rows:", X.shape[0])
st.write("Embedding dim:", X.shape[1])
st.write(f"{method} runtime: {runtime:.3f} seconds")


# Plot Data
plot_df = pd.DataFrame({
    "UMAP1": X_umap[:, 0],
    "UMAP2": X_umap[:, 1],
    "id": df["id"],
    "sequence": df["sequence"],
    "is_amp": df["is_amp"],
})


# Plot
fig = px.scatter(
    plot_df,
    x="UMAP1",
    y="UMAP2",
    color="is_amp",
    color_discrete_map={False: "blue", True: "red"},
    hover_data=["id", "sequence", "is_amp"],
    title=f"{method} projection",
    render_mode="webgl",
)

fig.update_traces(marker=dict(size=4, opacity=0.6))

st.plotly_chart(fig, width="stretch")


# Debug Table
with st.expander("Show first 10 peptides"):
    st.dataframe(df.head(10))


# Clear Cache Button
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")