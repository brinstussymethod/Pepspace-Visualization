import time
import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
import numpy as np
from utils.sphere_projection import project_embeddings_to_sphere


# CFG class: allows for reproducibility with the same results
class CFG:
    # Path files
    EMB_PATH = "embeddings/veltri/veltri_embeddings.npy"
    META_PATH = "embeddings/veltri/veltri_metadata.csv"
    VELTRI_PATH = "data/veltri/all_veltri.csv"
    # Default parameters (used as fallback if no Optuna JSON exists)
    n_neighbors = 5
    n_components = 2
    min_dist = 0.005527128631029834
    metric = "cosine"
    seed = 42


# --- Load Optuna params from JSON files ---
# Expects files named: optuna_results/2d_params.json, 3d_params.json, etc.
# Each file should have a "best_params" key with n_neighbors, min_dist, dens_lambda
OPTUNA_PARAMS = {}
for dim in [2, 3, 4, 5, 6, 7, 8]:
    path = f"optuna_results/{dim}d_params.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            OPTUNA_PARAMS[dim] = data["best_params"]


# --- Cached functions ---
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


# --- Page setup ---
st.set_page_config(page_title="UMAP Peptide Embeddings", layout="wide")
st.title("UMAP/DensMAP — ProteoGPT - Veltri Embeddings")


# --- File uploaders ---
st.sidebar.header("Upload Local Files")

uploaded_embeddings = st.sidebar.file_uploader("Upload embeddings (.npy)", type=["npy"])
uploaded_metadata = st.sidebar.file_uploader("Upload metadata (.csv)", type=["csv"])


# --- Data source selection ---
st.sidebar.header("Data Source")
meta_choice = st.sidebar.radio(
    "Metadata to use for labels/hover",
    ["Veltri metadata (recommended)", "Raw Veltri CSV (all_veltri.csv)"],
    index=0,
)

selected_meta_path = (
    CFG.META_PATH
    if meta_choice == "Veltri metadata (recommended)"
    else CFG.VELTRI_PATH
)


# --- Load embeddings ---
try:
    if uploaded_embeddings is not None:
        st.sidebar.success("Using uploaded embeddings")
        X = np.load(uploaded_embeddings).astype("float32")
        emb_source = "Uploaded .npy"
    else:
        X = np.load(CFG.EMB_PATH).astype("float32")
        emb_source = CFG.EMB_PATH
except Exception as e:
    st.error(f"Could not read embeddings: {e}")
    st.info(f"Default expected at: {CFG.EMB_PATH}")
    st.stop()

st.caption(f"Embeddings source: {emb_source}")


# --- Load metadata ---
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


# --- Column detection helper ---
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# Auto-detect columns
seq_col   = pick_col(meta, ["sequence", "aa_seq", "seq", "peptide", "peptide_sequence", "amp_sequence"])
id_col    = pick_col(meta, ["peptide_id", "id", "identifier", "name", "entry", "accession"])
label_col = pick_col(meta, ["source", "label", "class", "family", "dataset", "type"])

# Manual column override if auto-detection fails
st.sidebar.header("Column Override (if auto detection fails)")
cols = list(meta.columns)

if seq_col is None:
    st.sidebar.warning("Sequence column not detected automatically.")
    seq_col = st.sidebar.selectbox("Select sequence column", cols)

if id_col is None:
    id_choice = st.sidebar.selectbox("Select ID column (optional)", ["None"] + cols)
    if id_choice != "None":
        id_col = id_choice

if label_col is None:
    label_choice = st.sidebar.selectbox("Select label column (optional)", ["None"] + cols)
    if label_choice != "None":
        label_col = label_choice


# --- Build plotting dataframe ---
df = pd.DataFrame({
    "id":       meta[id_col].astype(str)    if id_col    else meta.index.astype(str),
    "label":    meta[label_col].astype(str) if label_col else "unknown",
    "sequence": meta[seq_col].astype(str)   if seq_col   else "",
})

# AMP detection
if "is_amp" in meta.columns:
    df["is_amp"] = meta["is_amp"].astype(bool)
elif "AMP" in meta.columns:
    df["is_amp"] = meta["AMP"].astype(int).eq(1)
else:
    df["is_amp"] = False


# --- Row safety check ---
nX, nM = X.shape[0], df.shape[0]
if nX != nM:
    st.warning(f"Row mismatch: embeddings {nX} vs metadata {nM}")
    n  = min(nX, nM)
    X  = X[:n]
    df = df.iloc[:n]


# --- Dimension selection ---
# User picks how many UMAP dimensions to compute (2–8)
st.sidebar.header("UMAP Settings")

CFG.n_components = st.sidebar.selectbox(
    "Dimensions",
    options=[2, 3, 4, 5, 6, 7, 8],
    index=0,
)

# Warn if no Optuna results exist for this dimension
if CFG.n_components not in OPTUNA_PARAMS:
    st.sidebar.warning(
        f"No Optuna results found for {CFG.n_components}D — parameters not validated."
    )


# --- Parameter inputs: synced slider + number input ---
# User can type any value directly; slider provides a quick visual control

st.sidebar.markdown("**n_neighbors**")
col1, col2 = st.sidebar.columns([3, 1])
nn = col1.slider("##nn_slider", 2, 200, CFG.n_neighbors, label_visibility="collapsed")
CFG.n_neighbors = col2.number_input(
    "##nn_input", value=nn, min_value=2, step=1, label_visibility="collapsed"
)

st.sidebar.markdown("**min_dist**")
col1, col2 = st.sidebar.columns([3, 1])
md = col1.slider("##md_slider", 0.0, 1.0, float(CFG.min_dist), 0.001, label_visibility="collapsed")
CFG.min_dist = col2.number_input(
    "##md_input", value=md, min_value=0.0, max_value=1.0, step=0.001,
    format="%.4f", label_visibility="collapsed"
)


# --- Embedding method ---
st.sidebar.header("Embedding Method")
method = st.sidebar.radio("Projection Type", ["UMAP", "densMAP"])

dens_lambda = dens_frac = dens_var_shift = None

if method == "densMAP":
    st.sidebar.markdown("**dens_lambda**")
    col1, col2 = st.sidebar.columns([3, 1])
    dl = col1.slider("##dl_slider", 0.0, 10.0, 2.0, label_visibility="collapsed")
    dens_lambda = col2.number_input(
        "##dl_input", value=dl, min_value=0.0, step=0.1, label_visibility="collapsed"
    )

    st.sidebar.markdown("**dens_frac**")
    col1, col2 = st.sidebar.columns([3, 1])
    df_ = col1.slider("##df_slider", 0.0, 1.0, 0.3, label_visibility="collapsed")
    dens_frac = col2.number_input(
        "##df_input", value=df_, min_value=0.0, max_value=1.0,
        step=0.01, label_visibility="collapsed"
    )

    st.sidebar.markdown("**dens_var_shift**")
    col1, col2 = st.sidebar.columns([3, 1])
    dvs = col1.slider("##dvs_slider", 0.0, 1.0, 0.1, label_visibility="collapsed")
    dens_var_shift = col2.number_input(
        "##dvs_input", value=dvs, min_value=0.0, max_value=1.0,
        step=0.01, label_visibility="collapsed"
    )


# --- 3D visualization controls (only shown for 3D–8D) ---
point_size = 3
opacity    = 0.7
zoom       = 1.5
if CFG.n_components >= 3:
    st.sidebar.header("3D Visualization")
    point_size = st.sidebar.slider("Point size", 1, 10, 3)
    opacity    = st.sidebar.slider("Opacity", 0.1, 1.0, 0.7, 0.05)
    # Zoom moves the camera closer/further from the sphere
    zoom       = st.sidebar.slider("Zoom", 0.5, 3.0, 1.5, 0.1)


# --- Run UMAP (cached per unique parameter combination) ---
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


# --- Baseline metrics ---
st.subheader("Baseline metrics")
st.write("Rows:", X.shape[0])
st.write("Embedding dim:", X.shape[1])
st.write(f"{method} runtime: {runtime:.3f} seconds")


# --- Plotting ---

if CFG.n_components == 2:
    # Standard 2D scatter plot
    plot_df = pd.DataFrame({
        "UMAP1":    X_umap[:, 0],
        "UMAP2":    X_umap[:, 1],
        "id":       df["id"],
        "sequence": df["sequence"],
        "is_amp":   df["is_amp"],
    })

    fig = px.scatter(
        plot_df,
        x="UMAP1", y="UMAP2",
        color="is_amp",
        color_discrete_map={False: "blue", True: "red"},
        hover_data=["id", "sequence", "is_amp"],
        title=f"{method} projection (2D)",
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)

else:
    # 3D–8D: spherize the UMAP output and visualize on a 3D unit sphere
    # For 3D: normalizes directly onto sphere
    # For 4D–8D: normalizes to hypersphere then stereographically projects down to 3D
    X_sphere = project_embeddings_to_sphere(X_umap)

    sphere_df = pd.DataFrame({
        "x":        X_sphere[:, 0],
        "y":        X_sphere[:, 1],
        "z":        X_sphere[:, 2],
        "id":       df["id"],
        "sequence": df["sequence"],
        "is_amp":   df["is_amp"],
    })

    fig = px.scatter_3d(
        sphere_df,
        x="x", y="y", z="z",
        color="is_amp",
        color_discrete_map={False: "blue", True: "red"},
        hover_data=["id", "sequence"],
        title=f"{method} projection ({CFG.n_components}D — Spherized)",
    )

    # Apply user-controlled point size and opacity
    fig.update_traces(marker=dict(size=point_size, opacity=opacity))

    # Force equal axes so sphere isn't squished, clean up gridlines,
    # and apply user-controlled zoom via camera eye distance
    fig.update_layout(
        height=700,  # taller plot for easier interaction
        scene=dict(
            aspectmode="cube",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        ),
        scene_camera=dict(
            eye=dict(x=zoom, y=zoom, z=zoom)
        ),
    )

    st.caption("💡 Tip: Two-finger scroll to zoom, drag to rotate, right-click drag to pan. Use ⛶ in the toolbar to fullscreen.")
    st.plotly_chart(fig, use_container_width=True)


# --- Clear cache ---
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")