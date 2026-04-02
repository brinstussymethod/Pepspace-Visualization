import numpy as np
import umap.umap_ as umap
import optuna
from sklearn.manifold import trustworthiness
import json
import os
from datetime import datetime

# Paths
EMB_PATH = "embeddings/veltri/veltri_embeddings.npy"
RESULTS_DIR = "optuna_results"
RESULT_FILE = os.path.join(RESULTS_DIR, "all_umap_results.json")

# Dimension to optimize — change this before each run
N_COMPONENTS = 5

# Create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load embeddings 
print("Loading embeddings...")
X = np.load(EMB_PATH).astype("float32")
print("Embedding shape:", X.shape)


# Optuna objective function 
def objective(trial):

    # Choose method
    method = trial.suggest_categorical("method", ["UMAP", "densMAP"])

    # Shared parameters
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    min_dist    = trial.suggest_float("min_dist", 0.0, 0.5)

    reducer_kwargs = dict(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=N_COMPONENTS,
        metric="cosine",
        random_state=42,
    )

    # densMAP-only parameter
    if method == "densMAP":
        dens_lambda = trial.suggest_float("dens_lambda", 0.1, 0.3)
        reducer_kwargs.update(
            densmap=True,
            dens_lambda=dens_lambda,
        )

    reducer = umap.UMAP(**reducer_kwargs)
    X_low   = reducer.fit_transform(X)
    score   = trustworthiness(X, X_low)

    return score


# Run optimizer 
print(f"\nStarting Optuna optimization for {N_COMPONENTS}D...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=75)

print("\nBest parameters:")
print(study.best_params)
print("\nBest trustworthiness score:")
print(study.best_value)


# Build result entry 
new_entry = {
    f"{N_COMPONENTS}d": {
        "best_params": study.best_params,
        "best_score":  study.best_value,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
}


# Load existing results file if it exists, then append
if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "r") as f:
        all_results = json.load(f)
else:
    all_results = {}

# Overwrite if this dimension was already run
all_results.update(new_entry)

with open(RESULT_FILE, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nSaved {N_COMPONENTS}D results to: {RESULT_FILE}")