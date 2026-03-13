import numpy as np
import umap.umap_ as umap
import optuna
from sklearn.manifold import trustworthiness
import json
import os

#paths
EMB_PATH = "embeddings/veltri/veltri_embeddings.npy"
RESULTS_DIR = "results"
RESULT_FILE = os.path.join(RESULTS_DIR, "best_umap_params.json")

# create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)


# Load embeddings
print("Loading embeddings...")

X = np.load(EMB_PATH).astype("float32")

print("Embedding shape:", X.shape)

# Optuna objectie function
def objective(trial):

    # choose method
    method = trial.suggest_categorical("method", ["UMAP", "densMAP"])

    # shared parameters
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)

    reducer_kwargs = dict(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )

    # densMAP-only parameter
    if method == "densMAP":

        dens_lambda = trial.suggest_float("dens_lambda", 0.1, 0.3)

        reducer_kwargs.update(
            densmap=True,
            dens_lambda=dens_lambda
        )

    reducer = umap.UMAP(**reducer_kwargs)

    # run dimensionality reduction
    X_low = reducer.fit_transform(X)

    # compute quality metric
    score = trustworthiness(X, X_low)

    return score


#Run optimizer
print("\nStarting Optuna optimization...")

study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

#Results
print("\nBest parameters:")
print(study.best_params)

print("\nBest trustworthiness score:")
print(study.best_value)

#Save results in a json file
with open(RESULT_FILE, "w") as f:
    json.dump(study.best_params, f, indent=4)

print("\nSaved best parameters to:", RESULT_FILE)