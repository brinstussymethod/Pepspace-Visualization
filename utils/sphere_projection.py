import numpy as np


def project_embeddings_to_sphere(X):
    """
    Normalize N-dimensional UMAP embeddings onto a unit hypersphere,
    then stereographically project down to 3D for visualization.

    Works for any input dimension 3-8.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_components) — output from UMAP

    Returns
    -------
    np.ndarray
        Shape (n_samples, 3) — points projected onto a 3D unit sphere
    """

    # Step 1: Normalize to unit length in native N dimensions
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Step 2: If already 3D, return as is
    if X_norm.shape[1] == 3:
        return X_norm

    # Step 3: Iteratively project from ND down to 3D via stereographic projection
    while X_norm.shape[1] > 3:
        w = X_norm[:, -1]  # take last dimension as projection pole
        X_norm = X_norm[:, :-1] / (1 - w[:, np.newaxis])
        # Re-normalize after each step to stay on the sphere
        X_norm = X_norm / np.linalg.norm(X_norm, axis=1, keepdims=True)

    return X_norm