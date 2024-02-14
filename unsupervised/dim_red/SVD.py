import numpy as np

class SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # compute the SVD
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        # store the first n_components singular vectors
        self.components = VT[:self.n_components, :]

        # store the first n_components singular values
        self.singular_values = S[:self.n_components]

    def transform(self, X):
        # center the data
        X_centered = X - self.mean

        # project the data onto the singular vectors
        X_transformed = np.dot(X_centered, self.components.T)

        return X_transformed

    def inverse_transform(self, X):
        # reconstruct from the transformed data
        X_reconstructed = np.dot(X, self.components) + self.mean

        return X_reconstructed