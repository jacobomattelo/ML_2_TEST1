import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class EnhancedSVD:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None
        self.U = None
        self.S = None
        self.VT = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, X):
        # handle missing data
        X_imputed = self.imputer.fit_transform(X)

        # scale the data
        X_scaled = self.scaler.fit_transform(X_imputed)

        # center the data
        self.mean = np.mean(X_scaled, axis=0)
        X_centered = X_scaled - self.mean

        # compute the SVD
        self.U, self.S, self.VT = np.linalg.svd(X_centered, full_matrices=False)

        # store the first n_components singular vectors and values
        self.components = self.VT[:self.n_components, :]
        self.singular_values = self.S[:self.n_components]

    def set_components(self, n_components):
        # Update the number of components
        self.n_components = n_components
        self.components = self.VT[:self.n_components, :]
        self.singular_values = self.S[:self.n_components]

    def transform(self, X):
        # handle missing data
        X_imputed = self.imputer.transform(X)

        # scale the data
        X_scaled = self.scaler.transform(X_imputed)

        # center the data
        X_centered = X_scaled - self.mean

        # project the data onto the singular vectors
        X_transformed = np.dot(X_centered, self.components.T)

        return X_transformed

    def inverse_transform(self, X):
        # reconstruct from the transformed data
        X_reconstructed = np.dot(X, self.components) + self.mean

        # inverse scale the data
        X_reconstructed = self.scaler.inverse_transform(X_reconstructed)

        return X_reconstructed

    def fit_transform(self, X):
        # first fit the model to the data
        self.fit(X)

        # then transform the data
        return self.transform(X)