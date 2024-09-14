import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components  # Number of components to keep
        self.mean = None  # Mean of the data
        self.components = None  # Principal components
        self.explained_variance = None  # Explained variance of each component

    def fit(self, X):
        """Fit the PCA model on the data."""
        # Step 1: Standardize the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Step 3: Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Step 4: Sort the eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Step 5: Select the top `n_components` eigenvectors
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
        
        self.components = eigenvectors
        self.explained_variance = eigenvalues

    def transform(self, X):
        """Transform the data to the new PCA space."""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """Fit the PCA model and transform the data."""
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio_(self):
        """Return the ratio of variance explained by each of the selected components."""
        total_variance = np.sum(self.explained_variance)
        return self.explained_variance / total_variance
