import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k  # Number of clusters
        self.max_iters = max_iters  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        self.centroids = None  # To store the centroid positions
        self.labels = None  # To store the label of each data point

    def fit(self, X):
        """Compute k-means clustering."""
        n_samples, n_features = X.shape
        
        # Initialize centroids by randomly selecting k data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Assign each sample to the nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Save the old centroids
            old_centroids = self.centroids.copy()
            
            # Compute the new centroids
            self.centroids = self._compute_centroids(X)
            
            # Check for convergence
            if self._is_converged(old_centroids, self.centroids):
                break

    def _assign_clusters(self, X):
        """Assign each data point to the closest centroid."""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        """Compute the distance from each point to each centroid."""
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return distances

    def _compute_centroids(self, X):
        """Compute the new centroids as the mean of all points assigned to each cluster."""
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        """Check if the centroids have moved less than the tolerance."""
        movement = np.linalg.norm(new_centroids - old_centroids, axis=1)
        return np.all(movement < self.tol)

    def predict(self, X):
        """Assign clusters to the input data."""
        return self._assign_clusters(X)

    def get_centroids(self):
        """Return the centroids of the clusters."""
        return self.centroids
