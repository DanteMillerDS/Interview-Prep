import numpy as np
from collections import Counter

class KNearestNeighborsRegressor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Store the training data."""
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_sample(self, x):
        """Predict the value for a single sample."""
        # Calculate distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the values of the k nearest neighbors
        k_nearest_values = [self.y_train[i] for i in k_indices]
        
        # Return the mean of the k nearest values
        return np.mean(k_nearest_values)

    def predict(self, X_test):
        """Predict the values for a given test set."""
        predictions = [self._predict_sample(x) for x in X_test]
        return np.array(predictions)