import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.means = {}  # Store means for each class
        self.covariance_matrix = None  # Shared covariance matrix across all classes
        self.class_priors = {}  # Store prior probabilities for each class
        self.classes = None  # The unique classes in the training set
        self.cov_inv = None  # Inverse of the shared covariance matrix

    def _calculate_priors(self, y_train):
        """Calculate prior probabilities for each class."""
        self.classes, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        for i, cls in enumerate(self.classes):
            self.class_priors[cls] = counts[i] / total_samples

    def _calculate_means(self, X_train, y_train):
        """Calculate the mean of each feature for each class."""
        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            self.means[cls] = np.mean(X_cls, axis=0)

    def _calculate_covariance_matrix(self, X_train, y_train):
        """Calculate the shared covariance matrix for all classes."""
        n_samples, n_features = X_train.shape
        covariance_matrix = np.zeros((n_features, n_features))
        
        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            mean_diff = X_cls - self.means[cls]
            covariance_matrix += np.dot(mean_diff.T, mean_diff)
        
        covariance_matrix /= (n_samples - len(self.classes))
        self.covariance_matrix = covariance_matrix
        self.cov_inv = np.linalg.inv(covariance_matrix)

    def fit(self, X_train, y_train):
        """Fit the LDA model based on the training data."""
        self._calculate_priors(y_train)
        self._calculate_means(X_train, y_train)
        self._calculate_covariance_matrix(X_train, y_train)

    def _discriminant_function(self, x, cls):
        """Calculate the discriminant function for a given class."""
        mean = self.means[cls]
        prior = self.class_priors[cls]
        return np.dot(x.T, np.dot(self.cov_inv, mean)) - 0.5 * np.dot(mean.T, np.dot(self.cov_inv, mean)) + np.log(prior)

    def _predict_sample(self, x):
        """Predict the class label for a single sample."""
        discriminants = {}
        for cls in self.classes:
            discriminants[cls] = self._discriminant_function(x, cls)
        return max(discriminants, key=discriminants.get)

    def predict(self, X_test):
        """Predict the class labels for a given test set."""
        predictions = [self._predict_sample(x) for x in X_test]
        return np.array(predictions)