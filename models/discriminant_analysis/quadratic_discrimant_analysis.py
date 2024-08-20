import numpy as np

class QuadraticDiscriminantAnalysis:
    def __init__(self):
        self.means = {}  # Store means for each class
        self.covariance_matrices = {}  # Store covariance matrix for each class
        self.class_priors = {}  # Store prior probabilities for each class
        self.classes = None  # The unique classes in the training set
        self.cov_invs = {}  # Inverse of the covariance matrix for each class
        self.cov_dets = {}  # Determinant of the covariance matrix for each class

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

    def _calculate_covariance_matrices(self, X_train, y_train):
        """Calculate the covariance matrix for each class."""
        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            mean_diff = X_cls - self.means[cls]
            cov_matrix = np.dot(mean_diff.T, mean_diff) / (len(X_cls) - 1)
            self.covariance_matrices[cls] = cov_matrix
            self.cov_invs[cls] = np.linalg.inv(cov_matrix)
            self.cov_dets[cls] = np.linalg.det(cov_matrix)

    def fit(self, X_train, y_train):
        """Fit the QDA model based on the training data."""
        self._calculate_priors(y_train)
        self._calculate_means(X_train, y_train)
        self._calculate_covariance_matrices(X_train, y_train)

    def _discriminant_function(self, x, cls):
        """Calculate the QDA discriminant function for a given class."""
        mean = self.means[cls]
        prior = self.class_priors[cls]
        cov_inv = self.cov_invs[cls]
        cov_det = self.cov_dets[cls]
        
        term1 = -0.5 * np.log(cov_det)
        term2 = -0.5 * np.dot((x - mean).T, np.dot(cov_inv, (x - mean)))
        term3 = np.log(prior)
        
        return term1 + term2 + term3

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