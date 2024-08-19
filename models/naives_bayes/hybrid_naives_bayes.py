import numpy as np

class HybridNaiveBayes:
    def __init__(self):
        self.gaussian_stats = {}  # To store means and variances for Gaussian Naive Bayes (numerical)
        self.bernoulli_probs = {}  # To store probabilities for Bernoulli Naive Bayes (binary)
        self.categorical_probs = {}  # To store probabilities for Multinomial Naive Bayes (categorical)
        self.class_priors = {}  # Store the prior probabilities for each class
        self.classes = None  # The unique classes in the training set

    def _calculate_priors(self, y_train):
        # Calculate prior probabilities P(Class)
        self.classes, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        for i, cls in enumerate(self.classes):
            self.class_priors[cls] = counts[i] / total_samples

    def _fit_gaussian(self, X_train, y_train, numerical_indices):
        # Calculate means and variances for numerical features using Gaussian Naive Bayes
        for cls in self.classes:
            X_cls = X_train[y_train == cls, :]  # Subset for each class
            self.gaussian_stats[cls] = {}
            for i in numerical_indices:
                feature = X_cls[:, i]
                self.gaussian_stats[cls][i] = {
                    'mean': np.mean(feature),
                    'var': np.var(feature)
                }

    def _fit_bernoulli(self, X_train, y_train, binary_indices):
        # Calculate probabilities for Bernoulli Naive Bayes (binary features)
        for cls in self.classes:
            X_cls = X_train[y_train == cls, :]  # Subset for each class
            self.bernoulli_probs[cls] = {}
            for i in binary_indices:
                feature = X_cls[:, i]
                # Bernoulli probability P(feature = 1 | class)
                self.bernoulli_probs[cls][i] = np.mean(feature)

    def _fit_multinomial(self, X_train, y_train, categorical_indices):
        # Calculate probabilities for Multinomial Naive Bayes (categorical features)
        for cls in self.classes:
            X_cls = X_train[y_train == cls, :]  # Subset for each class
            self.categorical_probs[cls] = {}
            for i in categorical_indices:
                feature = X_cls[:, i]
                unique_vals, counts = np.unique(feature, return_counts=True)
                self.categorical_probs[cls][i] = {
                    val: count / len(feature) for val, count in zip(unique_vals, counts)
                }

    def fit(self, X_train, y_train, feature_types):
        """
        Fit the Naive Bayes model based on the training data and feature types.
        Parameters:
        X_train: numpy array of shape (n_samples, n_features)
        y_train: numpy array of shape (n_samples,)
        feature_types: list specifying type for each feature: 'numerical', 'binary', or 'categorical'
        """
        numerical_indices = [i for i, t in enumerate(feature_types) if t == 'numerical']
        binary_indices = [i for i, t in enumerate(feature_types) if t == 'binary']
        categorical_indices = [i for i, t in enumerate(feature_types) if t == 'categorical']

        self._calculate_priors(y_train)

        if numerical_indices:
            self._fit_gaussian(X_train, y_train, numerical_indices)
        if binary_indices:
            self._fit_bernoulli(X_train, y_train, binary_indices)
        if categorical_indices:
            self._fit_multinomial(X_train, y_train, categorical_indices)

    def _gaussian_likelihood(self, x, mean, var):
        # Calculate Gaussian probability density function
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return coeff * exponent

    def _predict_sample(self, x, feature_types):
        posteriors = {}
        for cls in self.classes:
            # Start with the prior
            posterior = np.log(self.class_priors[cls])

            for i, feature_type in enumerate(feature_types):
                if feature_type == 'numerical':
                    mean = self.gaussian_stats[cls][i]['mean']
                    var = self.gaussian_stats[cls][i]['var']
                    posterior += np.log(self._gaussian_likelihood(x[i], mean, var))

                elif feature_type == 'binary':
                    prob = self.bernoulli_probs[cls][i]
                    if x[i] == 1:
                        posterior += np.log(prob)
                    else:
                        posterior += np.log(1 - prob)

                elif feature_type == 'categorical':
                    val_probs = self.categorical_probs[cls][i]
                    if x[i] in val_probs:
                        posterior += np.log(val_probs[x[i]])
                    else:
                        posterior += np.log(1e-6)  # Smoothing for unseen categories

            posteriors[cls] = posterior

        return max(posteriors, key=posteriors.get)

    def predict(self, X_test, feature_types):
        """
        Predict the class for each sample in X_test.
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        feature_types: list specifying type for each feature: 'numerical', 'binary', or 'categorical'
        Returns:
        predictions: numpy array of predicted class labels
        """
        predictions = [self._predict_sample(x, feature_types) for x in X_test]
        return np.array(predictions)