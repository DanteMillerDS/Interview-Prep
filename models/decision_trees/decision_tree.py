import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def _gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / m
        return 1 - np.sum(probabilities ** 2)

    def _information_gain(self, y, left_y, right_y):
        p = len(left_y) / len(y)
        return self._gini_impurity(y) - p * self._gini_impurity(left_y) - (1 - p) * self._gini_impurity(right_y)

    def _best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, -1
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = X[:, feature] >= threshold
                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = self._information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or num_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = np.argmax(np.bincount(y))
            return self.Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            leaf_value = np.argmax(np.bincount(y))
            return self.Node(value=leaf_value)

        left_mask = X[:, feature] < threshold
        right_mask = X[:, feature] >= threshold
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return self.Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X_train, y_train):
        self.tree = self._build_tree(X_train, y_train, 0)

    def _predict_one(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X_test):
        return np.array([self._predict_one(self.tree, x) for x in X_test])

    def predict_proba(self, X_test):
        # Simplified version for probabilities (not a standard method for decision trees)
        predictions = self.predict(X_test)
        return np.array([np.bincount(predictions, minlength=2) / len(predictions) for _ in predictions])
