import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _initialize_parameters(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def _decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def _hinge_loss_gradient(self, X, y, idx):
        condition = y[idx] * self._decision_function(X[idx]) >= 1
        if condition:
            return 2 * self.lambda_param * self.w, 0
        else:
            return 2 * self.lambda_param * self.w - np.dot(X[idx], y[idx]), y[idx]

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self._initialize_parameters(n_features)

        y_ = np.where(y_train <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                grad_w, grad_b = self._hinge_loss_gradient(X_train, y_, idx)
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self, X_test):
        approx = self._decision_function(X_test)
        return np.sign(approx)

    def predict_proba(self, X_test):
        decision_values = self._decision_function(X_test)
        return 1 / (1 + np.exp(-decision_values))  # Sigmoid for probability estimate
