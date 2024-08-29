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
        return np.dot(X, self.w) + self.b

    def _hinge_loss_gradient(self, X, y, idx):
        condition = y[idx] * self._decision_function(X[idx]) >= 1
        if condition:
            grad_w = 2 * self.lambda_param * self.w  # Regularization gradient only
            grad_b = 0  # No gradient for bias when condition is satisfied
        else:
            grad_w = 2 * self.lambda_param * self.w - np.dot(X[idx], y[idx])
            grad_b = -y[idx]
        return grad_w, grad_b

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

# Test cases for SVM model
class TestSVMModel:
    def __init__(self):
        self.model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)

    def test_train(self):
        # Create a simple dataset
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([1, 1, -1, -1, 1])

        # Train the model
        self.model.fit(X, y)

        # Check if weights and bias are set
        assert self.model.w is not None, "Weights should not be None"
        assert self.model.b is not None, "Bias should not be None"
        print("train method passed.")

    def test_predict(self):
        # Predict on new data
        X_test = np.array([[2, 3], [3, 4], [6, 7]])
        y_pred = self.model.predict(X_test)
        
        # Since we don't have a ground truth, let's check the sign of the predictions
        assert np.array_equal(np.sign(y_pred), np.sign(np.array([1, 1, 1]))), "Predictions are incorrect"
        print("predict method passed.")

    def test_predict_proba(self):
        # Predict probabilities on new data
        X_test = np.array([[2, 3], [3, 4], [6, 7]])
        y_proba = self.model.predict_proba(X_test)

        # The probabilities should be between 0 and 1
        assert np.all((y_proba >= 0) & (y_proba <= 1)), "Probabilities should be between 0 and 1"
        print("predict_proba method passed.")

    def run_all_tests(self):
        self.test_train()
        self.test_predict()
        self.test_predict_proba()
        print("All tests passed.")

# Instantiate and run the tests
tester = TestSVMModel()
tester.run_all_tests()
