import numpy as np

class RidgeRegressionModel:
    def __init__(self, l):
        self.w = None
        self.b = None
        self.l = l
    
    def train(self, X, y):
        # Add a column of ones to X to account for the bias term
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y = y.reshape(-1, 1)
        if self.l is None:
            self.l = 0
        identity = np.identity(X.shape[1])
        identity[-1, -1] = 0
        # Compute the weights using the normal equation
        self.w = np.linalg.inv(X.T @ X + self.l * identity) @ X.T @ y
        self.b = self.w[-1]  # Extract the bias term
        self.w = self.w[:-1]  # Keep the weights separate from the bias

    def predict(self, X):
        # Predict by adding the bias term manually
        y_pred = X @ self.w + self.b
        return y_pred

    def loss(self, y, y_pred):
        # Calculate the mean squared error
        return np.mean((y - y_pred) ** 2)

class TestRidgeRegressionModel:
    def __init__(self):
        self.model = RidgeRegressionModel(l=0.1)  # Initialize with a regularization parameter

    def test_train(self):
        # Create a simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        # Train the model
        self.model.train(X, y)

        # Check if weights are close to expected (considering regularization)
        expected_w = np.array([1.98])  # Slightly less than 2 due to regularization
        assert np.allclose(self.model.w, expected_w, atol=0.05), "Weight should be close to 2"
        assert np.isclose(self.model.b, 0, atol=0.1), "Bias should be close to 0"
        print("train method passed.")

    def test_predict(self):
        # Predict on new data
        X_test = np.array([[6], [7], [8]])
        y_pred = self.model.predict(X_test)

        # Expected predictions with regularization effect
        y_true = np.array([[11.9952], [13.9944], [15.9936]])  # Slightly less than [12, 14, 16]
        assert np.allclose(y_pred, y_true, atol=0.1), "Predictions are incorrect"
        print("predict method passed.")

    def test_loss(self):
        # Use the same data as test_predict
        X_test = np.array([[6], [7], [8]])
        y_true = np.array([[11.9952], [13.9944], [15.9936]])

        # Generate predictions
        y_pred = self.model.predict(X_test)

        # Calculate loss
        loss_value = self.model.loss(y_true, y_pred)
        # Expected loss should be very close to 0
        assert np.isclose(loss_value, 0, atol=0.05), "Loss should be close to 0"
        print("loss method passed.")

    def run_all_tests(self):
        self.test_train()
        self.test_predict()
        self.test_loss()
        print("All tests passed.")

# Instantiate and run the tests
tester = TestRidgeRegressionModel()
tester.run_all_tests()
