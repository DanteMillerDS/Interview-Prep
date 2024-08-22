import numpy as np

class LassoRegressionModel:
    def __init__(self, iterations, alpha):
        self.w = None
        self.b = 0
        self.iterations = iterations
        self.alpha = alpha

    def train(self, X, y):
        # Initialize weights and bias
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0

        # Reshape y to be a column vector
        y = y.reshape(-1, 1)
        
        # Training with Gradient Descent
        for iteration in range(self.iterations):
            y_pred = X @ self.w + self.b
            diff = y_pred - y

            # Compute gradients for w and b
            w_gradient = X.T @ diff / len(y) + self.alpha * np.sign(self.w)
            b_gradient = np.mean(diff)
            
            # Update weights and bias
            self.w -= self.alpha * w_gradient
            self.b -= self.alpha * b_gradient
            
            # Optionally, you could print the loss every few iterations
            if iteration % 100 == 0 or iteration == self.iterations - 1:
                loss = self.loss(y, y_pred) + self.alpha * np.sum(np.abs(self.w))
                print(f"Loss at iteration {iteration}: {loss}")

    def predict(self, X):
        # Predict by adding the bias term manually
        y_pred = X @ self.w + self.b
        return y_pred

    def loss(self, y, y_pred):
        # Calculate the mean squared error
        return np.mean((y - y_pred) ** 2)

class TestLassoRegressionModel:
    def __init__(self):
        self.model = LassoRegressionModel(None, None)

    def test_train(self):
        # Create a simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        # Train the model
        self.model.train(X, y)

        # Check if weights are correct (slope should be 2, bias should be 0)
        assert np.allclose(self.model.w, np.array([2])), "Weight should be 2"
        assert np.isclose(self.model.b, 0), "Bias should be 0"
        print("train method passed.")

    def test_predict(self):
        # Predict on new data
        X_test = np.array([[6], [7], [8]])
        y_pred = self.model.predict(X_test)

        # Expected predictions
        y_true = np.array([[12], [14], [16]])
        # Check if predictions are correct
        assert np.allclose(y_pred, y_true), "Predictions are incorrect"
        print("predict method passed.")

    def test_loss(self):
        # Use the same data as test_predict
        X_test = np.array([[6], [7], [8]])
        y_true = np.array([[12], [14], [16]])

        # Generate predictions
        y_pred = self.model.predict(X_test)

        # Calculate loss
        loss_value = self.model.loss(y_true, y_pred)
        # Expected loss should be 0 since the model fits the data perfectly
        assert np.isclose(loss_value, 0), "Loss should be 0"
        print("loss method passed.")

    def run_all_tests(self):
        self.test_train()
        self.test_predict()
        self.test_loss()
        print("All tests passed.")

# Instantiate and run the tests
tester = TestLassoRegressionModel()
tester.run_all_tests()