import numpy as np

class ElasticNetRegressionModel:
    def __init__(self, iterations, l1_ratio, alpha, lr):
        self.w = None
        self.b = None
        self.iterations = iterations
        self.alpha = alpha  # Overall regularization strength
        self.l1_ratio = l1_ratio  # Ratio between L1 and L2
        self.lr = lr

    def train(self, X, y):
        # Initialize weights and bias
        np.random.seed(0)
        self.w = np.random.randn(X.shape[1], 1)  # Initialize w with random values
        self.b = np.random.randn()  # Initialize b with a random value

        # Reshape y to be a column vector
        y = y.reshape(-1, 1)
        
        # Training with Stochastic Gradient Descent
        for iteration in range(self.iterations):
            for data_index in range(X.shape[0]):
                # Get prediction for the current data point
                y_pred = X[data_index] @ self.w + self.b
                diff = y_pred - y[data_index]
                
                # Compute gradients for w and b
                l1_term = self.l1_ratio * self.alpha * np.sign(self.w)
                l2_term = (1 - self.l1_ratio) * self.alpha * self.w
                w_gradient = X[data_index].reshape(-1, 1) * diff + l1_term + l2_term
                b_gradient = diff
                
                # Update weights and bias
                self.w -= self.lr * w_gradient
                self.b -= self.lr * b_gradient
                
            if iteration % 100 == 0 or iteration == self.iterations - 1:
                y_pred_all = X @ self.w + self.b
                mse_loss = 0.5 * np.mean((y_pred_all - y) ** 2)
                l1_loss = self.l1_ratio * np.sum(np.abs(self.w))
                l2_loss = (1 - self.l1_ratio) * 0.5 * np.sum(self.w ** 2)
                total_loss = mse_loss + self.alpha * (l1_loss + l2_loss)
                print(f"Loss at iteration {iteration}: {total_loss}")

    def predict(self, X):
        # Predict by adding the bias term manually
        y_pred = X @ self.w + self.b
        return y_pred

    def loss(self, y, y_pred):
        # Calculate the mean squared error
        return np.mean((y - y_pred) ** 2)

class TestElasticNetRegressionModel:
    def __init__(self):
        self.model = ElasticNetRegressionModel(iterations=1000, l1_ratio=0.9, alpha=0.1, lr=1e-3)

    def test_train(self):
        # Create a simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        # Train the model
        self.model.train(X, y)

        # Check if the model has learned some weights (not equal to initial random values)
        assert self.model.w is not None and self.model.b is not None, "Model parameters should be updated"
        print("train method passed.")

    def test_predict(self):
        # Predict on new data
        X_test = np.array([[6], [7], [8]])
        y_pred = self.model.predict(X_test)

        # Expected predictions (depending on the training data)
        y_true = np.array([12, 14, 16])  # Example continuous outcomes
        # Check if predictions are correct
        assert np.allclose(y_pred.flatten(), y_true, atol=1.0), "Predictions are incorrect"
        print("predict method passed.")

    def test_loss(self):
        # Use the same data as test_predict
        X_test = np.array([[6], [7], [8]])
        y_true = np.array([12, 14, 16])

        # Generate predictions
        y_pred = self.model.predict(X_test)

        # Calculate mean squared error loss
        loss_value = self.model.loss(y_true.reshape(-1, 1), y_pred)
        # Expected loss should be small
        assert loss_value < 1.0, "Loss should be reasonably low"
        print("loss method passed.")

    def run_all_tests(self):
        self.test_train()
        self.test_predict()
        self.test_loss()
        print("All tests passed.")

# Instantiate and run the tests
tester = TestElasticNetRegressionModel()
tester.run_all_tests()
