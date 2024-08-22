import numpy as np

class LassoRegressionModel:
    def __init__(self, iterations, l, alpha, lr):
        self.w = None
        self.b = None
        self.iterations = iterations
        self.alpha = alpha
        self.l = l
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
                w_gradient = X[data_index].reshape(-1, 1) * diff + self.l* np.abs(self.w) + self.alpha * np.sign(self.w)
                b_gradient = diff
                
                # Update weights and bias
                self.w -= self.lr * w_gradient
                self.b -= self.lr * b_gradient
                
            if iteration % 100 == 0 or iteration == self.iterations - 1:
                total_loss = 0
                for i in range(X.shape[0]):
                    y_pred = X[i] @ self.w + self.b
                    total_loss += 0.5 * (y_pred - y[i])**2
                total_loss += self.alpha * np.sum(np.abs(self.w))
                print(f"Loss at iteration {iteration}: {total_loss}")

    def predict(self, X):
        # Predict by adding the bias term manually
        y_pred = X @ self.w + self.b
        return y_pred

    def loss(self, y, y_pred):
        # Calculate the mean squared error
        return np.mean((y - y_pred) ** 2)

class TestLassoRegressionModel:
    def __init__(self):
        self.model = LassoRegressionModel(iterations=1000, alpha=0.1, lr=1e-3)

    def test_train(self):
        # Create a simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        # Train the model
        self.model.train(X, y)

        # Check if weights are approximately correct considering L1 regularization
        expected_w = np.array([[1.9]])  # Expect approximately 2, but Lasso might shrink it slightly
        assert np.allclose(self.model.w, expected_w, atol=0.1), "Weight should be close to 0.2"
        assert np.isclose(self.model.b, 0.2, atol=0.1), "Bias should be close to 0"
        print("train method passed.")

    def test_predict(self):
        # Predict on new data
        X_test = np.array([[6], [7], [8]])
        y_pred = self.model.predict(X_test)

        # Expected predictions
        y_true = np.array([[12], [14], [16]])
        # Check if predictions are correct
        assert np.allclose(y_pred, y_true, atol=1), "Predictions are incorrect"
        print("predict method passed.")

    def test_loss(self):
        # Use the same data as test_predict
        X_test = np.array([[6], [7], [8]])
        y_true = np.array([[12], [14], [16]])

        # Generate predictions
        y_pred = self.model.predict(X_test)

        # Calculate loss
        loss_value = self.model.loss(y_true, y_pred)
        # Expected loss should be small
        assert np.isclose(loss_value, 0, atol=0.1), "Loss should be close to 0"
        print("loss method passed.")

    def run_all_tests(self):
        self.test_train()
        self.test_predict()
        self.test_loss()
        print("All tests passed.")

# Instantiate and run the tests
tester = TestLassoRegressionModel()
tester.run_all_tests()
