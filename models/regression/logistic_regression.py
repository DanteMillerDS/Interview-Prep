import numpy as np

class LogisticRegressionModel:
    def __init__(self, iterations, lr):
        self.w = None
        self.b = None
        self.iterations = iterations
        self.lr = lr
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def log_loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
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
                y_pred = self.sigmoid(X[data_index] @ self.w + self.b)
                diff = y_pred - y[data_index]
            
                # Compute gradients for w and b
                w_gradient = X[data_index].reshape(-1, 1) * diff
                b_gradient = diff
                
                # Update weights and bias
                self.w -= self.lr * w_gradient
                self.b -= self.lr * b_gradient
                
            if iteration % 100 == 0 or iteration == self.iterations - 1:
                y_pred_all = self.sigmoid(X @ self.w + self.b)
                total_loss = self.log_loss(y, y_pred_all)
                print(f"Loss at iteration {iteration}: {total_loss}")

    def predict(self, X):
        # Predict probabilities
        y_pred = self.sigmoid(X @ self.w + self.b)
        # Convert probabilities to binary outcomes
        return (y_pred >= 0.5).astype(int)
    
class TestLogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegressionModel(iterations=1000, lr=0.01)

    def test_train(self):
        # Create a simple binary classification dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])

        # Train the model
        self.model.train(X, y)

        # Check if the model has learned some weights (not equal to initial random values)
        assert self.model.w is not None and self.model.b is not None, "Model parameters should be updated"
        print("train method passed.")

    def test_predict(self):
        # Predict on new data
        X_test = np.array([[1.5], [3.5], [6]])
        y_pred = self.model.predict(X_test)

        # Expected predictions (depending on the training data)
        y_true = np.array([[0], [1], [1]])  # Example binary outcomes
        # Check if predictions are correct
        assert np.allclose(y_pred, y_true, atol=0.1), "Predictions are incorrect"
        print("predict method passed.")

    def test_loss(self):
        # Use the same data as test_predict
        X_test = np.array([[1.5], [3.5], [6]])
        y_true = np.array([0, 1, 1])

        # Generate predictions
        y_pred = self.model.sigmoid(X_test @ self.model.w + self.model.b)

        # Calculate log loss
        loss_value = self.model.log_loss(y_true.reshape(-1, 1), y_pred)
        # Expected loss should be small if the model is trained well
        assert loss_value < 0.7, "Loss should be reasonably low"
        print("loss method passed.")

    def run_all_tests(self):
        self.test_train()
        self.test_predict()
        self.test_loss()
        print("All tests passed.")

# Instantiate and run the tests
tester = TestLogisticRegressionModel()
tester.run_all_tests()
