import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.noise_var = noise_var
        self.posterior_mean = None
        self.posterior_cov = None

    def train(self, X, y):
        if not prior mean
            self.prior_mean = np.zeros(n_features)
        if not prior_cov
            self.prior_cov = np.eye(n_features)
        
        

    def predict(self, X):
        # Predict by adding the bias term manually
        y_pred = X @ self.w + self.b
        return y_pred

    def loss(self, y, y_pred):
        # Calculate the mean squared error
        return np.mean((y - y_pred) ** 2)

class TestLinearRegressionModel:
    def __init__(self):
        self.model = LinearRegressionModel(None, None)

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
tester = TestLinearRegressionModel()
tester.run_all_tests()