import numpy as np

class BayesianLinearRegression:
    def __init__(self, prior_mean, prior_cov, noise_var):
        self.prior_mean = prior_mean     # Prior mean (mu_0)
        self.prior_cov = prior_cov       # Prior covariance matrix (Sigma_0)
        self.noise_var = noise_var       # Variance of the noise (sigma^2)
        self.posterior_mean = None       # Placeholder for posterior mean (mu_n)
        self.posterior_cov = None        # Placeholder for posterior covariance (Sigma_n)

    def train(self, X, y):
        # Inverting the prior covariance matrix (Sigma_0)
        prior_cov_inv = np.linalg.inv(self.prior_cov)
        
        # Calculating X^T * X
        XT_X = X.T @ X
        
        # Adding prior precision and data precision (posterior covariance)
        posterior_cov_inv = prior_cov_inv + (1 / self.noise_var) * XT_X
        
        # Inverting to get posterior covariance (Sigma_n)
        self.posterior_cov = np.linalg.inv(posterior_cov_inv)
        
        # Computing the posterior mean (mu_n)
        XT_y = X.T @ y
        self.posterior_mean = self.posterior_cov @ (prior_cov_inv @ self.prior_mean + (1 / self.noise_var) * XT_y)

    def predict(self, X_new):
        # Predicting the mean of the new data points
        y_pred_mean = X_new @ self.posterior_mean
        
        # Predicting the variance of the new data points
        y_pred_var = np.array([X_new[i] @ self.posterior_cov @ X_new[i].T for i in range(X_new.shape[0])]) + self.noise_var
        
        return y_pred_mean, y_pred_var

    def loss(self, y_true, y_pred):
        # Calculate the mean squared error
        return np.mean((y_true - y_pred) ** 2)

class TestBayesianLinearRegression:
    def __init__(self):
        prior_mean = np.zeros((1, 1))  # Prior mean for weights (1 feature)
        prior_cov = np.eye(1)          # Prior covariance matrix (identity matrix for simplicity)
        noise_var = 1                  # Variance of the noise
        self.model = BayesianLinearRegression(prior_mean, prior_cov, noise_var)

    def test_train(self):
        # Create a simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)

        # Train the model
        self.model.train(X, y)

        # Expected mean and covariance (can be manually calculated or approximated)
        expected_mean = np.array([[2]])  # Ideally close to 2 after training
        expected_cov = np.array([[0.2]]) # Posterior covariance should be reduced after training
        assert np.allclose(self.model.posterior_mean, expected_mean, atol=0.2), "Posterior mean should be close to expected"
        assert np.allclose(self.model.posterior_cov, expected_cov, atol=0.2), "Posterior covariance should be close to expected"
        print("train method passed.")

    def test_predict(self):
        # Predict on new data
        X_test = np.array([[6], [7], [8]])
        y_pred_mean, y_pred_var = self.model.predict(X_test)

        # Expected predictions (mean and variance)
        expected_mean = np.array([12, 14, 16]).reshape(-1, 1)
        assert np.allclose(y_pred_mean, expected_mean, atol=1), "Predicted mean is incorrect"
        print("predict method passed.")

    def test_loss(self):
        # Use the same data as test_predict
        X_test = np.array([[6], [7], [8]])
        y_true = np.array([12, 14, 16]).reshape(-1, 1)

        # Generate predictions
        y_pred_mean, _ = self.model.predict(X_test)

        # Calculate loss
        loss_value = self.model.loss(y_true, y_pred_mean)
        assert np.isclose(loss_value, 0, atol=0.1), "Loss should be close to 0"
        print("loss method passed.")

    def run_all_tests(self):
        self.test_train()
        self.test_predict()
        self.test_loss()
        print("All tests passed.")

# Instantiate and run the tests
tester = TestBayesianLinearRegression()
tester.run_all_tests()
