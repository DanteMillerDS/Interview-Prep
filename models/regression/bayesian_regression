import numpy as np

class BayesianLinearRegression:
    def __init__(self, prior_mean=None, prior_cov=None, noise_var=1.0):
        """
        Initialize the Bayesian Linear Regression model.
        
        Parameters:
        prior_mean: numpy array of shape (n_features,), prior mean of the weights
        prior_cov: numpy array of shape (n_features, n_features), prior covariance matrix of the weights
        noise_var: float, variance of the noise (assumed to be Gaussian)
        """
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.noise_var = noise_var
        self.posterior_mean = None
        self.posterior_cov = None
    
    def train(self, X_train, y_train):
        """
        Train the Bayesian regression model using the training data.
        
        Parameters:
        X_train: numpy array of shape (n_samples, n_features)
        y_train: numpy array of shape (n_samples,)
        """
        n_samples, n_features = X_train.shape
        
        if self.prior_mean is None:
            self.prior_mean = np.zeros(n_features)
        
        if self.prior_cov is None:
            self.prior_cov = np.eye(n_features)
        
        # Compute the posterior covariance
        prior_cov_inv = np.linalg.inv(self.prior_cov)
        posterior_cov_inv = prior_cov_inv + (1 / self.noise_var) * X_train.T.dot(X_train)
        self.posterior_cov = np.linalg.inv(posterior_cov_inv)
        
        # Compute the posterior mean
        self.posterior_mean = self.posterior_cov.dot(prior_cov_inv.dot(self.prior_mean) + (1 / self.noise_var) * X_train.T.dot(y_train))
    
    def predict(self, X_test):
        """
        Predict the target values using the trained model.
        
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        
        Returns:
        y_pred: numpy array of shape (n_samples,), mean predictions
        y_var: numpy array of shape (n_samples,), variance of predictions
        """
        # Predict the mean of the target distribution
        y_pred = X_test.dot(self.posterior_mean)
        
        # Predict the variance of the target distribution
        y_var = np.sum(X_test.dot(self.posterior_cov) * X_test, axis=1) + self.noise_var
        
        return y_pred, y_var
    
    def test(self, X_test, y_test):
        """
        Test the model using the test data and return the Mean Squared Error.
        
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        y_test: numpy array of shape (n_samples,)
        
        Returns:
        mse: float, Mean Squared Error of the predictions
        """
        y_pred, _ = self.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        return mse
