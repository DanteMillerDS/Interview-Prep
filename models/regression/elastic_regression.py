import numpy as np

class ElasticNetRegressionModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        """
        Initialize the Elastic Net Regression model.
        
        Parameters:
        alpha: float, regularization strength (default is 1.0)
        l1_ratio: float, the mix ratio between Lasso (L1) and Ridge (L2) regularization.
                  l1_ratio = 1.0 corresponds to Lasso, and l1_ratio = 0.0 corresponds to Ridge.
        max_iter: int, maximum number of iterations for coordinate descent (default is 1000)
        tol: float, tolerance for stopping criteria (default is 1e-4)
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
    
    def train(self, X_train, y_train):
        """
        Train the Elastic Net regression model using the training data.
        
        Parameters:
        X_train: numpy array of shape (n_samples, n_features)
        y_train: numpy array of shape (n_samples,)
        """
        n_samples, n_features = X_train.shape
        
        # Add bias term to the input features
        X_train_bias = np.c_[np.ones((n_samples, 1)), X_train]
        
        # Initialize weights
        self.weights = np.zeros(X_train_bias.shape[1])
        
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()
            
            for j in range(X_train_bias.shape[1]):
                residual = y_train - X_train_bias.dot(self.weights)
                rho_j = X_train_bias[:, j].dot(residual + self.weights[j] * X_train_bias[:, j])
                
                if j == 0:  # No regularization on the bias term
                    self.weights[j] = rho_j / X_train_bias[:, j].dot(X_train_bias[:, j])
                else:
                    # Apply both L1 (Lasso) and L2 (Ridge) penalties
                    self.weights[j] = self._soft_threshold(rho_j, self.alpha * self.l1_ratio) / (
                        X_train_bias[:, j].dot(X_train_bias[:, j]) + self.alpha * (1 - self.l1_ratio)
                    )
            
            # Check for convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                break
    
    def _soft_threshold(self, rho, alpha):
        """
        Soft thresholding function for Lasso part of Elastic Net.
        
        Parameters:
        rho: float, calculated residual
        alpha: float, regularization strength
        
        Returns:
        float, updated weight for the feature
        """
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        else:
            return 0.0
    
    def predict(self, X_test):
        """
        Predict the target values using the trained model.
        
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        
        Returns:
        y_pred: numpy array of shape (n_samples,)
        """
        n_samples = X_test.shape[0]
        
        # Add bias term to the input features
        X_test_bias = np.c_[np.ones((n_samples, 1)), X_test]
        
        # Predict using the elastic net regression model
        y_pred = X_test_bias.dot(self.weights)
        return y_pred
    
    def test(self, X_test, y_test):
        """
        Test the model using the test data and return the Mean Squared Error.
        
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        y_test: numpy array of shape (n_samples,)
        
        Returns:
        mse: float, Mean Squared Error of the predictions
        """
        y_pred = self.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        return mse
