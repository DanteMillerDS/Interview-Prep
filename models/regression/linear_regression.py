import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def train(self, X_train, y_train):
        """
        Train the linear regression model using the training data.
        
        Parameters:
        X_train: numpy array of shape (n_samples, n_features)
        y_train: numpy array of shape (n_samples,)
        """
        n_samples, n_features = X_train.shape
        
        # Add bias term to the input features
        X_train_bias = np.c_[np.ones((n_samples, 1)), X_train]
        
        # Calculate weights using the Normal Equation
        X_transpose = X_train_bias.T
        self.weights = np.linalg.inv(X_transpose.dot(X_train_bias)).dot(X_transpose).dot(y_train)
    
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
        
        # Predict using the linear model
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

 