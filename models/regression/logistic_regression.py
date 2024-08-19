import numpy as np

class LogisticRegressionModel:
    def __init__(self, max_iter=1000, learning_rate=0.01, tol=1e-4):
        """
        Initialize the Logistic Regression model.
        
        Parameters:
        max_iter: int, maximum number of iterations for gradient descent (default is 1000)
        learning_rate: float, learning rate for gradient descent (default is 0.01)
        tol: float, tolerance for stopping criteria (default is 1e-4)
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.weights = None
    
    def sigmoid(self, z):
        """
        Apply the sigmoid function to convert the linear output into probabilities.
        
        Parameters:
        z: numpy array, linear combination of input features and weights
        
        Returns:
        sigmoid: numpy array, output of the sigmoid function
        """
        return 1 / (1 + np.exp(-z))
    
    def train(self, X_train, y_train):
        """
        Train the Logistic Regression model using the training data.
        
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
            
            # Linear combination of input features and weights
            linear_model = np.dot(X_train_bias, self.weights)
            # Apply the sigmoid function
            y_pred = self.sigmoid(linear_model)
            
            # Compute the gradient of the Binary Cross Entropy loss
            gradient = np.dot(X_train_bias.T, (y_pred - y_train)) / n_samples
            
            # Update weights using gradient descent
            self.weights -= self.learning_rate * gradient
            
            # Check for convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                print(f"Converged after {iteration} iterations.")
                break
    
    def predict_proba(self, X_test):
        """
        Predict the probabilities of the target values using the trained model.
        
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        
        Returns:
        y_pred_proba: numpy array of shape (n_samples,)
        """
        n_samples = X_test.shape[0]
        
        # Add bias term to the input features
        X_test_bias = np.c_[np.ones((n_samples, 1)), X_test]
        
        # Predict the probabilities using the logistic regression model
        linear_model = np.dot(X_test_bias, self.weights)
        y_pred_proba = self.sigmoid(linear_model)
        
        return y_pred_proba
    
    def predict(self, X_test, threshold=0.5):
        """
        Predict the binary target values using the trained model.
        
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        threshold: float, decision threshold to classify the output (default is 0.5)
        
        Returns:
        y_pred: numpy array of shape (n_samples,)
        """
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        return y_pred
    
    def test(self, X_test, y_test):
        """
        Test the model using the test data and return the accuracy.
        
        Parameters:
        X_test: numpy array of shape (n_samples, n_features)
        y_test: numpy array of shape (n_samples,)
        
        Returns:
        accuracy: float, accuracy of the predictions
        """
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        return accuracy
