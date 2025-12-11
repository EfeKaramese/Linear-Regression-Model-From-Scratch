import numpy as np
class CustomLinearRegression:
    def __init__(self,learning_rate : float = 0.01, iterations : int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X : np.ndarray, y : np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        y = y.reshape(-1, 1)
        
        for i in range(self.iterations):
            y_pred = X.dot(self.weights) + self.bias
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            dw = (2 / n_samples) * X.T.dot(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db
            
    def predict(self,X : np.ndarray):
        return X.dot(self.weights)+self.bias


