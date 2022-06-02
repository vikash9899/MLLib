import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self) -> None:
        self.theta = None
        self.J_history = None
        self.alpha = None
        self.iteration = None

    def fit(self, X, y, alpha, num_iteration):
        if type(X).__module__ != np. __name__:
            raise TypeError("Please pass X as numpy array : ")

        if type(y).__module__ != np. __name__:
            raise TypeError("Please pass y as numpy array : ")

        if X.ndim != 2:
            raise ValueError("Input dimension of X must be 2")

        if y.ndim != 1:
            raise ValueError("Input dimension of y must be 1")

        X_train, mu, sigma = self.__featureNormalize(X)
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

        self.alpha = alpha
        self.iteration = num_iteration
        self.theta = np.zeros(X_train.shape[1])

        self.theta, self.J_history = self.__gradientDescent(
            X_train, y, self.theta, alpha, self.iteration)

    def predict(self, X):
        if type(X).__module__ != np. __name__:
            raise TypeError("Please pass X as numpy array : ")

        if X.ndim != 2:
            raise ValueError("Input dimension of X must be 2")

        X = np.hstack((np.ones((X.shape[0], 1)), X))

        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.matmul(X[i], self.theta)

        return y

    def loss_plot(self):
        plt.figure()
        plt.plot(np.arange(self.iteration), self.J_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

    def __featureNormalize(self, X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, ddof=1, axis=0)
        X_norm = (X - mu) / sigma
        return X_norm, mu, sigma

    def __computeCost(self, X, y, theta):
        h = np.dot(X, theta) - y
        J = np.dot(h, h) / (2 * X.shape[0])
        return J

    def __gradientDescent(self, X, y, theta, alpha, num_iters):
        J_history = np.zeros(num_iters)
        for i in range(num_iters):
            theta = (theta - (alpha / X.shape[0])
                     * np.dot(X.T, (np.dot(X, theta) - y)))
            J_history[i] = self.__computeCost(X, y, theta)
        return theta, J_history
