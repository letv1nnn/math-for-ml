import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinearRegressionOneVariable(object):

    def __init__(self, x: np.array, y: np.array, weight: float = 0.01, bias: float = 0.0, alpha: float = 0.001, iterations: int = 1000):
        self.x = self.normalization(x)
        self.y = y
        self.w = weight
        self.b = bias
        self.alpha = alpha
        self.iterations = iterations
        self.m = self.x.shape[0]

    def normalization(self, x: np.array):
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        return (x - self.x_mean) / self.x_std

    def cost_function(self):
        cost = sum([((self.w * self.x[i] + self.b) - self.y[i]) ** 2 for i in range(self.m)])
        return cost / (2 * self.m)

    def gradient_computation(self):
        der_w, der_b = 0.0, 0.0
        for i in range(self.m):
            func = (self.x[i] * self.w + self.b) - self.y[i]
            der_w += func * self.x[i]
            der_b += func
        return der_w / self.m, der_b / self.m

    def gradient_descent(self):
        min_cost = float("inf")
        for i in range(self.iterations):
            der_w, der_b = self.gradient_computation()
            self.w -= self.alpha * der_w
            self.b -= self.alpha * der_b

            if i % 100 == 0:
                cost = self.cost_function()
                if cost < min_cost:
                    min_cost = cost
                else:
                    print("Alpha is to large!")
                    break
                print(f"Cost = {cost}\tw = {self.w}\tb = {self.b}")

    def predict(self, x: float):
        x_normalized = (x - self.x_mean) / self.x_std
        return x_normalized * self.w + self.b

    def visualization(self):
        plt.scatter(self.x, self.y, c="r", marker='x', label="Training examples")
        func = self.w * self.x + self.b
        plt.plot(self.x, func, color="b", label="Model")

        plt.xlabel("Training features")
        plt.ylabel("Training targets")

        plt.legend()
        plt.show()


class LinearRegressionMultipleVariable(LinearRegressionOneVariable):

    def __init__(self, x: np.array, y: np.array, weight: np.array = None, bias: float = 0.0, alpha: float = 0.001, iterations: int = 1000):
        super().__init__(x[:, 0], y, weight=0.01, bias=bias, alpha=alpha, iterations=iterations)
        self.x = self.normalization(x)
        self.y = y.reshape(-1, 1)
        self.n_features = x.shape[1]
        self.w = weight if weight is not None else np.zeros((self.n_features, 1))
        self.m = len(y)

    def normalization(self, x: np.array):
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        return (x - self.x_mean) / self.x_std

    def cost_function(self):
        predictions = np.dot(self.x, self.w) + self.b
        error = predictions - self.y
        return np.sum(error ** 2) / (2 * self.m)

    def gradient_computation(self):
        predictions = np.dot(self.x, self.w) + self.b
        error = predictions - self.y
        dW = np.dot(self.x.T, error) / self.m
        db = np.sum(error) / self.m
        return dW, db

    def gradient_descent(self):
        min_cost = float("inf")
        for i in range(self.iterations):
            dW, db = self.gradient_computation()
            self.w -= self.alpha * dW
            self.b -= self.alpha * db

            if i % 100 == 0:
                cost = self.cost_function()
                if cost < min_cost:
                    min_cost = cost
                else:
                    print("Alpha is too large!")
                    break
                print(f"Iter {i}: Cost = {cost:.4f}, b = {self.b:.4f}")
                print("Weights:", self.w.flatten())

    def predict(self, x: np.array):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x_normalized = (x - self.x_mean) / self.x_std
        return np.dot(x_normalized, self.w) + self.b

    def visualization(self):
        if self.n_features == 1:
            super().visualization()
        elif self.n_features == 2:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.x[:, 0], self.x[:, 1], self.y, c='r', marker='o', label='Actual Data')
            x0_vals = np.linspace(min(self.x[:, 0]), max(self.x[:, 0]), 20)
            x1_vals = np.linspace(min(self.x[:, 1]), max(self.x[:, 1]), 20)
            x0_mesh, x1_mesh = np.meshgrid(x0_vals, x1_vals)
            y_pred = (self.w[0] * x0_mesh + self.w[1] * x1_mesh + self.b)
            ax.plot_surface(x0_mesh, x1_mesh, y_pred, alpha=0.5, cmap='viridis', label='Regression Plane')

            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Target')
            ax.set_title('Multiple Linear Regression (2 Features)')
            plt.legend()
            plt.show()
        else:
            print("Visualizing first two features (3D plot not possible for >2 features)")
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.x[:, 0], self.x[:, 1], self.y, c='r', marker='o')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Target')
            ax.set_title('First Two Features vs Target')
            plt.show()


'''
# Test for one variable linear regression

features = np.array([5, 10, 15, 25, 30, 40, 45])
targets = np.array([15, 20, 25, 35, 40, 50, 60])
lr = LinearRegressionOneVariable(features, targets, alpha=0.015)
lr.gradient_descent()
lr.visualization()

print(f"Predicted y is {lr.predict(17.5)}.")
'''

'''
# Test for multiple variables linear regression

features = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
targets = np.array([6, 15, 24])
model = LinearRegressionMultipleVariable(features, targets, alpha=0.01, iterations=1000)
model.gradient_descent()
model.visualization()

feature_test = np.array([[2, 3, 4], [5, 6, 7]])
print("Predictions:", model.predict(feature_test))
'''
