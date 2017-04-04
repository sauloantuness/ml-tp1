import numpy as np
import pandas as pd


def compute_cost(X, y, beta):
    total_cost = 0
    m, _ = X.shape
    for i in range(m):
        x_i = X[i, 1]
        y_i = y[i]
        b_0 = beta[0, 0]
        b_1 = beta[0, 1]
        total_cost += (b_0 + b_1 * x_i - y_i) ** 2

    return total_cost / (2 * m)


def gradient_descent2(X, y, theta, alpha, iters):
    '''
    alpha: learning rate
    iters: number of iterations
    OUTPUT:
    theta: learned parameters
    cost:  a vector with the cost at each training iteration
    '''
    # temp = np.matrix(np.zeros(theta.shape))
    # parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    m, _ = X.shape

    for i in range(iters):
        gradient_b0 = 0
        gradient_b1 = 0

        for j in range(m):
            gradient_b0 += (theta[0, 0] + theta[0, 1] * X[j, 1] - y[j, 0])
            gradient_b1 += X[j, 1] * (theta[0, 0] + theta[0, 1] * X[j, 1] - y[j, 0])

        gradient_b0 = gradient_b0 * 2.0 / m
        gradient_b1 = gradient_b1 * 2.0 / m

        theta[0, 0] = theta[0, 0] - alpha * gradient_b0
        theta[0, 1] = theta[0, 1] - alpha * gradient_b1

        cost[i]  = compute_cost(X, y, theta)

    return theta, cost

def gradient_descent(X, y, theta, alpha, iterations):
    """
    gradient_descent Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations
    m, _ = X.shape

    for iteration in range(iterations):
        hypothesis = X.dot(theta.T)
        loss = hypothesis - y
        gradient = X.T.dot(loss) / m
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history[iteration] = cost

    return theta, cost_history


data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
data.head()

data.insert(0, 'beta zero', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

X.head()
y.head()

X = np.matrix(X.values)
y = np.matrix(y.values)
beta = np.matrix(np.array([0.0, 0.0]))

X.shape, beta.shape, y.shape

alpha = 0.01
iters = 1500

g, cost = gradient_descent(X, y, beta, alpha, iters)

print g
print cost