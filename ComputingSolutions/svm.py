import numpy as np
import math


def kernel(x, y):
    return np.square(1 + np.sum(x * y))


def update_weight(x, y, k, step, alpha):
    # calculate and returns the gradient
    exp0 = 1 - (y[k] * y[0])
    exp1 = 2 * (y[k] * y[0]) * kernel(x[0], x[0]) * sum(alpha[i] * (y[i] * y[0]) for i in range(1, 4))
    exp2 = (y[k] * y[0]) * sum(alpha[i] * y[i] * kernel(x[i], x[0]) for i in range(1, 4))
    exp3 = y[k] * kernel(x[k], x[0]) * sum(alpha[i] * (y[i] * y[0]) for i in range(1, 4))
    exp4 = 2 * y[k] * sum(y[i] * kernel(x[i], x[k]) * alpha[i] for i in range(1, 4))

    objective = exp0 - (1 / 2) * (exp1 - 2 * y[0] * (exp2 + exp3) + exp4)
    logarithmic = (1 / alpha[k]) - y[k] * y[0] / sum((alpha[i] * (y[i] * y[0])) for i in range(1, 4))
    epsilon = (1 / step ** 2)

    grad = objective + epsilon * logarithmic

    return grad


def cal_err(x, y, step, alpha):
    # calculate and returns the loss function
    exp0 = -1 * sum(alpha[i] * (y[i] * y[0]) for i in range(1, 4)) + sum(alpha[i] for i in range(1, 4))
    exp1 = (-1 * sum(alpha[i] * (y[i] * y[0]) for i in range(1, 4))) ** 2 * kernel(x[0], x[0])
    exp2 = 2 * sum(-1 * alpha[i] * (y[i] * y[0]) for i in range(1, 4)) * y[0] * sum(alpha[i] * y[i] * kernel(x[i], x[0]) for i in range(1, 4))
    exp3 = sum(sum(alpha[i] * y[i] * kernel(x[i], x[j] * y[j] * alpha[j]) for j in range(1, 4)) for i in range(1, 4))
    objective = exp0 - (1 / 2) * (exp1 + exp2 + exp3)
    logarithmic = sum(math.log(alpha[i]) for i in range(1, 4)) + math.log(sum(-1 * alpha[i] * (y[i] * y[0]) for i in range(1, 4)))

    epsilon = (1 / step ** 2)
    loss = objective + epsilon * logarithmic
    return loss


def dual_svm(x, y):
    alpha = np.array([1.0, 1.0, 1.0, 1.0])
    learning_rate = 0.01
    threshold = 1e-10
    err_prev = 99999
    for steps in range(1, 10000):
        err_curr = cal_err(x, y, steps, alpha)
        if abs(err_prev - err_curr) <= threshold:
            alpha[0] = -1 * sum(alpha[i] * (y[i] * y[0]) for i in range(1, 4))
            return alpha
        err_prev = err_curr
        for n in range(1, 4):
            delta = update_weight(x, y, n, steps, alpha)
            while alpha[n] + learning_rate * delta < 0:
                # if alpha step out side the constraints region
                learning_rate /= 2
            alpha[n] = alpha[n] + learning_rate * delta
            learning_rate = min(learning_rate * 4, 0.01)
        print(steps, alpha)
    alpha[0] = -1 * sum(alpha[i] * (y[i] * y[0]) for i in range(1, 4))
    return alpha


if __name__ == "__main__":
    x = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    y = np.array([-1., 1., -1., 1.])
    kx = []
    w = dual_svm(x,y)
    print('Value of alpha is\t', w)
