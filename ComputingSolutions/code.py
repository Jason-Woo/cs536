import numpy as np
import matplotlib.pyplot as plt
import math


def data_generate(m):
    data_x = np.zeros((m, 21))
    data_y = np.zeros(m)
    for i in range(m):
        x = np.random.standard_normal(15)
        x_11 = x[0] + x[1] + np.random.normal(0, 0.1)
        x_12 = x[2] + x[3] + np.random.normal(0, 0.1)
        x_13 = x[3] + x[4] + np.random.normal(0, 0.1)
        x_14 = 0.1 * x[6] + np.random.normal(0, 0.1)
        x_15 = 2 * x[1] - 10 + np.random.normal(0, 0.1)
        x_new = np.array([x_11, x_12, x_13, x_14, x_15])
        x = np.concatenate(([1], x[:10], x_new, x[10:]))
        data_x[i] = x
        param = np.array([pow(0.6, i + 1) for i in range(10)])
        data_y[i] = 10 + np.dot(x[1:11], param) + np.random.normal(0, 0.1)
    return data_x, data_y


def naive_regression(x, y, gradient_descent=False):
    # gradient_descent set to true to use gradient descent
    # algebraic solution set to false for algebraic solution
    if gradient_descent:
        m = x.shape[0]
        w = np.zeros(21)
        iter_cnt = 0
        learning_rate = 0.009
        err_prev = 9999999
        threshold = 1e-10

        while True:
            y_predict = x.dot(w)
            err = np.sum((y - y_predict) ** 2) / m
            if err_prev - err < threshold:
                return w
            err_prev = err
            if iter_cnt % 1000 == 0:
                print(iter_cnt, err_prev)
            gradient = -2 / m * np.sum(x.transpose() * (y - y_predict), axis=1)
            w = w - learning_rate * gradient
            iter_cnt += 1
    else:
        # algebraic solution
        xtx = x.transpose().dot(x)
        return np.linalg.inv(xtx).dot(x.transpose()).dot(y)


def ridge_regression(x, y, gradient_descent=False, lmb=0.1):
    # gradient_descent set to true to use gradient descent
    # algebraic solution set to false for algebraic solution
    if gradient_descent:
        m = x.shape[0]
        w = np.zeros(21)
        iter_cnt = 0
        learning_rate = 0.009
        err_prev = 9999999
        threshold = 1e-30

        while True:
            y_predict = x.dot(w)
            err = np.sum((y - y_predict) ** 2) / m + lmb * np.linalg.norm(w, 2) ** 2
            if err_prev - err < threshold:
                return w
            err_prev = err
            if iter_cnt % 1000 == 0:
                print(iter_cnt, err_prev)

            gradient = -2 / m * np.sum(x.transpose() * (y - y_predict), axis=1) + 2 * lmb * w
            w = w - learning_rate * gradient
            iter_cnt += 1
    else:
        # algebraic solution
        xtx = x.transpose().dot(x)
        return np.linalg.inv(xtx + lmb * np.identity(21)).dot(x.transpose()).dot(y)


def lasso_regression(x, y, lmb=0.01):
    m = x.shape[0]
    w = np.ones(21)
    iter_cnt = 0
    err_prev = 9999999
    threshold = 1e-3

    while True:
        y_predict = x.dot(w)
        err = np.sum((y - y_predict) ** 2) / m + lmb * np.linalg.norm(w, 1)
        if err_prev - err < threshold:
            return w
        err_prev = err
        if iter_cnt % 10 == 0:
            print(iter_cnt, err_prev)
        for k in range(21):
            z_k = sum(np.transpose(x[:, k]) * x[:, k])
            w_k = 0
            for i in range(m):
                w_k += x[i, k] * (y[i] - sum(x[i, j] * w[j] for j in range(21) if j != k))
            if w_k < -lmb / 2:
                w_k = (w_k + lmb / 2) / z_k
            elif w_k > lmb / 2:
                w_k = (w_k - lmb / 2) / z_k
            else:
                w_k = 0
            w[k] = w_k
        iter_cnt += 1


def estimate_error(w, data_size):
    x, y = data_generate(data_size)
    x = x / np.linalg.norm(x, axis=0)
    y = y / np.linalg.norm(y, axis=0)
    y_predict = x.dot(w)
    mse = np.square(np.subtract(y, y_predict)).mean()
    return mse


if __name__ == '__main__':
    test_case = 3
    if test_case == 0:
        X, Y = data_generate(1000)
        w = naive_regression(X, Y, True)
        print(w)
        print(estimate_error(w, 10000))
    elif test_case == 1:
        iter_lmb = False
        if iter_lmb:
            l = np.linspace(0, 0.5, 200)
            err = []
            for tmp_l in l:
                X, Y = data_generate(1000)
                w = ridge_regression(X, Y, True, tmp_l)
                err.append(estimate_error(w, 10000))
            plt.plot(l, err)
            plt.show()
        else:
            X, Y = data_generate(1000)
            w = ridge_regression(X, Y, True, 0.005)
            print(w)
    elif test_case == 2:
        l = np.linspace(0, 100, 100)
        num_eliminate = []
        for tmp_l in l:
            X, Y = data_generate(1000)
            w = lasso_regression(X, Y, tmp_l)
            tmp_eliminate = 0
            for tmp_w in w:
                if tmp_w == 0:
                    tmp_eliminate += 1
            num_eliminate.append(tmp_eliminate)
        plt.plot(l, num_eliminate)
        plt.show()
    elif test_case == 3:
        iter_lmb = False
        if iter_lmb:
            l = np.linspace(0, 100, 100)
            err = []
            for tmp_l in l:
                X, Y = data_generate(1000)
                w = lasso_regression(X, Y, tmp_l)
                err.append(estimate_error(w, 10000))
            plt.plot(l, err)
            plt.show()
        else:
            X, Y = data_generate(1000)
            w = lasso_regression(X, Y, 0.1)
            print(w)

