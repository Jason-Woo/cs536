import numpy as np
import matplotlib.pyplot as plt


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
        threshold = 1e-7

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


def ridge_regression(x, y, gradient_descent=False, lmb=0):
    # gradient_descent set to true to use gradient descent
    # algebraic solution set to false for algebraic solution
    if gradient_descent:
        m = x.shape[0]
        w = np.zeros(21)
        iter_cnt = 0
        learning_rate = 0.009
        err_prev = 9999999
        threshold = 1e-7

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


def lasso_regression(x, y, lmb=1):
    m = x.shape[0]
    w = np.zeros(21)
    iter_cnt = 0
    err_prev = 9999999
    threshold = 1e-7

    while True:
        y_predict = x.dot(w)
        err = np.sum((y - y_predict) ** 2) / m + lmb * np.linalg.norm(w, 1) ** 2
        print(err_prev, err)
        if err_prev - err < threshold:
            return w
        err_prev = err
        if iter_cnt % 1 == 0:
            print(iter_cnt, err_prev)
        for k in range(21):
            w_k = np.transpose(x[:, k]).dot(y - x.dot(w) + w[k] * x[:, k])
            if w_k < -1 * lmb / 2:
                w_k = w_k + lmb / 2
            elif w_k > lmb / 2:
                w_k = w_k - lmb / 2
            else:
                w_k = 0
            w[k] = w_k


def estimate_error(w):
    w_real = np.array([pow(0.6, i + 1) for i in range(10)])
    w_real = np.concatenate(([10], w_real, np.ones(10)))
    mse = np.square(np.subtract(w_real, w)).mean()
    return mse


if __name__ == '__main__':
    test_case = 1
    if test_case == 0:
        X, Y = data_generate(1000)
        w = naive_regression(X, Y, False)
        print(w)
    elif test_case == 1:
        l = np.linspace(0, 0.5, 200)
        err = []
        for tmp_l in l:
            X, Y = data_generate(1000)
            w = ridge_regression(X, Y, False, tmp_l)
            err.append(estimate_error(w))
        plt.plot(l, err)
        plt.show()
    elif test_case == 2:
        X, Y = data_generate(1000)
        w = lasso_regression(X, Y)
        print(w)