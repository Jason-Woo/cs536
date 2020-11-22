import numpy as np
m = 200
w = 1
b = 5
sigma = 0.1


def data_generate():
    x = np.random.uniform(100, 102, m)
    epsilon = np.random.normal(0, sigma, m)
    y = w * x + 5 + epsilon
    x_prime = x - 101
    return x, x_prime, y


def regression(x, y):
    x_bar, y_bar = np.mean(x), np.mean(y)
    w_hat = np.sum(np.dot(x-x_bar, y)) / np.sum(np.dot(x-x_bar, x))
    b_hat = y_bar - w_hat * x_bar
    return w_hat, b_hat


if __name__ == '__main__':
    w_hat, w_hat_prime, b_hat, b_hat_prime = [], [], [], []
    for i in range(1000):
        x, x_prime, y = data_generate()
        w_hat_tmp, b_hat_tmp = regression(x, y)
        w_hat.append(w_hat_tmp)
        b_hat.append(b_hat_tmp)
        w_hat_tmp, b_hat_tmp = regression(x_prime, y)
        w_hat_prime.append(w_hat_tmp)
        b_hat_prime.append(b_hat_tmp)

    print("Expected value")
    print(np.mean(w_hat), np.mean(b_hat))
    print(np.mean(w_hat_prime), np.mean(b_hat_prime))
    print("variance")
    print(np.var(w_hat), np.var(b_hat))
    print(np.var(w_hat_prime), np.var(b_hat_prime))
