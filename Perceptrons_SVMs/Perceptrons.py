import numpy as np
import matplotlib.pyplot as plt

def data_generate(k, m, epsilon):
    data_x, data_y = [], []
    curr_size = 0
    while curr_size < m:
        z = np.random.normal(0, 1, k)
        norm_2 = np.linalg.norm(z, 2)
        x = z / norm_2
        if abs(x[k - 1]) >= epsilon:
            y = 1 if x[k - 1] >= epsilon else -1
            data_x.append(x)
            data_y.append(y)
            curr_size += 1
    return data_x, data_y


def perceptron(data_x, data_y):
    weight = np.zeros(data_x.shape[1])
    bias = 0
    step_cnt = 0
    while step_cnt <= 10000:
        err_list = []
        for i in range(len(data_x)):
            y_predict = np.dot(weight, data_x[i]) + bias
            label_predict = 1 if y_predict > 0 else -1
            label_real = data_y[i]
            if label_predict != label_real:
                err_list.append(i)
        if len(err_list) == 0:
            return step_cnt, weight, bias
        for err in err_list:
            weight += data_y[err] * data_x[err]
            bias += data_y[err]
        step_cnt += 1
    return step_cnt, weight, bias


def compute_error(weight, bias):
    k = len(weight)
    idea_weight = np.zeros(k)
    idea_weight[-1] = 1
    idea_bias = 0
    err = np.linalg.norm((weight - idea_weight), 2) ** 2 + (idea_bias - bias) ** 2
    return err


if __name__ == "__main__":
    test_id = 2
    plt_id = 1
    if test_id == 0:
        m = [i for i in range(10, 1000, 10)]
        step_cnt = []
        err = []
        repeat = 100
        for tmp_m in m:
            average_w = np.zeros(5)
            average_b = 0
            print(tmp_m)
            tmp_cnt = 0
            tmp_err = 0
            for _ in range(repeat):
                data_x, data_y = data_generate(5, tmp_m, 0.1)
                cnt, w, b = perceptron(np.array(data_x), np.array(data_y))
                average_w += w
                average_b += b
                tmp_cnt += cnt
            tmp_cnt /= repeat
            average_w /= repeat
            average_b /= repeat
            step_cnt.append(tmp_cnt)
            err.append(compute_error(average_w, average_b))
        if plt_id == 0:
            plt.plot(m, step_cnt)
        else:
            plt.plot(m, err)
        plt.show()
    if test_id == 1:
        k = [i for i in range(2, 800, 10)]
        step_cnt = []
        err = []
        repeat = 100
        for tmp_k in k:
            print(tmp_k)
            average_w = np.zeros(tmp_k)
            average_b = 0
            tmp_cnt = 0
            tmp_err = 0
            for _ in range(repeat):
                data_x, data_y = data_generate(tmp_k, 100, 0.1)
                cnt, w, b = perceptron(np.array(data_x), np.array(data_y))
                average_w += w
                average_b += b
                tmp_cnt += cnt
            tmp_cnt /= repeat
            average_w /= repeat
            average_b /= repeat
            step_cnt.append(tmp_cnt)
            err.append(compute_error(average_w, average_b))
        if plt_id == 0:
            plt.plot(k, step_cnt)
        else:
            plt.plot(k, err)
        plt.show()
    if test_id == 2:
        epsilon = [i for i in np.arange(0.01, 0.95, 0.01)]
        step_cnt = []
        err = []
        average_w = np.zeros(5)
        average_b = 0
        repeat = 100
        for tmp_epsilon in epsilon:
            print(tmp_epsilon)
            tmp_cnt = 0
            tmp_err = 0
            for _ in range(repeat):
                data_x, data_y = data_generate(5, 100, tmp_epsilon)
                cnt, w, b = perceptron(np.array(data_x), np.array(data_y))
                average_w += w
                average_b += b
                tmp_cnt += cnt
            tmp_cnt /= repeat
            average_w /= repeat
            average_b /= repeat
            step_cnt.append(tmp_cnt)
            err.append(compute_error(average_w, average_b))
        if plt_id == 0:
            plt.plot(epsilon, step_cnt)
        else:
            plt.plot(epsilon, err)
        plt.show()