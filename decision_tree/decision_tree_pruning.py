import random
from math import log
import copy
from tree_visualization import createPlot


def data_generator(m):
    data_x = []
    data_y = []

    for _ in range(m):
        x = []
        rad = random.random()
        x.append(1) if rad < 0.5 else x.append(0)

        for _ in range(1, 15):
            rad = random.random()
            x.append(1 - x[-1]) if rad < 0.25 else x.append(x[-1])
        for _ in range(15, 21):
            rad = random.random()
            x.append(1) if rad < 0.5 else x.append(0)
        data_x.append(x)
        if x[0] == 0:
            y = 1 if sum(x[1:8]) > 3 else 0
        elif x[0] == 1:
            y = 1 if sum(x[8:15]) > 3 else 0
        data_y.append(y)
    return data_x, data_y


def find_key_id3(data_x, data_y):
    p_1 = sum(data_y) / len(data_y)
    p_0 = 1 - p_1
    entropy_base = -1 * (p_0 * log(p_0) + p_1 * log(p_1))
    max_gain, max_key = 0, -1
    for i in range(len(data_x[0])):
        p_0_0, p_0_1, p_1_0, p_1_1 = 0, 0, 0, 0
        for j in range(len(data_x)):
            if data_x[j][i] == 0:
                if data_y[j] == 1:
                    p_0_1 += 1
                elif data_y[j] == 0:
                    p_0_0 += 1
            elif data_x[j][i] == 1:
                if data_y[j] == 1:
                    p_1_1 += 1
                elif data_y[j] == 0:
                    p_1_0 += 1
        if p_0_0 == 0 or p_0_1 == 0:
            entropy_new1 = 0
        else:
            entropy_new1 = -1 * (p_0_0 / (p_0_0 + p_0_1) * log(p_0_0 / (p_0_0 + p_0_1)) + p_0_1 / (p_0_0 + p_0_1) * log(p_0_1 / (p_0_0 + p_0_1)))
        if p_1_0 == 0 or p_1_1 == 0:
            entropy_new2 = 0
        else:
            entropy_new2 = -1 * (p_1_0 / (p_1_0 + p_1_1) * log(p_1_0 / (p_1_0 + p_1_1)) + p_1_1 / (p_1_0 + p_1_1) * log(p_1_1 / (p_1_0 + p_1_1)))
        info_gain = entropy_base - ((p_0_1 + p_0_0) / len(data_x) * entropy_new1 + (p_1_1 + p_1_0) / len(data_x) * entropy_new2)
        if info_gain > max_gain:
            max_gain = info_gain
            max_key = i
    return max_key


class tree_node():
    def __init__(self, key, left=None, right=None, label=-1):
        self.key = key
        self.left = left
        self.right = right
        self.label = label


def split_data(data_x, data_y, key):
    data_x1, data_x2, data_y1, data_y2 = [], [], [], []
    for i in range(len(data_x)):
        if data_x[i][key] == 0:
            data_x1.append(copy.deepcopy(data_x[i][:key] + data_x[i][key + 1:]))
            data_y1.append(copy.deepcopy(data_y[i]))
        elif data_x[i][key] == 1:
            data_x2.append(copy.deepcopy(data_x[i][:key] + data_x[i][key + 1:]))
            data_y2.append(copy.deepcopy(data_y[i]))
    return data_x1, data_x2, data_y1, data_y2


def build_tree(data_x, data_y, label_list, depth, s):
    if len(data_x) == 0:
        return None
    elif sum(data_y) == 0 or sum(data_y) == len(data_y):
        leaf = tree_node(-1, label=data_y[0])
        return leaf
    elif len(data_x[0]) == 0 or depth == 0 or len(data_y) <= s:
        lbl = 1 if 2 * sum(data_y) > len(data_y) else 0
        leaf = tree_node(-1, label=lbl)
        return leaf
    else:
        key = find_key_id3(data_x, data_y)
        data_x1, data_x2, data_y1, data_y2 = split_data(data_x, data_y, key)
        node = tree_node(label_list[key])
        new_label_list = copy.deepcopy(label_list[:key] + label_list[key + 1:])
        left_node = build_tree(data_x1, data_y1, new_label_list, depth - 1)
        right_node = build_tree(data_x2, data_y2, new_label_list, depth - 1)
        node.left = left_node
        node.right = right_node
        return node


def test_tree(data_test, tree):
    res = []
    for data in data_test:
        tmp_tree = tree
        while tmp_tree.key != -1:
            if data[tmp_tree.key] == 0:
                tmp_tree = tmp_tree.left
            else:
                tmp_tree = tmp_tree.right
        res.append(tmp_tree.label)
    return res


def cal_acc(real_label, predicted_label):
    err = 0
    for i in range(len(real_label)):
        err += abs(real_label[i] - predicted_label[i])
    err /= len(real_label)
    return err


if __name__ == '__main__':
    m = 8000
    d = 3
    s = 5
    data_x, data_y = data_generator(m)
    label_list = [i for i in range(21)]
    print(data_x)
    print(data_y)
    decision_tree = build_tree(data_x, data_y, label_list, d, s)
    predict_y = test_tree(data_x, decision_tree)
    print(predict_y)
    createPlot(decision_tree)