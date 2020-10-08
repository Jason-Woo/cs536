import random
from math import log
import copy
import matplotlib.pyplot as plt
import numpy as np

from tree_visualization import createPlot


def data_generator(m):
    """
        Generate training/testing data based on given rules
        ----------
        Parameters
        m: integer
            size of the data
        ----------
        Return
        data_x: list of size m*20
            feature values of data
        data_y: list of size m
            the label of each data
    """
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
    """
        Return the feature with largest info-gain
        ----------
        Parameters
        data_x: 2-d list
            feature values of data
        data_y: list
            the label of each data
        ----------
        Return
        max_key: integer
            the index of feature with largest info-gain of data_x
    """
    g = []
    p_1 = sum(data_y) / len(data_y)
    p_0 = (len(data_y)-sum(data_y)) / len(data_y)
    entropy_base = -1 * (p_0 * log(p_0, 2) + p_1 * log(p_1, 2))
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
            entropy_new1 = -1 * (p_0_0 / (p_0_0 + p_0_1) * log(p_0_0 / (p_0_0 + p_0_1), 2) + p_0_1 / (p_0_0 + p_0_1) * log(p_0_1 / (p_0_0 + p_0_1), 2))
        if p_1_0 == 0 or p_1_1 == 0:
            entropy_new2 = 0
        else:
            entropy_new2 = -1 * (p_1_0 / (p_1_0 + p_1_1) * log(p_1_0 / (p_1_0 + p_1_1), 2) + p_1_1 / (p_1_0 + p_1_1) * log(p_1_1 / (p_1_0 + p_1_1), 2))
        info_gain = entropy_base - ((p_0_1 + p_0_0) / len(data_x) * entropy_new1 + (p_1_1 + p_1_0) / len(data_x) * entropy_new2)
        g.append(info_gain)
        if info_gain > max_gain:
            max_gain = info_gain
            max_key = i
    print(g)
    if max_key == 2:
        print(data_x)
        print(data_y)
    return max_key


class tree_node():
    def __init__(self, key, left=None, right=None, label=-1):
        self.key = key  # For leaf, key = -1
        self.left = left
        self.right = right
        self.label = label  # For tree node, label = -1


def split_data(data_x, data_y, key):
    """
        split data set into two parts based on the value of given feature
        ----------
        Parameters
        data_x: 2-d list
            feature values of data
        data_y: list
            the label of each data
        key: integer
            the feature that the partition based on
        ----------
        Return
        data_x1, data_x2, data_y1, data_y2: lists
            (data_x1, data_y1) and (data_x2, data_y2) is the partition result
    """
    data_x1, data_x2, data_y1, data_y2 = [], [], [], []
    for i in range(len(data_x)):
        if data_x[i][key] == 0:
            data_x1.append(copy.deepcopy(data_x[i][:key] + data_x[i][key + 1:]))
            data_y1.append(copy.deepcopy(data_y[i]))
        elif data_x[i][key] == 1:
            data_x2.append(copy.deepcopy(data_x[i][:key] + data_x[i][key + 1:]))
            data_y2.append(copy.deepcopy(data_y[i]))
    return data_x1, data_x2, data_y1, data_y2


def build_tree(data_x, data_y, label_list, depth, threshold):
    """
        split data set into two parts based on the value of given feature
        ----------
        Parameters
        data_x: 2-d list
            feature values of data
        data_y: list
            the label of each data
        label list: list
            the name of features of data_x
        depth: integer
            The maximum depth we allowed for the tree
        threshold: integer
            The minimum size of data we allowed for each node
        ----------
        Return
        node: tree_node object
            the decision tree build with the training data
    """
    if sum(data_y) == 0 or sum(data_y) == len(data_y):   # All data in the node have same label, then it is a leaf node
        leaf = tree_node(-1, label=data_y[0])
        return leaf
    elif len(data_x[0]) == 0 or depth == 0 or len(data_y) <= threshold:
        # All feature have been use, or we are reaching maximal depth or minimal size of data, then it is a leaf node
        lbl = 1 if 2 * sum(data_y) > len(data_y) else 0
        leaf = tree_node(-1, label=lbl)
        return leaf
    else:
        key = find_key_id3(data_x, data_y)
        data_x1, data_x2, data_y1, data_y2 = split_data(data_x, data_y, key)
        node = tree_node(label_list[key])
        new_label_list = copy.deepcopy(label_list[:key] + label_list[key + 1:])
        left_node = build_tree(data_x1, data_y1, new_label_list, depth - 1, threshold)
        right_node = build_tree(data_x2, data_y2, new_label_list, depth - 1, threshold)
        node.left = left_node
        node.right = right_node
        return node


def test_tree(data_test, tree):
    """
        Predict label of test data using the decision tree
        ----------
        Parameters
        data_test: 2-d list
            feature values of testing data
        tree: tree_node object
            decision tree used for prediction
        ----------
        Return
        res: list
            predicted label of the testing data
    """
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
    """
        Calculate the accuracy based on real labels and predicted labels
        ----------
        Parameters
        real_label: list
            the real label of testing data set
        predicted_label: list
            the label generated by the decision tree
        ----------
        Return
        err: float
            accuracy of the prediction
    """
    err = 0
    for i in range(len(real_label)):
        err += abs(real_label[i] - predicted_label[i])
    err /= len(real_label)
    return err


def num_irrelevant(tree):
    """
        Calculate number of irrelevant variables in the tree
        ----------
        Parameters
        tree: tree_node object
            decision tree
        ----------
        Return
        cnt: integer
            the number of irrelevant variables in the tree
    """
    cnt = 0
    if tree.left.key != -1:
        cnt += num_irrelevant(tree.left)
    if tree.right.key != -1:
        cnt += num_irrelevant(tree.right)
    irr_id = [i for i in range(15, 21)]
    if tree.key in irr_id:
        cnt += 1
    return cnt


if __name__ == '__main__':
    test_id = 0
    if test_id == 0:
        m = 1000
        d = -1
        s = -1
        # data_x, data_y = data_generator(m)
        data_x = np.transpose(np.load('x.npy')).tolist()
        data_y = np.load('y.npy').tolist()
        label_list = [i for i in range(21)]
        # print(data_x)
        # print(data_y)

        decision_tree = build_tree(data_x, data_y, label_list, d, s)
        predict_y = test_tree(data_x, decision_tree)
        print(predict_y)
        createPlot(decision_tree)
        print(cal_acc(data_y, predict_y))
    elif test_id == 1:
        num_irr = []
        m = [j for j in range(10, 1001, 10)]
        for tmp_m in m:
            irr_cnt = 0
            for i in range(100):
                d = -1
                s = -1
                data_x, data_y = data_generator(tmp_m)
                label_list = [i for i in range(21)]
                decision_tree = build_tree(data_x, data_y, label_list, d, s)
                irr_cnt += num_irrelevant(decision_tree)
            irr_cnt /= 100
            num_irr.append(irr_cnt)
            print(irr_cnt)
        plt.plot(m, num_irr)
        plt.show()
    elif test_id == 2:
        m = 10000
        d = [i for i in range(0, 21)]
        s = -1
        err_train, err_test = [], []
        for tmp_d in d:
            print(tmp_d)
            tmp_err_train, tmp_err_test = 0, 0
            data_x, data_y = data_generator(m)
            train_x, train_y = data_x[:8000], data_y[:8000]
            test_x, test_y = data_x[8000:], data_y[8000:]
            label_list = [i for i in range(21)]
            decision_tree = build_tree(train_x, train_y, label_list, tmp_d, s)
            predcit_train = test_tree(train_x, decision_tree)
            predcit_test = test_tree(test_x, decision_tree)
            err_train.append(cal_acc(train_y, predcit_train))
            err_test.append(cal_acc(test_y, predcit_test))
        plt.plot(d, err_train, color='black', label='train')
        plt.plot(d, err_test, color='blue', label='test')
        plt.legend()
        plt.show()
    elif test_id == 3:
        m = 10000
        d = -1
        s = [i for i in range(1, 4000, 50)]
        err_train, err_test = [], []
        for tmp_s in s:
            print(tmp_s)
            tmp_err_train, tmp_err_test = 0, 0
            data_x, data_y = data_generator(m)
            train_x, train_y = data_x[:8000], data_y[:8000]
            test_x, test_y = data_x[8000:], data_y[8000:]
            label_list = [i for i in range(21)]
            decision_tree = build_tree(train_x, train_y, label_list, d, tmp_s)
            predcit_train = test_tree(train_x, decision_tree)
            predcit_test = test_tree(test_x, decision_tree)
            err_train.append(cal_acc(train_y, predcit_train))
            err_test.append(cal_acc(test_y, predcit_test))
        plt.plot(s, err_train, color='black', label='train')
        plt.plot(s, err_test, color='blue', label='test')
        plt.legend()
        plt.show()
