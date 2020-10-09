import random
from math import log
import copy
from tree_visualization import createPlot
import matplotlib.pyplot as plt


def data_generator(m, k):
    """
        Generate training/testing data based on given rules
        ----------
        Parameters
        m: integer
            size of the data
        k: integer
            size of features of each data
        ----------
        Return
        data_x: list of size m*k
            feature values of data
        data_y: list of size m
            the label of each data
    """
    data_x, data_y = [], []
    w2 = 0  # denominator of w_i
    for i in range(2, k + 1):
        w2 += pow(0.9, i)

    for _ in range(m):
        x = []
        rad = random.random()
        x.append(1) if rad < 0.5 else x.append(0)

        for _ in range(k - 1):
            rad = random.random()
            x.append(1 - x[-1]) if rad < 0.25 else x.append(x[-1])
        data_x.append(x)
        tmp_y = 0
        for i in range(2, k + 1):
            tmp_y += x[i - 1] * pow(0.9, i)
        tmp_y /= w2
        data_y.append(x[0]) if tmp_y >= 0.5 else data_y.append(1 - x[0])

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
    p_1 = sum(data_y) / len(data_y)
    p_0 = 1 - p_1
    entropy_base = -1 * (p_0 * log(p_0, 2) + p_1 * log(p_1, 2))  # Entropy of the label
    max_gain, max_key = 0, -1  # store maximum info-gain and index of corresponding feature
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
            entropy_new1 = 0  # entropy of subset feature value = 0
        else:
            entropy_new1 = -1 * (p_0_0 / (p_0_0 + p_0_1) * log(p_0_0 / (p_0_0 + p_0_1), 2) + p_0_1 / (p_0_0 + p_0_1) * log(
                p_0_1 / (p_0_0 + p_0_1), 2))
        if p_1_0 == 0 or p_1_1 == 0:
            entropy_new2 = 0  # entropy of subset feature value = 1
        else:
            entropy_new2 = -1 * (p_1_0 / (p_1_0 + p_1_1) * log(p_1_0 / (p_1_0 + p_1_1), 2) + p_1_1 / (p_1_0 + p_1_1) * log(
                p_1_1 / (p_1_0 + p_1_1), 2))
        info_gain = entropy_base - (
                    (p_0_1 + p_0_0) / len(data_x) * entropy_new1 + (p_1_1 + p_1_0) / len(data_x) * entropy_new2)
        if info_gain > max_gain:
            max_gain = info_gain
            max_key = i
    return max_key


def find_key_c45(data_x, data_y):
    """
        Return the feature with largest info-gain ratio
        ----------
        Parameters
        data_x: 2-d list
            feature values of data
        data_y: list
            the label of each data
        ----------
        Return
        max_key: integer
            the index of feature with largest info-gain ratio of data_x
    """
    p_1 = sum(data_y) / len(data_y)
    p_0 = 1 - p_1
    entropy_base = -1 * (p_0 * log(p_0, 2) + p_1 * log(p_1, 2))  # Entropy of the label
    max_gain, max_key = 0, -1  # store maximum info-gain and index of corresponding feature
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
            entropy_new1 = 0  # entropy of subset feature value = 0
        else:
            entropy_new1 = -1 * (p_0_0 / (p_0_0 + p_0_1) * log(p_0_0 / (p_0_0 + p_0_1), 2) + p_0_1 / (p_0_0 + p_0_1) * log(
                p_0_1 / (p_0_0 + p_0_1), 2))
        if p_1_0 == 0 or p_1_1 == 0:
            entropy_new2 = 0  # entropy of subset feature value = 1
        else:
            entropy_new2 = -1 * (p_1_0 / (p_1_0 + p_1_1) * log(p_1_0 / (p_1_0 + p_1_1), 2) + p_1_1 / (p_1_0 + p_1_1) * log(
                p_1_1 / (p_1_0 + p_1_1), 2))
        info_gain = entropy_base - (
                    (p_0_1 + p_0_0) / len(data_x) * entropy_new1 + (p_1_1 + p_1_0) / len(data_x) * entropy_new2)
        if p_0_1 + p_0_0 == 0 or p_1_1 + p_1_0 == 0:
            split_info = 1
        else:
            split_info = -1 * ((p_0_1 + p_0_0) / len(data_x) * log((p_0_1 + p_0_0) / len(data_x), 2) + (p_1_1 + p_1_0) / len(
            data_x) * log((p_1_1 + p_1_0) / len(data_x), 2))
        gain_ratio = info_gain / split_info  # info-gain ratio
        if gain_ratio > max_gain:
            max_gain = gain_ratio
            max_key = i
    return max_key


def find_key_cart(data_x, data_y):
    """
        Return the feature with smallest gini
        ----------
        Parameters
        data_x: 2-d list
            feature values of data
        data_y: list
            the label of each data
        ----------
        Return
        max_key: integer
            the index of feature with smallest gini of data_x
    """
    min_gini, min_key = 99999, -1  # store minimum gini and index of corresponding feature
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
        if p_1_1 + p_1_0 == 0:
            gini1 = 0
        else:
            gini1 = 1 - pow(p_1_1 / (p_1_1 + p_1_0), 2) - pow(p_1_0 / (p_1_1 + p_1_0), 2)
        if p_0_0 + p_0_1 == 0:
            gini0 = 0
        else:
            gini0 = 1 - pow(p_0_0 / (p_0_0 + p_0_1), 2) - pow(p_0_1 / (p_0_0 + p_0_1), 2)
        gini = (p_1_0 + p_1_1) / (p_1_0 + p_1_1 + p_0_1 + p_0_0) * gini1 + (p_0_0 + p_0_1) / (
                    p_1_0 + p_1_1 + p_0_1 + p_0_0) * gini0
        if gini < min_gini:
            min_gini = gini
            min_key = i
    return min_key


class tree_node():
    def __init__(self, key, left=None, right=None, label=-1):
        self.key = key  # the selected feature of a tree node
        self.left = left  # the left child of a tree node
        self.right = right  # the right child of the tree node
        self.label = label  # the label of the leaf node


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


def build_tree(data_x, data_y, label_list, method='id3'):
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
        method: string
            method we use to build the tree, choose from: id3, c4.5 and cart
        ----------
        Return
        node: tree_node object
            the decision tree build with the training data
    """
    if len(data_y) == 0:
        return None
    if sum(data_y) == 0 or sum(data_y) == len(data_y):  # All data in the node have same label, then it is a leaf node
        leaf = tree_node(-1, label=data_y[0])
        return leaf
    elif len(data_x[0]) == 0:  # All feature have been use and reach the bottom of the tree, then it is a leaf node
        lbl = 1 if 2 * sum(data_y) > len(data_y) else 0  # voting for leaf label
        leaf = tree_node(-1, label=lbl)
        return leaf
    else:  # If it is a tree node, keep separating data set and build the tree
        if method == 'c4.5':
            key = find_key_c45(data_x, data_y)
        elif method == 'cart':
            key = find_key_cart(data_x, data_y)
        else:
            key = find_key_id3(data_x, data_y)
        data_x1, data_x2, data_y1, data_y2 = split_data(data_x, data_y, key)
        node = tree_node(label_list[key])
        new_label_list = copy.deepcopy(label_list[:key] + label_list[key + 1:])
        left_node = build_tree(data_x1, data_y1, new_label_list)
        right_node = build_tree(data_x2, data_y2, new_label_list)
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
        while tmp_tree.key != -1:  # keep diving until reach leaf node
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


if __name__ == '__main__':
    test_id = 1
    if test_id == 0:
        m, k = 30, 4
        method = 'c4.5'
        data_x, data_y = data_generator(m, k)
        label_list = [i for i in range(k)]
        for i in range(3):
            for j in range(10):
                print(data_x[i * 10 + j], data_y[i * 10 + j], end=' ,')
            print('\n')
        decision_tree = build_tree(data_x, data_y, label_list, method)
        predict_y = test_tree(data_x, decision_tree)

        print('training error is ', cal_acc(data_y, predict_y))
        createPlot(decision_tree)

    elif test_id == 1:
        test_case = 1000
        m, k = 30, 4
        method = 'id3'
        data_x, data_y = data_generator(m, k)
        label_list = [i for i in range(k)]
        decision_tree = build_tree(data_x, data_y, label_list, method)
        error = 0
        for _ in range(test_case):
            data_x, data_y = data_generator(m, k)
            predict_y = test_tree(data_x, decision_tree)
            error += cal_acc(data_y, predict_y)
        error /= test_case
        print('Error is ', error)

    elif test_id == 2:
        test_case = 100
        k = 10
        method = 'id3'
        err = []
        m = [i for i in range(1, 500)]
        for tmp_m in m:
            data_x, data_y = data_generator(tmp_m, k)
            label_list = [i for i in range(k)]
            decision_tree = build_tree(data_x, data_y, label_list, method)
            error = 0
            for _ in range(test_case):
                data_x, data_y = data_generator(tmp_m, k)
                predict_y = test_tree(data_x, decision_tree)
                error += cal_acc(data_y, predict_y)
            error /= test_case
            err.append(error)

        plt.plot(m, err)
        plt.show()

    elif test_id == 3:
        test_case = 100
        k = 10
        method = ['id3', 'c4.5', 'cart']
        color = ['black', 'skyblue', 'red']
        label = ['id3', 'c4.5', 'CART']
        m = [i for i in range(1, 1000, 10)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        for i in range(3):
            err = []
            for tmp_m in m:
                print(tmp_m)
                data_x, data_y = data_generator(tmp_m, k)
                label_list = [i for i in range(k)]
                decision_tree = build_tree(data_x, data_y, label_list, method[i])
                error = 0
                for _ in range(test_case):
                    data_x, data_y = data_generator(tmp_m, k)
                    predict_y = test_tree(data_x, decision_tree)
                    error += cal_acc(data_y, predict_y)
                error /= test_case
                err.append(error)
            plt.plot(m, err, color=color[i], label=label[i])
        plt.legend()
        plt.show()
