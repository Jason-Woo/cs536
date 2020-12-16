from math import log
import copy
import numpy as np


class TreeNode:
    def __init__(self, key, left=None, right=None, label=-1):
        self.key = key  # For leaf, key = -1
        self.left = left
        self.right = right
        self.label = label  # For tree node, label = -1


class DecisionTree:
    def __init__(self, method='id3', max_depth=-1, min_num=-1):
        self.method = method
        self.max_depth = max_depth
        self.min_num = min_num
        self.tree = None

    def find_key_id3(self, data_x, data_y):
        label_dict = {}
        for label in data_y:
            label_dict[label] = label_dict.get(label, 0) + 1
        label_cnt = list(label_dict.values())
        entropy0 = 0
        for i in range(len(label_cnt)):
            frac = label_cnt[i] / sum(label_cnt)
            entropy0 -= frac * log(frac, 2)
        max_info_gain = -1
        max_idx = -1
        keys = None
        for i in range(len(data_x[0])):
            info_gain = entropy0
            feature = {}
            for feat in np.array(data_x)[:, i]:
                feature[feat] = feature.get(feat, 0) + 1
            feature_cnt = list(feature.values())
            feature_val = list(feature.keys())
            for k, feat in enumerate(feature_val):
                label_dict2 = {}
                for j, label in enumerate(data_y):
                    if data_x[j][i] == feat:
                        label_dict2[label] = label_dict2.get(label, 0) + 1
                label_cnt2 = list(label_dict2.values())
                entropy_tmp = 0
                for j in range(len(label_cnt2)):
                    frac = label_cnt2[j] / sum(label_cnt2)
                    entropy_tmp -= frac * log(frac, 2)
                info_gain -= feature_cnt[k] / sum(feature_cnt) * entropy_tmp
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_idx = i
                keys = feature_val
        return max_idx, keys

    def split_data(self, data_x, data_y, idx, key):
        data_x_split, data_y_split = [], []
        for i in range(len(key)):
            data_x_split.append([])
            data_y_split.append([])
        for i in range(len(data_x)):
            key.index()

    def build_tree(self, data_x, data_y, label_list, depth, threshold):
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
            leaf = TreeNode(-1, label=data_y[0])
            return leaf
        elif len(data_x[0]) == 0 or depth == 0 or len(data_y) <= threshold:
            # All feature have been use, or we are reaching maximal depth or minimal size of data, then it is a leaf node
            lbl = 1 if 2 * sum(data_y) > len(data_y) else 0
            leaf = TreeNode(-1, label=lbl)
            return leaf
        else:
            key = -1
            if self.method == 'id3':
                key = self.find_key_id3(data_x, data_y)
            data_x1, data_x2, data_y1, data_y2 = self.split_data(data_x, data_y, key)
            node = TreeNode(label_list[key])
            new_label_list = copy.deepcopy(label_list[:key] + label_list[key + 1:])
            left_node = self.build_tree(data_x1, data_y1, new_label_list, depth - 1, threshold)
            right_node = self.build_tree(data_x2, data_y2, new_label_list, depth - 1, threshold)
            node.left = left_node
            node.right = right_node
            return node

    def train(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y, [], self.max_depth, self.min_num)

    def predict(self, data_test):
        """
            Predict label of test data using the decision tree
            ----------
            Parameters
            data_test: 2-d list
                feature values of testing data
            ----------
            Return
            res: list
                predicted label of the testing data
        """
        res = []
        for data in data_test:
            tmp_tree = self.tree
            while tmp_tree.key != -1:
                if data[tmp_tree.key] == 0:
                    tmp_tree = tmp_tree.left
                else:
                    tmp_tree = tmp_tree.right
            res.append(tmp_tree.label)
        return res

