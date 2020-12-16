from math import log
import copy
import numpy as np


class TreeNode:
    def __init__(self, key, label=-1):
        self.key = key  # For leaf, key = -1
        self.children = []
        self.children_label = []
        self.label = label  # For tree node, label = -1

    def add_child(self, child, label):
        self.children.append(child)
        self.children_label.append(label)


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
            j = key.index(data_x[i][idx])
            data_x_split[j].append(copy.deepcopy(data_x[i][:idx] + data_x[i][idx + 1:]))
            data_y_split[j].append(copy.deepcopy(data_y[i]))
        return data_x_split, data_y_split

    def build_tree(self, data_x, data_y, label_list, depth, threshold):
        if len(list(set(data_y))) == 1:  # All data in the node have same label, then it is a leaf node
            leaf = TreeNode(-1, label=data_y[0])
            return leaf
        elif len(data_x[0]) == 0 or depth == 0 or len(data_y) <= threshold:
            # All feature have been use, or we are reaching maximal depth or minimal size of data, then it is a leaf node
            label_cnt = {}
            for label in data_y:
                label_cnt[label] = label_cnt.get(label, 0) + 1
            label_max = max(label_cnt.items(), key=lambda x: x[1])[0]
            leaf = TreeNode(-1, label=label_max)
            return leaf
        else:
            idx, key = self.find_key_id3(data_x, data_y)
            data_x_split, data_y_split = self.split_data(data_x, data_y, idx, key)
            node = TreeNode(label_list[idx])
            new_label_list = copy.deepcopy(label_list[:key] + label_list[key + 1:])
            for i in range(len(data_x_split)):
                children_node = self.build_tree(data_x_split[i], data_y_split[i], new_label_list, depth - 1, threshold)
                node.add_child(children_node, key[i])
            return node

    def train(self, data_x, data_y):
        label_list = [i for i in range(len(data_x[0]))]
        self.tree = self.build_tree(data_x, data_y, [], self.max_depth, self.min_num)

    def predict(self, data_test):
        return 0

