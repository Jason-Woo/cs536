from knn import *
from regression import *
from decision_tree import *

import numpy as np
import csv


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def generate_data(dataset, cols, k=10):
    with open(dataset, 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    data_full = np.array(rows[1:])
    np.random.shuffle(data_full)
    data_y = data_full[:, cols]
    data_x = np.delete(data_full, cols, axis=1)
    feature_type_x = get_feature_type(data_x)
    feature_type_y = get_feature_type(data_y)
    for i, f in enumerate(feature_type_x):
        if f == 'string':
            encoded_x = one_hot_encoding(data_x[:, i])
            data_x = np.delete(data_x, i, axis=1)
            data_x = np.hstack((data_x, encoded_x))
            feature_type_x = np.delete(feature_type_x, i)
            feature_type_x = np.hstack((feature_type_x, np.array(['discrete_num'] * len(encoded_x[0]))))
    data_x = data_x.astype(np.float64)
    for i in range(len(data_x[0])):
        data_x[:, i], _ = normalization(data_x[:, i])
    data_x_split = np.split(data_x, k)
    data_y_split = np.split(data_y, k)
    return data_x_split, data_y_split, feature_type_x, feature_type_y


def cal_acc(label_real, label_predict, task):
    if task == 'regression':
        return np.square(np.subtract(label_real, label_predict)).mean()
    elif task == 'classification_num':
        cnt = 0
        for i in range(len(label_real)):
            if float(label_real[i]) == float(label_predict[i]):
                cnt += 1
        return cnt / len(label_real)
    elif task == 'classification_str':
        cnt = 0
        for i in range(len(label_real)):
            if str(label_real[i]) == str(label_predict[i]):
                cnt += 1
        return cnt / len(label_real)


def get_feature_type(data):
    feature_type = []
    for i in range(len(data[0])):
        if is_number(data[0][i]):
            tmp_col = list(map(float, data[:, i]))
            if len(set(tmp_col[:50])) <= 10:
                feature_type.append('discrete_num')
            else:
                feature_type.append('continuous_num')
        else:
            feature_type.append('string')
    return np.array(feature_type)


def get_task_priority(tasks):
    priority = []
    for i, task in enumerate(tasks):
        if task == 'discrete_num' or 'string':
            priority = [i] + priority
        elif task == 'continuous_num':
            priority = priority + [i]
    return priority


def normalization(data):
    data_min = min(data)
    data_max = max(data)
    data_mean = np.mean(data)
    if data_max == data_min:
        return np.ones(len(data)), [data_max, 0]
    else:
        data = (data - data_mean) / (data_max - data_min)
        return data, [data_max - data_min, data_mean]


def normalization_reverse(data, info):
    data_max_min = info[0]
    data_mean = info[1]
    data = data * data_max_min + data_mean
    return data


def one_hot_encoding(data):
    encoded_data = []
    value = list(set(data))
    value_cnt = len(value)
    for i in range(len(data)):
        data_tmp = np.zeros(value_cnt)
        for j in range(value_cnt):
            if data[i] == value[j]:
                data_tmp[j] = 1
                encoded_data.append(data_tmp)
                continue
    return encoded_data


if __name__ == '__main__':
    dataset_path = 'Skyserver_SQL2_27_2018 6_51_39 PM.csv'
    dataset_path1 = 'test.csv'
    target_col = [13, 10]
    cross_validation_size = 2
    regression_model = 'regression'
    classification_model = 'decision_tree'

    data_x, data_y, f_x, f_y = generate_data(dataset_path1, target_col, cross_validation_size)
    task_priority = get_task_priority(f_y)

    acc_regression, acc_classification = 0, 0
    num_regression, num_classification = 0, 0
    for i in range(cross_validation_size):
        print("Cross Validation fold ", i)
        training_data = np.vstack(data_x[:i] + data_x[i + 1:])
        testing_data = data_x[i]
        training_label = np.vstack(data_y[:i] + data_y[i + 1:]).transpose()
        testing_label = data_y[i].transpose()
        for task in task_priority:
            print("Processing column ", task, " ", f_y[task])
            predict_label = None
            if f_y[task] == 'discrete_num' or f_y[task] == 'string':
                if classification_model == 'knn':
                    model = KNN(50)
                    predict_label = model.train_and_predict(training_data, training_label[task], testing_data)
                if classification_model == 'decision_tree':
                    model = DecisionTree()
                    predict_label = model.train_and_predict(training_data, training_label[task], testing_data)
                tmp_acc = 0
                if f_y[task] == 'discrete_num':
                    tmp_acc = cal_acc(testing_label[task], predict_label, 'classification_num')
                    acc_classification += tmp_acc
                    num_classification += 1
                elif f_y[task] == 'string':
                    tmp_acc = cal_acc(testing_label[task], predict_label, 'classification_str')
                    acc_classification += tmp_acc
                    num_classification += 1
                print("acc_classification = ", tmp_acc)

            elif f_y[task] == 'continuous_num':
                training_label_normalized, meta_data = normalization(training_label[task].astype(np.float64))
                if regression_model == 'regression':
                    model = Regression()
                    model.train(training_data, training_label_normalized)
                    predict_label_normalized = model.predict(testing_data)
                    predict_label = normalization_reverse(predict_label_normalized, meta_data)
                tmp_acc = cal_acc(testing_label[task].astype(np.float64), predict_label, 'regression')
                acc_regression += tmp_acc
                num_regression += 1
                print("acc_regression = ", tmp_acc)

            if f_y[task] == 'string':
                new_label = np.hstack((training_label[task], predict_label))
                encoded_label = np.array(one_hot_encoding(new_label))
                training_data = np.hstack((training_data, encoded_label[:len(training_label[task])]))
                testing_data = np.hstack((testing_data, encoded_label[len(training_label[task]):]))
            elif f_y[task] == 'discrete_num' or 'continuous_num':
                new_label = np.hstack((training_label[task].astype(np.float64), predict_label.astype(np.float64)))
                normalized_label, _ = normalization(new_label)
                normalized_label = normalized_label[:, np.newaxis]
                training_data = np.hstack((training_data, normalized_label[:len(training_label[task])]))
                testing_data = np.hstack((testing_data, normalized_label[len(training_label[task]):]))
    if num_classification != 0:
        acc_classification /= num_classification
        print(acc_classification)
    if num_regression != 0:
        acc_regression /= num_regression
        print(acc_regression)






