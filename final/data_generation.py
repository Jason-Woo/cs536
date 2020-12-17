from knn import *
from regression import *
from decision_tree import *
from random_forest import *
from basic_completion import *
from math import sqrt

import numpy as np
import csv
import random


def is_number(s):
    """
        Judge whether the given string can be converted into a number
        ----------
        Parameters
        s: string
        ----------
        Return
        whether s can be be converted into a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_feature_type(data):
    """
        Get the type of all features
        ----------
        Parameters
        data: Dataset
        ----------
        Return
        List of feature types of each columns
    """
    feature_type = []
    for i in range(len(data[0])):
        if is_number(data[0][i]):
            tmp_col = list(map(float, data[:, i]))
            if len(set(tmp_col)) <= 10:
                feature_type.append('discrete_num')
            else:
                feature_type.append('continuous_num')
        else:
            feature_type.append('string')
    return np.array(feature_type)


def basic_generate(dataset, size):
    """
        Generate data according to assumed distribution
        ----------
        Parameters
        dataset: Real dataset
        size: size of generated dataset
        ----------
        Return
        The generated dataset
    """
    feature_type = get_feature_type(dataset)
    data_generate = []
    for i in range(len(feature_type)):
        tmp_data = []
        if feature_type[i] == 'discrete_num' or 'string':
            ele_set = list(set(dataset[:, i]))
            for j in range(size):
                # Assume that discrete features satisfy distribution
                tmp_data.append(random.sample(ele_set, 1)[0])
        elif feature_type[i] == 'continuous_num':
            min_val = min(dataset[:, i].astype(np.float64))
            max_val = max(dataset[:, i].astype(np.float64))
            for j in range(size):
                # Assume that continuous features satisfy normal distribution
                tmp_data.append(str(random.uniform(min_val, max_val)))
        data_generate.append(np.array(tmp_data))
    return np.array(data_generate).transpose()


def one_hot_encoding(data):
    """
        One-hot encoding the feature
        ----------
        Parameters
        Data: Dataset column
        ----------
        Return
        The encoded columns
    """
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


def normalization(data):
    """
        normalize every column of the data
        ----------
        Parameters
        Data: Dataset column
        ----------
        Return
        The normalized dataset
        Meta data for recovery
    """
    data_min = min(data)
    data_max = max(data)
    data_mean = np.mean(data)
    if data_max == data_min:
        return np.ones(len(data)), [data_max, 0]
    else:
        data = (data - data_mean) / (data_max - data_min)
        return data, [data_max - data_min, data_mean]


def normalization_reverse(data, info):
    """
        Recover normalized data to real data
        ----------
        Parameters
        Data: Dataset column
        Meta data for recovery
        ----------
        Return
        The real dataset
    """
    data_max_min = info[0]
    data_mean = info[1]
    data = data * data_max_min + data_mean
    return data


def generate_data(data_full, cols):
    """
        Split dataset into data and label
        ----------
        Parameters
        dataset: The raw dataset
        col: The columns we use as label
        ----------
        Return
        data_x: Data
        data_y: Label
        feature_type_x: List of feature type of data
        feature_type_y: List of feature type of label
    """
    cols = [cols]
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
    return data_x, data_y, feature_type_x, feature_type_y


def data_completion(data_x, data_y, f_y, task, regression_model, classification_model, n):
    """
        Main function of data completion
        ----------
        Parameters
        data_x: Data
        Data_y: labels
        f_y: type of labels
        task: target column
        regression_model: model used for regression tasks
        classification_model: model used for classification tasks ,
        n: size of the real datset
        ----------
        Return
        Predicted label of target column
    """
    training_data = data_x[:n, :]
    testing_data = data_x[n:,:]
    training_label = data_y[:n, :].transpose()
    predict_label = None
    if f_y[task] == 'discrete_num' or f_y[task] == 'string':
        if classification_model == 'knn':
            model = KNN(20)
            predict_label = model.train_and_predict(training_data, training_label[task], testing_data)
        elif classification_model == 'decision_tree':
            model = DecisionTree(max_depth=10, k=10)
            predict_label = model.train_and_predict(training_data, training_label[task], testing_data)
        elif classification_model == 'random_forest':
            model = RandomForest(8)
            print(training_label[task])
            predict_label = model.train_and_predict(training_data, training_label[task], testing_data)
        elif classification_model == 'basic_completion':
            model = BasicCompletion('discrete')
            predict_label = model.predict(training_label[task], testing_data)
    elif f_y[task] == 'continuous_num':
        training_label_normalized, meta_data = normalization(training_label[task].astype(np.float64))
        if regression_model == 'naive_regression':
            model = Regression(method='naive')
            model.train(training_data, training_label_normalized)
            predict_label_normalized = model.predict(testing_data)
            predict_label = normalization_reverse(predict_label_normalized, meta_data)
        elif regression_model == 'ridge_regression':
            model = Regression(method='ridge')
            model.train(training_data, training_label_normalized)
            predict_label_normalized = model.predict(testing_data)
            predict_label = normalization_reverse(predict_label_normalized, meta_data)
        elif regression_model == 'lasso_regression':
            model = Regression(method='lasso')
            model.train(training_data, training_label_normalized)
            predict_label_normalized = model.predict(testing_data)
            predict_label = normalization_reverse(predict_label_normalized, meta_data)
        elif classification_model == 'basic_completion':
            model = BasicCompletion('continuous')
            predict_label = model.predict(training_label[task], testing_data)
    return predict_label


if __name__ == '__main__':
    dataset_path = 'Skyserver_SQL2_27_2018 6_51_39 PM.csv'
    generate_data_size = 50
    regression_model = 'lasso_regression'
    classification_model = 'random_forest'
    with open(dataset_path, 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    data_full = np.array(rows[1:])
    basic_data = basic_generate(data_full, generate_data_size)

    for i in range(len(data_full[0])):
        print("Predicting row", i)
        x, y, f_x, f_y = generate_data(np.vstack((data_full, basic_data)), i)
        label = data_completion(x, y, f_y, 0, regression_model, classification_model, len(data_full))
        basic_data[:, i] = label
    np.savetxt("generate.csv", basic_data, delimiter=",", fmt='%s')












