from knn import *
from regression import *
from decision_tree import *
from random_forest import *
from basic_completion import *
from math import sqrt

import numpy as np
import csv


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


def generate_data(dataset, cols, k=10):
    """
        Split dataset into data and label, split the data into k parts
        ----------
        Parameters
        dataset: The raw dataset
        col: The columns we use as label
        k: Number of parts we want to split
        ----------
        Return
        data_x_split: Data (k parts)
        data_y_split: Label (k parts)
        feature_type_x: List of feature type of data
        feature_type_y: List of feature type of label
    """
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
        if f == 'string':  # One hot encoding if string
            encoded_x = one_hot_encoding(data_x[:, i])
            data_x = np.delete(data_x, i, axis=1)
            data_x = np.hstack((data_x, encoded_x))
            feature_type_x = np.delete(feature_type_x, i)
            feature_type_x = np.hstack((feature_type_x, np.array(['discrete_num'] * len(encoded_x[0]))))
    data_x = data_x.astype(np.float64)
    for i in range(len(data_x[0])):  # Normalizing all columns
        data_x[:, i], _ = normalization(data_x[:, i])
    data_x_split = np.split(data_x, k)
    data_y_split = np.split(data_y, k)
    return data_x_split, data_y_split, feature_type_x, feature_type_y


def cal_acc(label_real, label_predict, task):
    """
        Calculate accuracy / RMSE
        ----------
        Parameters
        label_real: Real label
        label_predict: Predicted label
        task: regression/classification
        ----------
        Return
        accuracy for classification tasks
        RMSE for regression tasks
    """
    if task == 'regression':
        return sqrt(np.square(np.subtract(label_real, label_predict)).mean())
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
            if len(set(tmp_col[:50])) <= 10:
                feature_type.append('discrete_num')
            else:
                feature_type.append('continuous_num')
        else:
            feature_type.append('string')
    return np.array(feature_type)


def get_task_priority(tasks):
    """
        Get the executing sequence of the tasks
        ----------
        Parameters
        tasks: List of tasks
        ----------
        Return
        The executing sequence of the tasks
    """
    priority = []
    for i, task in enumerate(tasks):
        if task == 'discrete_num' or 'string':  # classification goes first
            priority = [i] + priority
        elif task == 'continuous_num':  # regression goes last
            priority = priority + [i]
    return priority


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


def data_completion(data_x, data_y, f_y, regression_model, classification_model, cross_validation_size, task_priority):
    """
        Main function of data completion
        ----------
        Parameters
        data_x: Data
        Data_y: labels
        f_y: type of labels
        regression_model: model used for regression tasks
        classification_model: model used for classification tasks ,
        cross_validation_size: k value of cross_validation
        task_priority: The executing sequence of the tasks
        ----------
        Return
        Accuracy
    """
    acc_regression, acc_classification = 0, 0
    num_regression, num_classification = 0, 0
    for i in range(cross_validation_size):
        print("Cross Validation fold ", i)
        # Make training and testing dataset
        training_data = np.vstack(data_x[:i] + data_x[i + 1:])
        testing_data = data_x[i]
        training_label = np.vstack(data_y[:i] + data_y[i + 1:]).transpose()
        testing_label = data_y[i].transpose()
        for task in task_priority:
            print("Processing column ", task, " ", f_y[task])
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
                    predict_label = model.train_and_predict(training_data, training_label[task], testing_data)
                elif classification_model == 'basic_completion':
                    model = BasicCompletion('discrete')
                    predict_label = model.predict(training_label[task], testing_data)
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
                tmp_acc = cal_acc(testing_label[task].astype(np.float64), predict_label, 'regression')
                acc_regression += tmp_acc
                num_regression += 1
                print("acc_regression = ", tmp_acc)

            # Encoding/normalizing the new column
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
    # Store the accuracy/RMSE
    acc = [-1, -1]
    have_task = [0, 0]
    if num_classification != 0:
        acc_classification /= num_classification
        have_task[0] = 1
        acc[0] = acc_classification
    if num_regression != 0:
        acc_regression /= num_regression
        have_task[1] = 1
        acc[1] = acc_regression
    return have_task, acc


if __name__ == '__main__':
    dataset_path = 'Skyserver_SQL2_27_2018 6_51_39 PM.csv'
    target_col = [17]  # List specifying the missing column
    cross_validation_size = 10
    regression_model_list = ['naive_regression', 'ridge_regression', 'lasso_regression', 'basic_completion']
    classification_model_list = ['knn', 'decision_tree', 'random_forest', 'basic_completion']
    find_best_model = False  # Set to True to iterate over the model list

    best_regression_model, best_classification_model = None, None
    best_regression_loss, best_classification_acc = 1e20, 0

    x, y, f_x, f_y = generate_data(dataset_path, target_col, cross_validation_size)
    task_priority = get_task_priority(f_y)

    if find_best_model:
        # iterate over the model list
        for i in range(3):
            status, accuracy = data_completion(x, y, f_y, regression_model_list[i], classification_model_list[i], cross_validation_size, task_priority)
            if status[1] == 1:
                if accuracy[1] < best_regression_loss:
                    best_regression_loss = accuracy[1]
                    best_regression_model = regression_model_list[i]
            if status[0] == 1:
                if accuracy[0] > best_classification_acc:
                    best_classification_acc = accuracy[0]
                    best_classification_model = classification_model_list[i]
        print("-----------------------------------")
        if best_regression_model is not None:
            print("Best regression model is ", best_regression_loss)
            print("RMSE = ", best_regression_loss)
        if best_classification_model is not None:
            print("Best classification model is ", best_classification_model)
            print("Accuracy = ", best_classification_acc)
    else:
        # Use selected model
        regression_model = 'basic_completion'
        classification_model = 'basic_completion'
        status, accuracy = data_completion(x, y, f_y, regression_model, classification_model, cross_validation_size, task_priority)
        print("-----------------------------------")
        if status[1] == 1:
            print("For regression mode, RMSE = ", accuracy[1])
        if status[0] == 1:
            print("For classification model, Accuracy = ", accuracy[0])
