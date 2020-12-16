from decision_tree import *
import random


class RandomForest:
    def __init__(self, num_feature, k, forest_size=50):
        self.num_feature = num_feature
        self.forest_size = forest_size
        self.k = k

    def continuous_to_discrete(self, data):
        feature_type = get_feature_type(data)
        for i in range(len(feature_type)):
            if feature_type[i] == 'continuous_num':
                min_val = min(data[:, i])
                max_val = max(data[:, i])
                interval = (max_val - min_val) / self.k
                for j in range(len(data)):
                    if data[j][i] == max_val:
                        data[j][i] = self.k - 1
                    else:
                        data[j][i] = (data[j][i] - min_val) // interval
        return data

    def sample_data(self, data_train, label, data_test):
        m, n = data_train.shape
        label_list = [i for i in range(n)]
        sample_feature = sorted(random.sample(label_list, 5))
        data_train_sample = np.zeros((m, n))
        label_sample = []
        data_test_sample = np.zeros((m, n))
        for i in range(m):
            j = np.random.randint(0, m)
            for k, feature in enumerate(sample_feature):
                data_train_sample[i][k] = data_train[j][feature]
            label_sample.append(label[j])
        for i in range(len(data_test)):
            for k, feature in enumerate(sample_feature):
                data_test_sample[i][k] = data_train[i][feature]
        return data_train_sample, np.array(label_sample), data_test_sample

    def train_and_predict(self, data_train, label, data_test):
        predictions_list = []
        for _ in range(len(data_test)):
            predictions_list.append({})
        for _ in range(self.forest_size):
            data_dis = self.continuous_to_discrete(np.vstack((data_train, data_test)))
            data_train_dis = data_dis[:len(data_train)]
            data_test_dis = data_dis[len(data_train):]
            training_data, training_label, testing_data = self.sample_data(data_train_dis, label, data_test_dis)
            model = DecisionTree()
            label_predict = model.train_and_predict(training_data, training_label, testing_data)
            for i in range(len(data_test)):
                predictions_list[i][label_predict[i]] = predictions_list[i].get(label_predict[i], 0) + 1
        label_final = []
        for i in range(len(data_test)):
            label_final.append(max(predictions_list[i].items(), key=lambda x: x[1])[0])
        return np.array(label_final)
