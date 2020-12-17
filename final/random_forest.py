from decision_tree import *
import random


class RandomForest:
    def __init__(self, num_feature, k=10, forest_size=40):
        self.num_feature = num_feature
        self.forest_size = forest_size
        self.k = k

    def continuous_to_discrete(self, data):
        """
            Convert continuous feature to discrete feature
            ----------
            Parameters
            data: dataset
            ----------
            Return
            dataset after convert
        """
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
        """
            Sample dataset for training and testing
            ----------
            Parameters
            data_train: Original training data
            label:  Original label
            data_test: Original testing data
            ----------
            Return
            Sampled training data
            Sampled label
            Sampled testing data
        """
        m, n = data_train.shape
        label_list = [i for i in range(n)]

        # pick random features
        sample_feature = sorted(random.sample(label_list, self.num_feature))

        data_train_sample = np.zeros((m, self.num_feature))
        label_sample = []
        data_test_sample = np.zeros((len(data_test), self.num_feature))
        for i in range(m):
            # pick m random rows
            j = np.random.randint(0, m)
            for k, feature in enumerate(sample_feature):
                # pick feature
                data_train_sample[i][k] = data_train[j][feature]
            label_sample.append(label[i])
        for i in range(len(data_test)):
            for k, feature in enumerate(sample_feature):
                data_test_sample[i][k] = data_test[i][feature]
        return data_train_sample, np.array(label_sample), data_test_sample

    def train_and_predict(self, data_train, label, data_test):
        predictions_list = []
        for _ in range(len(data_test)):
            predictions_list.append({})
        data_dis = self.continuous_to_discrete(np.vstack((data_train, data_test)))
        data_train_dis = data_dis[:len(data_train)]
        data_test_dis = data_dis[len(data_train):]
        for t in range(self.forest_size):
            print('Building tree No.', t)
            training_data, training_label, testing_data = self.sample_data(data_train_dis, label, data_test_dis)
            model = DecisionTree()
            label_predict = model.train_and_predict(training_data, training_label, testing_data)
            for i in range(len(data_test)):
                predictions_list[i][label_predict[i]] = predictions_list[i].get(label_predict[i], 0) + 1
        label_final = []

        for i in range(len(data_test)):
            label_final.append(max(predictions_list[i].items(), key=lambda x: x[1])[0])
        return np.array(label_final)
