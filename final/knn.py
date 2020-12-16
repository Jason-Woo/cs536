import math
import numpy as np


class KNN:
    def __init__(self, k, method='Euclidean'):
        self.k = k
        self.method = method

    def euclidean_distance(self, arr1, arr2):
        return np.sqrt(np.sum((arr1 - arr2) ** 2))
        # return np.linalg.norm(arr1 - arr2, ord=2)

    def manhattan_distance(self, arr1, arr2):
        return np.linalg.norm(arr1 - arr2, ord=1)

    def train_and_predict(self, data_x, label_x, data_y):
        label_predict = []
        n = data_x.shape[0]
        for i, data in enumerate(data_y):
            if i % 100 == 0:
                print("KNN ", i, "/", len(data_y))
            if self.method == 'Euclidean':
                distance = [[self.euclidean_distance(data_x[j], data), j] for j in range(n)]
            elif self.method == 'Manhattan':
                distance = [[self.manhattan_distance(data_x[j], data), j]for j in range(n)]
            else:
                distance = []
            distance_sorted = [j[1] for j in sorted(distance)]
            k_nearest = {}
            for key in distance_sorted[:self.k]:
                k_nearest[key] = k_nearest.get(key, 0) + 1
            nearest = max(k_nearest.items(), key=lambda x: x[1])[0]
            label_predict.append(label_x[nearest])
        return np.array(label_predict)

