import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score


class KMeans:
    def __init__(self, n_clusters, data, epochs=300, epsila=0.0001):
        self.K = n_clusters
        self.data = data
        self.epochs = epochs
        self.epsila = epsila
        
        np.random.seed(42)
        random_index = np.random.choice(self.data.shape[0], self.K, replace=False)
        
        self.weights = self.data[random_index]
        self.new_weights = self.weights.copy()
        # print(f'weights:{self.weights}')
        
    def fit(self):
        self.label = self.E_distance(self.weights)
        self.new_label = self.label.copy()
        for epoch in range(self.epochs):
            # print(f'epoch:{epoch}')
            #更新群聚中心
            for k in range(self.K):
                self.new_weights[k,:] = np.mean(self.data[self.label==k], axis=0)
            # print(self.new_weights)
            # print(f'new weight{self.new_weights}')
            # print(f'new_weights-weights{self.new_weights-self.weights}')
            #計算新的群聚中心的位移量
            max_change = np.linalg.norm(self.new_weights - self.weights, axis=1).max()
            self.new_label = self.E_distance(self.new_weights)
            #判斷是否要提前終止
            if max_change<self.epsila or np.equal(self.new_label, self.label).all():
                break
            self.label = self.new_label.copy()
            self.weights = self.new_weights.copy()

        # 這裡計算每一群的標準差 (σ)
        cluster_std = []
        for i in range(self.K):
            cluster_data = self.data[self.label == i]
            mean_distance = np.mean(np.linalg.norm(cluster_data - self.weights[i], axis=1))
            cluster_std.append(mean_distance)

        cluster_std = np.array(cluster_std)
        # print(self.new_weights,cluster_std)
        
        silhouette_avg = silhouette_score(self.data, self.label)
        # print("輪廓係數:", silhouette_avg)
        return self.new_weights, cluster_std
        
        
    def E_distance(self, weights):
        distances = np.zeros((self.data.shape[0], weights.shape[0]))
        for k in range(self.K):
            # print(f'self.data - self.weights[{k}]{self.data - self.weights[k]}')
            distances[:,k] = np.linalg.norm(self.data - weights[k], axis=1)
        # print(f'distances:{distances}')
        label = np.argmin(distances, axis=1)
        # print(f'label:{label}')
        return label
