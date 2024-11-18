import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


class KMeans:
    def __init__(self, n_clusters, data, epochs=300, epsilon=0.0001):
        """
        Initialize the KMeans clustering algorithm.

        Parameters:
        - n_clusters: int, the number of clusters.
        - data: np.ndarray, the dataset to cluster.
        - epochs: int, maximum number of iterations (default=300).
        - epsilon: float, convergence threshold (default=0.0001).
        """
        self.K = n_clusters
        self.data = data
        self.epochs = epochs
        self.epsilon = epsilon

        # Initialize cluster centers by selecting random data points
        np.random.seed(42)
        random_indices = np.random.choice(self.data.shape[0], self.K, replace=False)
        self.weights = self.data[random_indices]
        self.new_weights = self.weights.copy()

    def fit(self):
        """
        Fit the KMeans model to the data.

        Returns:
        - new_weights: np.ndarray, the final cluster centers.
        - cluster_std: np.ndarray, the standard deviation of each cluster.
        """
        # Initial assignment of labels based on current weights
        self.label = self._assign_labels(self.weights)
        self.new_label = self.label.copy()

        for epoch in range(self.epochs):
            # Update each cluster center
            for k in range(self.K):
                cluster_data = self.data[self.label == k]
                if len(cluster_data) > 0:
                    self.new_weights[k, :] = np.mean(cluster_data, axis=0)

            # Compute the maximum shift among all cluster centers
            max_change = np.linalg.norm(self.new_weights - self.weights, axis=1).max()

            # Reassign labels based on updated cluster centers
            self.new_label = self._assign_labels(self.new_weights)

            # Check for convergence
            if max_change < self.epsilon or np.array_equal(self.new_label, self.label):
                # print(f"Converged at epoch {epoch}")
                break

            # Update weights and labels for next iteration
            self.weights = self.new_weights.copy()
            self.label = self.new_label.copy()

        # Calculate standard deviation for each cluster
        cluster_std = self._calculate_cluster_std()

        # Compute the overall silhouette score
        silhouette_avg = silhouette_score(self.data, self.label)
        # print("Silhouette Coefficient:", silhouette_avg)

        return self.new_weights, cluster_std

    def _assign_labels(self, weights):
        """
        Assign each data point to the nearest cluster center.

        Parameters:
        - weights: np.ndarray, current cluster centers.

        Returns:
        - labels: np.ndarray, cluster labels for each data point.
        """
        # Calculate distances from each point to each cluster center
        distances = np.linalg.norm(self.data[:, np.newaxis] - weights, axis=2)
        # Assign labels based on the nearest cluster center
        labels = np.argmin(distances, axis=1)
        return labels

    def _calculate_cluster_std(self):
        """
        Calculate the standard deviation of each cluster.

        Returns:
        - cluster_std: np.ndarray, standard deviations for each cluster.
        """
        cluster_std = []
        for i in range(self.K):
            cluster_data = self.data[self.label == i]
            if len(cluster_data) > 0:
                # Compute mean distance of points in the cluster to the cluster center
                mean_distance = np.mean(np.linalg.norm(cluster_data - self.weights[i], axis=1))
                cluster_std.append(mean_distance)
            else:
                # If a cluster has no points, set its standard deviation to 0
                cluster_std.append(0.0)
        return np.array(cluster_std)


def plot_clusters(data, labels, centers, silhouette_avg, cluster_std):
    """
    Plot the clustered data along with cluster centers and silhouette score.

    Parameters:
    - data: np.ndarray, the dataset.
    - labels: np.ndarray, cluster labels for each data point.
    - centers: np.ndarray, coordinates of cluster centers.
    - silhouette_avg: float, average silhouette score.
    - cluster_std: np.ndarray, standard deviation of each cluster.
    """
    plt.figure(figsize=(12, 6))

    # Scatter plot of data points colored by cluster label
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster Label')

    # Display silhouette score
    plt.subplot(1, 2, 2)
    plt.bar(range(len(cluster_std)), cluster_std, color='skyblue')
    plt.title('Cluster Standard Deviations')
    plt.xlabel('Cluster')
    plt.ylabel('Standard Deviation')
    plt.xticks(range(len(cluster_std)), [f'Cluster {i}' for i in range(len(cluster_std))])

    plt.suptitle(f'KMeans Clustering Results\nSilhouette Coefficient: {silhouette_avg:.2f}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Generate synthetic data using make_blobs
    data, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=4, data=data)
    centers, cluster_std = kmeans.fit()

    # Compute silhouette score for plotting
    silhouette_avg = silhouette_score(data, kmeans.label)

    # Plot the clustering results
    plot_clusters(data, kmeans.label, centers, silhouette_avg, cluster_std)