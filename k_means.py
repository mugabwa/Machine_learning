import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample
np.random.seed(42)
def edistance(x1,x2):
    return np.square(np.sum(pow(x1-x2,2)))

class KMeans:
    def __init__(self, K=2, iters = 100, plot_steps=False):
        self.K = K
        self.iters = iters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.plot_steps = plot_steps

    def predict(self, x):
        self.x = x
        self.no_sample, self.no_features = x.shape

        random_sample_idx = np.random.choice(self.no_sample,self.K,replace=False)
        self.centroids = [self.x[idx] for idx in random_sample_idx]


        #optimize
        for _ in range(self.iters):
            # update cluster
            self.clusters = self.create_cluster(self.centroids)
            if self.plot_steps:
                self.plot()
            # update centroid
            centroids_old = self.centroids
            self.centroids = self.get_centroid(self.clusters)
            if self.plot_steps:
                self.plot()
            #Convergence
            if self.is_converged(centroids_old, self.centroids):
                break

        #return cluster labels
        return self.get_culster_labels(self.clusters)

    def create_cluster(self, centroid):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self.closest_cent(sample,centroid)
            clusters[centroid_idx].append(idx)
        return clusters

    def closest_cent(self, sample, cent):
        distance = [edistance(sample, point) for point in cent]
        closest_idx = np.argmin(distance)
        return closest_idx
        
    def get_centroid(self, clusters):
        cent = np.zeros((self.K, self.no_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            cent[cluster_idx] = cluster_mean
        return cent

    def is_converged(self, cent_old, cent_new):
        distance = [edistance(cent_old[i], cent_new[i]) for i in range(self.K)]
        return sum(distance) == 0

    def get_culster_labels(self, clusters):
        labels = np.empty(self.no_sample)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
            return labels

    ## Ploting
    def plot(self):
        fig, axis = plt.subplots(figsize = (12,8))
        for i, idx in enumerate(self.clusters):
            point = self.x[idx].T
            axis.scatter(*point)

        for point in self.centroids:
            axis.scatter(*point, marker="x",color="black",linewidth=2)
        plt.show()
