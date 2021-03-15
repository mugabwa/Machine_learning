import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from k_means import KMeans
# x,y = make_blobs(centers=3, n_samples=100,n_features=2,shuffle=True,random_state=42)
# x,y = make_blobs(centers=3, n_samples=100,n_features=2)
x = [[1,1],[2,1],[4,3],[5,4],[1,3],[4,3],[9,8]]
y = np.array(x)
print(y.shape)
clusters = 2
# clusters = len(np.unique(y))
# print(clusters)
k = KMeans(K=clusters, iters=150, plot_steps=True)
# y_pred = k.predict(x)
y_pred = k.predict(y)
# k.plot()
# print(x)