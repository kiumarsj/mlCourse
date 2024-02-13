# k-means
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# generating sample data

n_samples = 300
random_state = 20
x,y = make_blobs(n_samples=n_samples, random_state=random_state)

# performing k-means clustering
# kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
km = KMeans(n_clusters=3, init='k-means++', random_state=random_state, n_init=10, max_iter=100)
km.fit(x)
y_km = km.predict(x)

# visualizing the clusters
plt.scatter(x[:,0], x[:,1], c=y_km, s=50, cmap='viridis')
centers = km.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.75)
plt.tight_layout()
plt.show()