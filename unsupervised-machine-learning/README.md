<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">Unsupervised Machine Learning</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe unsupervised machine learning approach and explain the types of unsupervised machine learning algorithms.

## An Introduction to Unsupervised Machine Learning
Unsupervised Machine Learning is a type of machine learning technique where models are trained on datasets without labeled responses. The algorithm tries to learn the inherent structure, patterns, or distributions in the data to derive meaningful insights.

## Key Features of Unsupervised Learning
- **No Labels Required:** Works on unlabeled data.
- **Pattern Discovery:** Identifies hidden patterns or intrinsic structures in input data.
- **Dimensionality Reduction:** Reduces the number of features while retaining significant information.
- **Clustering:** Groups data points based on similarities.

## Clustering
Clustering algorithms partition data into groups based on similarity. There are two types of clustering algorithms.

### 1. K-Means Clustering Algorithms
- K-Means is a widely-used unsupervised learning algorithm designed for clustering tasks.  
- It partitions a dataset into `k` clusters, each represented by its centroid.  
- The goal is to minimize the within-cluster variance by iteratively assigning data points to clusters and recalculating centroids.  
- Applications:
  - **Customer Segmentation**: Grouping customers for targeted marketing strategies.  
  - **Image Segmentation**: Dividing an image into meaningful regions.  
  - **Document Clustering**: Organizing text documents by topics.  
  - **Anomaly Detection**: Identifying data points that deviate significantly from cluster norms.  
  - **Healthcare**: Grouping patients with similar health conditions for better treatment plans.  

#### Demo of a K-Means Clustering program: Data Grouping
This model groups the given data points into 2 clusters.
```python
from sklearn.cluster import KMeans
import numpy as np

# Sample dataset
data = np.array([[1, 2], [3, 4], [5, 6], [8, 9], [10, 11]])

# Apply K-Means Clustering
model = KMeans(n_clusters=2, random_state=42)
model.fit(data)

# Print cluster assignments
print("Cluster Labels:", model.labels_)
```

### 2. Hierarchical Clustering Algorithms
- Hierarchical clustering is an unsupervised learning algorithm used for clustering tasks.  
- Unlike partitioning methods like K-Means, hierarchical clustering builds a hierarchy of clusters, represented as a tree structure called a dendrogram.  
- This approach does not require the user to specify the number of clusters in advance.  
- Types of Hierarchical Clustering:
  - **Agglomerative (Bottom-Up)**: Starts with each data point as an individual cluster and iteratively merges the closest clusters until all points belong to a single cluster.  
  - **Divisive (Top-Down)**: Starts with all data points in a single cluster and recursively splits clusters until each point is its own cluster.  
- Steps for Agglomerative Clustering:
  - **Initialization**: Treat each data point as an individual cluster.  
  - **Distance Calculation**: Compute pairwise distances between all clusters.  
  - **Merging Clusters**: Merge the two clusters with the smallest distance.  
  - **Update Distance Matrix**: Recalculate distances between the newly formed cluster and remaining clusters.  
  - **Repeat**: Continue merging clusters until only one cluster remains.

#### Demo of a Hierarchical Clustering program: Dendrogram Visualization
This code creates a dendrogram to visualize hierarchical clustering of data.

```python
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Sample dataset
data = np.array([[1, 2], [3, 4], [5, 6], [8, 9], [10, 11]])

# Perform hierarchical clustering
linked = linkage(data, method='ward')

# Plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
```

## Use Cases of Unsupervised Learning
- 🧑‍🤝‍🧑 **Market Segmentation:** Identifying customer groups with similar behavior.
- 🚨 **Anomaly Detection:** Spotting unusual patterns in datasets for fraud detection or system monitoring.
- 🛒 **Recommendation Systems:** Suggesting items to users by finding similar users or products.
- 🧬 **Genomics:** Understanding genetic data by identifying groups of genes with similar expressions.
- 🖼️ **Image Compression:** Reducing the size of image data using dimensionality reduction techniques.

