<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">K-Nearest Neighbors</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe K-nearest neighbors algorithms used for supervised machine learning.

## An Introduction to K-Nearest Neighbors
K-Nearest Neighbors (KNN) is a non-parametric, instance-based machine learning algorithm used for classification and regression tasks. It operates on the principle of similarity: a data point is predicted to belong to a category or have a value similar to its nearest neighbors in the dataset. KNN is widely used for its simplicity and effectiveness across various real-world applications.

## Key Concepts
-   KNN does not require explicit model-building or parameter learning during the training phase. Instead, it stores the entire dataset as a reference for predictions.
-   KNN uses distance metrics such as Euclidean, Manhattan, and Minkowski to identify the nearest neighbors.
-   The parameter `k` determines how many neighbors are considered for predictions:
    -   Small `k` values may lead to overfitting.
    -   Large `k` values may underfit the data.
-   Feature scaling is critical for KNN because it relies on distance calculations. Normalization or standardization ensures all features contribute equally.
-   Weighted neighbors assign more influence to closer neighbors, improving prediction accuracy in many cases.

## Core Assumptions
-   Assumes that similar data points are located near each other in the feature space.
-   Assumes that the chosen distance metric appropriately represents similarity for the dataset.
-   Performance depends heavily on careful preprocessing and hyperparameter selection.

## Types of KNN
- **Classification KNN**: Predicts discrete categories based on the most frequent class among the `k` nearest neighbors. For example, predicting whether a user will purchase a product based on browsing behavior and pricing.
- **Regression KNN**: Predicts continuous values by averaging (or weighting) the values of the `k` nearest neighbors. For example, estimating the total cart value for a user based on similar customers.

### **How KNN Works**
-   **Training Phase**: KNN does not involve explicit training. Instead, the entire dataset is stored as a reference.
-   **Prediction Phase**:
    -   For a new data point, calculate its distance to all other points in the dataset using a chosen metric (e.g., Euclidean).
    -   Identify the `k` nearest neighbors.
    -   For classification, assign the most frequent class among the neighbors.
    -   For regression, calculate the average (or weighted average) of the neighbors' values.

## Steps to Build a KNN Model
1.  Choose the value of `k` based on the dataset's characteristics.
2.  Select an appropriate distance metric for similarity measurement.
3.  Preprocess data by scaling features and removing irrelevant attributes.
4.  Validate the model using cross-validation to ensure optimal hyperparameter settings.

## Demo: Data Classifier
This model classifies a new data point by checking its 3 nearest neighbors.
```python
from sklearn.neighbors import KNeighborsClassifier

# Train KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Make a prediction
print("Prediction:", model.predict([[30, 50000]])[0])
```

## Applications
-   In healthcare, predicting patient diagnoses based on similar cases.
-   In marketing, segmenting customers and predicting purchase behavior.
-   In finance, identifying fraudulent transactions through anomaly detection.
-   In e-commerce, recommending products based on user similarity.

## Advantages
-   Simple and easy to understand without requiring complex parameter tuning.
-   Flexible, handling both classification and regression tasks effectively.
-   Non-parametric, making no assumptions about the underlying data distribution.
-   Adapts dynamically to new data without retraining.

## Limitations
-   Computationally expensive for large datasets, requiring optimizations like approximate nearest neighbors.
-   Performance is sensitive to irrelevant or noisy features, necessitating careful preprocessing.
-   Struggles in high-dimensional datasets due to the curse of dimensionality.
-   Requires significant memory to store the entire dataset for predictions.

## Common Challenges with KNN
-   **Computational Complexity**: Distance calculations for every query can be slow for large datasets.
-   **Memory Usage**: Storing the entire dataset for predictions can be resource-intensive.
-   **Irrelevant Features**: Noisy or irrelevant features can degrade prediction accuracy.
-   **Curse of Dimensionality**: In high-dimensional spaces, distance metrics become less effective at identifying true nearest neighbors.

## 🗣️ Discussion Activity: K-Nearest Neighbors
As a class or in small groups via breakout rooms, discuss the following questions. Be prepared to share your ideas!
1.  How could ShopSmart use KNN for personalized recommendations or customer segmentation?
2.  What challenges might arise when scaling KNN for large datasets?
