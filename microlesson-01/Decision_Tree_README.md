
# Decision Tree 

## 1. Introduction to Decision Trees
- A decision tree is a supervised machine learning algorithm used for both classification and regression tasks.
- It splits the data into subsets based on the feature values, resulting in a tree-like structure of decisions.

### Types of Decision Trees:
1. **Classification Tree**: Predicts categorical outcomes.
2. **Regression Tree**: Predicts continuous values.

## 2. Building a Decision Tree
- **Steps**:
  1. Select the best attribute for splitting the data.
  2. Split the dataset into subsets based on the selected attribute.
  3. Repeat the process until a stopping condition is met.

### Criteria for Stopping:
- Maximum depth of the tree.
- Minimum number of samples per leaf.
- No further improvement in splits.

### Gini Impurity
- Gini Impurity is a metric used to evaluate the quality of a split in the dataset.
- It measures the likelihood of incorrect classification of a randomly chosen element.

### Example in Python:
```python
# Example: Using Gini Impurity as the criterion
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy with Gini: {accuracy_score(y_test, y_pred):.2f}")
```

### Entropy and Information Gain
- Entropy measures the impurity in the dataset, and Information Gain is used to determine the best attribute to split.

### Example in Python:
```python
# Calculating Entropy and Information Gain
import numpy as np

def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(parent_entropy, left_child, right_child):
    total_samples = len(left_child) + len(right_child)
    left_entropy = calculate_entropy(left_child)
    right_entropy = calculate_entropy(right_child)

    weighted_entropy = (len(left_child) / total_samples) * left_entropy +                        (len(right_child) / total_samples) * right_entropy
    return parent_entropy - weighted_entropy

# Example usage
parent_entropy = calculate_entropy(y_train)
left_child, right_child = y_train[:30], y_train[30:]
info_gain = information_gain(parent_entropy, left_child, right_child)
print(f"Information Gain: {info_gain:.2f}")
```

## 3. Pruning Techniques
- **Pre-pruning**: Stops tree growth early by imposing constraints.
- **Post-pruning**: Trims the tree after it is built to remove unnecessary nodes.

## 4. Advantages and Disadvantages
### Advantages:
- Easy to interpret and visualize.
- Handles both categorical and continuous data.

### Disadvantages:
- Prone to overfitting.
- Sensitive to small changes in data.

## 5. Implementation
Here is a Python implementation using Scikit-learn:

### Step 1: Import Necessary Libraries
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
```

### Step 2: Load the Dataset
```python
# Loading the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
```

### Step 3: Split the Data into Training and Testing Sets
```python
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 4: Create and Train the Decision Tree Classifier
```python
# Creating a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)

# Training the model
clf.fit(X_train, y_train)
```

### Step 5: Make Predictions and Evaluate the Model
```python
# Making predictions
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Step 6: Visualize the Decision Tree
```python
# Visualizing the Decision Tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

## 6. Evaluation Metrics
- **Classification Metrics**:
  - Accuracy, Precision, Recall, F1-score.
- **Regression Metrics**:
  - RMSE, MAE, R-squared.

## 7. Real-world Applications
- Fraud detection.
- Customer segmentation.
- Predicting housing prices.

## 8. Optimizations and Enhancements
- **Ensemble Methods**:
  - Random Forest.
  - Gradient Boosting.
- **Hyperparameter Tuning**:
  - Grid Search.
  - Randomized Search.

---

### For more details, explore:
- [Scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Visualizing Decision Trees](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html)
