<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">Decision Trees</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe decision tree algorithms used for supervised machine learning.

## An Introduction to Decision Trees
Decision Trees are supervised machine learning algorithms used for both classification and regression tasks. They build a hierarchical tree structure by recursively splitting the dataset into subsets based on feature values. Decision Trees are interpretable and flexible, making them popular for a wide range of applications.

## Key Concepts
-   A Decision Tree consists of nodes that split data based on feature values, branches representing decision rules, and leaf nodes containing predictions.
-   The splitting process continues until a stopping condition is met, such as reaching a maximum tree depth or achieving a minimum number of samples in leaf nodes.
-   Decision Trees can handle both numerical and categorical data.
-   They are capable of capturing non-linear relationships and do not require feature scaling.

## Core Assumptions
-   Assumes that splits in the data can effectively separate outcomes based on the chosen features.
-   Assumes that features used for splitting have meaningful relationships with the target variable
-   Decision Trees are prone to overfitting, which requires pruning or regularization techniques to address.

## How Decision Trees Work
-   **Root Node**: Represents the entire dataset and begins the recursive splitting process.
-   **Decision Nodes**: Split the data based on feature thresholds or categories.
-   **Leaf Nodes**: Contain the final predictions or outcomes for the data subsets.
-   **Splitting Criteria**: Measures like Gini Impurity, Entropy, or Reduction in Variance are used to evaluate the quality of splits.

## Types of Decision Trees
- **Classification Trees**: Predict discrete outcomes or categories (like yes/no).
- **Regression Trees**: Predict continuous outcomes.
- **Hybrid Trees**: Handle mixed data types, predicting both categorical and continuous outcomes in a single model.

## Steps to Build a Decision Tree
1.  Define the objective (classification or regression).
2.  Select a splitting criterion (e.g., Gini Impurity, Entropy).
3.  Split the dataset iteratively by choosing features that maximize the improvement in the chosen criterion.
4.  Stop splitting based on predefined conditions (e.g., maximum depth, minimum samples per node).

## Demo: Loan Approval Prediction
This decision tree model predicts whether a person gets loan approval based on age and income.

```python
from sklearn.tree import DecisionTreeClassifier

# Dataset: Age & Income vs. Loan Approval
X = np.array([[25, 40000], [35, 60000], [45, 80000], [20, 20000]])
y = np.array([0, 1, 1, 0])

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Make a prediction
print("Prediction for age 30, income 50000:", model.predict([[30, 50000]])[0])
```

## Applications
-   In healthcare, diagnosing diseases or predicting patient outcomes.
-   In finance, credit scoring, risk assessment, and fraud detection.
-   In marketing, segmenting customers and predicting purchasing behavior.
-   In operations, optimizing processes and supply chain management.

### Advantages
-   Highly interpretable and transparent for decision-making.
-   Handles both numerical and categorical features.
-   Robust to missing values and outliers.
-   Captures non-linear relationships effectively.

## Limitations
-   Prone to overfitting without pruning or regularization.
-   Sensitive to noise and small data variations.
-   Can struggle with imbalanced datasets unless adjustments are made. 

## Common Challenges with Decision Trees
-   **Overfitting**: Trees that grow too deep can overfit the training data, reducing generalization performance.
-   **Sensitivity to Small Changes**: Small variations in data can lead to significantly different tree structures.
-   **Imbalanced Data**: Disproportionate class distributions can skew splits toward the majority class.

## 🗣️ Discussion Activity: Decision Trees

As a class or in small groups via breakout rooms, discuss the following questions. Be prepared to share your ideas!
1.  How could ShopSmart use decision trees to improve its marketing strategies?
2.  What types of data and features would be most useful for building an effective decision tree model at ShopSmart?
