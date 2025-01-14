
#  Support Vector Machine (SVM)

## Overview
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification, regression, and outlier detection tasks. It is known for its effectiveness in high-dimensional spaces and its ability to handle non-linear decision boundaries using kernel functions.

---

## What is SVM?
- SVM aims to find the optimal hyperplane that separates data points of different classes with the maximum margin.
- For non-linearly separable data, SVM uses kernel tricks to map data into higher dimensions where a linear separator can be applied.

---

## Key Features
1. **Maximum Margin**: Ensures robustness and generalization by maximizing the margin between classes.
2. **Kernel Trick**: Allows SVM to handle non-linear decision boundaries effectively.
3. **Support Vectors**: Relies only on the critical data points (support vectors) to define the decision boundary.
4. **Versatility**: Applicable to both linear and non-linear problems.

---

## How It Works
1. **Hyperplane**: Separates data points into distinct classes.
2. **Margin**: Distance between the hyperplane and the closest data points from each class.
3. **Support Vectors**: Data points that influence the position and orientation of the hyperplane.
4. **Kernel Functions**:
   - Linear: For linearly separable data.
   - Polynomial: For complex, polynomial decision boundaries.
   - RBF (Gaussian): For highly non-linear decision boundaries.
   - Sigmoid: For specific applications like neural networks.

---

## Advantages
- Effective in high-dimensional spaces.
- Works well for both linear and non-linear problems.
- Robust to overfitting, especially in high-dimensional datasets.

---

## Disadvantages
- Computationally intensive for large datasets.
- Performance depends on the proper choice of kernel and parameters.
- Sensitive to noisy data and overlapping classes.

---

## Applications
1. Text classification (e.g., spam detection).
2. Image classification.
3. Medical diagnosis.
4. Bioinformatics (e.g., protein classification).

---


## Hyperparameters
1. **C**: Regularization parameter; balances margin size and misclassification.
2. **kernel**: Defines the type of hyperplane (e.g., linear, RBF, polynomial).
3. **gamma**: Kernel coefficient for non-linear hyperplanes.
4. **degree**: Degree of the polynomial kernel (if used).

---

## Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1 Score.

---

## References
1. "A Tutorial on Support Vector Machines for Pattern Recognition", *Data Mining and Knowledge Discovery*, 1998. [Link](https://link.springer.com/article/10.1023/A:1009715923555)
2. "Support Vector Machines Explained", *arXiv preprint*, 2003. [Link](https://arxiv.org/abs/0907.2878)

This lesson provides a comprehensive guide to understanding and implementing Support Vector Machines in machine learning tasks.
