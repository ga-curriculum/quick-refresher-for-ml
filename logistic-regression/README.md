<h1>
  <span class="headline">Quick Refresher for Ml</span>
  <span class="subhead">Logistic Regression</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe logistic regression algorithms used for supervised machine learning.

## An Introduction to Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification tasks. It predicts the probability of an event occurring and uses this probability to classify data into discrete categories. Despite its name, Logistic Regression is not a regression algorithm in the traditional sense but a classification technique.

## Key Concepts
-   Logistic Regression uses the sigmoid function to map any real-valued input into a range between 0 and 1, making it suitable for probability estimation. The sigmoid function ensures predictions remain within valid probability limits.
-   Logistic Regression transforms the linear combination of input features into a probability score. This score is then used to determine class membership based on a threshold (commonly 0.5).
-   Logistic Regression is primarily used for binary classification problems, where the dependent variable has two classes (e.g., 0 and 1). The output is the probability of belonging to the positive class.
-   The decision boundary is a threshold that separates classes. Inputs with probabilities above the threshold are classified into one class (e.g., 1), while those below it belong to the other class.
-   Logistic Regression assumes that the relationship between the independent variables and the log odds of the dependent variable is linear.

## Core Assumptions
-   The dependent variable is binary or categorical.
-   Observations are independent, with no dependencies between data points.
-   Predictors are linearly related to the log of odds.
-   Predictors should not exhibit high multicollinearity to avoid unreliable coefficient estimation.
-   Outliers should be minimal as they can distort the decision boundary.

## Types of Logistic Regression
- **Binary Logistic Regression**: Used when the dependent variable has two possible outcomes (yes/no, 0/1).
- **Multinomial Logistic Regression**: Handles classification problems with three or more categories without any ordinal relationship. (like predicting one of several product categories).
- **Ordinal Logistic Regression**: Used for dependent variables with ordered categories, such as ratings (like low, medium, high).
- **Regularized Logistic Regression**: Incorporates L1 (Lasso) and L2 (Ridge) penalties to prevent overfitting and improve generalization.
  -   L1 regularization (Lasso) performs feature selection by shrinking some coefficients to zero.
  -   L2 regularization (Ridge) reduces overfitting by penalizing large coefficients.
  -   Elastic Net combines L1 and L2 regularization for a balanced approach.

## Demo: Binary Classification
This model predicts a binary class (0 or 1) based on input features.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample dataset: Age vs. Loan Approval (1 = Approved, 0 = Denied)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Predict class for a new data point
print("Predicted class for X=2.5:", model.predict([[2.5]])[0])
```

## Applications
-   Predicting customer churn in subscription-based services.
-   Classifying emails as spam or non-spam.
-   Diagnosing diseases based on medical test results.
-   Estimating the likelihood of loan defaults in finance.
-   Identifying fraudulent transactions in e-commerce.

## Advantages
-   Logistic Regression is simple and easy to implement, making it an excellent choice for baseline classification tasks.
-   It provides interpretable results by offering insights into the relationship between predictors and the likelihood of outcomes.
-   It is computationally efficient, making it suitable for large datasets.
-   Logistic Regression outputs probabilities, allowing for more nuanced decision-making beyond binary classifications.
-   It is versatile and can handle both binary and multiclass classification problems with extensions like multinomial logistic regression.

## Limitations
-   Logistic Regression assumes a linear relationship between predictors and the log odds, which may not hold for complex or non-linear data.
-   It is sensitive to outliers, which can significantly affect the decision boundary and model coefficients.
-   Logistic Regression struggles with imbalanced datasets, as the majority class can dominate predictions without resampling or weighting techniques.
-   It requires careful feature engineering and preprocessing, as irrelevant or noisy predictors can reduce model performance.

## Common Challenges with Logistic Regression
-   **Imbalanced Data**: Class imbalance can skew predictions toward the majority class.
-   **Multicollinearity**: Correlated predictors can distort coefficient estimates and reduce model reliability.
-   **Outliers**: Extreme values can disproportionately influence model performance.
-   **Threshold Selection**: Determining the optimal decision threshold is crucial for balancing precision and recall.

## 🗣️ Discussion Activity: Logistic Regression

As a class or in small groups via breakout rooms, discuss the following questions. Be prepared to share your ideas!
1.  How could ShopSmart use logistic regression to predict customer behavior, such as purchase likelihood or churn?
2.  What types of logistic regression might be most useful for different use cases at ShopSmart (e.g., binary for purchase prediction, multinomial for product category preferences)?