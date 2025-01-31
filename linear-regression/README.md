<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">Linear Regression</span>
</h1>

**Learning objective:** By the end of this lesson, students will be able to describe linear regression algorithms used for supervised machine learning.

## An Introduction to Linear Regression

Linear Regression is a supervised learning algorithm used to predict continuous outcomes by modeling the relationship between one or more independent variables (features) and a dependent variable (target). It serves as a foundation for many machine learning models and provides insights into the relationships between variables. Using ShopSmart, an e-commerce company, as a case study, we can explore its applications and variations.

## Key Concepts
- Linear Regression predicts the dependent variable as a linear combination of independent variables plus an intercept.
- It assumes a linear relationship between the dependent and independent variables.
- The model works for both simple (single variable) and multiple (multi-variable) regression scenarios.
- In simple linear regression, the relationship is represented as a straight line.
- In multiple linear regression, the relationship is represented as a plane or hyperplane.
- The model finds the best-fit line by minimizing the error between predicted and actual values.
- It identifies the contribution of each independent variable through coefficients (weights).
- Linear Regression is sensitive to outliers, which can distort predictions.

## Core Assumptions
- The relationship between variables is linear.
- Observations are independent of each other.
- The variance of residuals (errors) is constant across all levels of independent variables (homoscedasticity).
- Residuals are normally distributed.
- Independent variables are not highly correlated (no multicollinearity).
 
## Types of Linear Regression
- **Simple Linear Regression**: Models the relationship between one independent variable (feature) and one dependent variable (target).
- **Multiple Linear Regression**: Extends simple linear regression to include multiple independent variables.
- **Polynomial Regression**: Models a non-linear relationship between the independent variable(s) and the dependent variable by incorporating polynomial terms (e.g., squared or cubed variables).
- **Ridge Regression (L2 Regularization)**: Penalizes large coefficients in the regression model to reduce overfitting, especially when features are highly correlated (multicollinearity).
- **Lasso Regression (L1 Regularization)**: Shrinks some coefficients to zero, effectively selecting only the most important features for the model.
- **Elastic Net Regression**: Combines Ridge (L2) and Lasso (L1) regularization to balance feature selection and model robustness.

## Applications
- Predicting recovery time in healthcare based on age and treatment type.
- Forecasting stock prices in finance using historical data and market trends.
- Estimating house prices in real estate based on location, size, and features.
- Analyzing sales trends in marketing based on advertising spend and seasonal data.
- Predicting student performance in education based on study hours and attendance.

## Demo: Predicting House Prices
This code trains a linear regression model to predict house prices based on square footage.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset: Square footage vs. House price
X = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
y = np.array([200000, 250000, 300000, 350000, 400000])

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict house price for 2200 sq ft
predicted_price = model.predict([[2200]])
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

## Advantages
- Simple and interpretable.
- Easy to implement and computationally efficient.
- Provides insights into the relationships between variables.

## Limitations
- Assumes a linear relationship between variables.
- Sensitive to outliers, which can distort results.
- Struggles with multicollinearity, leading to unreliable coefficients.
- Performs poorly on non-linear problems without feature transformation.

## Common Challenges with Linear Regression
- **Outliers** can heavily influence the regression line and distort predictions.
- **Multicollinearity** makes it difficult to determine the true effect of independent variables.
- **Overfitting** occurs when the model performs well on training data but poorly on unseen data.

## 🗣️ Discussion Activity: Linear Regression
As a class or in small groups via breakout rooms, discuss the following questions. Be prepared to share your ideas!
  1. How could ShopSmart use each type of linear regression to analyze and optimize different product features, such as pricing, customer behavior, and inventory management?
  2. Can you think of specific scenarios where each regression type would be most effective?