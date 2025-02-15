<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">XGBoost</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe XGBoost algorithms used for supervised machine learning.

## An Introduction to XGBoost

XGBoost (eXtreme Gradient Boosting) is a powerful and efficient implementation of gradient boosting machines. It's known for its speed and performance, particularly in structured/tabular data problems. XGBoost builds an ensemble of weak prediction models, typically decision trees, in a sequential manner to create a strong predictive model.

## Key Concepts
- XGBoost uses gradient boosting, where each new model tries to correct the errors of previous models.
- It can handle both regression and classification tasks
- Features built-in regularization to prevent overfitting.
- Efficiently handles missing values.
- Provides parallel and distributed computing capabilities.
- Uses a unique tree pruning approach.

## Core Assumptions
- Features should be numeric or properly encoded categorical variables.
- The relationship between features and target can be modeled through boosted trees.
- The loss function should be differentiable.
- Data quality impacts model performance significantly.

## Types of Problems XGBoost Solves
- **Regression**: Predicting continuous values (e.g., sales forecasting).
- **Binary Classification**: Predicting between two classes (e.g., churn prediction).
- **Multi-class Classification**: Predicting among multiple classes.
- **Ranking**: Ordering items by relevance.

## Demo: Customer Churn Prediction
This example shows how to use XGBoost for binary classification:

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create DMatrix
dtrain = xgb.DMatrix(X, label=y)

# Set parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# Make predictions
predictions_prob = model.predict(xgb.DMatrix(X))
predictions = (predictions_prob > 0.5).astype(int)

# Print performance report
print("Model Performance:")
print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y, predictions))

# Print predictions vs actual values
print("First 10 predictions (probabilities):")
print(predictions_prob[:10])
print("\nFirst 10 actual values:")
print(y[:10])
```

## Applications
- **Finance**: Credit scoring, fraud detection, stock price prediction.
- **Retail**: Customer segmentation, demand forecasting, recommendation systems.
- **Healthcare**: Disease prediction, patient risk assessment.
- **Marketing**: Campaign response prediction, customer lifetime value estimation.
- **Manufacturing**: Quality control, predictive maintenance.

## Advantages
- Superior performance on structured/tabular data.
- Built-in handling of missing values.
- Efficient memory usage and fast training.
- Regularization to prevent overfitting.
- Feature importance scoring.
- Handles both numerical and categorical features.

## Limitations
- Requires careful parameter tuning.
- Can be computationally intensive for very large datasets.
- Less effective for unstructured data (images, text, etc.).
- May overfit on small datasets without proper regularization.
- Black-box nature makes interpretability challenging.

## Common Challenges with XGBoost
- **Parameter Tuning**: Finding optimal hyperparameters requires experimentation.
- **Memory Usage**: Large datasets can strain memory resources.
- **Overfitting**: Need to balance model complexity with regularization.
- **Feature Engineering**: Still requires good feature preparation.

## 🗣️ Discussion Activity: XGBoost Applications

As a class or in small groups via breakout rooms, discuss the following questions. Be prepared to share your ideas!
1. How could ShopSmart use XGBoost to improve its business operations?
2. What specific problems at ShopSmart would be best suited for XGBoost versus other algorithms? 