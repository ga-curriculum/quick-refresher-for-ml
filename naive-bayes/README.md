<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">Naive Bayes</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe naive bayes algorithms used for supervised machine learning.


## Naive Bayes
Naive Bayes is a family of simple yet powerful probabilistic algorithms based on Bayes' Theorem with the assumption of independence between features. Despite its simplicity, Naive Bayes performs exceptionally well for tasks like text classification, spam filtering, and sentiment analysis.

## Key Concepts
-   **Bayes' Theorem**: Calculates the probability of an event based on prior knowledge of related events.
-   **Feature Independence Assumption**: Assumes all features contribute independently to the outcome, which rarely holds but still delivers strong results in practice.
-   **Prior and Likelihood**:
    -   **Prior**: The initial probability of each class based on training data.
    -   **Likelihood**: The probability of the data point given a class.

## Core Assumptions
-   Assumes that all features are conditionally independent given the class label.
-   Assumes sufficient training data is available to estimate probabilities reliably.
-   Performs best when features have a strong individual relationship with the target variable.

### Types of Naive Bayes
- **Gaussian Naive Bayes**: Used for continuous features assumed to follow a normal distribution. For example, predicting customer satisfaction scores based on numerical feedback.
- **Multinomial Naive Bayes**: Suitable for discrete data like word counts, commonly used in text classification. For example, categorizing product reviews into sentiment categories (positive, neutral, negative).
- **Bernoulli Naive Bayes**: Designed for binary or boolean feature vectors.For example, identifying whether a user review contains spam-related keywords (yes/no).

## How Naive Bayes Works

-   **Training Phase**:
    1.  Calculate the prior probabilities of each class based on the training dataset.
    2.  Compute the likelihood of each feature given each class.
    3.  Store these probabilities for prediction.
-   **Prediction Phase**:
    1.  For a new data point, compute the posterior probability for each class using the stored priors and likelihoods.
    2.  Assign the class with the highest posterior probability.

## Demo: Probabilistic Classification
This model predicts a class based on probability distributions of the features.
```python
from sklearn.naive_bayes import GaussianNB

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X, y)

# Make a prediction
print("Prediction:", model.predict([[30, 50000]])[0])
```

## Applications
-   In text processing, spam detection, and sentiment analysis.
-   In e-commerce, product categorization and fraud detection.
-   In marketing, predicting customer churn and segmenting users for targeted campaigns.
-   In healthcare, diagnosing diseases based on symptoms.

## Advantages
-   Simple and fast to implement and execute.
-   Effective for high-dimensional data, such as text classification.
-   Robust to irrelevant features, as they minimally impact predictions.
-   Provides probabilistic output, offering a measure of certainty in predictions.

## Limitations
-   Relies on the unrealistic assumption of feature independence.
-   Sensitive to zero probabilities for unseen feature values.
-   Struggles with datasets where features are highly correlated.
-   May produce probabilities that are not well-calibrated.

## Common Challenges with Naive Bayes
-   **Strong Independence Assumption**: This assumption may not hold in real-world datasets, reducing performance in some cases.
-   **Zero Frequency Problem**: Feature values not observed during training lead to zero probability; this can be addressed with Laplace Smoothing.
-   **Limited to Linearly Separable Data**: Performs poorly when classes are not linearly separable.
-   **Misleading Probabilities**: Outputs are not calibrated and may not reflect true confidence levels.

## 🗣️ Discussion Activity: Naive Bayes
As a class or in small groups via breakout rooms, discuss the following questions. Be prepared to share your ideas!
1.  How could ShopSmart use Naive Bayes to analyze customer reviews or detect fraudulent activity?
2.  What limitations might arise when applying Naive Bayes to product categorization?