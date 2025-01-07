
# README: Understanding Logistic Regression

Logistic Regression is one of the fundamental algorithms in machine learning. Despite its name, it is used primarily for classification tasks, not regression. It predicts the probability of an outcome falling into one of two categories (binary classification).

---

## What is Logistic Regression?

Logistic Regression is a statistical method that models the probability of a binary outcome based on one or more input features. The algorithm uses a logistic function (also known as a sigmoid function) to map predicted values to probabilities between 0 and 1.

### Key Features:
- **Output**: Predicts probabilities between 0 and 1.
- **Binary Classification**: Commonly used for tasks like spam detection or disease prediction.
- **Linear Decision Boundary**: Works well when the relationship between input features and the target is linear.

---

## How Logistic Regression Works

1. **Input Features**: Takes the input features (X) and calculates a weighted sum (linear combination):
   
   \[ z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b \]
   
   Where \( w \) represents weights, and \( b \) is the bias term.

2. **Sigmoid Function**: Applies the sigmoid function to map \( z \) into a probability:
   
   \[ P(y=1|X) = \frac{1}{1 + e^{-z}} \]

3. **Thresholding**: Uses a threshold (e.g., 0.5) to classify probabilities as either class 0 or class 1.

---

## Example: Logistic Regression in Python

Below is a simple implementation of Logistic Regression from scratch using Python:

### Step-by-Step Code Example

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Generate a simple dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Input feature (hours studied)
y = (X > 5).astype(int).ravel()  # Output label (pass=1, fail=0)

# Step 2: Add a bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding bias as the first column

# Step 3: Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 4: Implement Logistic Regression
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])  # Initialize weights
        for _ in range(self.epochs):
            z = np.dot(X, self.weights)  # Linear combination
            predictions = sigmoid(z)
            gradient = np.dot(X.T, (predictions - y)) / y.size  # Gradient calculation
            self.weights -= self.learning_rate * gradient  # Update weights

    def predict(self, X):
        z = np.dot(X, self.weights)
        probabilities = sigmoid(z)
        return (probabilities >= 0.5).astype(int)  # Apply threshold

# Step 5: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

---

## Applications of Logistic Regression

1. **Healthcare**: Predicting whether a patient has a disease (yes/no).
2. **Finance**: Classifying loan applications as approved or rejected.
3. **Marketing**: Identifying potential customers for a product.

---

## Advantages
- Easy to implement and interpret.
- Works well for linearly separable data.
- Computationally efficient.

## Limitations
- Struggles with non-linear relationships.
- Sensitive to outliers.

---

## Conclusion

Logistic Regression is a powerful yet simple algorithm for binary classification tasks. By understanding its principles and implementing it from scratch, you can gain deeper insights into machine learning and prepare for more advanced algorithms.

---

Feel free to explore and experiment with the code to solidify your understanding of Logistic Regression!
