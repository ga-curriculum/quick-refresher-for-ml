# Base Vocabulary with Code Foundation for Supervised Learning

Welcome to the **Base Vocabulary with Code Foundation for Supervised Learning** repository. This document serves as a quick reference guide to foundational technical terms used in supervised machine learning. Whether you're a beginner or an advanced practitioner, this vocabulary will help you understand key concepts and enhance your comprehension of ML models and workflows.

## Glossary of Terms

1. **Supervised Learning**: A type of machine learning where the model is trained on labeled data.
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression

   # Example Dataset
   X, y = load_data()

   # Splitting the dataset
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Training a supervised model
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

2. **Feature**: An individual measurable property or characteristic used as input to a model.
   ```python
   # Example: Features in a dataset
   features = dataset[['age', 'income', 'education_level']]
   ```

3. **Label**: The output or target variable in supervised learning, used to train the model.
   ```python
   # Example: Label in a dataset
   labels = dataset['target_variable']
   ```

4. **Training Data**: The dataset used to train a machine learning model.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   ```

5. **Test Data**: A separate dataset used to evaluate the performance of a trained model.
   ```python
   # Example: Using test data to evaluate a model
   accuracy = model.score(X_test, y_test)
   ```

6. **Validation Data**: A subset of data used to fine-tune model hyperparameters during training.
   ```python
   from sklearn.model_selection import GridSearchCV

   # Example: Hyperparameter tuning with validation data
   param_grid = {'C': [0.1, 1, 10]}
   grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
   grid_search.fit(X_train, y_train)
   ```

7. **Regularization**: A technique used to prevent overfitting by adding a penalty term to the loss function.
   ```python
   # Example: L2 Regularization in Logistic Regression
   model = LogisticRegression(penalty='l2', C=0.1)
   ```

8. **Overfitting**: A scenario where a model performs well on training data but poorly on unseen data due to excessive complexity.
   ```python
   # Example: Regularization to prevent overfitting
   model = LogisticRegression(C=0.1)  # Adding regularization
   model.fit(X_train, y_train)
   ```

9. **Underfitting**: A scenario where a model is too simple to capture the underlying patterns in the data.
   ```python
   # Example: Increasing model complexity to address underfitting
   model = LogisticRegression(max_iter=500)
   model.fit(X_train, y_train)
   ```

10. **Model**: A mathematical representation of a real-world process that is trained on data to make predictions.
   ```python
   # Example: Logistic Regression Model
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

11. **Algorithm**: A step-by-step procedure used to train a machine learning model.
    ```python
    # Example: Training a Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    ```

12. **Loss Function**: A mathematical function that measures the error between predicted and actual labels.
    ```python
    # Example: Mean Squared Error Loss Function
    from sklearn.metrics import mean_squared_error
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    ```

13. **Gradient Descent**: An optimization algorithm used to minimize the loss function by iteratively updating model parameters.
    ```python
    # Example: Gradient Descent for Linear Regression
    from sklearn.linear_model import SGDRegressor
    model = SGDRegressor()
    model.fit(X_train, y_train)
    ```

14. **Hyperparameter**: A parameter whose value is set before the training process begins, such as learning rate or batch size.
    ```python
    # Example: Setting hyperparameters in Logistic Regression
    model = LogisticRegression(C=0.1, max_iter=500)
    ```

15. **Cross-Validation**: A technique for assessing model performance by splitting data into multiple training and testing subsets.
    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation Scores:", scores)
    ```

16. **Confusion Matrix**: A table used to evaluate the performance of a classification model by summarizing true positives, true negatives, false positives, and false negatives.
    ```python
    from sklearn.metrics import confusion_matrix
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    ```

17. **Precision**: The ratio of true positive predictions to the total number of positive predictions made by the model.
    ```python
    from sklearn.metrics import precision_score
    precision = precision_score(y_test, predictions)
    ```

18. **Recall**: The ratio of true positive predictions to the total number of actual positive instances.
    ```python
    from sklearn.metrics import recall_score
    recall = recall_score(y_test, predictions)
    ```

19. **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of model performance.
    ```python
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, predictions)
    ```

20. **ROC Curve**: A graphical representation of a classifier's performance across different thresholds, plotting the true positive rate against the false positive rate.
    ```python
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    ```
