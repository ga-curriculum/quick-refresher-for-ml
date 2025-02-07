<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">Supervised Machine Learning</span>
</h1>

**Learning objective:** By the end of this lesson, students will be able to describe supervised machine learning approach and list the major algorithms used for supervised machine learning.

## Supervised Machine Learning

Supervised Machine Learning involves building models that learn from labeled datasets to make predictions or decisions. This repository provides an introduction to supervised learning with examples of both classification and regression tasks.

Supervised learning is a type of machine learning where the model is trained on labeled data. In this context, "labeled data" means that each training example is paired with an output label. The model uses this data to learn the mapping function from inputs to outputs, enabling it to make predictions on new, unseen data.



## Types of Supervised Learning

Supervised Machine Learning is a foundational approach in artificial intelligence, where algorithms are trained to map input data to output labels using a labeled dataset. The process involves identifying patterns and relationships within the data to make predictions or decisions. There are two primary types of tasks in supervised learning:

1. **Classification**: Involves predicting categorical labels. Examples include spam detection, image recognition, and disease diagnosis.

2. **Regression**: Involves predicting continuous values. Examples include house price prediction, stock price forecasting, and weather prediction.

Supervised learning is widely used due to its effectiveness and reliability in solving real-world problems such as **fraud detection, customer segmentation, and predictive maintenance**.

## Major Algorithms in Supervised Learning

- **Linear Regression:** Predicts a continuous numerical value based on the relationship between input and output variables.  
- **Logistic Regression:** Classifies data into binary categories using a sigmoid function to estimate probabilities.  
- **Decision Trees:** Uses a tree-like model of decisions and their possible consequences to classify or predict outcomes.  
- **Random Forest:** An ensemble of multiple decision trees that improves accuracy and reduces overfitting.  
- **Support Vector Machine:** Finds the optimal hyperplane to separate data into distinct classes with maximum margin.  
- **K-Nearest Neighbors:** Classifies data points based on the majority class of their k-nearest neighbors.  
- **Naive Bayes:** A probabilistic classifier based on Bayes’ theorem, assuming feature independence.

## Let's look at each in more detail with some examples:

1\. [Linear Regression](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/linear-regression/README.md) 📈
------------------------

**Core Concept:**\
Linear regression predicts continuous outcomes by modeling the relationship between one or more independent variables (features) and a target variable. The model finds the best-fit line (or hyperplane) by minimizing the error between predicted and actual values.

**ShopSmart Example:**\
Imagine ShopSmart wants to forecast monthly sales revenue. By analyzing past data on advertising spend, website traffic, and seasonal trends, linear regression can predict future revenue---helping the team plan budgets and optimize marketing efforts.

* * * * *

2\. [Logistic Regression](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/logistic-regression/README.md) 🔢
--------------------------

**Core Concept:**\
Logistic regression is designed for classification. Instead of predicting a continuous value, it estimates the probability that a given input belongs to a particular category using the logistic (sigmoid) function. A threshold (typically 0.5) is applied to decide the final class.

**ShopSmart Example:**\
Consider predicting whether a website visitor will complete a purchase. By examining features such as time on site, number of pages viewed, and past purchase behavior, logistic regression outputs the probability of conversion---enabling ShopSmart to target customers with personalized incentives if they're on the fence.

* * * * *

3\. [Decision Trees](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/decision-trees/README.md) 🌳
---------------------

**Core Concept:**\
Decision trees use a tree-like structure to make decisions. The data is split at each node based on a feature value, leading to branches that culminate in leaf nodes where predictions are made. They're especially useful for their interpretability and straightforward decision rules.

**ShopSmart Example:**\
ShopSmart could build a decision tree to determine which customers should receive a special promotion. The tree might split the data based on factors like purchase history, average order value, and visit frequency---resulting in clear rules such as:\
*"If a customer has made more than three purchases and spends over $100 per order, then target them for premium offers."*

* * * * *

4\. [Random Forest](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/random-forest/README.md) 🌲
--------------------

**Core Concept:**\
Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs to improve predictive accuracy and control overfitting. For classification, it uses majority voting; for regression, it averages the predictions of its trees.

**ShopSmart Example:**\
Imagine ShopSmart needs a robust forecast of product demand. By analyzing features like historical sales, seasonality, and product reviews, a Random Forest model aggregates the predictions from multiple trees to provide a more reliable demand forecast---helping with inventory planning and supply chain management.

* * * * *

5\. [Support Vector Machines (SVM)](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/support-vector-machine/README.md)**** 💡
------------------------------------

**Core Concept:**\
SVMs are powerful classifiers that find the optimal hyperplane which best separates data into distinct classes with the maximum margin. They can handle non-linear relationships through kernel functions that transform the data into higher dimensions.

**ShopSmart Example:**\
For fraud detection, ShopSmart can employ an SVM to classify transactions as fraudulent or legitimate. By analyzing transaction features---such as purchase amount, time of purchase, and user behavior---the SVM isolates the "support vectors" (critical cases) to build a robust boundary between normal and suspicious activities.

* * * * *

6\. [K-Nearest Neighbors (KNN)](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/k-nearest-neighbors/README.md) 👥
--------------------------------

**Core Concept:**\
KNN is an instance-based algorithm that predicts the outcome for a new data point by identifying the 'k' closest examples in the training dataset. For classification, it assigns the most frequent class among the neighbors; for regression, it averages their values.

**ShopSmart Example:**\
Imagine ShopSmart wants to personalize product recommendations. KNN can find customers with similar browsing and purchasing histories and recommend products that those similar customers have loved. It's a straightforward, "find your neighbors" approach that drives more tailored suggestions.

* * * * *

7\. [Naive Bayes](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/naive-bayes/README.md) 🧮
------------------

**Core Concept:**\
Naive Bayes uses Bayes' Theorem to calculate the probability of each class based on the input features---assuming that each feature contributes independently. Despite this "naive" assumption, it's incredibly efficient, especially for text data.

**ShopSmart Example:**\
ShopSmart can apply Naive Bayes to automatically categorize customer reviews as positive or negative. By analyzing word frequencies and sentiment indicators in the reviews, the model helps quickly gauge customer satisfaction and highlight areas for improvement.

* * * * *

Conclusion 🎉
-------------

Each supervised learning algorithm brings something unique to the table:

| **Algorithm** | **Type** | **Primary Use** |
| --- | --- | --- |
| **Linear Regression** | Regression | Predicting continuous values | 
| **Logistic Regression** | Classification | Binary/multi-class classification |
| **Decision Trees** | Classification/Regression | Rule-based predictions | 
| **Random Forest** | Ensemble (Decision Trees) | Improved prediction & reduced overfitting | 
| **SVM** | Classification/Regression | Complex classification tasks | 
| **KNN** | Instance-based | Classification/Regression | 
| **Naive Bayes** | Probabilistic | Text classification, sentiment analysis |

