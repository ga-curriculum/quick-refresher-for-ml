<h1>
  <span class="headline">[Quick Refresher to Machine Learning]</span>
  <span class="subhead">Supervised Machine Learning Microlesson 01</span>
</h1>


**Learning objective:**

1. Refresh understanding of supervised learning, focusing on its key characteristics and applications in classification and regression tasks.

2. Refresh understanding of unsupervised learning, focusing on clustering and dimensionality reduction techniques.

3. Refresh understanding of reinforcement learning, with an emphasis on policy-based and reward-driven decision-making systems.

4. Relate refreshed ML concepts to real-world use cases in AI-driven solutions, setting the stage for advanced architectures like transformers and GANs.

5. Build a strong base for exploring advanced AI topics in subsequent modules.

# Supervised Machine Learning

Supervised Machine Learning involves building models that learn from labeled datasets to make predictions or decisions. This repository provides an introduction to supervised learning with examples of both classification and regression tasks.

## Table of Contents

1. [Introduction](#introduction)
2. [Base vocabulary and Base Code for Supervised ML](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/microlesson-01/README_Base_Vocabulary_Supervised_Learning.md)
3. [Types of Supervised Machine Learning](#summary-of-supervised-machine-learning)
4. [Major Supervised Machine Learning Algorithms](#major-supervised-machine-learning-algorithms)
5. [Prerequisites](#prerequisites)
6. [Dataset Resources](#dataset-resources)


---

## Introduction

Supervised learning is a type of machine learning where the model is trained on labeled data. In this context, "labeled data" means that each training example is paired with an output label. The model uses this data to learn the mapping function from inputs to outputs, enabling it to make predictions on new, unseen data.

---
## Two types of Sypervised Machine Learning 

Supervised Machine Learning is a foundational approach in artificial intelligence, where algorithms are trained to map input data to output labels using a labeled dataset. The process involves identifying patterns and relationships within the data to make predictions or decisions. There are two primary types of tasks in supervised learning:

1. **Classification**: Involves predicting categorical labels. Examples include spam detection, image recognition, and disease diagnosis.
2. **Regression**: Involves predicting continuous values. Examples include house price prediction, stock price forecasting, and weather prediction.

Supervised learning is widely used due to its effectiveness and reliability in solving real-world problems such as fraud detection, customer segmentation, and predictive maintenance.

# Comparative Analysis of Supervised, Unsupervised, and Reinforcement Machine Learning

| **Category**           | **Supervised Learning**                                   | **Unsupervised Learning**                                | **Reinforcement Learning**                            |
|------------------------|---------------------------------------------------------|--------------------------------------------------------|-----------------------------------------------------|
| **Definition**         | Predicts outcomes using labeled data.                   | Identifies patterns in unlabeled data.                 | Trains agents to maximize rewards through actions.   |
| **Objective**          | Minimize prediction error and generalize to new data.  | Discover hidden patterns or latent structures.         | Maximize long-term cumulative rewards.              |
| **Data Requirements**  | Requires large, labeled datasets.                      | Operates on raw, unlabeled data.                       | Interacts with an environment to generate data.      |
| **Algorithms**         | Random Forest, Gradient Boosting, Transformers.        | DBSCAN, Variational Autoencoders (VAEs), GANs.         | Proximal Policy Optimization (PPO), AlphaZero.      |
| **Applications**       | Personalized recommendations, medical diagnosis.       | Genomics, customer segmentation, fraud detection.      | Self-driving cars, financial portfolio management.   |
| **Strengths**          | High precision with labeled data, interpretable models.| Identifies unknown relationships, reduced preprocessing.| Effective in dynamic, sequential environments.       |
| **Limitations**        | Labeled data dependency, scalability issues.            | Results can be vague; limited real-world usage.        | High computational cost; environment-sensitive.      |
| **Scalability**        | Scales well with distributed training (e.g., GPUs).     | Limited by algorithm complexity (e.g., clustering).    | Resource-heavy; often requires simulation setups.    |
| **Learning Type**      | Predictive (maps inputs to outputs).                    | Descriptive (finds structure in data).                 | Prescriptive (takes actions for optimal results).    |
| **Recent Advances**    | Pre-trained models like BERT, GPT for NLP.             | Advances in generative models like DALL-E, StyleGAN.   | Real-world RL in robotics and autonomous systems.    |
| **Interpretability**   | High with simpler models, challenging for deep models. | Often low; results require domain knowledge to analyze.| Policy outcomes interpretable; underlying process opaque. |

This advanced tabular format highlights key distinctions and innovations, offering deeper insights into the three paradigms of machine learning.


---

## Major Supervised Machine Learning Algorithms

This section describes 10 major supervised machine learning algorithms, along with their key characteristics and applications:

1. **[Linear Regression](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/microlesson-01/Linear_Regression_README.md)**
   - Predicts continuous values by establishing a linear relationship between the input features and the target variable.
   - **Applications**: House price prediction, stock price forecasting.

2. **[Logistic Regression](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/microlesson-01/logistic_regression_readme1.md)**
   - Used for binary and multi-class classification tasks. Estimates probabilities using a logistic function.
   - **Applications**: Spam detection, credit risk analysis.

3. **[Decision Tree](https://git.generalassemb.ly/modular-curriculum-all-courses/quick-refresher-for-ml/blob/main/microlesson-01/Decision_Tree_README.md)**
   - A tree-based model that splits the data into subsets based on feature conditions. Works for both classification and regression.
   - **Applications**: Customer segmentation, fraud detection.

4. **Random Forest**
   - An ensemble method that builds multiple decision trees and combines their outputs to improve accuracy.
   - **Applications**: Loan approval, product recommendation.

5. **Support Vector Machine (SVM)**
   - Classifies data by finding the hyperplane that best separates classes. Also used for regression tasks.
   - **Applications**: Image recognition, text categorization.

6. **K-Nearest Neighbors (KNN)**
   - A non-parametric algorithm that classifies or predicts based on the closest training examples in the feature space.
   - **Applications**: Handwriting detection, recommendation systems.

7. **Naive Bayes**
   - Based on Bayes' theorem, assumes independence between features. Commonly used for classification.
   - **Applications**: Sentiment analysis, email classification.

8. **Gradient Boosting Machines (GBM)**
   - An ensemble method that builds models sequentially to correct errors from previous models.
   - **Applications**: Web search ranking, healthcare prediction.

9. **XGBoost**
   - A highly efficient implementation of gradient boosting, known for its speed and accuracy.
   - **Applications**: Competition-winning solutions in Kaggle, predictive maintenance.


---

## Prerequisites

- Python 3.7+
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

---

## Dataset Resources

Here are some useful resources for datasets commonly used in supervised machine learning:

1. **Iris Dataset**: Built-in dataset from `sklearn` for classification tasks.
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

2. **Boston Housing Dataset**: Built-in dataset from `sklearn` for regression tasks.
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing)

3. **MNIST Dataset**: Widely used for digit classification tasks.
   - Source: [MNIST Database](http://yann.lecun.com/exdb/mnist/)

4. **CIFAR-10 Dataset**: Used for object recognition and image classification.
   - Source: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

5. **Kaggle Datasets**: A rich source of datasets for various machine learning tasks.
   - Source: [Kaggle Datasets](https://www.kaggle.com/datasets)

6. **Google Dataset Search**: Search for datasets across domains.
   - Source: [Google Dataset Search](https://datasetsearch.research.google.com/)

7. **Data.gov**: Open government data for a variety of domains.
   - Source: [Data.gov](https://www.data.gov/)

8. **UCI Machine Learning Repository**: A comprehensive collection of datasets for different machine learning tasks.
   - Source: [UCI Repository](https://archive.ics.uci.edu/ml/index.php)

