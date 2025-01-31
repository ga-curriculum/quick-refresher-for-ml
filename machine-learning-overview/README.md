<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">Machine Learning Overview</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to define machine learning and explain the three types learning. 


## Introduction to Machine Learning (ML)

Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to automatically learn from data and improve their performance over time without being explicitly programmed. It focuses on creating algorithms that can:

- Recognize patterns
- Make decisions
- Solve problems based on input data

## Types of Machine Learning

| **Category**           | **Supervised Learning**                                   | **Unsupervised Learning**                                | **Reinforcement Learning**                            |
|------------------------|---------------------------------------------------------|--------------------------------------------------------|-----------------------------------------------------|
| **Definition**         | Predicts outcomes using labeled data.                   | Identifies patterns in unlabeled data.                 | Trains agents to maximize rewards through actions.   |
| **Objective**          | Minimize prediction error and generalize to new data.  | Discover hidden patterns or latent structures.         | Maximize long-term cumulative rewards.              |
| **Data Requirements**  | Requires large, labeled datasets.                      | Operates on raw, unlabeled data.                       | Interacts with an environment to generate data.      |
| **Applications**       | Personalized recommendations, medical diagnosis.       | Genomics, customer segmentation, fraud detection.      | Self-driving cars, financial portfolio management.   |
| **Strengths**          | High precision with labeled data, interpretable models.| Identifies unknown relationships, reduced preprocessing.| Effective in dynamic, sequential environments.       |
| **Limitations**        | Labeled data dependency, scalability issues.            | Results can be vague; limited real-world usage.        | High computational cost; environment-sensitive.      |
| **Scalability**        | Scales well with distributed training (e.g., GPUs).     | Limited by algorithm complexity (e.g., clustering).    | Resource-heavy; often requires simulation setups.    |
| **Learning Type**      | Predictive (maps inputs to outputs).                    | Descriptive (finds structure in data).                 | Prescriptive (takes actions for optimal results).    |
| **Interpretability**   | High with simpler models, challenging for deep models. | Often low; results require domain knowledge to analyze.| Policy outcomes interpretable; underlying process opaque. |

## Machine Learning Types and Their Real-World Applications

1. **Supervised Machine Learning**  
   - Used for tasks with labeled data like classification (spam detection) and regression (price prediction).  
   - Powers applications in fraud detection, recommendation systems, and predictive analytics.  
   - Widely used in industries like finance, healthcare, and e-commerce.  

2. **Unsupervised Machine Learning**  
   - Works with unlabeled data for clustering (customer segmentation) and dimensionality reduction (PCA).  
   - Applied in anomaly detection, market basket analysis, and data exploration.  
   - Used in marketing, cybersecurity, and biological research.  

3. **Reinforcement Machine Learning**  
   - Focuses on learning optimal actions through rewards in dynamic environments.  
   - Powers robotics, game AI, and autonomous vehicles.  
   - Effective in supply chain optimization and personalized recommendations.  


## Introduction to ShopSmart
Throughout this session, as we encounter various machine learning concepts, we are going to use ShopSmart as a sample application to test our understanding of the concepts that we are going to learn.


ShopSmart is a platform designed to make your online shopping smarter and more efficient. It offers tools and insights that help you save time, find the best deals, and keep track of your purchases.

### Key Features
- **Personalized Recommendations**: Discover items tailored to your interests.  
- **Price Alerts**: Set alerts to catch price drops and get the best deals.  
- **Spending Insights**: Stay on top of your shopping habits and budgeting.  
- **Product Reviews**: Read and trust verified product reviews to make informed decisions.

Throughout this session, we’ll introduce each of these features step-by-step, and we’ll revisit them as you progress. By the end, you’ll have a deep understanding of how to get the most out of ShopSmart. 

Let’s get started!
