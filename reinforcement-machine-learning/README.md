<h1>
  <span class="headline">Quick Refresher for ML</span>
  <span class="subhead">Reinforcement Machine Learning</span>
</h1>

**Learning objective:** By the end of this lesson, you'll be able to describe reinforcement machine learning approach and explain the types of reinforcement machine learning algorithms.

## An Introduction to Reinforcement Machine Learning

Reinforcement Machine Learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards. It is inspired by behavioral psychology and is used in tasks where sequential decision-making is critical.


## Key Features
- **Agent-Environment Interaction:** The agent learns by interacting with the environment.
- **Exploration vs. Exploitation:** The agent explores new actions while exploiting known rewards.
- **Reward Signal:** Guides the agent's learning process based on feedback.
- **Sequential Decision-Making:** Focuses on long-term cumulative rewards.


## Key Terminologies
- **Agent:** The decision-maker.
- **Environment:** The system with which the agent interacts.
- **Action (A):** Choices the agent can make.
- **State (S):** Representation of the environment at a given time.
- **Reward (R):** Feedback signal for the agent's actions.
- **Policy (π):** Strategy that the agent follows to decide actions.
- **Value Function:** Measures the long-term reward of states.


## Common Algorithms

### 1. Model-Free Methods

#### Q-Learning
It's an off-policy algorithm that learns the value of actions without a model of the environment. In the code below, the Q-table update mimics how an agent learns rewards for different actions.
```python
import numpy as np

# Initialize Q-table (5 states, 2 actions)
Q = np.zeros((5, 2))

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Sample update (state, action, reward, next_state)
state, action, reward, next_state = 0, 1, 10, 1
Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

# Print updated Q-table
print("Updated Q-table:\n", Q)
```
#### SARSA (State-Action-Reward-State-Action)
It's a variation of Q-learning that uses an on-policy algorithm that updates action-value based on the current policy. The code below, updates the Q-values based on the next action taken (on-policy).
```python
# Same Q-learning setup
next_action = 0  # Assume next action is 0
Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

# Print updated Q-table
print("Updated Q-table (SARSA):\n", Q)
```


### 2. Policy Gradient Methods

- **REINFORCE**: Directly optimizes the policy by following the gradient of expected rewards.
- **Actor-Critic**: Combines policy-based (actor) and value-based (critic) methods for stability and efficiency.

The deep learning model, shown below, is optimized using policy gradient methods in reinforcement learning.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple policy neural network
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),  # Input shape is an example
    Dense(2, activation='softmax')  # Output layer for action probabilities
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')

# Display model summary
model.summary()
```


## Use Cases of Reinforcement Learning

- 🎮 **Gaming:** Mastering complex games like chess, Go, and video games.
- 🤖 **Robotics:** Training robots to perform tasks such as navigation and manipulation.
- 🚗 **Self-Driving Cars:** Decision-making for navigation and obstacle avoidance.
- 💰 **Finance:** Portfolio optimization and automated trading.
- 🏥 **Healthcare:** Personalized treatment planning and drug discovery.

