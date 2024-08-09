# Q-Learning on Frozen Lake
This repository contains an implementation of the Q-learning algorithm from scratch. The algorithm's performance is tested on the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment.

## Q-Learning Implementation
The pseudocode for the Q-Learning algorithm is as follows:

<img src="/doc/qlearning_psdcode.png">

Here's the revised version with improved grammar and clarity:

Exploration allows an agent to enhance its current knowledge about each action, ideally leading to long-term benefits. By improving the accuracy of the estimated action-values, exploration enables an agent to make more informed decisions in the future.

Exploitation, on the other hand, chooses the greedy action to maximize immediate reward based on the agent’s current action-value estimates. However, solely relying on exploitation may not always yield the highest reward and can lead to sub-optimal behavior.

The Epsilon-Greedy method is a simple approach to balance exploration and exploitation by randomly choosing between the two strategies.

$$
\pi(s) = 
\begin{cases} 
\text{argmax}_a \, Q(s, a) & \text{with probability } 1 - \epsilon \\
\text{random action} & \text{with probability } \epsilon 
\end{cases}
$$

By choosing a higher ϵ at the beginning of the learning process, we ensure that the agent has adequate exploration. As ϵ decays over time, the agent begins to exploit what it has learned.

The Q-learning algorithm is implemented as follows:
```python
class QLearner:
    def __init__(self, env, decay_alpha=False):
        self.env = env
        num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.q_table = np.zeros((num_states, self.num_actions))
        self.decay_alpha = decay_alpha


    def update_q_table(self, state, next_state, reward, action):
        self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state,:]) - self.q_table[state][action])

    def behavior_policy(self, state):

        if np.random.uniform() <= self.epsilon:
          action = self.env.action_space.sample()
        else:
          action = self.target_policy(state)

        return action

    def target_policy(self, state):
        max_indices = np.where(self.q_table[state] == np.max(self.q_table[state]))[0]
        return np.random.choice(max_indices)

    def update_param(self):
        self.epsilon = max(self.epsilon * self.xi, 0.01)
        if self.decay_alpha:
          self.alpha *= self.xi

    def reset(self, alpha=0.1, gamma=0.9, epsilon=0.9, xi=0.999, seed=456):
        np.random.seed(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi
```
## Results
### Random Agent
The results of running a random agent are depicted below for comparison.

| Reward Plot | Video |
| --- | --- |
| <img src="/plot/random_reward.png"> | <img src="/doc/random.gif"> |

