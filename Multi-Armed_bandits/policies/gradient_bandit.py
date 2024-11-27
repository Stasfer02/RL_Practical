"""
Implementation of the Gradient bandit policy, AKA: Action preference with baseline.
"""

"""
Implementation of the epsilon-greedy policy
"""

from policies.abstract_policy import Policy
import numpy as np

class GradientBandit(Policy):
    """
    Gradient-bandit policy.
    """
    def __init__(self, epsilon: float, num_arms: int):
        
        self.epsilon = epsilon
        self.num_arms = num_arms
        self.q_values = np.zeros(num_arms)
        self.arm_cnts = np.zeros(num_arms)

    
    def select_action(self) -> int:
        # random value between 0 and 1
        i = np.random.rand()

        if i < self.epsilon:
            # random selection: Exploration
            action = np.random.choice(self.num_arms)
        else:
            # greedy selection: Exploitation
            action = np.argmax(self.q_values)
        
        return action
    
    def update(self, arm: int, reward: float) -> None:
        self.arm_cnts[arm] += 1

        # update corresponding Q-value using Sample-average method.
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_cnts[arm]

    def __str__(self) -> str:
        return "Gradient-bandit"
