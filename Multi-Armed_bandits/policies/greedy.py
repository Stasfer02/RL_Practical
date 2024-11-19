"""
Implementation of the greedy policy.
"""

from policies.abstract_policy import Policy
import numpy as np



class Greedy(Policy):
    """
    Greedy action-selection.
    """
    def __init__(self, num_arms: int, learning_rate: float):
        """
        initialize with the number of arms in the bandit problem and the learning rate (alfa) for the weighted-average method.s
        """
        self.num_arms = num_arms
        self.q_values = np.zeros(num_arms)
        self.learning_rate = learning_rate

    
    def select_action(self) -> int:
        # Greedy: we always select the action with the highest q-value
        action = np.argmax(self.q_values)
        return action

    def update(self, arm: int, reward: float) -> None:

        # update Q-value using the Weighted-Average method.
        self.q_values[arm] += self.learning_rate * (reward - self.q_values[arm])
        pass