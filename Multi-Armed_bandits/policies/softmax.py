"""
Implementation of the Softmax policy.
"""

from policies.abstract_policy import Policy
import numpy as np

class Softmax(Policy):
    def __init__(self, num_arms: int, temperature: float):
        self.num_arms = num_arms
        self.temperature = temperature
        self.q_values = np.zeros(num_arms)
        self.arm_cnts = np.zeros(num_arms)


    def select_action(self) -> int:

        estimates = []
        for i in range(self.num_arms):
            estimate = np.exp(self.q_values[i]/self.temperature)
            estimates.append(estimate)
        
        probabilities = []
        for estimate in estimates:
            probabilities.append(estimate/np.sum(estimates))

        return np.random.choice(self.num_arms, p= probabilities)

    def update(self, arm: int, reward: float) -> None:
        self.arm_cnts[arm] += 1
        
        # Update Q-value using Sample-Average method.
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_cnts[arm]