"""
Implementation of the Upper Confidence Bound (UCB) policy.
"""

from policies.abstract_policy import Policy
import numpy as np

class UCB(Policy):
    def __init__(self, num_arms: int, conf_level: float):
        self.num_arms = num_arms
        self.conf_level = conf_level
        self.total_cnt = 0
        self.q_values = np.zeros(num_arms)
        self.arm_cnts = np.zeros(num_arms)
        

    
    def select_action(self) -> int:

        self.total_cnt += 1
        
        estimates = []
        for i in range(self.num_arms):
            estimates.append(self.q_values[i] + self.conf_level * np.sqrt(np.log(self.total_cnt) / self.arm_cnts[i]) )

        # greedy selection
        return np.argmax(estimates)

    def update(self, arm: int, reward: float) -> None:
        self.arm_cnts[arm] += 1

        self.q_values[arm] = (reward - self.q_values[arm]) / self.arm_cnts[arm]
        pass

    def __str__(self) -> str:
        return "UCB"