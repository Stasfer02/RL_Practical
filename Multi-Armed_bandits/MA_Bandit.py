"""
This is the Multi-Armed bandit class. 
"""

from typing import List
import numpy as np


class MultiArmedBandit:
    def __init__(self, num_arms: int, means: List[float], stds: List[float]):
        """
        take in the number of arms and the corresponding lists for mean and STD values (created in main)
        """
        self.num_arms = num_arms
        self.means = means
        self.stds = stds
        

    def pull_arm(self, arm: int) -> float:
        """
        Specific arm is pulled. 

        We draw the reward using the arm index for the mean/STD lists. 

        Return the reward (drawn from normal distribution)
        """
        reward = np.random.normal(loc= self.means[arm], scale= self.stds[arm])
        return reward