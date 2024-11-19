from abstract_policy import Policy
import numpy as np



class Greedy(Policy):
    """
    Greedy action-selection algorithm.
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.q_values = np.zeros(num_arms)

    
    def select_action(self):
        # Greedy: we always select the action with the highest q-value
        action = np.argmax(self.q_values)

        return action

    def update(self, arm, reward):
        # TODO
        pass