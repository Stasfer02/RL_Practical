from abstract_policy import Policy
import numpy as np

class Epsilon_Greedy(Policy):
    """
    Epsilon greedy action-selection algorithm.
    """
    def __init__(self, epsilon, num_arms):
        
        self.epsilon = epsilon
        self.num_arms = num_arms
        self.q_values = np.zeros(num_arms)

    
    def select_action(self):
        # random value between 0 and 1
        i = np.random.rand()

        if i < self.epsilon:
            # random selection: Exploration
            action = np.random.choice(self.num_arms)
        else:
            # greedy selection: Exploitation
            action = np.argmax(self.q_values)
        
        return action
    
    def update(self, arm, reward):
        # TODO
        pass
