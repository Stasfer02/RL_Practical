"""
Main running file.
"""
import numpy as np
from MA_Bandit import MultiArmedBandit


def main() -> None:

    # specify the number of arms
    num_arms = 10

    # initialize reward distributions
    bandit_means = np.random.normal(loc= 0, scale= 1,size= num_arms).tolist()
    bandit_stds = [1] * num_arms
    #print(bandit_means, "\n", bandit_stds)

    # create our bandit
    bandit = MultiArmedBandit(num_arms, bandit_means, bandit_stds)
    
    # select a policy

    # simulate game
    n_runs = 500
    
    # plotting
    

if __name__ == "__main__":
    main()