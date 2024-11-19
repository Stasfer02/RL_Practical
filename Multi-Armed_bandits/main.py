"""
Main running file.
"""
import numpy as np
from MA_Bandit import MultiArmedBandit


def main() -> None:

    # specify the number of arms
    num_arms = 10

    # amount of runs
    n_runs = 500

    # amount of timesteps per run
    t_steps = 1000
    
    for n in range(n_runs):
        # perform n runs, with a new bandit for each.

        # initialize reward distributions
        bandit_means = np.random.normal(loc= 0, scale= 1,size= num_arms).tolist()
        bandit_stds = [1] * num_arms
        #print(bandit_means, "\n", bandit_stds)

        # create our bandit
        bandit = MultiArmedBandit(num_arms, bandit_means, bandit_stds)

        for t in range(t_steps):
            # train algorithm of choice on the current bandit
            
            pass

        # plotting

        pass


if __name__ == "__main__":
    main()