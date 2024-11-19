"""
Main running file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MA_Bandit import MultiArmedBandit
from policies.epsilon_greedy import Epsilon_Greedy
from policies.greedy import Greedy
from policies.softmax import Softmax
from policies.UCB import UCB

def main() -> None:
    # specify the number of arms
    num_arms = 10

    # amount of runs
    n_runs = 1
    # amount of timesteps per run
    t_steps = 1000
    
    df_rewards = pd.DataFrame()
    
    
    for n in range(n_runs):
        # perform n runs, with a new bandit for each.

        # initialize reward distributions for bandit arms
        bandit_means = np.random.normal(loc= 0, scale= 1,size= num_arms).tolist()
        bandit_stds = [1] * num_arms
        #print(bandit_means, "\n", bandit_stds)

        # create our bandit
        bandit = MultiArmedBandit(num_arms, bandit_means, bandit_stds)

        # choose policy
        epsilon = 0.1
        policy = Epsilon_Greedy(epsilon,num_arms)

        # keep track of rewards
        reward_list = []

        # for each run, perform the timesteps
        for t in range(t_steps):
            # train policy of choice on the current bandit
            action = policy.select_action()
            reward = bandit.pull_arm(action)
            policy.update(action,reward)

            reward_list.append(reward)
        # stack reward array onto dataframe of total rewards.
        df_rewards = pd.concat([df_rewards, pd.DataFrame([reward_list])])
    
    
    averages = df_rewards.mean(axis=0).transpose()
    plt.figure()
    plt.plot(averages)
    plt.show()

    

if __name__ == "__main__":
    main()