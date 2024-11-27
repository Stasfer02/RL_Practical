import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MA_Bandit import MultiArmedBandit
from policies.epsilon_greedy import Epsilon_Greedy
from policies.greedy import Greedy
from policies.softmax import Softmax
from policies.UCB import UCB
import matplotlib.pyplot as plt
import seaborn as sns

num_arms = 10
config = [{'policy': Greedy, 'custom':'learning_rate'},
          {'policy':Epsilon_Greedy, 'custom':'epsilon'},
          {'policy':UCB, 'custom':'conf_level'},
          {'policy':Softmax, 'custom':'temperature'}]
x_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
x_values_str = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]

# amount of timesteps per run
t_steps = 1000

plot1 = pd.DataFrame()
df_rewards = pd.DataFrame()

np.random.seed(42)

final_plot = pd.DataFrame(columns=[x['policy'].__str__() for x in config])

for p in range(len(config)):
    bandit_means = np.random.normal(loc= 0, scale= 1,size= num_arms).tolist()
    bandit_stds = [1] * num_arms

    bandit = MultiArmedBandit(num_arms, bandit_means, bandit_stds)

    p_plot = []

    for x in x_values:
        print({config[p].get('custom'):x})
        policy = config[p].get('policy')(**{config[p].get('custom'):x, 'num_arms':num_arms}) 
        reward_list = []
        for t in range(t_steps):
            action = policy.select_action()
            reward = bandit.pull_arm(action)
            policy.update(action,reward)

            reward_list.append(reward)
        p_plot.append(pd.DataFrame(reward_list).mean())
    final_plot[str(config[p].get('policy'))]=p_plot
    
plt.figure()
for p in config:
    plt.plot(x_values_str,final_plot[str(p['policy'])], label=p['policy'].__str__())
plt.xlabel("Hyperparameter Value")
plt.ylabel("Average Reward")
plt.title("Evaluation of Exploration Methods on 10-Armed Bandit")
plt.legend()

plt.grid(True, linestyle="--", alpha=0.6)
plt.show()