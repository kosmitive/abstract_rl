import numpy as np

from abstract_rl.src.plots.plot import Plot


class RewardValueFunctionPlot(Plot):
    """A plot capable of displaying a graph along with the value function
    and the actions used in each state."""

    def __init__(self):
        super().__init__('Average Total Reward', None)

    def update(self, stoch_rewards, num_evals):

        plt = super().get_axes()
        plt.cla()

        # stochastic policy
        mean_reward = stoch_rewards[:, 0]
        std_reward = stoch_rewards[:, 1]
        offset = 1.96 * std_reward / np.sqrt(num_evals)
        plt.plot(mean_reward, label="sto", color="#550000")
        plt.fill_between(np.arange(len(mean_reward)), mean_reward - offset, mean_reward + offset, facecolor="#d46a6a9a")

        # labels
        plt.set_xlabel("t")
        plt.set_ylabel("r")