import numpy as np
from numpy import ndarray

from abstract_rl.plots.plot import Plot


class ReturnPlot(Plot):
    """A plot of the average episodic return and its confidence band."""

    def __init__(self) -> None:
        super().__init__("Average Total Reward", None)

    def update(self, stoch_rewards: ndarray, num_evals: int) -> None:

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
