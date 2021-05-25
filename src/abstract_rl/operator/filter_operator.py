import numpy as np
import torch
from torch.distributions import transforms

from abstract_rl.src.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.src.operator.trajectory_operator import TrajectoryOperator


class FilterOperator(TrajectoryOperator):
    """
    Represents a simple filter operator using the integrated filter from the environment.
    """

    def __init__(self, mc):
        """
        Initializes a new filter operator
        :param mc: The model configuration to use.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get('conf')
        self.conf = conf
        self.env = mc['env']

        # get discount
        self.filter_states = conf['states']
        self.filter_rewards = conf['rewards']

    def transform(self, trajectory):
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # deactivate grad
        with torch.no_grad():

            if self.filter_states:
                if 'unfiltered_states' not in trajectory:
                    trajectory['unfiltered_states'] = trajectory._states
                    trajectory._states = self.env.state_filter(trajectory['unfiltered_states'], False)

            if self.filter_rewards:
                if 'unfiltered_rewards' not in trajectory:
                    trajectory['unfiltered_rewards'] = trajectory._rewards
                    trajectory._rewards = self.env.reward_filter(trajectory['unfiltered_rewards'], False)