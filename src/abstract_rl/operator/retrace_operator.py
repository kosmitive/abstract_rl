import torch
from pandas.tests.extension.numpy_.test_numpy_nested import np

from abstract_rl.src.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.src.operator.trajectory_operator import TrajectoryOperator
from abstract_rl.src.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


class RetraceOperator(TrajectoryOperator):
    """
    Represents the retrace operator from https://arxiv.org/abs/1606.02647.
    """

    def __repr__(self):
        return "retrace"

    def __init__(self, mc):
        """
        Initializes a new retrace operator.
        :param mc: The model configuration with env and so on.
        """
        assert isinstance(mc, ModelConfiguration)
        conf = mc.get('conf')
        self.conf = conf
        self.mc = mc
        self.policy = mc.get('policy', False)
        self.q_network = mc.get('q_network', True)

        env = mc['env']
        self.discount = env.discount()
        self.env = env

        self.ret_lambda = torch.Tensor([self.conf['lambda']])
        self.ret_bootstrap = self.conf.get_root('mem_bootstrap_steps')
        self.ret_add_acts = self.conf['add_acts']

    def train(self):
        """
        Train the operator, or more specifically the q network.
        """
        self.q_network.train()

    def transform(self, trajectory):
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # first rolling estimate of the exp q function
        discount = self.discount

        # it is not necessary to correctly log the grads
        with torch.no_grad():

            # get truncated length
            tl = len(trajectory)

            # obtain q and v value estimates
            states = torch.Tensor(trajectory.states)
            actions = torch.Tensor(trajectory.actions)
            q_vals = self.q_network.q_val(states, actions)

            # calculate expectation of next state
            nxt_states = torch.Tensor(trajectory.next_states)
            dones = torch.Tensor(trajectory.dones)
            nxt_v_vals = self.q_network.v_val(nxt_states)
            nxt_v_vals[-1] *= (1 - dones[-1])

            # get import. sampling coefficients
            c = self._def_bounded_importance_sampling_coefficients(trajectory)

            # calc td errors for all t
            r = torch.Tensor(trajectory.rewards).view([-1, 1])
            td_errors = r + discount * nxt_v_vals

            td_errors_q = td_errors - q_vals
            gammas = torch.Tensor(self.discount ** np.arange(self.ret_bootstrap))

            # backwards iterate for all q values
            tq = q_vals
            for t in range(tl-self.ret_bootstrap+1):

                # extract coefficients and build up
                extracted_c = c[t:t+self.ret_bootstrap-1, 0]
                one_c = torch.cat([torch.ones(1), extracted_c])
                cum_c = torch.cumprod(one_c, 0)

                # get all fields
                coeff = cum_c * gammas * td_errors_q[t:t+self.ret_bootstrap]
                tq[t] += torch.sum(coeff)

            trajectory['q'] = q_vals
            trajectory['tq'] = tq

    def _run_retrace(self, trajectory):
        """
        Executes retrace on the trajectory.
        :param trajectory: The trajectory to execute retrace on.
        :return: The retraced q values.
        """

    def _def_bounded_importance_sampling_coefficients(self, trajectory):
        """
        Calculated the bounded importance sampling coefficients.
        :param trajectory: The trajectory for which the coefficients have to be calculated
        :return: A vector containing the coefficients.
        """

        # obtain the bounded importance sampling coefficients coefficients
        beh_lls = torch.Tensor(trajectory.log_likelihoods)
        beh_lls = beh_lls.view([-1, 1])
        s = torch.Tensor(trajectory.states)
        a = torch.Tensor(trajectory.actions)

        pi_ll = self.policy.log_prob(a, states=s)
        r = torch.exp(pi_ll - beh_lls)
        c = torch.min(r, torch.ones(r.size())) * self.ret_lambda
        return c
