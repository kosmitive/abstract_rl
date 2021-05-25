import torch
import torch.optim as optim
from pandas.tests.extension.numpy_.test_numpy_nested import np

from abstract_rl.src.algorithms.continuous.policy_gradient.policy_optimization_algorithm import PolicyOptimizationAlgorithm
from abstract_rl.src.misc.cli_printer import CliPrinter
from abstract_rl.src.misc.flat_params import get_flat_params_from


class MPOAlgorithm(PolicyOptimizationAlgorithm):
    """
    Implementation of Maximum A Posteriori Policy optimization (mpo), e.g. see the original paper for more
    information https://arxiv.org/abs/1806.06920. It estimates the q values using a critic based on retrace, which
    is not inherent to this algorithm, so it is not defined in this class. From these q values a variational
    distribution is fitted. This 'reweighted' policy is then used to fit the current iterated policy to the optimal
    one in a supervised fashion.
    """

    def __repr__(self): return f"Maximum A Posteriori Policy Optimization"

    def __init__(self, mc):
        """
        Initializes a new maximum a posteriori policy optimization algorithm.
        :param mc: The model configuration, containing all important elements, like env or similar.
        """
        PolicyOptimizationAlgorithm.__init__(self, mc)

        # create lagrange multiplier
        policy = self.mc.get('policy', target=False)
        num_lagrange_multipliers = policy.get_suff_stats_dim()

        # inner and outer learning rated, trust regions and batch size shortcuts
        self.outer_lr = self.conf['outer_lr']
        self.inner_lr = self.conf['inner_lr']
        self.mode = self.conf['mode']
        self.trust_regions = self.conf['trust_regions']
        self.split_objective = self.conf['coordinate_ascent_objective']
        self.split_constraint = self.conf['coordinate_ascent_constraint']
        self.cli = CliPrinter().instance
        self.l2_reg = self.conf['l2_reg']

        num_params = 1 if not self.split_constraint else num_lagrange_multipliers
        self.lagrange_mults = [torch.nn.Parameter(torch.ones(1), requires_grad=True) for _ in range(num_params)]
        self.v = self.sample_v()

        self.fmtstr = "%10i | %10.5f" + num_params * " | %10.5f x %10.5f"
        self.titlestr = "%10s | %10s" + num_params * " | %23s"

    def sample_v(self):
        """Sample new trust regions.

        :return: the new sampled trust regions.
        """

        samp_trust_regions = self.trust_regions
        v = [None] * len(samp_trust_regions)
        for d in range(len(samp_trust_regions)):
            if isinstance(samp_trust_regions[d], list):
                v[d] = np.random.uniform(samp_trust_regions[d][0], samp_trust_regions[d][1])
            else:
                v[d] = samp_trust_regions[d]

        self.v = v
        return self.v

    def shuffle_up_stats(self, batch):
        """Use this to create a list, where each sufficient statistic from the
        normal policy is mixed with the target ones, so the objective and
        constraint can be effectively splitted up.

        :param batch: The batch to use for obtaining the sufficient statistics
        :return:
        """

        # get target and correct policy
        t_states = torch.Tensor(batch.states)
        t_policy = self.mc.get('policy', target=True)
        policy = self.mc.get('policy', target=False)

        # target stats
        target_suff_stats = t_policy.forward(t_states).detach()
        suff_stats = policy.forward(t_states)
        l_suff_stats = []
        l_suff_stats.append(suff_stats)

        # for each dimension
        num_dims = target_suff_stats.size()[1]
        for d in range(num_dims):
            lst_els = []
            if d > 0: lst_els.append(target_suff_stats[:, :d])
            lst_els.append(suff_stats[:, d:d + 1].view([-1, 1]))
            if d < num_dims - 1: lst_els.append(target_suff_stats[:, d + 1:])
            l_suff_stats.append(torch.cat(lst_els, dim=1))

        return l_suff_stats

    def main_obj(self, batch, l_suff_stats):
        """Obtain the objective based on the list of sufficient statistics.

        :param batch: The batch to use.
        :param l_suff_stats: A list of sufficient statistics.
        :return A list of constraints
        """

        # split objective if wanted
        t_states = torch.Tensor(batch.states)
        t_policy = self.mc.get('policy', target=True)

        target_suff_stats = t_policy.forward(t_states).detach()
        if self.split_objective:

            # build the objective in a coordinate ascent fashion
            # iterate over the dimensions
            num_dims = target_suff_stats.size()[1]
            obj = torch.Tensor(torch.zeros(1))
            for d in range(num_dims):
                part_obj = self.__lagrange_obj(batch, l_suff_stats[d + 1])
                obj = obj + part_obj

        else:
            obj = self.__lagrange_obj(batch, l_suff_stats[0])

        return obj

    def main_constraints(self, batch, l_suff_stats):
        """Obtain all constraints using the list of sufficient statistics and the batch.

        :param batch: The batch to use.
        :param l_suff_stats: A list of sufficient statistics.
        :return A list of constraints
        """

        if self.mode == 'parametric':
            kl = 'i-projection'
        else:
            kl = 'm-projection'
        policy = self.mc.get('policy', target=False)
        t_states = torch.Tensor(batch.states)
        t_policy = self.mc.get('policy', target=True)
        target_suff_stats = t_policy.forward(t_states).detach()

        # split constraint if wanted
        if self.split_constraint:

            num_dims = target_suff_stats.size()[1]
            constraints = []
            for d in range(num_dims):
                c = (self.v[d] - policy.kl_divergence(batch, l_suff_stats[d+1], kl, 'mean'))
                constraints.append(c)

        else:
            constraints = [(self.v[0] - policy.kl_divergence(batch, l_suff_stats[0], kl, 'mean'))]

        return constraints

    def obj(self, batch):
        """
        Calculate the objective based on the objective and the splitted kl divergence constraints.
        :param batch: Batch to estimate objective and constraints
        :return: A un-detached version of the objective
        """

        l_suff_stats = self.shuffle_up_stats(batch)
        obj = self.main_obj(batch, l_suff_stats)
        constraints = self.main_constraints(batch, l_suff_stats)
        for lm, c in zip(self.lagrange_mults, constraints):
            obj = obj + torch.exp(lm) * c

        return obj

    def perform_mult_steps(self, num_steps, tc, batch_size):
        """
        Maximization step. Use the reweighted samples to solve a trust region optimization problem. Note that the
        constrain is effectively decoupled, so convergence of mean and variance can be controlled separately. A
        objective is created based on lagrangian multipliers and the theory of min max optimization.
        """

        self.sample_v()
        self.cli.empty().print(self.titlestr % ("k", "objective", *[f"lm_{i} x constr_{i}" for i in range(len(self.lagrange_mults))]))

        # perform a min step
        for s in range(num_steps):
            batch = tc.sample(batch_size, ['add_act', 'grad_ranking', 'add_pi'])
            self.perform_step(s, batch)

    def perform_step(self, step, batch):
        """
        Maximization step. Use the reweighted samples to solve a trust region optimization problem. Note that the
        constrain is effectively decoupled, so convergence of mean and variance can be controlled separately. A
        objective is created based on lagrangian multipliers and the theory of min max optimization.
        """

        # perform a min step
        self.__max_step(batch)
        self.__min_step(batch)

        l_suff_stats = self.shuffle_up_stats(batch)
        obj = self.main_obj(batch, l_suff_stats)
        constraints = self.main_constraints(batch, l_suff_stats)

        self.cli.print(self.fmtstr % (step, obj.detach().numpy(), *[item for sublist in zip(self.lagrange_mults, constraints) for item in sublist]))

    def __lagrange_obj(self, batch, suff_stats):
        """
        Calculated the reweighted objective needed for the min max problem. This is only partly 
        the objective as the lagrange constraints are missing.
        :param batch: The batch to use for estimating the objective.
        :param suff_stats: A matrix containing for each element of the batch a vector of the 
               sufficient statistics.
        :return: The calculated objective.
        """

        policy = self.mc.get('policy', target=True)
        policy = self.mc.get('policy', target=False)

        # build regularization for mean
        if self.mode == 'parametric':

            act = torch.Tensor(batch.actions)
            adv = torch.Tensor(batch['a'])

            # sample sufficient statistics
            ll = policy.log_prob(act, suff_stats=suff_stats)
            rew_vals = torch.exp(ll) * adv
            obj = torch.mean(rew_vals)

        else:

            t_add_act = torch.Tensor(batch['add_act'])
            t_add_q = torch.Tensor(batch['grad_ranking'])

            # sample sufficient statistics
            ll = policy.log_prob(t_add_act, suff_stats=suff_stats)
            rew_vals = torch.exp(t_add_q) * ll
            obj = torch.mean(rew_vals)

        w = get_flat_params_from(policy)
        return obj - self.l2_reg * torch.norm(w)

    def __max_step(self, batch):
        """
        Solves the minimization problem targeting the policy parameters only.
        """

        # short hand and optimizer
        policy = self.mc.get('policy', target=False)
        policy_parameters = list(policy.parameters())
        opt_outer = optim.Adam(policy_parameters, lr=self.outer_lr)

        # optimize
        opt_outer.zero_grad()
        obj = -self.obj(batch)
        obj.backward()
        opt_outer.step()

    def __min_step(self, batch):
        """
        Solves the minimization problem targeting the lagrange multiplies. Note that they have to be
        positive.
        """

        # short hand and optimizer
        opt_inner = optim.Adam(self.lagrange_mults, lr=self.inner_lr)

        # training step
        opt_inner.zero_grad()
        obj = self.obj(batch)
        obj.backward()
        opt_inner.step()
