from os.path import join

import numpy as np
import torch
import os

from abstract_rl.src.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm
from abstract_rl.src.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.src.misc.cli_printer import CliPrinter

# create the environment in initial state.
from abstract_rl.src.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.src.misc.data_logger import DataLogger
from abstract_rl.src.misc.mixed import default_settings
from abstract_rl.src.operator.filter_operator import FilterOperator
from abstract_rl.src.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.src.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.src.plots.reward_plot import RewardValueFunctionPlot
from abstract_rl.src.plots.trajectory_plot import TrajectoryPlot
from abstract_rl.src.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy
from abstract_rl.src.value_functions.fc_v_networks import FullyVNeuralNet


# title of the script
def run_trpo(conf):
    """
    Runs the passed config off policy.
    :param conf: The run configuration.
    """

    default_settings()

    # create a model configuration
    conf['alg'] = 'trpo'
    mc = ModelConfiguration(conf)
    conf = mc['conf']
    cli = CliPrinter().instance

    seed = conf['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_epochs = conf['num_epochs']
    max_steps = conf['max_steps']

    # make default instance available
    logger = DataLogger(mc.data_dir)
    mc.add_main('log', logger)

    # create an environment wrapper
    with conf.ns('env'):
        render = conf['render']
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)

    # define namespace for v
    with conf.ns('val'):
        sync_steps = conf['sync_steps']
        v_network = FullyVNeuralNet(mc)
        mc.add_syncable('v_network', v_network, sync_steps)

    # policy neural_modules
    with conf.ns('policy'):
        policy_batch_size = conf['batch_size']
        sync_steps = conf['sync_steps']
        policy = UnivariateGaussianPolicy(mc)
        mc.add_syncable('policy', policy, sync_steps)

    with conf.ns('filter'):
        filter = FilterOperator(mc)
        mc.add_main('fil', filter)

    # create algo
    with conf.ns('tdop'):
        td_op = TemporalDifferenceOperator(mc)
        mc.add_main('td', td_op)

    with conf.ns('gae'):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
        mc.add_main('gae', gae)

    alg_name = conf['alg']
    with conf.ns(alg_name):
        alg = TRPOAlgorithm(mc)
        mc.add_main('alg', alg)

    # names for rewards
    stoch_reward = 'sreward'

    # create a new reward plot
    rew_plot = RewardValueFunctionPlot()
    logger.create_field(stoch_reward, 2)

    # repeat for the number of given steps
    for e in range(num_epochs):

        # initial print of the episode
        if e % 100 == 0: cli.new_epoch(mc.run_root, e)
        cli.big_header(f"epoch {e}")

        # --------------------------------
        # test policy
        # --------------------------------

        with cli.header("stochastic evaluation", 1):
            tc = env.execute_policy(policy, max_steps, policy_batch_size, exploration=True, render=render, rew_field_name=stoch_reward)
            filter.transform_all(tc)

        with cli.header("store model & generate plots", 1):

            # make and store reward plot
            res_log = logger.get([stoch_reward], e+1)
            rew_plot.update(*res_log, len(tc))
            rew_plot.save(join(mc.plot_root, 'performance.eps'))

            # save last model
            mc.store('last')
            cli.empty().print(f"last model saved")

        # --------------------------------
        # calc value fns and policies
        # --------------------------------

        with cli.header("p-step", True):

            # estimate v values by sample return
            td_op.transform_all(tc)
            gae.transform_all(tc)

            batch = tc.to_batch(['a'])
            alg.perform_tr_sgd(0, batch, batch)

        with cli.header("v-step", True):
            batch = tc.to_batch(['tv'])
            v_network.perform_sgd(0, batch)

        # copy target_networks
        mc.sync()
        e += 1