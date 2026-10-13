from os.path import join
from typing import Any

import numpy as np
from numpy import ndarray

from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm

# create the environment in initial state.
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.data_logger import DataLogger
from abstract_rl.misc.mixed import default_settings
from abstract_rl.operator.filter_operator import FilterOperator
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.plots.reward_plot import RewardValueFunctionPlot
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet


# title of the script
def run_trpo(conf: dict[str, Any]) -> ndarray:
    """
    Runs the passed config off policy.
    :param conf: The run configuration.
    """

    # create a model configuration
    conf.setdefault("alg", "trpo")
    mc = ModelConfiguration(conf)
    run_conf = mc["conf"]
    default_settings(run_conf.get_root("torch_threads") if "torch_threads" in run_conf else 1)
    cli = CliPrinter().instance

    seed = run_conf["seed"]
    np.random.seed(seed)
    num_epochs = run_conf["num_epochs"]
    max_steps = run_conf["max_steps"]

    # make default instance available
    logger = DataLogger(mc.data_dir)
    mc.add_main("log", logger)

    # create an environment wrapper
    with run_conf.ns("env"):
        render = run_conf["render"]
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)

    # define namespace for v
    with run_conf.ns("val"):
        sync_steps = run_conf["sync_steps"]
        v_network = FullyVNeuralNet(mc)
        mc.add_syncable("v_network", v_network, sync_steps)

    # policy neural_modules
    with run_conf.ns("policy"):
        policy_batch_size = run_conf["batch_size"]
        sync_steps = run_conf["sync_steps"]
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, sync_steps)

    with run_conf.ns("filter"):
        filter = FilterOperator(mc)
        mc.add_main("fil", filter)

    # create algo
    with run_conf.ns("tdop"):
        td_op = TemporalDifferenceOperator(mc)
        mc.add_main("td", td_op)

    with run_conf.ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
        mc.add_main("gae", gae)

    alg_name = run_conf["alg"]
    with run_conf.ns(alg_name):
        alg = TRPOAlgorithm(mc)
        mc.add_main("alg", alg)

    # names for rewards
    stoch_reward = "sreward"

    # create a new reward plot
    rew_plot = RewardValueFunctionPlot()
    logger.create_field(stoch_reward, 2)
    # number of environment steps collected in each epoch (for plotting learning curves over samples)
    logger.create_field("steps", 1)

    # repeat for the number of given steps
    # optional budget of environment steps: stop before num_epochs once reached, the last epoch is shortened
    max_env_steps = run_conf.get("max_env_steps", None)
    epoch_steps = policy_batch_size
    for e in range(num_epochs):
        # initial print of the episode
        if e % 100 == 0:
            cli.new_epoch(mc.run_root, e)
        cli.big_header(f"epoch {e}")
        if max_env_steps:
            done_steps = getattr(env, "total_steps", 0)
            if done_steps >= max_env_steps:
                break
            policy_batch_size = min(epoch_steps, max_env_steps - done_steps)

        # --------------------------------
        # test policy
        # --------------------------------

        with cli.header("stochastic evaluation", 1):
            tc = env.execute_policy(
                policy, max_steps, policy_batch_size, exploration=True, render=render, rew_field_name=stoch_reward
            )
            filter.transform_all(tc)

        with cli.header("store model & generate plots", 1):
            # make and store reward plot
            res_log = logger.get([stoch_reward], e + 1)
            rew_plot.update(*res_log, len(tc))  # type: ignore[call-arg]
            rew_plot.save(join(mc.plot_root, "performance.png"))

            # save last model
            mc.store("last")
            cli.empty().print("last model saved")

        # --------------------------------
        # calc value fns and policies
        # --------------------------------

        with cli.header("p-step", True):
            # estimate v values by sample return
            td_op.transform_all(tc)
            gae.transform_all(tc)

            batch = tc.to_batch(["a"])
            alg.perform_tr_sgd(0, batch, batch)

        with cli.header("v-step", True):
            batch = tc.to_batch(["tv"])
            v_network.perform_sgd(0, batch)

        # copy target_networks
        mc.sync()

    env.close()
    rew_plot.close()
    return logger.get([stoch_reward], num_epochs)[0]
