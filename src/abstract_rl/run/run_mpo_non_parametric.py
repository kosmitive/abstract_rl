from os.path import join
from typing import Any

import numpy as np
import torch
from numpy import ndarray

from abstract_rl.algorithms.continuous.policy_gradient.mpo import (
    MPOAlgorithm,
)

# create the environment in initial state.
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.env.env_runner import EnvRunner
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.data_logger import DataLogger
from abstract_rl.misc.mixed import default_settings
from abstract_rl.operator.filter_operator import FilterOperator
from abstract_rl.operator.resample_actions_operator import ActionSamplingOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.operator.reweighting_operator import VariationalReweightingOperator
from abstract_rl.plots.return_plot import ReturnPlot
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.q_network import QNetwork

# title of the script


def run_mpo_non_parametric(conf: dict[str, Any]) -> ndarray:
    """
    Runs the passed config off policy.
    :param conf: The run configuration.
    """

    # create a model configuration
    conf.setdefault("alg", "mpo")
    mc = ModelConfiguration(conf)
    run_conf = mc["conf"]
    default_settings(run_conf.get_root("torch_threads") if "torch_threads" in run_conf else 1)
    cli = CliPrinter().instance

    seed = run_conf["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_epochs = run_conf["num_epochs"]
    max_steps = run_conf["max_steps"]

    # make default instance available
    logger = DataLogger(mc.data_dir)
    mc.add_main("log", logger)

    # create an environment wrapper
    with run_conf.ns("env"):
        render = run_conf["render"]
        num_new_steps = run_conf["sample"]
        num_traj = run_conf["traj_evaluations"]
        env = EnvRunner(mc)
        mc.add_environment(env)

    # create the memory and get some settings
    with run_conf.ns("tc"):
        tc_size = run_conf["size"]
        tc = TrajectoryCollection(env, tc_size)
        mc.add_main("tc", tc)

    # policy neural_modules
    with run_conf.ns("policy"):
        policy_opt_size = run_conf["opt_steps"]
        policy_batch_size = run_conf["batch_size"]
        sync_steps = run_conf["sync_steps"]
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, sync_steps)

    # get obj_watcher
    alg_name = run_conf["alg"]
    with run_conf.ns(alg_name):
        alg = MPOAlgorithm(mc)
        mc.add_main("alg", alg)

    with run_conf.ns("val"):
        q_opt_steps = run_conf["opt_steps"]
        q_batch_size = run_conf["batch_size"]
        sync_steps = run_conf["sync_steps"]
        q_network = QNetwork(mc)
        mc.add_syncable("q_network", q_network, sync_steps)

    with run_conf.ns("filter"):
        filter = FilterOperator(mc)
        mc.add_main("fil", filter)

    # push retrace
    with run_conf.ns("ret"):
        ret = RetraceOperator(mc)
        mc.add_main("ret", ret)

    # push resample
    with run_conf.ns("res"):
        res = ActionSamplingOperator(mc)
        mc.add_main("res", res)

    # push resample
    with run_conf.ns("rnk"):
        rnk = VariationalReweightingOperator(mc)
        mc.add_main("rnk", rnk)

    # names for rewards
    stoch_reward = "sreward"

    # create a new reward plot
    rew_plot = ReturnPlot()
    logger.create_field(stoch_reward, 2)
    # number of environment steps collected in each epoch (for plotting learning curves over samples)
    logger.create_field("steps", 1)

    # if reloaded load old values
    e = 0

    # repeat for the number of given steps
    # optional budget of environment steps: stop before num_epochs once reached, the last epoch is shortened
    max_env_steps = run_conf.get("max_env_steps", None)
    epoch_steps = num_new_steps
    while e < num_epochs:
        # initial print of the episode
        if e % 100 == 0:
            cli.new_epoch(mc.run_root, e)
        cli.big_header(f"epoch {e}")
        if max_env_steps:
            done_steps = getattr(env, "total_steps", 0)
            if done_steps >= max_env_steps:
                break
            num_new_steps = min(epoch_steps, max_env_steps - done_steps)

        # --------------------------------
        # test policy
        # --------------------------------

        # first of all run the deterministic policy afterwards
        # run the stochastic ones and save them in the memory.

        with cli.header("stochastic evaluation", 1):
            new_tc = env.execute_policy(
                policy, max_steps, num_new_steps, exploration=True, render=render, rew_field_name=stoch_reward
            )
            tc.merge(new_tc)
            filter.transform_all(tc)

        with cli.header("store model & generate plots", 1):
            # make and store reward plot
            res_log = logger.get([stoch_reward], e + 1)
            rew_plot.update(*res_log, num_traj)  # type: ignore[call-arg]
            rew_plot.save(join(mc.plot_root, "performance.png"))

            # save last model
            mc.store("last")
            cli.empty().print("last model saved")

        # --------------------------------
        # calc value fns and policies
        # --------------------------------

        with cli.header("e-step", 1):
            # resample extra actions and q values
            res.transform_all(tc)
            rnk.hook()
            rnk.transform_all(tc)

        with cli.header("m-step", 1):
            alg.perform_mult_steps(policy_opt_size, tc, policy_batch_size)

        with cli.header("v-step", 1):
            ret.transform_all(tc)
            q_network.perform_mult_sgd(q_opt_steps, tc, q_batch_size, ["tq", "q"])

        # copy target_networks
        mc.sync()
        e += 1

    env.close()
    rew_plot.close()
    return logger.get([stoch_reward], num_epochs)[0]
