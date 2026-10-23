from os.path import join
from typing import Any

import mlx.core as mx
import numpy as np

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
)

# create the environment in initial state.
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.data_logger import DataLogger
from abstract_rl.misc.mixed import default_settings
from abstract_rl.operator.filter_operator import FilterOperator
from abstract_rl.operator.np_rew_operator import VariationalRankingOperator
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.plots.reward_plot import RewardValueFunctionPlot
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet


# title of the script
def run_mpo_non_parametric(conf: Any) -> np.ndarray:
    """
    Runs the passed config off policy.
    :param conf: The run configuration.
    """

    # create a model configuration
    conf.setdefault("alg", "mpo")
    mc = ModelConfiguration(conf)
    conf = mc["conf"]
    # auto: MPO fits the Q function on large batches of the replay memory, that is faster on the GPU
    default_settings(conf.get_root("device") if "device" in conf else "cpu", prefer_gpu=True)
    cli = CliPrinter().instance

    seed = conf["seed"]
    mx.random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    num_epochs = conf["num_epochs"]
    max_steps = conf["max_steps"]

    # make default instance available
    logger = DataLogger(mc.data_dir)
    mc.add_main("log", logger)

    # create an environment wrapper
    with conf.ns("env"):
        render = conf["render"]
        num_new_steps = conf["sample"]
        num_traj = conf["traj_evaluations"]
        env = RolloutEnvWrapper(mc)
        mc.add_environment(env)

    # create the memory and get some settings
    with conf.ns("tc"):
        tc_size = conf["size"]
        tc = TrajectoryCollection(env, tc_size)
        mc.add_main("tc", tc)

    # policy neural_modules
    with conf.ns("policy"):
        policy_opt_size = conf["opt_steps"]
        policy_batch_size = conf["batch_size"]
        sync_steps = conf["sync_steps"]
        policy = select_policy(mc)
        mc.add_syncable("policy", policy, sync_steps)

    # get obj_watcher
    alg_name = conf["alg"]
    with conf.ns(alg_name):
        alg = MPOAlgorithm(mc)
        mc.add_main("alg", alg)

    with conf.ns("val"):
        q_opt_steps = conf["opt_steps"]
        q_batch_size = conf["batch_size"]
        sync_steps = conf["sync_steps"]
        q_network = FullyQNeuralNet(mc)
        mc.add_syncable("q_network", q_network, sync_steps)

    with conf.ns("filter"):
        filter = FilterOperator(mc)
        mc.add_main("fil", filter)

    # push retrace
    with conf.ns("ret"):
        ret = RetraceOperator(mc)
        mc.add_main("ret", ret)

    # push resample
    with conf.ns("res"):
        res = ResampleOperator(mc)
        mc.add_main("res", res)

    # push resample
    with conf.ns("rnk"):
        rnk = VariationalRankingOperator(mc)
        mc.add_main("rnk", rnk)

    # names for rewards
    stoch_reward = "sreward"

    # create a new reward plot
    rew_plot = RewardValueFunctionPlot()
    logger.create_field(stoch_reward, 2)
    # number of environment steps collected in each epoch (for plotting learning curves over samples)
    logger.create_field("steps", 1)

    # if reloaded load old values
    e = 0

    # repeat for the number of given steps
    # optional budget of environment steps: stop before num_epochs once reached, the last epoch is shortened
    max_env_steps = conf.get("max_env_steps", None)
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

            # save last model and keep it as best model if its epoch return is the highest so far
            mc.store("last")
            mc.store_best(float(res_log[0][e, 0]), e)
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
