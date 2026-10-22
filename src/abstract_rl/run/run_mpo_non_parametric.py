from functools import partial
from os.path import join
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.algorithms.continuous.policy_gradient.maximum_a_posteriori_policy_optimization_algorithm import (
    MPOAlgorithm,
)

# create the environment in initial state.
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.data_structures.temporal_difference_data.trajectory_collection import TrajectoryCollection
from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.mixed import default_settings
from abstract_rl.operator.filter_operator import FilterOperator
from abstract_rl.operator.np_rew_operator import VariationalRankingOperator
from abstract_rl.operator.resample_actions_operator import ResampleOperator
from abstract_rl.operator.retrace_operator import RetraceOperator
from abstract_rl.plots.reward_plot import RewardValueFunctionPlot
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_seeds import SeedRuns
from abstract_rl.value_functions.fc_q_networks import FullyQNeuralNet

# batch sizes the jitted epoch was traced for (a second epoch with the same batch size must not retrace)
EPOCH_TRACES: list[int] = []

# number of valid transitions in the memory of every seed after every epoch of the last run
LAST_MEMORY_SIZES: list[int] = []


# title of the script
def run_mpo_non_parametric(conf: Any) -> np.ndarray:
    """
    Runs the passed config off policy, every seed of ``seeds`` at once.
    :param conf: The run configuration.
    :return: The logged [mean, std] return of every seed and epoch, shape [seeds, epochs, 2].
    """

    # create a model configuration per seed, the first one wires the shared components
    conf = dict(conf)
    conf.setdefault("alg", "mpo")
    runs = SeedRuns(conf, "mpo")
    mc = runs.mc
    conf = mc["conf"]
    default_settings(conf.get_root("torch_threads") if "torch_threads" in conf else 1)
    cli = CliPrinter().instance
    LAST_MEMORY_SIZES.clear()

    num_epochs = conf["num_epochs"]
    max_steps = conf["max_steps"]
    epochs_per_call = int(conf.get("epochs_per_call", 1))

    # create an environment wrapper
    with conf.ns("env"):
        num_new_steps = conf["sample"]
        num_traj = conf["traj_evaluations"]
        env = RolloutEnvWrapper(mc)
        mc.add_environment(env)
    for k, v in env.init_state().items():
        mc.add_main(k, runs.broadcast(v))
    mc.add_main("key", runs.keys)

    # create the memory and get some settings
    with conf.ns("tc"):
        tc_size = conf["size"]
    num_envs = env.num_envs(num_new_steps, max_steps)
    # the compacted ring needs room for the kept transitions and two rollouts (the insert wraps around once)
    capacity = tc_size + 2 * (num_new_steps + num_envs * max_steps)

    # policy neural_modules
    with conf.ns("policy"):
        policy_opt_size = conf["opt_steps"]
        policy_batch_size = conf["batch_size"]
        sync_steps = conf["sync_steps"]
        policy = runs.build(lambda key: select_policy(mc, jax.random.fold_in(key, 2)))
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
        q_network = runs.build(lambda key: FullyQNeuralNet(mc, jax.random.fold_in(key, 1)))
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

    # persistent optimiser states, lagrange multipliers and temperature of every seed
    mc.add_main("q_network_opt", jax.vmap(lambda q: q.init_opt())(q_network))
    for k, v in jax.vmap(alg.init_state)(policy).items():
        mc.add_main(k, v)
    mc.add_main("eta", runs.broadcast(rnk.b.init_state()["eta"]))

    def annotate(state: ModelState, tc: TrajectoryCollection, key: Any) -> TrajectoryCollection:
        k_res, k_ret = jax.random.split(key)
        tc = res.transform_all(state, tc, k_res)
        tc = rnk.transform_all(state, tc)
        return ret.transform_all(state, tc, k_ret)

    # the memory holds every annotation of the operators from the start, so the jitted carry keeps its structure
    empty = TrajectoryCollection.memory(env, capacity, env.observation_dim, env.action_dim)
    template = jax.eval_shape(lambda s: annotate(s, empty, jax.random.key(0)), runs.seed_slice(runs.state(), 0))
    template_ann = {
        k: jax.ShapeDtypeStruct(v.shape[2:], v.dtype)
        for k, v in template.trajectories()._annotations.items()
        if k not in ("insert_id", "time")
    }
    mc.add_main("tc", runs.broadcast(empty.with_annotations(template_ann)))

    def epoch(state: ModelState, batch_size: int) -> tuple[ModelState, dict[str, Any]]:
        """One epoch of one seed: rollout into the memory, e-step, m-step on the policy, retrace fit of q, sync."""
        key, k_rollout, k_annotate, k_m, k_ret, k_q = jax.random.split(state.get("key"), 6)
        pre = {"policy": state.get("policy"), "q_network": state.get("q_network")}

        # run the stochastic policy and save the trajectories in the memory; filters see the new trajectories once
        state, new_tc, rollout = env.execute_policy(
            state, k_rollout, state.get("policy"), max_steps, batch_size, exploration=True, num_envs=num_envs
        )
        # the stored checkpoint pairs the models the rollout ran with with the statistics the rollout updated
        pre["obs_rms"] = state.get("obs_rms")
        new_tc = filter.transform_all(state, new_tc)
        tc = state.get("tc").merge(new_tc, tc_size)

        # e-step: resample extra actions and q values, temperature (on the merged memory), ranking
        k_res, _ = jax.random.split(k_annotate)
        tc = res.transform_all(state, tc, k_res)
        state = rnk.hook(state, tc)
        tc = rnk.transform_all(state, tc)

        # m-step
        state, m_step = alg.perform_mult_steps(state, k_m, policy_opt_size, tc, policy_batch_size)

        # v-step
        tc = ret.transform_all(state, tc, k_ret)
        state, v_step = state.get("q_network").perform_mult_sgd(state, k_q, q_opt_steps, tc, q_batch_size, ["tq", "q"])

        # copy target_networks
        state = state.set("tc", tc).sync().set("key", key)
        memory_size = jnp.sum(tc.trajectories().mask)
        return state, {"rollout": rollout, "m_step": m_step, "v_step": v_step, "pre": pre, "eta": state.get("eta"),
                       "memory_size": memory_size}

    def chunk(state: ModelState, batch_size: int, num: int) -> tuple[ModelState, dict[str, Any]]:
        EPOCH_TRACES.append(batch_size)
        state, metrics = jax.lax.scan(lambda s, _: epoch(s, batch_size), state, None, length=num)
        return state, metrics

    compiled: dict[tuple[int, int], Any] = {}

    def run_chunk(state: ModelState, batch_size: int, num: int) -> tuple[ModelState, dict[str, Any]]:
        if (batch_size, num) not in compiled:
            fn = jax.vmap(partial(chunk, batch_size=batch_size, num=num))
            compiled[(batch_size, num)] = jax.jit(fn, donate_argnums=0)
        return compiled[(batch_size, num)](state)

    # names for rewards
    stoch_reward = "sreward"

    # create a new reward plot and the logged fields of every seed
    rew_plots = [RewardValueFunctionPlot() for _ in runs.mcs]
    for seed_mc in runs.mcs:
        seed_mc["log"].create_field(stoch_reward, 2)
        # number of environment steps collected in each epoch (for plotting learning curves over samples)
        seed_mc["log"].create_field("steps", 1)
        # temperature of the non-parametric e-step
        seed_mc["log"].create_field("eta", 1)

    state = runs.state()

    # repeat for the number of given steps
    # optional budget of environment steps: stop before num_epochs once reached, the last epoch is shortened
    max_env_steps = conf.get("max_env_steps", None)
    epoch_steps = num_new_steps
    total_steps = np.zeros(len(runs.seeds), np.int64)
    e = 0
    # join the checkpoint writer also when training fails, re-raising failed writes
    try:
        while e < num_epochs:
            # initial print of the episode
            if e % 100 == 0:
                cli.new_epoch(mc.run_root, e)
            cli.big_header(f"epoch {e}")
            num = min(epochs_per_call, num_epochs - e)
            num_new_steps = epoch_steps
            if max_env_steps:
                done_steps = int(total_steps.max())
                if done_steps >= max_env_steps:
                    break
                num_new_steps = min(epoch_steps, max_env_steps - done_steps)
                if num_new_steps < epoch_steps:
                    num = 1
                else:
                    # a chunk of full epochs must not overshoot the budget, the shortened last epoch runs on its own
                    num = max(1, min(num, (max_env_steps - done_steps) // epoch_steps))

            state, metrics = run_chunk(state, num_new_steps, num)
            metrics = jax.device_get(metrics)
            LAST_MEMORY_SIZES.extend(int(m) for m in np.asarray(metrics["memory_size"]).reshape(-1))

            for c in range(num):
                # --------------------------------
                # test policy
                # --------------------------------

                # run the stochastic ones and save them in the memory.
                with cli.header("stochastic evaluation", 1):
                    for i, seed_mc in enumerate(runs.mcs):
                        rollout = jax.tree.map(lambda x, i=i, c=c: x[i, c], metrics["rollout"])
                        returns = rollout["episode_returns"][rollout["episode_valid"]]
                        # the logged return is the one of the environment, not of the (scaled) training reward
                        mean_return, std_return = float(np.mean(returns)), float(np.std(returns))
                        seed_mc["log"].log({stoch_reward: [mean_return, std_return], "steps": [int(rollout["steps"])]})
                        total_steps[i] += int(rollout["steps"])
                        cli.print(
                            f"seed {runs.seeds[i]:5d} | episodes {len(returns):4d} | "
                            f"steps {int(rollout['steps']):7d} | return {mean_return:12.4f} +- {std_return:10.4f}"
                        )

                with cli.header("store model & generate plots", 1):
                    for i, seed_mc in enumerate(runs.mcs):
                        # make and store reward plot
                        res_log = seed_mc["log"].get([stoch_reward], e + 1)
                        rew_plots[i].update(*res_log, num_traj)  # type: ignore[call-arg]
                        rew_plots[i].save(join(seed_mc.plot_root, "performance.png"))

                # --------------------------------
                # calc value fns and policies
                # --------------------------------

                with cli.header("e-step", 1):
                    # the temperature of the non-parametric e-step, fitted on the merged memory
                    for i, seed_mc in enumerate(runs.mcs):
                        eta = float(metrics["eta"][i, c, 0])
                        seed_mc["log"].log({"eta": [eta]})
                        cli.print(f"seed {runs.seeds[i]:5d} | eta {eta:12.6f}")

                with cli.header("m-step", 1):
                    # the objective, multipliers and constraints of the min step (spec P4), every 10 steps of seed 0
                    cli.empty().print(
                        alg.titlestr % ("k", "objective", *[f"lm_{j} x constr_{j}" for j in range(alg.num_params)])
                    )
                    m_step = jax.tree.map(lambda x, c=c: x[0, c], metrics["m_step"])
                    for s in range(0, len(m_step["obj"]), 10):
                        vals = [float(v) for pair in zip(m_step["lagrange_mults"][s], m_step["constraints"][s],
                                                         strict=True) for v in pair]
                        cli.print(alg.fmtstr % (s, float(m_step["obj"][s]), *vals))

                with cli.header("v-step", True):
                    cli.print(FullyQNeuralNet.titlestr % ("k", "pre_objective", "post_objective")).line()
                    k_pre_post = metrics["v_step"]["k_pre_post"][0, c]
                    for j in np.flatnonzero(~np.isnan(k_pre_post[:, 1])):
                        cli.print(FullyQNeuralNet.fmtstr % (j, *k_pre_post[j]))
                e += 1

            # save last model (the models the last rollout ran with), keep it as best model if its return is the highest
            last = num - 1
            for seed_mc in runs.mcs:
                seed_mc.overall_step = e
            runs.store("last", jax.tree.map(lambda x, last=last: x[:, last], metrics["pre"]))
            for i, seed_mc in enumerate(runs.mcs):
                runs.store_best(i, float(seed_mc["log"].get([stoch_reward], e)[0][e - 1, 0]), e - 1)
            cli.empty().print("last model saved")
    finally:
        runs.close()
    for plot in rew_plots:
        plot.close()
    return np.stack([seed_mc["log"].get([stoch_reward], e)[0] for seed_mc in runs.mcs])
