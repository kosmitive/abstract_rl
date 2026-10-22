from functools import partial
from os.path import join
from typing import Any

import jax
import numpy as np

from abstract_rl.algorithms.continuous.policy_gradient.trust_region_policy_optimization import TRPOAlgorithm

# create the environment in initial state.
from abstract_rl.data_structures.abstract_conf.model_state import ModelState
from abstract_rl.env.rollout_env_wrapper import RolloutEnvWrapper
from abstract_rl.misc.cli_printer import CliPrinter
from abstract_rl.misc.mixed import default_settings
from abstract_rl.operator.filter_operator import FilterOperator
from abstract_rl.operator.gae_operator import GeneralizedAdvantagesEstimationOperator
from abstract_rl.operator.td_operator import TemporalDifferenceOperator
from abstract_rl.plots.reward_plot import RewardValueFunctionPlot
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_seeds import SeedRuns
from abstract_rl.value_functions.fc_v_networks import FullyVNeuralNet

# batch sizes the jitted epoch was traced for (a second epoch with the same batch size must not retrace)
EPOCH_TRACES: list[int] = []

# one line of the policy step's line search table: step, old_obj, obj, lr, ratio, act_improv, exp_improv, distance
P_ROW_FORMAT = "{:10d} | {:14.10f} | {:14.10f} | {:10.5f} | {:14.10f} | {:14.10f} | {:14.10f} | {:10.6f}"


# title of the script
def run_trpo(conf: Any) -> np.ndarray:
    """
    Runs the passed config on policy, every seed of ``seeds`` at once.
    :param conf: The run configuration.
    :return: The logged [mean, std] return of every seed and epoch, shape [seeds, epochs, 2].
    """

    # create a model configuration per seed, the first one wires the shared components
    conf = dict(conf)
    conf.setdefault("alg", "trpo")
    runs = SeedRuns(conf, "trpo")
    mc = runs.mc
    conf = mc["conf"]
    default_settings(conf.get_root("torch_threads") if "torch_threads" in conf else 1)
    cli = CliPrinter().instance

    num_epochs = conf["num_epochs"]
    max_steps = conf["max_steps"]
    epochs_per_call = int(conf.get("epochs_per_call", 1))

    # create an environment wrapper
    with conf.ns("env"):
        env = RolloutEnvWrapper(mc)
        mc.add_environment(env)
    for k, v in env.init_state().items():
        mc.add_main(k, runs.broadcast(v))
    mc.add_main("key", runs.keys)

    # define namespace for v
    with conf.ns("val"):
        sync_steps = conf["sync_steps"]
        v_opt_steps = conf["opt_steps"]
        v_batch_size = conf["batch_size"]
        v_network = runs.build(lambda key: FullyVNeuralNet(mc, jax.random.fold_in(key, 1)))
        mc.add_syncable("v_network", v_network, sync_steps)
        mc.add_main("v_network_opt", jax.vmap(lambda v: v.init_opt())(v_network))

    # policy neural_modules
    with conf.ns("policy"):
        policy_batch_size = conf["batch_size"]
        sync_steps = conf["sync_steps"]
        policy = runs.build(lambda key: select_policy(mc, jax.random.fold_in(key, 2)))
        mc.add_syncable("policy", policy, sync_steps)

    with conf.ns("filter"):
        filter = FilterOperator(mc)
        mc.add_main("fil", filter)

    # create algo
    with conf.ns("tdop"):
        td_op = TemporalDifferenceOperator(mc)
        mc.add_main("td", td_op)

    with conf.ns("gae"):
        gae = GeneralizedAdvantagesEstimationOperator(mc)
        mc.add_main("gae", gae)

    alg_name = conf["alg"]
    with conf.ns(alg_name):
        alg = TRPOAlgorithm(mc)
        mc.add_main("alg", alg)

    def epoch(state: ModelState, batch_size: int) -> tuple[ModelState, dict[str, Any]]:
        """One epoch of one seed: rollout, advantages, trust region step on the policy, fit of v, sync."""
        key, k_rollout, k_v = jax.random.split(state.get("key"), 3)
        pre = {"policy": state.get("policy"), "v_network": state.get("v_network"), "obs_rms": state.get("obs_rms")}

        # test policy
        state, tc, rollout = env.execute_policy(
            state, k_rollout, state.get("policy"), max_steps, batch_size, exploration=True
        )
        tc = filter.transform_all(state, tc)

        # calc value fns and policies: estimate v values by sample return
        tc = td_op.transform_all(state, tc)
        tc = gae.transform_all(state, tc)
        batch = tc.to_batch(["a"])
        state, p_step = alg.perform_tr_sgd(state, 0, batch, batch)

        # v-step
        state, v_step = state.get("v_network").perform_mult_sgd(state, k_v, v_opt_steps, tc, v_batch_size, ["tv"])

        # copy target_networks
        state = state.sync()
        state = state.set("key", key)
        return state, {"rollout": rollout, "p_step": p_step, "v_step": v_step, "pre": pre}

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

    state = runs.state()
    best_reward = np.full(len(runs.seeds), np.finfo(np.float64).min)

    # repeat for the number of given steps
    # optional budget of environment steps: stop before num_epochs once reached, the last epoch is shortened
    max_env_steps = conf.get("max_env_steps", None)
    epoch_steps = policy_batch_size
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
            policy_batch_size = epoch_steps
            if max_env_steps:
                done_steps = int(total_steps.max())
                if done_steps >= max_env_steps:
                    break
                policy_batch_size = min(epoch_steps, max_env_steps - done_steps)
                if policy_batch_size < epoch_steps:
                    num = 1

            state, metrics = run_chunk(state, policy_batch_size, num)
            metrics = jax.device_get(metrics)

            for c in range(num):
                with cli.header("stochastic evaluation", 1):
                    for i, seed_mc in enumerate(runs.mcs):
                        rollout = jax.tree.map(lambda x, i=i, c=c: x[i, c], metrics["rollout"])
                        returns = rollout["episode_returns"][rollout["episode_valid"]]
                        # the logged return is the one of the environment, not of the (scaled) training reward
                        mean_return, std_return = float(np.mean(returns)), float(np.std(returns))
                        best_reward[i] = max(best_reward[i], mean_return)
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
                        rew_plots[i].update(*res_log, int(np.sum(metrics["rollout"]["episode_valid"][i, c])))  # type: ignore[call-arg]
                        rew_plots[i].save(join(seed_mc.plot_root, "performance.png"))

                with cli.header("p-step", True):
                    rows = metrics["p_step"]["rows"][0, c]
                    for row in rows[~np.isnan(rows[:, 0])]:
                        cli.print(P_ROW_FORMAT.format(int(row[0]), *row[1:]))
                    if not bool(metrics["p_step"]["accepted"][0, c]):
                        cli.line().print("FAILED")
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
