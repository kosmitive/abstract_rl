"""Each benchmark run stores at most the environment step budget, plus at most one parallel round.

The budget is the environment ``steps`` field. One round is ``n_envs * max_steps``: the transitions
a full wave of parallel episodes can add. The collector stores exactly the request, so the
projected run lands on the budget and the round is only the allowed slack.
"""

import math

import gymnasium as gym
import pytest

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper, collection_request
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.scripts import hydra_conf

PAIRS = [(env, alg) for env in hydra_conf.environments() for alg in ("trpo", "mpo")]

# ceil(steps / env_sample). MountainCar MPO used to list 1000 epochs; the 400000-step budget
# already stopped that run at 400.
EXPECTED_EPOCHS = {
    ("Pendulum-v1", "trpo"): 200,
    ("Pendulum-v1", "mpo"): 800,
    ("MountainCarContinuous-v0", "trpo"): 100,
    ("MountainCarContinuous-v0", "mpo"): 400,
    ("InvertedPendulum-v5", "trpo"): 75,
    ("InvertedPendulum-v5", "mpo"): 150,
    ("InvertedDoublePendulum-v5", "trpo"): 75,
    ("InvertedDoublePendulum-v5", "mpo"): 600,
    ("HalfCheetah-v5", "trpo"): 150,
    ("HalfCheetah-v5", "mpo"): 1200,
    ("Humanoid-v5", "trpo"): 200,
    ("Humanoid-v5", "mpo"): 3000,
}


def _project(num_epochs: int, epoch_steps: int, budget: int, slack: int) -> int:
    """Steps the training loop would store if every collection overshoots its request by ``slack``."""
    done = 0
    for _ in range(num_epochs):
        request = collection_request(done, epoch_steps, budget)
        if request is None:
            break
        done += request + slack
    return done


@pytest.mark.parametrize(("env_id", "alg"), PAIRS)
def test_stored_env_steps_stay_within_one_round_of_the_budget(env_id, alg, tmp_path):
    try:
        probe = gym.make(env_id)
    except gym.error.Error as exc:
        pytest.skip(f"{env_id} is not installed ({exc})")
    probe.close()

    conf = hydra_conf.make_config(env_id, alg, 1, [f"data_root={tmp_path}"])
    conf["alg"] = alg
    budget = hydra_conf.steps()[env_id]
    assert conf["steps"] == budget
    epoch_steps = int(conf["env_sample"])
    max_steps = int(conf["max_steps"])
    num_epochs = math.ceil(budget / epoch_steps)
    assert num_epochs == EXPECTED_EPOCHS[(env_id, alg)]

    mc = ModelConfiguration(conf)
    with mc["conf"].ns("env"):
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)
    with mc["conf"].ns("policy"):
        policy = select_policy(mc)
    n_envs = env._resolve_n_envs(epoch_steps, max_steps, render=False)
    one_round = n_envs * max_steps
    try:
        collection = env.execute_policy(policy, max_steps, epoch_steps, exploration=True)
    finally:
        env.close()

    collected = sum(len(traj) for traj in collection)
    # Exactly the request. The bound the run is allowed to miss by is one parallel round.
    assert collected == epoch_steps
    assert 0 <= collected - epoch_steps <= one_round
    stored = _project(num_epochs, epoch_steps, budget, collected - epoch_steps)
    assert stored <= budget + one_round
    assert stored == min(num_epochs * epoch_steps, budget)
