import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.policy.continuous.select_policy import select_policy
from abstract_rl.run.run_seeds import SeedRuns


def add_env(runs):
    with runs.mc["conf"].ns("env"):
        runs.mc.add_environment(MCMCEnvWrapper(runs.mc))


def test_each_seed_has_its_own_run_directory(trpo_conf):
    runs = SeedRuns({**trpo_conf, "seeds": [3, 7]}, "trpo")
    add_env(runs)
    assert runs.seeds == [3, 7] and len({mc.run_root for mc in runs.mcs}) == 2
    with runs.mc["conf"].ns("policy"):
        policies = runs.build(lambda k: select_policy(runs.mc, k))
    assert policies.network.layers[0].weight.shape[0] == 2
    assert not np.allclose(policies.network.layers[0].weight[0], policies.network.layers[0].weight[1])
    runs.store("last", {"policy": policies})
    runs.store_best(0, -5.0, 0)
    runs.close()
    for i, mc in enumerate(runs.mcs):
        conf = json.loads(Path(f"{mc.run_root}/run.conf").read_text())
        assert conf["seed"] == runs.seeds[i]  # D3: the real seed, not a random overwrite
        with runs.mc["conf"].ns("policy"):
            fresh = select_policy(runs.mc, jax.random.key(0))
        loaded = eqx.tree_deserialise_leaves(f"{mc.run_root}/model/last/policy.eqx", fresh)
        np.testing.assert_allclose(loaded.network.layers[0].weight, policies.network.layers[0].weight[i])
    assert json.loads(Path(runs.mcs[0].model_root + "/best.json").read_text())["epoch"] == 0


def test_state_has_distinct_target_buffers(trpo_conf):
    runs = SeedRuns({**trpo_conf, "seeds": [1, 2]}, "trpo")
    add_env(runs)
    with runs.mc["conf"].ns("policy"):
        policies = runs.build(lambda k: select_policy(runs.mc, k))
        runs.mc.add_syncable("policy", policies, 1)
    st = runs.state()
    assert st.overall_step.shape == (2,)
    online, target = st.get("policy"), st.get("policy", True)
    assert online.network.layers[0].weight is not target.network.layers[0].weight
    # donation of the whole state must not hit "donate the same buffer twice"
    jax.jit(jax.vmap(lambda s: s.sync()), donate_argnums=0)(st)
    runs.close()


def test_broadcast_adds_the_seed_axis(trpo_conf):
    runs = SeedRuns({**trpo_conf, "seeds": [1, 2, 3]}, "trpo")
    out = runs.broadcast({"x": jnp.zeros(4)})
    assert out["x"].shape == (3, 4)
    runs.close()
