"""Observation normalisation: moments, act/eval agreement, and the policy checkpoint."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from abstract_rl.misc.obs_norm import ObsNorm, normalize_obs
from abstract_rl.policy.continuous.gaussian_math import mode as gaussian_mode
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy
from abstract_rl.scripts.eval_balance import load_policy

jax.config.update("jax_enable_x64", True)


def _welford(rows: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sequential Welford. The reference for the parallel merge."""
    count = 0.0
    mean = np.zeros(rows.shape[1], dtype=rows.dtype)
    moment = np.zeros(rows.shape[1], dtype=rows.dtype)
    for row in rows:
        count += 1.0
        delta = row - mean
        mean = mean + delta / count
        moment = moment + delta * (row - mean)
    return count, mean, moment


def _float64_norm(dim: int) -> ObsNorm:
    return ObsNorm(
        count=jnp.zeros((), dtype=jnp.float64),
        mean=jnp.zeros((dim,), dtype=jnp.float64),
        m2=jnp.zeros((dim,), dtype=jnp.float64),
        clip=10.0,
        eps=1e-8,
        enabled=True,
    )


def test_welford_matches_numpy_and_a_split_merge() -> None:
    rows = np.random.default_rng(0).normal(size=(137, 5)).astype(np.float64)
    norm = _float64_norm(5)
    merged = norm.update(rows)
    count, mean, moment = _welford(rows)
    assert float(merged.count) == count
    assert np.allclose(merged.mean, mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(merged.m2, moment, rtol=1e-10, atol=1e-10)
    split = norm.update(rows[:60]).update(rows[60:])
    assert np.allclose(split.mean, merged.mean, rtol=1e-10, atol=1e-10)
    assert np.allclose(split.m2, merged.m2, rtol=1e-10, atol=1e-10)
    assert float(split.count) == float(merged.count)


def test_normalize_clips_and_does_not_rewrite_the_source() -> None:
    norm = ObsNorm(
        count=jnp.asarray(10.0, dtype=jnp.float32),
        mean=jnp.zeros((1,), dtype=jnp.float32),
        m2=jnp.asarray([9.0], dtype=jnp.float32),
        clip=10.0,
        eps=1e-8,
        enabled=True,
    )
    source = jnp.asarray([[100.0], [-100.0], [0.0]], dtype=jnp.float32)
    normalised = np.asarray(normalize_obs(norm, source))
    assert np.allclose(source, jnp.asarray([[100.0], [-100.0], [0.0]]))
    assert normalised[0, 0] == np.float32(10.0)
    assert normalised[1, 0] == np.float32(-10.0)
    assert abs(float(normalised[2, 0])) < 1e-5
    cold = ObsNorm.create(1, True, 10.0)
    untouched = jnp.asarray([[3.0]], dtype=jnp.float32)
    assert np.allclose(normalize_obs(cold, untouched), untouched)
    disabled = ObsNorm.create(1, False, 10.0)
    assert normalize_obs(disabled, untouched) is untouched
    assert disabled.update(untouched) is disabled


def test_new_statistics_renormalise_an_old_row() -> None:
    """A stored row stays raw, so a later aggregate changes how it is read."""
    rng = np.random.default_rng(1)
    first = ObsNorm.create(2, True, 10.0).update(rng.normal(size=(40, 2)).astype(np.float32))
    raw = jnp.asarray(rng.normal(size=(5, 2)), dtype=jnp.float32)
    stored = jnp.array(raw)
    early = np.asarray(normalize_obs(first, stored))
    later = np.asarray(normalize_obs(first.update(rng.normal(size=(40, 2)).astype(np.float32) + 4.0), stored))
    assert np.allclose(stored, raw)
    assert not np.allclose(early, later, atol=1e-3)


def _pendulum_policy(make_pendulum_mc):
    mc = make_pendulum_mc(obs_norm=True, obs_clip=10.0)
    with mc["conf"].ns("policy"):
        policy = UnivariateGaussianPolicy(mc)
    policy.obs_norm = eqx.tree_at(
        lambda tree: (tree.count, tree.mean, tree.m2),
        policy.obs_norm,
        (
            jnp.asarray(40.0, dtype=jnp.float32),
            jnp.asarray([0.2, -0.4, 0.1], dtype=jnp.float32),
            jnp.asarray([12.0, 30.0, 8.0], dtype=jnp.float32),
        ),
    )
    policy._affine_id = -1
    return mc, policy


def _manual(rows: np.ndarray, norm: ObsNorm) -> np.ndarray:
    count = float(np.asarray(norm.count))
    variance = np.asarray(norm.m2, dtype=np.float64) / (count - 1.0)
    inverse = 1.0 / (np.sqrt(variance) + norm.eps)
    scaled = (rows - np.asarray(norm.mean, dtype=np.float64)) * inverse
    return np.clip(scaled, -norm.clip, norm.clip).astype(np.float32)


def test_act_matches_the_normalised_mode_and_eval_reloads_it(make_pendulum_mc) -> None:
    mc, policy = _pendulum_policy(make_pendulum_mc)
    raw = np.random.default_rng(2).normal(size=(8, 3)).astype(np.float32)
    actions, _log_density = policy.act(raw, deterministic=True)
    via_mode = np.asarray(policy.mode(states=raw))
    via_gaussian = np.asarray(gaussian_mode(policy.params, jnp.asarray(_manual(raw, policy.obs_norm))))
    assert np.allclose(actions, via_mode, atol=1e-5)
    assert np.allclose(via_mode, via_gaussian, atol=1e-5)
    assert not np.allclose(raw, _manual(raw, policy.obs_norm), atol=1e-2)

    mc.add_syncable("policy", policy, 1)
    mc.store("last")
    loaded_env, loaded = load_policy(mc.run_root)
    try:
        assert np.allclose(loaded.obs_norm.mean, policy.obs_norm.mean)
        assert np.allclose(loaded.obs_norm.m2, policy.obs_norm.m2)
        assert float(loaded.obs_norm.count) == float(policy.obs_norm.count)
        assert np.allclose(np.asarray(loaded.mode(states=raw)), via_mode, atol=1e-5)
        reloaded_actions, _ = loaded.act(raw, deterministic=True)
        assert np.allclose(reloaded_actions, via_mode, atol=1e-5)
    finally:
        loaded_env.close()
        mc.env.close()


def test_jax_rollout_stores_raw_observations_and_acts_on_the_normalised_ones(make_pendulum_mc) -> None:
    mc, policy = _pendulum_policy(make_pendulum_mc)
    collection = mc.env.execute_policy(policy, 16, 16, exploration=False)
    trajectory = collection.trajectories()[0]
    raw = np.asarray(trajectory.states[:8], dtype=np.float32)
    assert not np.allclose(raw, _manual(raw, policy.obs_norm), atol=1e-2)
    assert np.allclose(np.asarray(trajectory.actions[:8]), np.asarray(policy.mode(states=raw)), atol=1e-4)
    mc.env.close()
