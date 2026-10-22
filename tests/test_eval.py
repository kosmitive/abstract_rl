import numpy as np

from abstract_rl.run.run_trpo import run_trpo
from abstract_rl.scripts import eval_balance, eval_returns


def test_eval_uses_frozen_obs_rms_and_seeded_starts(trpo_conf, tmp_path):
    run_trpo({**trpo_conf, "obs_norm": True})
    (run_dir,) = tmp_path.glob("Pendulum-v1/trpo/*")
    env, policy, state = eval_balance.load_policy(str(run_dir))
    stored = np.load(run_dir / "model/obs_rms.npz")
    np.testing.assert_allclose(state.get("obs_rms").mean, stored["mean"], rtol=1e-6)
    r1, t1 = eval_balance.run_episode(env, policy, state, 10_000)
    r2, t2 = eval_balance.run_episode(env, policy, state, 10_000)
    assert r1 == r2 and t1.shape == (200,)  # greedy and seeded: identical episodes, stats never change
    np.testing.assert_array_equal(t1, t2)
    name, returns = eval_returns.evaluate(str(run_dir), episodes=2)
    assert name == "Pendulum-v1" and returns.shape == (2,)


def test_pendulum_gif_frames(trpo_conf, tmp_path):
    run_trpo(trpo_conf)
    (run_dir,) = tmp_path.glob("Pendulum-v1/trpo/*")
    gif = tmp_path / "x.gif"
    eval_balance.save_gif(str(run_dir), str(gif))
    assert gif.exists() and gif.stat().st_size > 0
