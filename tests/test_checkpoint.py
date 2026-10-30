"""A run resumed from its checkpoint continues exactly as the uninterrupted run."""

import copy
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import abstract_rl.run.run_mpo_non_parametric as run_module
from abstract_rl.run.checkpoint import CHECKPOINT, has_checkpoint, latest_resumable_run
from abstract_rl.run.run_mpo_non_parametric import run_mpo_non_parametric


class Interrupt(Exception):
    pass


def _conf(mpo_conf, root: Path, **overrides):
    conf = copy.deepcopy(mpo_conf)
    conf.update({"num_epochs": 6, "data_root": str(root), **overrides})
    return conf


def _run_dir(root: Path) -> Path:
    (run_dir,) = root.glob("Pendulum-v1/mpo/*")
    return run_dir


def _outputs(run_dir: Path) -> dict:
    out = {f: (run_dir / "data" / f).read_text() for f in ("sreward.txt", "eta.txt", "steps.txt")}
    for name in ("policy", "q_network"):
        out.update(
            {f"{name}.{k}": np.array(v) for k, v in mx.load(str(run_dir / f"model/last/{name}.safetensors")).items()}
        )
    return out


@pytest.mark.parametrize("background", [False, True], ids=["main_process", "fit_worker"])
def test_resumed_run_matches_uninterrupted_run(mpo_conf, tmp_path, monkeypatch, background):
    overrides = {"val_background_fit": background, "obs_norm": True}
    returns = run_mpo_non_parametric(_conf(mpo_conf, tmp_path / "a", **overrides))
    expected = _outputs(_run_dir(tmp_path / "a"))

    # checkpoints after epochs 2 and 4, the run stops before the second one, so the logs are two epochs ahead of it
    save = run_module.save_checkpoint
    calls = []

    def save_then_stop(mc, epoch):
        calls.append(epoch)
        if len(calls) == 2:
            raise Interrupt
        return save(mc, epoch)

    monkeypatch.setattr(run_module, "save_checkpoint", save_then_stop)
    conf = _conf(mpo_conf, tmp_path / "b", checkpoint_every=2, **overrides)
    with pytest.raises(Interrupt):
        run_mpo_non_parametric(copy.deepcopy(conf))
    monkeypatch.setattr(run_module, "save_checkpoint", save)
    run_dir = _run_dir(tmp_path / "b")
    assert has_checkpoint(str(run_dir))
    assert len((run_dir / "data/sreward.txt").read_text().split()) == 2 * 4

    resumed = run_mpo_non_parametric({**conf, "resume": True})
    assert _run_dir(tmp_path / "b") == run_dir
    np.testing.assert_array_equal(resumed, returns)
    got = _outputs(run_dir)
    assert got.keys() == expected.keys()
    for k, v in expected.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(got[k], v, err_msg=k)
        else:
            assert got[k] == v, k
    # the checkpoint after epoch 4 replaced the one after epoch 2
    assert sorted(p.name for p in run_dir.iterdir() if p.name.startswith(CHECKPOINT)) == [CHECKPOINT]


def test_no_checkpoint_by_default(mpo_conf, tmp_path):
    run_mpo_non_parametric(_conf(mpo_conf, tmp_path, num_epochs=3))
    run_dir = _run_dir(tmp_path)
    assert not list(run_dir.glob(f"{CHECKPOINT}*"))
    # resume: true without a checkpoint starts a new run
    run_mpo_non_parametric(_conf(mpo_conf, tmp_path, num_epochs=1, resume=True))
    assert len(list(tmp_path.glob("Pendulum-v1/mpo/*"))) == 2


def test_latest_resumable_run_skips_incomplete_checkpoints(tmp_path):
    for name, ckpt in [("20260101-000000", "checkpoint"), ("20260102-000000", "checkpoint.old"), ("20260103", "")]:
        (tmp_path / name).mkdir()
        if ckpt:
            (tmp_path / name / ckpt).mkdir()
            (tmp_path / name / ckpt / "state.pkl").write_bytes(b"")
    # a save interrupted before its state was written
    (tmp_path / "20260104-000000" / "checkpoint.tmp").mkdir(parents=True)
    assert latest_resumable_run(str(tmp_path)) == str(tmp_path / "20260102-000000")
