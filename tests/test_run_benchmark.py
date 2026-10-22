import json

import pytest

from abstract_rl.scripts import run_benchmark


def test_benchmark_cell_and_report(tmp_path):
    root = tmp_path / "bench"
    args = ["--envs", "pendulum", "--algs", "trpo", "--seeds", "1", "2", "--root", str(root), "--episodes", "1"]
    args += ["--set", "max_env_steps=400", "alg.val_opt_steps=5"]
    run_benchmark.main(args)
    runs = sorted((root / "runs" / "pendulum_trpo" / "Pendulum-v1" / "trpo").iterdir())
    assert len(runs) == 2 and all((r / "eval.json").exists() for r in runs)
    timing = json.loads((root / "timing" / "pendulum_trpo_seeds1-2.json").read_text())
    assert timing["env_steps"] == [400, 400] and timing["env_steps_per_second"] > 0

    # runs of a requested seed are only replaced with --fresh, other seeds of the cell can be added
    with pytest.raises(SystemExit):
        run_benchmark.main(args)
    run_benchmark.main([*args, "--fresh"])
    assert len(list((root / "runs" / "pendulum_trpo" / "Pendulum-v1" / "trpo").iterdir())) == 2

    run_benchmark.main(["--report", "--root", str(root), "--out", str(tmp_path / "docs")])
    assert len(json.loads((root / "timing.json").read_text())) == 1
    table = (tmp_path / "docs" / "benchmark.md").read_text()
    assert "| Pendulum-v1 (≥ -150) | TRPO (abstract_rl) | 2 | 400 |" in table
    assert (tmp_path / "docs" / "benchmark.png").exists()
