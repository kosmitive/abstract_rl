# Run

There are two training loops, one for Trust Region Policy Optimization (`run_trpo.py`) and one for Maximum A
Posteriori Policy Optimization (`run_mpo_non_parametric.py`):

The MPO loop consists of two steps, in the first one (E-step) additional actions are sampled from the target policy,
evaluated with the target q function and reweighted non-parametrically. Afterwards the current policy is fitted in a
supervised fashion to the reweighted samples (M-step) and the q function is updated off-policy using retrace.

These loops are used by the configurations in the scripts folder.

# Replay a stored policy

```
uv run python -m abstract_rl.run.run_policy data/Pendulum-v1/mpo/<run> --episodes 10 [--render]
```
