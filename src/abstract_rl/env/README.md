# Environment

`RolloutEnvWrapper` wraps the JAX environments of `jax_envs.py` (Gymnax, Brax, MJX/Playground) and runs a policy for a
given number of transitions. The rollout runs in E lanes inside one jitted loop (E follows from `policy_batch_size`
and `max_steps`), and the observation statistics live in the traced model state.

`Humanoid-v5` is the exception: it is Gymnasium's MuJoCo env stepped on the host (`host_gym.py`, the `gym` extra),
because an MJX or Brax humanoid is not Gymnasium v5. The MPO learner is still the vmapped JAX update. The rollout calls
`execute_policy_host`; `execute_policy` stays the jitted path used by every other env.

```
env = RolloutEnvWrapper(mc)
state, tc, info = env.execute_policy(state, key, policy, max_steps, policy_batch_size, exploration=True)
```

`state` is the traced `ModelState`, `key` a JAX PRNG key, and `tc` the resulting `TrajectoryCollection` with one
episode per lane.
