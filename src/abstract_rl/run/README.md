# Run

There are basically three scripts to run the algorithms. One runs the Trust Region Policy Optimization, the
next one is then Maximum A Posteriori Policy Optimization:

This version consists of two steps, in the first one additional actions are sampled from the current policy and the current
policy is fitted in a supervised fashion to the value function. Afterwards the value function is updated in off policy fashion.

These scripts are used by the configurations in the scripts folder.

# MPO Best Policy Cartpole:

To start the best policy on Cartpole with MPO simply call:

```
python abstract_rl/src/run/run_policy.py
```