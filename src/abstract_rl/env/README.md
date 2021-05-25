# Environment

In order to speed up the handling of the creation of executed policy a wrapper for 
gym environments is introduced it can be used to execute a policy for a given number of time steps.


```
[...]

# create an environment wrapper
with conf.ns('env'):
    num_traj = conf['traj_evaluations']
    env = MCMCEnvWrapper(mc)
    mc.add_environment(env)

[...]

tc = env.execute_policy(policy, max_steps, policy_batch_size, exploration=True, render=False, rew_field_name=stoch_reward)
    
[...]
```