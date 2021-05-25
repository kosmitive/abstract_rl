# Configurations

The model or run configuration is given as a dictionary encapsulated inside of a hierarchical 
configuration. Let's see how to use it to access several lines of a configuration:

```
c = {

    # environment settings
    'env_name': 'Qube-v0',
    'env_angle_to_sin_cos': 0,

    'ret_add_acts': 20,
    'res_add_acts': 20,
}

sc = SharedConf(c)
print(sc['env_name]) # Qube-v0

with sc.ns('env'):
    print(sc['name']) # Qube-v0
```

# Shared Memory / Model Configuration

In general important elements of a run can be accessed via the Model Configuration. Note that
there is a specific type of objects called Syncable, which are automatically synced at the 
given rate, when sync() is called.

```
    mc = ModelConfiguration(conf)
    
    # create an environment wrapper
    with conf.ns('env'):
        num_traj = conf['traj_evaluations']
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)

    # create the memory and get some settings
    with conf.ns('tc'):
        tc_size = conf['size']
        tc = TrajectoryCollection(env, tc_size)
        mc.add_main('tc', tc)

    # policy neural_modules
    with conf.ns('policy'):
        policy_batch_size = conf['batch_size']
        sync_steps = conf['sync_steps']
        policy = UnivariateGaussianPolicy(mc) # select_policy(mc)
        mc.add_syncable('policy', policy, sync_steps)
```