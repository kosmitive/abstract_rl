# Circular Tail Memory (NOT USED).

 It is important to note that trajectories, can be saved in a somewhat efficient fashion in a 
ring buffer. To support a high performance architecture, one gets direct write access to the memory.

A new memory can be created by:

```
from abstract_rl.data_structures.memory.circular_tail_memory import CircularTailMemory

conf = {}

[...]

# Somewhere in advance an environment has to be set. Therefore reference to the
# README in the '<abstract_rl>/abstract_rl/env folder'. Unless this is specified
# it is not possible to create the memory.

[...]


# add this to the shared configuration
mem_config = {

    # create off policy memory
    'mem_size': 100000,
    'mem_bootstrap_steps': 4
    
}
conf.update(mem_config)

# create a model configuration
mc = ModelConfiguration(conf)

# create the memory and get some settings
with conf.namespace('mem'):
    mem = CircularTailMemory(mc)
    mc.add_main('mem', mem)
```

