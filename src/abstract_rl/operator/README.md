# Operator System

In general a operator system is used. The basic principle starts on the concept of trajectories. 
They are usually collected inside a TrajectoryCollection or the CircularTailMemory. A operator is
a function transforming a trajectory to  a special annotated one, e.g.

```
tc = TrajectoryCollection(...)
op = Operator(...)
op.transform_all(tc)
```

These annotations are saved in the trajectory and are stored until the next time an operation
is executed on that same trajectory. It is very easy to apply them in abstract scripts to define the
algorithmic logic. It is very easy to write a operator.

Here is an example of a simple filter operator, which can be used as described above to apply them to
trajectories.

```
class FilterOperator(TrajectoryOperator):
    """
    Represents a generalized advantage estimation operator based upon https://arxiv.org/abs/1506.02438.
    """

    def __init__(self, mc):
        """
        [...]

    def transform(self, trajectory):
        """
        Transform a trajectory with the current instance of the evaluation operator.
        :param trajectory: trajectory to transform.
        """

        # deactivate grad
        with torch.no_grad():

            if self.filter_states:
                if 'unfiltered_states' not in trajectory:
                    trajectory['unfiltered_states'] = trajectory._states
                    trajectory._states = self.env.state_filter(trajectory['unfiltered_states'], False)

            if self.filter_rewards:
                if 'unfiltered_rewards' not in trajectory:
                    trajectory['unfiltered_rewards'] = trajectory._rewards
                    trajectory._rewards = self.env.reward_filter(trajectory['unfiltered_rewards'], False)
```