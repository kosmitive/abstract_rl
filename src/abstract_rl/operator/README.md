# Operator System

The operator system builds on trajectories. They are collected in a `TrajectoryCollection`, which stacks E lanes (one
episode each) as JAX arrays. An operator transforms one lane into an annotated one; `transform_all` vmaps it over all
lanes, e.g.

```
op = Operator(...)
tc = op.transform_all(state, tc)
```

`state` is the traced `ModelState` (parameters, filters, statistics). Operators are pure: they return a new collection
instead of mutating, so they can be used inside jitted code and composed in the algorithm scripts.

Here is a sketch of a simple operator. Only `transform` for a single lane needs to be written:

```
class MyOperator(TrajectoryOperator):
    def transform(self, state: ModelState, trajectory: Trajectory) -> Trajectory:
        # trajectory holds one lane; return the annotated trajectory
        ...
```
