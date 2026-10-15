# Operator System

Operators transform trajectories stored in a `TrajectoryCollection`. Annotations stay on the
trajectory until the next operator runs.

```
tc = TrajectoryCollection(...)
op = Operator(...)
op.transform_all(tc)
```

A training loop is a sequence of these calls. The operators in this package are temporal difference,
generalised advantage estimation, Retrace, action resampling and the MPO E-step weights.
