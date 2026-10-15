# Policy

There are two types of policies currently implemented.

# Continuous Policy

Three Gaussian policies share the `ContinuousPolicy` interface (all can be squashed with tanh into the action bounds):
`UnivariateGaussianPolicy` for one-dimensional actions, `DiagonalGaussianPolicy` for `action_dim > 1` (independent
dimensions, per-dimension bounds, log-density summed over dimensions with the tanh Jacobian per dimension), and
`FullCovarianceGaussianPolicy`, a state-dependent Cholesky covariance. MPO defaults to the Cholesky policy.
`select_policy(mc)` uses that policy when `cov_type` is `full`, and otherwise picks the univariate or diagonal
policy from the action space. There are various methods
implemented up on it. For example the KL-divergence, Entropy and several other methods. Sometimes it makes 
sense that these methods are overwritten by special functions, e.g. when the KL-divergence can
be solved in a closed solution, as it is for Gaussians. The included EnvironmentWrapper accepts a policy
executes it and returns the obtained results.

# Non Parametric Policy 

This is used by the E-Step in MPO. It basically consists of samples and the Non Parametric Policy Builder.

