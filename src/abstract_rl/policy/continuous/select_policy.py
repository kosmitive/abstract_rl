from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.full_covariance_gaussian_policy import FullCovarianceGaussianPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def select_policy(
    mc: ModelConfiguration,
) -> UnivariateGaussianPolicy | DiagonalGaussianPolicy | FullCovarianceGaussianPolicy:
    """Create the Gaussian configured in the current namespace of the model configuration.

    ``cov_type`` (``policy_cov_type`` on the top level) is ``full``, ``single``, ``fixed`` or
    ``state``. ``full`` is the state-dependent Cholesky policy. The other three use the
    univariate policy when the action is one dimensional and the diagonal policy otherwise.

    Args:
        mc: The model configuration with the environment already registered.

    Returns:
        The new policy.
    """
    cov_type = str(mc["conf"]["cov_type"])
    if cov_type == "full":
        return FullCovarianceGaussianPolicy(mc)
    if cov_type not in ("single", "fixed", "state"):
        raise NotImplementedError(f"unknown covariance type {cov_type}")
    if mc.env.action_dim == 1:
        return UnivariateGaussianPolicy(mc)
    return DiagonalGaussianPolicy(mc)
