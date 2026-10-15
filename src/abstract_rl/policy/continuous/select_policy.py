from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.policy.continuous.diagonal_gaussian_policy import DiagonalGaussianPolicy
from abstract_rl.policy.continuous.full_covariance_gaussian_policy import FullCovarianceGaussianPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def select_policy(
    mc: ModelConfiguration,
) -> UnivariateGaussianPolicy | DiagonalGaussianPolicy | FullCovarianceGaussianPolicy:
    """
    Creates the gaussian policy configured in the current namespace of the model configuration.

    The config key ``type`` (``policy_type`` on the top level) is ``univariate``, ``diagonal``,
    ``full`` or ``auto`` (default): auto uses the univariate policy for one dimensional and the
    diagonal policy for multi dimensional actions. ``cov_type: full`` selects the Cholesky policy.
    :param mc: The model configuration with the environment already registered.
    :return: The new policy.
    """
    kind = mc["conf"].get("type", "auto")
    if kind == "full":
        mc["conf"]["cov_type"] = "full"
    if str(mc["conf"]["cov_type"]) == "full":
        return FullCovarianceGaussianPolicy(mc)
    if kind == "auto":
        kind = "univariate" if mc.env.action_dim == 1 else "diagonal"
    if kind == "univariate":
        return UnivariateGaussianPolicy(mc)
    if kind == "diagonal":
        return DiagonalGaussianPolicy(mc)
    raise NotImplementedError(f"unknown policy type {kind}")
