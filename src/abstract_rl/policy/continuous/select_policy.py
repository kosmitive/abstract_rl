from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.policy.continuous.gaussian_policy import GaussianPolicy
from abstract_rl.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def select_policy(mc: ModelConfiguration) -> UnivariateGaussianPolicy | GaussianPolicy:
    """
    Creates the gaussian policy configured in the current namespace of the model configuration.

    The config key ``type`` (``policy_type`` on the top level) is ``univariate``, ``diagonal`` or ``auto`` (default):
    auto uses the univariate policy for one dimensional and the diagonal policy for multi dimensional actions.
    ``cov_type`` ``full`` is a state-dependent Cholesky factor and uses the diagonal policy.
    :param mc: The model configuration with the environment already registered.
    :return: The new policy.
    """
    if mc["conf"].get("cov_type", "single") == "full":
        return GaussianPolicy(mc)
    kind = mc["conf"].get("type", "auto")
    if kind == "auto":
        kind = "univariate" if mc.env.action_dim == 1 else "diagonal"
    if kind == "univariate":
        return UnivariateGaussianPolicy(mc)
    if kind == "diagonal":
        return GaussianPolicy(mc)
    raise NotImplementedError(f"unknown policy type {kind}")
