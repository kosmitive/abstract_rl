"""Base class for the policy updates."""

from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration


class PolicyOptimizationAlgorithm:
    """Policy update. Subclasses implement the concrete step."""

    def __init__(self, mc: ModelConfiguration) -> None:
        assert isinstance(mc, ModelConfiguration)
        self.mc = mc
        self.conf = mc["conf"]
