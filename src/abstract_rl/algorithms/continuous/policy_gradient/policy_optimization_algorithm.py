from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration


class PolicyOptimizationAlgorithm:
    """
    Represents a policy optimization algorithm. It is splitted into two phases for the  scheme from the original
    papers of relative entropy policy search and maximum a posteriori policy optimization. Note that algorithms like
    trust region policy optimization or proximal policy optimization can be implemented efficiently.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        """
        Initializes a new policy approximation algorithm. It makes the model config and from that also the running
        config available to the derived class itself.
        :param mc: The model configuration, containing all important elements, like env or similar.
        """
        assert isinstance(mc, ModelConfiguration)
        self.mc = mc
        self.conf = mc["conf"]
