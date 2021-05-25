class NonParametricPolicy:
    """
    Represents a non parametric policy, which basically saves only samples.
    """

    def __init__(self, samples):
        """
        Initializes a new non parametric policy.
        :param samples: The samples to add to the policy
        """
        self.samples = samples


