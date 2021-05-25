import copy


class Syncable:
    """
    Sync able for use with the model configuration. This can be for example a neural network policy or
    a general value network.
    """

    def clone(self):
        """
        Clones the current instance.
        :return: A new identical clones instance of the current.
        """

        # first of all clone the module using the subclass and
        # then fill all hierarchically lower elements.
        v = copy.deepcopy(self)
        self.sync(v)

        return v

    def sync(self, v):
        """
        Copies the content of the current instance to the supplied parameter instance.
        :param v: The other instance, where the current values should be copied to.
        """
        raise NotImplementedError
