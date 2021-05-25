import os
from os.path import join

import numpy as np


class DataLogger:
    """
    Logs data to a csv file during the epochs
    """

    def __init__(self, fp):

        self._data = {}
        self._filepath = fp
        self._header = {}

    def create_field(self, name, dim):
        """Create a new field internally for logging.

        :param name: The name of the field.
        :param dim: The dimension to use.
        """
        self._header[name] = dim
        self._data[name] = []

    def get(self, names, until):
        """
        Obtain the given names until the given until value.
        :param names: The name to extract
        :param until: Until when should they be extracted.
        :return: The extracted values.
        """

        no_list = False
        if not isinstance(names, list):
            no_list = True
            names = []

        res = []
        for name in names:
            res.append(np.stack(self._data[name][:until], axis=0))

        if no_list:
            res = res[0]

        return res

    def log(self, vals):
        """
        Log the given values.
        :param vals: The values to log
        """

        for key, value in vals.items():
            vp_value = np.asarray(value)
            assert len(vp_value) == self._header[key]
            assert vp_value.ndim == 1
            self._data[key].append(value)

            path = join(self._filepath, f'{key}.txt')
            with open(path, 'a') as handle:
                np.savetxt(handle, value)