import threading
from os.path import join
from typing import Any

import numpy as np
from numpy import float64, ndarray


class DataLogger:
    """
    Logs data to a csv file during the epochs
    """

    def __init__(self, fp: str) -> None:

        self._data: dict[str, list[Any]] = {}
        self._filepath = fp
        self._header: dict[str, int] = {}
        # the stale-policy rollout logs an epoch while the learner logs the temperature
        self._lock = threading.Lock()

    def create_field(self, name: str, dim: int) -> None:
        """Create a new field internally for logging.

        :param name: The name of the field.
        :param dim: The dimension to use.
        """
        self._header[name] = dim
        self._data[name] = []

    def has_field(self, name: str) -> bool:
        """Whether a field with this name was created."""
        return name in self._header

    def get(self, names: list[str], until: int) -> Any:
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

        res: Any = []
        for name in names:
            res.append(np.stack(self._data[name][:until], axis=0))

        if no_list:
            res = res[0]

        return res

    def log(self, vals: dict[str, ndarray | list[float64] | list[int]]) -> None:
        """
        Log the given values.
        :param vals: The values to log
        """

        with self._lock:
            for key, value in vals.items():
                vp_value = np.asarray(value)
                assert len(vp_value) == self._header[key]
                assert vp_value.ndim == 1
                self._data[key].append(value)

                path = join(self._filepath, f"{key}.txt")
                with open(path, "a") as handle:
                    np.savetxt(handle, value)
