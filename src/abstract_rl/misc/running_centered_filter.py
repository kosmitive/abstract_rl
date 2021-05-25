import numpy as np


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
# https://github.com/ikostrikov/pytorch-trpo/blob/master/running_state.py
import torch

from abstract_rl.src.misc.cli_printer import CliPrinter


class RunningCenteredFilter:

    def __init__(self, name, dim, std_dev_noise=1e-8, clip=5):

        # build running variables
        self.cli = CliPrinter().instance
        self._n = 0
        self._dim = dim
        self._m = np.zeros(dim)
        self._s = np.zeros(dim)
        self._name = name
        self._clip = clip
        self._std_dev_noise = std_dev_noise

    def register(self, xs):
        if len(xs.shape) == 1: xs = xs.reshape([-1, self._dim])

        # for all datapoints
        for i in range(len(xs)):
            s = xs[i]
            self._n += 1

            # trivial
            if self._n == 1:
                self._m[...] = s

            # update
            else:
                old_mean = self._m.copy()
                self._m[...] = old_mean + (s - old_mean) / self._n
                self._s[...] = self._s + (s - old_mean) * (s - self._m)

    def __call__(self, x, update=True):
        if update: self.register(x)
        x = (x - self.mean) / (self.std + self._std_dev_noise)
        x = np.clip(x, -self._clip, self._clip)
        return x

    @property
    def mean(self):
        return self._m

    @property
    def var(self):
        return self._s / (self._n - 1) if self._n > 1 else self._m * self._m

    @property
    def std(self):
        return np.sqrt(self.var)

    def print_block(self):
        self.cli.empty().line(self._name).line()
        self.cli.print(f"mean \t\t=\t{self.mean}")
        self.cli.print(f"std \t\t=\t{self.std}")
