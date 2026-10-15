"""Temperature of the non-parametric E-step."""

from typing import Any

import numpy as np
import optax

from abstract_rl.algorithms.mixed.non_parametric_trust_region_optimizatioon import (
    positive_temperature,
    temperature_raw,
)
from abstract_rl.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.misc.cli_printer import CliPrinter


class NonParametricVariationalPolicyBuilder:
    """Fits the temperature η of the non-parametric variational distribution on the replay.

    η is ``softplus`` of an unconstrained parameter. Each M-step minibatch takes one Adam step
    on the dual, and this object keeps that parameter and its Adam state across epochs.
    """

    def __init__(self, mc: ModelConfiguration) -> None:
        self.mc = mc
        conf = mc["conf"]
        self.non_parametric_variational_eps = conf["eps"]
        self.non_parametric_variational_lr = conf["lr"]
        self.non_parametric_variational_init_eta = conf["init_eta"]
        self.eta_raw: Any = temperature_raw(float(self.non_parametric_variational_init_eta))
        self.tx_eta: optax.GradientTransformation = optax.adam(float(self.non_parametric_variational_lr))
        self.eta_opt: Any = self.tx_eta.init(self.eta_raw)
        self.eta = np.asarray([float(positive_temperature(self.eta_raw))], dtype=np.float32)
        self.log = self.mc["log"]
        self.log.create_field("eta", 1)

    def calc_eta(self) -> None:
        """Log the temperature left by the previous M-step.

        The dual steps run on each M-step minibatch. This does not start a new solve.
        """
        current = float(positive_temperature(self.eta_raw))
        self.eta = np.asarray([current], dtype=np.float32)
        self.log.log({"eta": self.eta.reshape(-1)})
        CliPrinter().instance.print(f"eta = {current:.6f} (sgd dual, updated in the M-step)")
