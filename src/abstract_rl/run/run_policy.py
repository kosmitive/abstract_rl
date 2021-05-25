import sys
sys.path.append('.')

import json
from os.path import join

import torch

from abstract_rl.src.data_structures.abstract_conf.hierarchical_conf import SharedConf
from abstract_rl.src.data_structures.abstract_conf.model_configuration import ModelConfiguration
from abstract_rl.src.env.mcmc_env_wrapper import MCMCEnvWrapper
from abstract_rl.src.policy.continuous.univariate_gaussian_policy import UnivariateGaussianPolicy


def run_policy(dir):
    """
    Runs the passed config off policy.
    :param conf: The run configuration.
    """

    # create a model configuration

    # load the config as a json file
    with open(join(dir, "run.conf"), "r") as file_conf:
        file_str = file_conf.read()
        conf = json.loads(file_str)

    mc = ModelConfiguration(conf)
    conf = mc['conf']

    # create an environment wrapper
    with conf.ns('env'):
        env = MCMCEnvWrapper(mc)
        mc.add_environment(env)

    # policy neural_modules
    with conf.ns('policy'):
        policy = UnivariateGaussianPolicy(mc)
        policy.load_state_dict(torch.load(join(dir, "model/last", f"policy.data")))

    while True: env.execute_policy_once(10000, policy, True, True)


run_policy('best/CartpoleSwingLong-v0/mpo/saved')