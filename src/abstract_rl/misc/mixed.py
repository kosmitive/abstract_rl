import numpy as np
import torch
import torch.nn as nn


def default_settings():
    """
    Apply default settings to all.
    :return:
    """
    float_formatter = lambda x: "% 10.4f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})
    #torch.set_printoptions(profile='full')


def select_activation(name):
    """
    Select a activation based on the name.
    :param name: The name of the activation, e.g. 'prelu'.
    :return: A object of the activation.
    """
    if name == 'none' or name is None:
        return nn.Sequential()

    elif name.lower() == 'softplus':
        return nn.Softplus()

    elif name.lower() == 'elu':
        return nn.ELU()

    elif name.lower() == 'prelu':
        return nn.PReLU()

    elif name.lower() == 'relu':
        return nn.ReLU()

    elif name.lower() == 'lrelu':
        return nn.LeakyReLU()

    elif name.lower() == 'tanh':
        return nn.Tanh()

    elif name == 'sigmoid':
        return nn.Sigmoid()
