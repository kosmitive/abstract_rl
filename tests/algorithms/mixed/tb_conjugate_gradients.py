#  MIT License
#
#  Copyright (c) 2019 Markus Semmler
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
from copy import copy

import numpy as np
import pytest
import torch
import random
import torch.distributions as dist

from abstract_rl.src.algorithms.mixed.conjugate_gradient import conjugate_gradient


def test_mult_conjugate_gradients():

    n = 3
    A = torch.eye(n)
    b = torch.ones(n)
    _test_conjugate_gradients(A, b, 20, 1e-3)


def _test_conjugate_gradients(A, b, max_k=20, damping=1e-5):

    def fnA(v): return A @ v
    x = conjugate_gradient(fnA, b, max_k, damping)
    diff = torch.norm(fnA(x) - b)
    fmtstr = "%10i %10.3g %10.3g"
    assert diff < 1e-3
