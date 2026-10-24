"""Custom Metal kernels for steps that are bound by the number of MLX operations, not by the GPU.

Each MLX operation costs about 1-2 us of host time to encode, so loops of many small steps (Adam on the temperature
dual) spend most of their time encoding. The kernels here do the work of such a step in one or two dispatches.
"""

import math
from functools import cache
from typing import Any

import mlx.core as mx

_THREADS = 256

# Per state s (one thread per row of the centered Q values qc = Q - max_a Q, so the row maximum is 0):
#   l_s = log sum_a exp(qc / eta),  e_s = sum_a softmax_a(qc / eta) qc,
# summed over the threadgroup into partials [2 * threadgroups].
_DUAL_ROWS = """
    uint gid = thread_position_in_grid.x;
    uint sid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    float inv_eta = metal::exp(-state[0]);
    float l = 0.0f;
    float e = 0.0f;
    if (gid < uint(N)) {
        float s = 0.0f;
        float w = 0.0f;
        const device float* row = qc + gid * A;
        for (int a = 0; a < A; ++a) {
            float x = row[a];
            float ex = metal::exp(x * inv_eta);
            s += ex;
            w += ex * x;
        }
        l = metal::log(s);
        e = w / s;
    }
    threadgroup float shared[64];
    l = simd_sum(l);
    e = simd_sum(e);
    if (lane == 0) {
        shared[2 * sid] = l;
        shared[2 * sid + 1] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        uint groups = threads_per_threadgroup.x / threads_per_simdgroup;
        float l2 = lane < groups ? shared[2 * lane] : 0.0f;
        float e2 = lane < groups ? shared[2 * lane + 1] : 0.0f;
        l2 = simd_sum(l2);
        e2 = simd_sum(e2);
        if (lane == 0) {
            partials[2 * threadgroup_position_in_grid.x] = l2;
            partials[2 * threadgroup_position_in_grid.x + 1] = e2;
        }
    }
"""

# Sums the partials and does one Adam step (torch.optim.Adam, betas 0.9, 0.999, eps 1e-8) on the log temperature.
# state = [log_eta, m, v, dual, step], consts = [eps, lr, log(A), mean_s max_a Q]. With eta = exp(log_eta):
#   dual = eta eps + mean max Q + eta (mean_s l_s - log A),
#   d dual / d log_eta = eta (eps + mean_s l_s - log A) - mean_s e_s
_DUAL_ADAM = """
    uint lid = thread_position_in_threadgroup.x;
    uint sid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    float l = 0.0f;
    float e = 0.0f;
    for (uint i = lid; i < uint(P); i += threads_per_threadgroup.x) {
        l += partials[2 * i];
        e += partials[2 * i + 1];
    }
    threadgroup float shared[64];
    l = simd_sum(l);
    e = simd_sum(e);
    if (lane == 0) {
        shared[2 * sid] = l;
        shared[2 * sid + 1] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint groups = threads_per_threadgroup.x / threads_per_simdgroup;
        float l_sum = 0.0f;
        float e_sum = 0.0f;
        for (uint k = 0; k < groups; ++k) {
            l_sum += shared[2 * k];
            e_sum += shared[2 * k + 1];
        }
        float log_eta = state[0];
        float t = state[4] + 1.0f;
        float eta = metal::exp(log_eta);
        float eps = consts[0];
        float mean_l = l_sum / float(N) - consts[2];
        float grad = eta * (eps + mean_l) - e_sum / float(N);
        float m = 0.9f * state[1] + 0.1f * grad;
        float v = 0.999f * state[2] + 0.001f * grad * grad;
        float denom = metal::sqrt(v) / metal::sqrt(1.0f - metal::pow(0.999f, t)) + 1e-8f;
        out[0] = log_eta - (consts[1] / (1.0f - metal::pow(0.9f, t))) * m / denom;
        out[1] = m;
        out[2] = v;
        out[3] = eta * eps + consts[3] + eta * mean_l;
        out[4] = t;
    }
"""


@cache
def _dual_kernels() -> tuple[Any, Any]:
    """The kernels are built on first use, MLX without Metal (Linux) can import this module."""
    rows = mx.fast.metal_kernel(
        name="temperature_dual_rows", input_names=["qc", "state"], output_names=["partials"], source=_DUAL_ROWS
    )
    adam = mx.fast.metal_kernel(
        name="temperature_dual_adam",
        input_names=["partials", "state", "consts"],
        output_names=["out"],
        source=_DUAL_ADAM,
    )
    return rows, adam


def temperature_dual_adam(
    q_centered: mx.array, mean_max_q: mx.array, eps: float, lr: float, eta0: float, steps: int
) -> tuple[mx.array, mx.array]:
    """
    Adam on the log temperature of the dual of temperature_dual on the GPU, two dispatches per step instead of the
    operations of the autodiff graph. Same update as _adam_dual_step, the sums run in another order (float noise).
    :param q_centered: A float32 tensor [NxA] of Q(s, a) - max_a Q(s, a).
    :param mean_max_q: The mean over the states of max_a Q(s, a).
    :param eps: The KL bound.
    :param lr: The Adam learning rate.
    :param eta0: The initial temperature.
    :param steps: The number of Adam steps.
    :return: The log temperature and the dual of the last step (before its update), arrays of shape [1].
    """
    n, a = q_centered.shape
    groups = (n + _THREADS - 1) // _THREADS
    consts = mx.concatenate([mx.array([eps, lr, math.log(a)], dtype=mx.float32), mean_max_q.reshape(1)])
    state = mx.array([math.log(eta0), 0.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    dual_rows, dual_adam = _dual_kernels()
    for _ in range(steps):
        (partials,) = dual_rows(
            inputs=[q_centered, state],
            template=[("N", n), ("A", a)],
            grid=(groups * _THREADS, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(2 * groups,)],
            output_dtypes=[mx.float32],
        )
        (state,) = dual_adam(
            inputs=[partials, state, consts],
            template=[("N", n), ("P", groups)],
            grid=(_THREADS, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(5,)],
            output_dtypes=[mx.float32],
        )
    return state[0:1], state[3:4]
