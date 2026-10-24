"""Custom Metal kernels for steps that are bound by the number of MLX operations, not by the GPU.

Each MLX operation costs about 1-2 us of host time to encode, so loops of many small steps (Adam on the temperature
dual) spend most of their time encoding. The kernels here do the work of such a step in one or two dispatches.
"""

import math
from functools import cache
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map_with_path

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


# Parameters per dispatch of FusedAdam, each takes 7 buffers (parameter, gradient, moments in and out) and a kernel
# takes at most 31
_ADAM_GROUP = 3


def _adam_source(sizes: tuple[int, ...]) -> str:
    """
    The update of Adam (bias corrected) of the flattened parameters p0, p1, ... with the given sizes, one element per
    thread. The operations and their order are the ones of the compiled kernel of optim.Adam.apply_single, one
    statement per operation, so the results are the same bits.
    """
    lines = [
        "uint i = thread_position_in_grid.x;",
        "float b1 = consts[0], ob1 = consts[1], b2 = consts[2], ob2 = consts[3], eps = consts[4];",
        "float c1 = consts[5], c2 = consts[6];",
    ]
    offset = 0
    for k, size in enumerate(sizes):
        lines += [
            f"{'if' if k == 0 else 'else if'} (i < {offset + size}u) {{",
            f"    uint j = i - {offset}u;",
            f"    float g = g{k}[j];",
            f"    float t0 = b1 * m{k}[j];",
            "    float t1 = ob1 * g;",
            "    float m = t0 + t1;",
            f"    float t2 = b2 * v{k}[j];",
            "    float t3 = g * g;",
            "    float t4 = ob2 * t3;",
            "    float v = t2 + t4;",
            "    float num = c1 * m;",
            "    float t5 = metal::precise::sqrt(v);",
            "    float t6 = t5 * c2;",
            "    float den = t6 + eps;",
            "    float upd = num / den;",
            f"    m_out{k}[j] = m;",
            f"    v_out{k}[j] = v;",
            f"    p_out{k}[j] = p{k}[j] - upd;",
            "}",
        ]
        offset += size
    return "\n".join(lines)


@cache
def _adam_kernel(sizes: tuple[int, ...]) -> Any:
    n = len(sizes)
    return mx.fast.metal_kernel(
        name="fused_adam_" + "_".join(str(s) for s in sizes),
        input_names=[f"{t}{k}" for t in "pgmv" for k in range(n)] + ["consts"],
        output_names=[f"{t}_out{k}" for t in "pmv" for k in range(n)],
        source=_adam_source(sizes),
    )


class FusedAdam(optim.Adam):
    """
    optim.Adam with the update of all float32 parameters in a few Metal dispatches instead of one per parameter and
    the operations around it, the results are the same bits. Each MLX operation costs host time to encode, which
    bounds loops of many small Adam steps. Other devices, dtypes and settings use optim.Adam.
    """

    def apply_gradients(self, gradients: dict, parameters: dict) -> Any:
        grads: list[tuple[str, mx.array]] = tree_flatten(gradients)  # type: ignore[assignment]
        if (
            not self.bias_correction
            or self._schedulers
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
            or any(g.dtype != mx.float32 for _, g in grads)
        ):
            return super().apply_gradients(gradients, parameters)
        if not self._initialized:
            self.init(gradients)
        self.state["step"] = self.step + 1

        keys = [k for k, _ in grads]
        flat_grads = dict(grads)
        flat_params = dict(tree_flatten(parameters))
        flat_state = dict(tree_flatten(self.state))
        # the scalars of apply_single, computed with the same operations
        lr = self.learning_rate.astype(mx.float32)
        b1, b2 = self.betas
        step = self.step
        c1 = (lr / (1 - b1**step)).astype(mx.float32)
        c2 = mx.rsqrt(1 - b2**step).astype(mx.float32)
        consts = mx.concatenate(
            [mx.array([b1, 1 - b1, b2, 1 - b2, self.eps], dtype=mx.float32), c1.reshape(1), c2.reshape(1)]
        )

        new: dict[str, mx.array] = {}
        for a in range(0, len(keys), _ADAM_GROUP):
            group = keys[a : a + _ADAM_GROUP]
            n = len(group)
            g = [flat_grads[k] for k in group]
            sizes = tuple(x.size for x in g)
            outs = _adam_kernel(sizes)(
                inputs=[flat_params[k] for k in group]
                + g
                + [flat_state[k + ".m"] for k in group]
                + [flat_state[k + ".v"] for k in group]
                + [consts],
                grid=(sum(sizes), 1, 1),
                threadgroup=(_THREADS, 1, 1),
                output_shapes=[x.shape for x in g] * 3,
                output_dtypes=[mx.float32] * (3 * n),
            )
            for i, k in enumerate(group):
                new[k] = outs[i]
                new[k + ".m"] = outs[n + i]
                new[k + ".v"] = outs[2 * n + i]

        def set_moments(path: str, _: mx.array, state: dict) -> None:
            state["m"] = new[path + ".m"]
            state["v"] = new[path + ".v"]

        # the moments of a parameter are the leaves m and v of the dict at its path in the state
        tree_map_with_path(set_moments, gradients, self.state)
        return tree_map_with_path(lambda path, _: new[path], gradients)
