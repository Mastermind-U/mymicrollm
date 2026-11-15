"""Experimental Taichi implementation of the manually differentiated model.

All activations and gradients stay on the selected Taichi device. Matrix
products use simple kernels intended as a correctness baseline, not a tuned
replacement for a vendor BLAS.
"""

import math
from typing import Any

import numpy as np
import taichi as ti
from numpy.typing import NDArray

from microllm.model import ModelConfig, Transformer


@ti.kernel
def _mm(  # noqa: ANN202
    a: ti.types.ndarray(dtype=ti.f32, ndim=2),
    b: ti.types.ndarray(dtype=ti.f32, ndim=2),
    c: ti.types.ndarray(dtype=ti.f32, ndim=2),
    m: ti.i32,
    n: ti.i32,
    k: ti.i32,
    ta: ti.i32,
    tb: ti.i32,
):
    for i, j in ti.ndrange(m, n):
        s = 0.0
        for z in range(k):
            av = 0.0
            bv = 0.0
            if ta:  # noqa: SIM108
                av = a[z, i]
            else:
                av = a[i, z]
            if tb:  # noqa: SIM108
                bv = b[j, z]
            else:
                bv = b[z, j]
            s += av * bv
        c[i, j] = s


@ti.kernel
def _add(  # noqa: ANN202
    a: ti.types.ndarray(dtype=ti.f32, ndim=2),
    b: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for i, j in out:
        out[i, j] = a[i, j] + b[i, j]


@ti.kernel
def _embed(  # noqa: ANN202
    tokens: ti.types.ndarray(dtype=ti.i32, ndim=2),
    emb: ti.types.ndarray(dtype=ti.f32, ndim=2),
    pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    length: ti.i32,
):
    for i, j in out:
        out[i, j] = (
            emb[tokens[i // length, i % length], j] + pos[i % length, j]
        )


@ti.kernel
def _embed_backward(  # noqa: ANN202
    tokens: ti.types.ndarray(dtype=ti.i32, ndim=2),
    dx: ti.types.ndarray(dtype=ti.f32, ndim=2),
    de: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dp: ti.types.ndarray(dtype=ti.f32, ndim=2),
    length: ti.i32,
):
    for i, j in dp:
        dp[i, j] = 0.0
    for i, j in dx:
        ti.atomic_add(de[tokens[i // length, i % length], j], dx[i, j])
        ti.atomic_add(dp[i % length, j], dx[i, j])


@ti.kernel
def _norm(  # noqa: ANN202
    x: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.types.ndarray(dtype=ti.f32, ndim=1),
    y: ti.types.ndarray(dtype=ti.f32, ndim=2),
    inv: ti.types.ndarray(dtype=ti.f32, ndim=1),
    eps: ti.f32,
    width: ti.i32,
):
    for i in range(x.shape[0]):
        s = 0.0
        for j in range(width):
            s += x[i, j] * x[i, j]
        inv[i] = ti.rsqrt(s / ti.cast(width, ti.f32) + eps)
    for i, j in y:
        y[i, j] = x[i, j] * inv[i] * w[j]


@ti.kernel
def _norm_backward(  # noqa: ANN202
    x: ti.types.ndarray(dtype=ti.f32, ndim=2),
    w: ti.types.ndarray(dtype=ti.f32, ndim=1),
    inv: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dy: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dx: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dw: ti.types.ndarray(dtype=ti.f32, ndim=1),
    width: ti.i32,
):
    for j in range(width):
        s = 0.0
        for i in range(x.shape[0]):
            s += dy[i, j] * x[i, j] * inv[i]
        dw[j] = s
    for i in range(x.shape[0]):
        p = 0.0
        for j in range(width):
            p += dy[i, j] * w[j] * x[i, j]
        p = p / ti.cast(width, ti.f32) * inv[i] * inv[i]
        for j in range(width):
            dx[i, j] = inv[i] * (dy[i, j] * w[j] - x[i, j] * p)


@ti.kernel
def _silu(  # noqa: ANN202
    gate: ti.types.ndarray(dtype=ti.f32, ndim=2),
    up: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for i, j in out:
        g = gate[i, j]
        out[i, j] = g / (1.0 + ti.exp(-g)) * up[i, j]


@ti.kernel
def _silu_backward(  # noqa: ANN202
    gate: ti.types.ndarray(dtype=ti.f32, ndim=2),
    up: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dy: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dg: ti.types.ndarray(dtype=ti.f32, ndim=2),
    du: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for i, j in dy:
        g = gate[i, j]
        s = 1.0 / (1.0 + ti.exp(-g))
        dg[i, j] = dy[i, j] * up[i, j] * s * (1.0 + g * (1.0 - s))
        du[i, j] = dy[i, j] * g * s


@ti.kernel
def _attention(  # noqa: ANN202
    q: ti.types.ndarray(dtype=ti.f32, ndim=2),
    k: ti.types.ndarray(dtype=ti.f32, ndim=2),
    v: ti.types.ndarray(dtype=ti.f32, ndim=2),
    p: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    length: ti.i32,
    heads: ti.i32,
    hd: ti.i32,
    scale: ti.f32,
):
    for r in range(p.shape[0]):
        b = r // (heads * length)
        h = (r // length) % heads
        t = r % length
        maxval = -1.0e30
        for s in range(t + 1):
            score = 0.0
            for d in range(hd):
                score += (
                    q[b * length + t, h * hd + d]
                    * k[b * length + s, h * hd + d]
                )
            p[r, s] = score * scale
            maxval = ti.max(maxval, p[r, s])
        total = 0.0
        for s in range(length):
            val = 0.0
            if s <= t:
                val = ti.exp(p[r, s] - maxval)
            p[r, s] = val
            total += val
        for s in range(length):
            p[r, s] /= total
        for d in range(hd):
            val = 0.0
            for s in range(t + 1):
                val += p[r, s] * v[b * length + s, h * hd + d]
            out[b * length + t, h * hd + d] = val


@ti.kernel
def _attention_dp(  # noqa: ANN202
    dout: ti.types.ndarray(dtype=ti.f32, ndim=2),
    v: ti.types.ndarray(dtype=ti.f32, ndim=2),
    p: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ds: ti.types.ndarray(dtype=ti.f32, ndim=2),
    length: ti.i32,
    heads: ti.i32,
    hd: ti.i32,
    scale: ti.f32,
):
    for r in range(p.shape[0]):
        b = r // (heads * length)
        h = (r // length) % heads
        t = r % length
        dot = 0.0
        for s in range(length):
            val = 0.0
            for d in range(hd):
                val += (
                    dout[b * length + t, h * hd + d]
                    * v[b * length + s, h * hd + d]
                )
            ds[r, s] = val
            dot += val * p[r, s]
        for s in range(length):
            ds[r, s] = p[r, s] * (ds[r, s] - dot) * scale


@ti.kernel
def _attention_dqkv(  # noqa: ANN202
    q: ti.types.ndarray(dtype=ti.f32, ndim=2),
    k: ti.types.ndarray(dtype=ti.f32, ndim=2),
    p: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ds: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dout: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dq: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dk: ti.types.ndarray(dtype=ti.f32, ndim=2),
    dv: ti.types.ndarray(dtype=ti.f32, ndim=2),
    length: ti.i32,
    heads: ti.i32,
    hd: ti.i32,
):
    for row, col in dq:
        b = row // length
        t = row % length
        h = col // hd
        base = (b * heads + h) * length
        aq = 0.0
        ak = 0.0
        av = 0.0
        for s in range(length):
            aq += ds[base + t, s] * k[b * length + s, col]
            ak += ds[base + s, t] * q[b * length + s, col]
            av += p[base + s, t] * dout[b * length + s, col]
        dq[row, col] = aq
        dk[row, col] = ak
        dv[row, col] = av


@ti.kernel
def _cross_entropy(  # noqa: ANN202
    logits: ti.types.ndarray(dtype=ti.f32, ndim=2),
    target: ti.types.ndarray(dtype=ti.i32, ndim=2),
    grad: ti.types.ndarray(dtype=ti.f32, ndim=2),
    losses: ti.types.ndarray(dtype=ti.f32, ndim=1),
    length: ti.i32,
):
    for i in range(logits.shape[0]):
        mx = -1.0e30
        for j in range(logits.shape[1]):
            mx = ti.max(mx, logits[i, j])
        total = 0.0
        for j in range(logits.shape[1]):
            val = ti.exp(logits[i, j] - mx)
            grad[i, j] = val
            total += val
        correct = target[i // length, i % length]
        losses[i] = ti.log(total) + mx - logits[i, correct]
        for j in range(logits.shape[1]):
            grad[i, j] = (
                grad[i, j] / total - ti.cast(j == correct, ti.f32)
            ) / ti.cast(logits.shape[0], ti.f32)


class TaichiTransformer:
    """Naive GPU baseline preserving the NumPy model's equations and names."""

    def __init__(
        self,
        config: ModelConfig,
        parameters: dict[str, NDArray[np.float32]] | None = None,
    ) -> None:
        self.config = config
        if parameters is None:
            parameters = Transformer(config).parameters
        self.parameters = {
            name: self._from_numpy(value) for name, value in parameters.items()
        }
        self.gradients = {
            name: ti.ndarray(dtype=ti.f32, shape=value.shape)
            for name, value in parameters.items()
        }

    @staticmethod
    def _from_numpy(a: NDArray[np.float32]) -> Any:
        out = ti.ndarray(dtype=ti.f32, shape=a.shape)
        out.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return out

    def load_parameters(self, params: dict[str, NDArray[np.float32]]) -> None:
        if set(params) != set(self.parameters):
            raise ValueError("Parameter names do not match")
        for name, value in params.items():
            if tuple(value.shape) != tuple(self.parameters[name].shape):
                raise ValueError(f"Invalid shape for {name}")
            self.parameters[name].from_numpy(
                np.ascontiguousarray(value, dtype=np.float32)
            )

    def export_parameters(self) -> dict[str, NDArray[np.float32]]:
        return {
            name: value.to_numpy() for name, value in self.parameters.items()
        }

    @staticmethod
    def _empty(shape: tuple[int, ...]) -> Any:
        return ti.ndarray(dtype=ti.f32, shape=shape)

    def _mat(self, a: Any, b: Any, ta: bool = False, tb: bool = False) -> Any:
        m = a.shape[1] if ta else a.shape[0]
        k = a.shape[0] if ta else a.shape[1]
        n = b.shape[0] if tb else b.shape[1]
        out = self._empty((m, n))
        _mm(a, b, out, m, n, k, int(ta), int(tb))
        return out

    def _sum(self, a: Any, b: Any) -> Any:
        out = self._empty(a.shape)
        _add(a, b, out)
        return out

    def _norm(self, x: Any, weight: Any) -> tuple[Any, tuple[Any, Any]]:
        y = self._empty(x.shape)
        inv = self._empty((x.shape[0],))
        _norm(x, weight, y, inv, self.config.norm_eps, x.shape[1])
        return y, (x, inv)

    def _norm_back(
        self, dy: Any, weight: Any, cache: tuple[Any, Any], name: str
    ) -> Any:
        x, inv = cache
        dx = self._empty(x.shape)
        _norm_backward(
            x, weight, inv, dy, dx, self.gradients[name], x.shape[1]
        )
        return dx

    def _forward(
        self, tokens: NDArray[np.int64]
    ) -> tuple[Any, Any, Any, list[Any], Any]:
        if (
            tokens.ndim != 2
            or tokens.dtype.kind not in "iu"
            or not tokens.size
            or tokens.shape[1] > self.config.context_length
            or np.any(tokens < 0)
            or np.any(tokens >= self.config.vocab_size)
        ):
            raise ValueError("Invalid tokens")
        b, t = tokens.shape
        d, h = self.config.d_model, self.config.n_heads
        ids = ti.ndarray(dtype=ti.i32, shape=tokens.shape)
        ids.from_numpy(np.ascontiguousarray(tokens, dtype=np.int32))
        w = self.parameters
        x = self._empty((b * t, d))
        _embed(ids, w["token_embedding"], w["position_embedding"], x, t)
        caches = []
        for layer in range(self.config.n_layers):
            prefix = f"layers.{layer}."
            a, norm1 = self._norm(x, w[prefix + "attention_norm"])
            q, k, v = (
                self._mat(a, w[prefix + name])
                for name in ("query", "key", "value")
            )
            p = self._empty((b * h * t, t))
            combined = self._empty((b * t, d))
            _attention(
                q, k, v, p, combined, t, h, d // h, 1.0 / math.sqrt(d // h)
            )
            residual = self._sum(x, self._mat(combined, w[prefix + "output"]))
            n, norm2 = self._norm(residual, w[prefix + "feedforward_norm"])
            gate, up = (
                self._mat(n, w[prefix + "gate"]),
                self._mat(n, w[prefix + "up"]),
            )
            act = self._empty(gate.shape)
            _silu(gate, up, act)
            x = self._sum(residual, self._mat(act, w[prefix + "down"]))
            caches.append(
                (norm1, a, q, k, v, p, combined, norm2, n, gate, up, act)
            )
        normalized, final_norm = self._norm(x, w["final_norm"])
        return (
            self._mat(normalized, w["token_embedding"], tb=True),
            normalized,
            final_norm,
            caches,
            ids,
        )

    def _loss(
        self,
        tokens: NDArray[np.int64],
        targets: NDArray[np.int64],
        backward: bool,
    ) -> float:
        if (
            targets.shape != tokens.shape
            or targets.dtype.kind not in "iu"
            or np.any(targets < 0)
            or np.any(targets >= self.config.vocab_size)
        ):
            raise ValueError("Invalid targets")
        logits, normed, final_norm, caches, ids = self._forward(tokens)
        target = ti.ndarray(dtype=ti.i32, shape=targets.shape)
        target.from_numpy(np.ascontiguousarray(targets, dtype=np.int32))
        dl = self._empty(logits.shape)
        losses = self._empty((logits.shape[0],))
        _cross_entropy(logits, target, dl, losses, tokens.shape[1])
        loss = float(np.mean(losses.to_numpy(), dtype=np.float64))
        if not backward:
            return loss
        w, g = self.parameters, self.gradients
        _mm(
            dl,
            normed,
            g["token_embedding"],
            w["token_embedding"].shape[0],
            w["token_embedding"].shape[1],
            logits.shape[0],
            1,
            0,
        )
        dx = self._norm_back(
            self._mat(dl, w["token_embedding"]),
            w["final_norm"],
            final_norm,
            "final_norm",
        )
        t, h, d = tokens.shape[1], self.config.n_heads, self.config.d_model
        for layer in reversed(range(self.config.n_layers)):
            prefix = f"layers.{layer}."
            norm1, a, q, k, v, p, combined, norm2, n, gate, up, act = caches[
                layer
            ]
            _mm(
                act,
                dx,
                g[prefix + "down"],
                act.shape[1],
                dx.shape[1],
                dx.shape[0],
                1,
                0,
            )
            dact = self._mat(dx, w[prefix + "down"], tb=True)
            dg, du = self._empty(gate.shape), self._empty(up.shape)
            _silu_backward(gate, up, dact, dg, du)
            for name, delta in (("gate", dg), ("up", du)):
                _mm(
                    n,
                    delta,
                    g[prefix + name],
                    n.shape[1],
                    delta.shape[1],
                    n.shape[0],
                    1,
                    0,
                )
            dn = self._sum(
                self._mat(dg, w[prefix + "gate"], tb=True),
                self._mat(du, w[prefix + "up"], tb=True),
            )
            dr = self._sum(
                dx,
                self._norm_back(
                    dn,
                    w[prefix + "feedforward_norm"],
                    norm2,
                    prefix + "feedforward_norm",
                ),
            )
            _mm(
                combined,
                dr,
                g[prefix + "output"],
                combined.shape[1],
                dr.shape[1],
                dr.shape[0],
                1,
                0,
            )
            dc = self._mat(dr, w[prefix + "output"], tb=True)
            ds = self._empty(p.shape)
            _attention_dp(dc, v, p, ds, t, h, d // h, 1.0 / math.sqrt(d // h))
            dq, dk, dv = (
                self._empty(q.shape),
                self._empty(k.shape),
                self._empty(v.shape),
            )
            _attention_dqkv(q, k, p, ds, dc, dq, dk, dv, t, h, d // h)
            da = None
            for name, delta in (("query", dq), ("key", dk), ("value", dv)):
                _mm(
                    a,
                    delta,
                    g[prefix + name],
                    a.shape[1],
                    delta.shape[1],
                    a.shape[0],
                    1,
                    0,
                )
                part = self._mat(delta, w[prefix + name], tb=True)
                da = part if da is None else self._sum(da, part)
            dx = self._sum(
                dr,
                self._norm_back(
                    da,
                    w[prefix + "attention_norm"],
                    norm1,
                    prefix + "attention_norm",
                ),
            )
        _embed_backward(
            ids, dx, g["token_embedding"], g["position_embedding"], t
        )
        return loss

    def loss_and_gradients(
        self, tokens: NDArray[np.int64], targets: NDArray[np.int64]
    ) -> float:
        return self._loss(tokens, targets, True)

    def loss(
        self, tokens: NDArray[np.int64], targets: NDArray[np.int64]
    ) -> float:
        return self._loss(tokens, targets, False)
