"""Reference integrator — the algorithm the GPU runs, in float64 numpy.

State convention (identical on the GPU):
  * the Sun is integrated on its own in float64 (in the browser: JS doubles);
  * every star is stored RELATIVE to the Sun: Δx = x − x☉, Δv = v − v☉,
    in galactocentric axes (astropy v4.0), pc and pc/Myr;
  * one step of kick-drift-kick leapfrog with step h (h = ±DT_MYR):
        Δv += h/2 · (a(x☉ₖ + Δx) − a☉ₖ)
        Δx += h · Δv
        Δv += h/2 · (a(x☉ₖ₊₁ + Δx) − a☉ₖ₊₁)
    which is algebraically identical to leapfrogging star and Sun separately
    and subtracting, but keeps nearby stars precise in float32;
  * the state at time t is DEFINED as n = ⌊|t|/DT⌋ steps from t = 0 followed
    by a cubic Hermite interpolation to the fractional step — so every visitor
    to the same deep link sees the same sky, whatever path they scrubbed.
For |t| < DT the very first step is a pure straight line to first order, so
the short-span straight-line check is a property of the scheme, not a
separate code path.
"""
import numpy as np

from . import config as C
from .potential import interp_accel, table_float32

KV = C.KMS_TO_PCMYR

_TAB = None


def table():
    global _TAB
    if _TAB is None:
        _TAB = table_float32().astype(np.float64)
    return _TAB


def accel(pos):
    return interp_accel(pos, table())


def sun_orbit(sun_pos, sun_vel_kms, n_steps, h):
    """Sun's grid states: pos[k], vel[k] (pc/Myr), acc[k] at t = k·h."""
    x = np.array(sun_pos, float)
    v = np.array(sun_vel_kms, float) * KV
    a = accel(x)
    P, V, A = [x.copy()], [v.copy()], [a.copy()]
    for _ in range(n_steps):
        v = v + 0.5 * h * a
        x = x + h * v
        a = accel(x)
        v = v + 0.5 * h * a
        P.append(x.copy()); V.append(v.copy()); A.append(a.copy())
    return np.array(P), np.array(V), np.array(A)


def step_rel(dx, dv, sun_k, sun_k1, h):
    """One KDK step of Sun-relative states. sun_k = (pos, acc) at the step ends."""
    dv = dv + 0.5 * h * (accel(sun_k[0] + dx) - sun_k[1])
    dx = dx + h * dv
    dv = dv + 0.5 * h * (accel(sun_k1[0] + dx) - sun_k1[1])
    return dx, dv


def hermite(xa, va, xb, vb, h, f):
    """Cubic Hermite position/velocity at fraction f ∈ [0,1] of a step of length h."""
    f2, f3 = f * f, f * f * f
    h00, h10, h01, h11 = 2 * f3 - 3 * f2 + 1, f3 - 2 * f2 + f, -2 * f3 + 3 * f2, f3 - f2
    x = h00 * xa + h10 * h * va + h01 * xb + h11 * h * vb
    d00, d10, d01, d11 = 6 * f2 - 6 * f, 3 * f2 - 4 * f + 1, -6 * f2 + 6 * f, 3 * f2 - 2 * f
    v = (d00 * xa + d01 * xb) / h + d10 * va + d11 * vb
    return x, v


class Model:
    """Sun + Sun-relative stars, evaluated at arbitrary t (Myr)."""

    def __init__(self, sun_pos, sun_vel_kms, t_max=C.T_MAX_MYR, dt=C.DT_MYR):
        self.dt = dt
        self.n = int(np.ceil(t_max / dt)) + 1
        self.fwd = sun_orbit(sun_pos, sun_vel_kms, self.n, +dt)
        self.bwd = sun_orbit(sun_pos, sun_vel_kms, self.n, -dt)

    def sun(self, k, sign):
        P, V, A = self.fwd if sign > 0 else self.bwd
        return P[k], V[k], A[k]

    def sun_at(self, t):
        sign = 1 if t >= 0 else -1
        n, f = divmod(abs(t), self.dt)
        n = int(n)
        P, V, _ = self.fwd if sign > 0 else self.bwd
        return hermite(P[n], V[n], P[n + 1], V[n + 1], sign * self.dt, f / self.dt)

    def stars_at(self, t, dx0, dv0_kms, return_vel=False):
        """Sun-relative positions (pc) [and velocities, pc/Myr] at time t."""
        sign = 1 if t >= 0 else -1
        h = sign * self.dt
        n, f = divmod(abs(t), self.dt)
        n = int(n)
        dx, dv = np.array(dx0, float), np.array(dv0_kms, float) * KV
        for k in range(n):
            dx, dv = self._step(dx, dv, k, sign, h)
        xb, vb = self._step(dx, dv, n, sign, h)
        x, v = hermite(dx, dv, xb, vb, h, f / self.dt)
        return (x, v) if return_vel else x

    def trajectory(self, dx0, dv0_kms, t_end, samples_per_step=50):
        """Dense Sun-relative trajectory 0 → t_end (for closest-approach searches)."""
        sign = 1 if t_end >= 0 else -1
        h = sign * self.dt
        n = int(np.ceil(abs(t_end) / self.dt))
        dx, dv = np.array(dx0, float), np.array(dv0_kms, float) * KV
        T, X = [], []
        fs = np.linspace(0, 1, samples_per_step, endpoint=False)
        for k in range(n):
            xb, vb = self._step(dx, dv, k, sign, h)
            for f in fs:
                x, _ = hermite(dx, dv, xb, vb, h, f)
                T.append(sign * (k + f) * self.dt)
                X.append(x)
            dx, dv = xb, vb
        return np.array(T), np.array(X)

    def _step(self, dx, dv, k, sign, h):
        pk, _, ak = self.sun(k, sign)
        pk1, _, ak1 = self.sun(k + 1, sign)
        return step_rel(dx, dv, (pk, ak), (pk1, ak1), h)
