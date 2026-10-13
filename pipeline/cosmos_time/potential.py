"""McMillan (2017) Milky Way potential, tabulated for the GPU.

galpy evaluates McMillan17 through SCF/DiskSCF expansions at ~10 ms per force,
far too slow for a million particles. So we tabulate it once:

  1. evaluate the potential on a GRID_N x GRID_N grid that is uniform in
     u = asinh(R / A_R) and v = asinh(|z| / A_Z) — dense near the centre and
     the plane, sparse in the halo;
  2. fit a quintic spline to the (mirrored, so derivatives vanish on the axes)
     grid and differentiate it to get F_R and F_z at the nodes;
  3. at run time, every consumer (GPU shader, browser JS, this module) does the
     SAME bilinear interpolation of the force table in (u, v).

`interp_accel` below is the reference implementation of step 3; the shader in
time/js/potential.js and the JS twin must match it line for line.
Units throughout: pc, Myr, pc/Myr, pc/Myr^2.
"""
import os
from multiprocessing import Pool

import numpy as np

from . import config as C

DU = np.arcsinh(C.GRID_R_MAX / C.GRID_A_R) / (C.GRID_N - 1)
DV = np.arcsinh(C.GRID_Z_MAX / C.GRID_A_Z) / (C.GRID_N - 1)
CACHE_FILE = C.CACHE / 'mcmillan17_grid.npz'


def node_coords():
    u = np.arange(C.GRID_N) * DU
    v = np.arange(C.GRID_N) * DV
    return C.GRID_A_R * np.sinh(u), C.GRID_A_Z * np.sinh(v)      # R (pc), z (pc)


def _galpy():
    import warnings
    warnings.filterwarnings('ignore')
    from galpy.potential.mwpotentials import McMillan17
    from galpy.util.conversion import get_physical
    phys = get_physical(McMillan17)
    return McMillan17, phys['ro'] * 1000.0, phys['vo']          # ro in pc, vo in km/s


def _phi_rows(args):
    """Potential (pc/Myr)^2 on rows of the grid — runs in a worker process."""
    R, zs = args
    from galpy.potential import evaluatePotentials
    pot, ro, vo = _galpy()
    out = np.empty((len(zs), len(R)))
    for k, z in enumerate(zs):
        out[k] = evaluatePotentials(pot, np.maximum(R, 1e-3) / ro, np.full_like(R, z) / ro,
                                    use_physical=False) * (vo * C.KMS_TO_PCMYR) ** 2
    return out


def galpy_forces(R, z):
    """Direct galpy forces (pc/Myr^2) — the ground truth the table is tested against."""
    from galpy.potential import evaluateRforces, evaluatezforces
    pot, ro, vo = _galpy()
    f = vo * C.KMS_TO_PCMYR
    f = f * f / ro
    R = np.maximum(np.asarray(R, float), 1e-3)
    return (evaluateRforces(pot, R / ro, np.asarray(z) / ro, use_physical=False) * f,
            evaluatezforces(pot, R / ro, np.asarray(z) / ro, use_physical=False) * f)


def galpy_potential(R, z):
    from galpy.potential import evaluatePotentials
    pot, ro, vo = _galpy()
    return evaluatePotentials(pot, np.maximum(np.asarray(R, float), 1e-3) / ro,
                              np.asarray(z) / ro, use_physical=False) * (vo * C.KMS_TO_PCMYR) ** 2


def build(workers=None):
    """Tabulate Φ, F_R, F_z. Cached; ~5 min on 4 cores."""
    if CACHE_FILE.exists():
        d = np.load(CACHE_FILE)
        return d['phi'], d['fr'], d['fz']
    from scipy.interpolate import RectBivariateSpline
    R, z = node_coords()
    workers = workers or os.cpu_count()
    chunks = np.array_split(z, workers * 8)
    with Pool(workers) as pool:
        phi = np.vstack(pool.map(_phi_rows, [(R, zc) for zc in chunks]))   # [v, u]
    # mirror across both axes: Φ is even in R and z, and so in u and v
    u = np.arange(C.GRID_N) * DU
    v = np.arange(C.GRID_N) * DV
    uf = np.concatenate([-u[:0:-1], u])
    vf = np.concatenate([-v[:0:-1], v])
    pf = np.concatenate([phi[:0:-1], phi], axis=0)
    pf = np.concatenate([pf[:, :0:-1], pf], axis=1)
    spl = RectBivariateSpline(vf, uf, pf, kx=5, ky=5, s=0)
    dphi_dv = spl(v, u, dx=1, dy=0)
    dphi_du = spl(v, u, dx=0, dy=1)
    fr = -dphi_du / np.sqrt(R[None, :] ** 2 + C.GRID_A_R ** 2)
    fz = -dphi_dv / np.sqrt(z[:, None] ** 2 + C.GRID_A_Z ** 2)
    C.CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE_FILE, phi=phi, fr=fr, fz=fz)
    return phi, fr, fz


def table_float32():
    """Interleaved [v][u][F_R, F_z] float32 — exactly the bytes the GPU gets."""
    _, fr, fz = build()
    t = np.empty((C.GRID_N, C.GRID_N, 2), np.float32)
    t[..., 0] = fr
    t[..., 1] = fz
    return t


def _bilinear(tab, fu, fv):
    n = C.GRID_N
    fu = np.clip(fu, 0.0, n - 1.000001)
    fv = np.clip(fv, 0.0, n - 1.000001)
    i = fu.astype(np.int64)
    j = fv.astype(np.int64)
    a = (fu - i)[..., None]
    b = (fv - j)[..., None]
    return ((tab[j, i] * (1 - a) + tab[j, i + 1] * a) * (1 - b)
            + (tab[j + 1, i] * (1 - a) + tab[j + 1, i + 1] * a) * b)


def interp_accel(pos, tab):
    """Acceleration (pc/Myr^2) at galactocentric positions pos[..., 3] (pc).

    The reference implementation of the run-time lookup. tab comes from
    table_float32() (cast to float64 here; the GPU keeps float32)."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    R = np.sqrt(x * x + y * y)
    fu = np.arcsinh(R / C.GRID_A_R) / DU
    fv = np.arcsinh(np.abs(z) / C.GRID_A_Z) / DV
    f = _bilinear(tab, fu, fv)
    fr, fz = f[..., 0], f[..., 1] * np.sign(z)
    inv = np.where(R > 0, 1.0 / np.maximum(R, 1e-12), 0.0)
    return np.stack([fr * x * inv, fr * y * inv, fz], axis=-1)


def interp_phi(pos):
    """Bilinear Φ from the cached grid — energy diagnostics only (not shipped)."""
    phi, _, _ = build()
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    R = np.sqrt(x * x + y * y)
    fu = np.arcsinh(R / C.GRID_A_R) / DU
    fv = np.arcsinh(np.abs(z) / C.GRID_A_Z) / DV
    return _bilinear(phi[..., None], fu, fv)[..., 0]


if __name__ == '__main__':
    import time
    t0 = time.time()
    build()
    print(f'potential table built in {time.time() - t0:.0f} s → {CACHE_FILE}')
