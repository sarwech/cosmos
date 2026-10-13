"""Force-table accuracy against direct galpy evaluation (writes a report)."""
import json

import numpy as np

from . import config as C
from .potential import galpy_forces, interp_accel, table_float32


def run(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    # disk region where stars live (log-uniform R 0.5-30 kpc, |z| sinh-spread to 5 kpc)
    R = 10 ** rng.uniform(np.log10(500), np.log10(30_000), n)
    z = np.sinh(rng.uniform(-np.arcsinh(5000 / 100), np.arcsinh(5000 / 100), n)) * 100
    tab = table_float32().astype(np.float64)
    a = interp_accel(np.stack([R, np.zeros(n), z], -1), tab)
    fr, fz = galpy_forces(R, z)
    mag = np.hypot(fr, fz)
    err = np.hypot(a[:, 0] - fr, a[:, 2] - fz) / mag
    sol = (np.abs(R - 8122) < 1500) & (np.abs(z) < 1000)
    rep = {'n': n, 'median_rel_err': float(np.median(err)), 'p99_rel_err': float(np.percentile(err, 99)),
           'max_rel_err': float(err.max()), 'worst_R_pc': float(R[err.argmax()]), 'worst_z_pc': float(z[err.argmax()]),
           'solar_annulus_max_rel_err': float(err[sol].max()), 'solar_annulus_n': int(sol.sum())}
    C.REPORTS.mkdir(exist_ok=True)
    (C.REPORTS / 'table_accuracy.json').write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == '__main__':
    print(run())
