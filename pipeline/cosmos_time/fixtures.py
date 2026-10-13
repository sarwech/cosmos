"""Reference positions for the browser parity tests (time/test/).

For a fixed sample of stars, the Sun-relative positions the reference
integrator gives at a handful of times, computed from the DECODED packed data
— exactly the initial conditions the GPU starts from. The browser tests
(GPU float32 and the JS float64 twin) must reproduce these.
"""
import json

import numpy as np

from . import config as C
from . import pack
from .catalogue import solar_state
from .orbit import Model
from .validate import BARNARD, GL710

TIMES = [0.0097, -0.05, 1.2934, 10.05, -37.33, 100.0, -250.0, 249.95]
N_MAX_INDEX = 20_000                   # the browser test loads the first 20k stars


def write():
    d = pack.read()
    sun = solar_state()
    model = Model(sun['pos_pc'], sun['vel_kms'])
    rng = np.random.default_rng(11)
    idx = [int(np.flatnonzero(d['source_id'] == s)[0]) for s in (BARNARD, GL710)]
    idx += list(range(0, 40))                                       # brightest tracked stars
    idx += sorted(rng.choice(np.arange(40, N_MAX_INDEX), 160, replace=False).tolist())
    idx = [i for i in dict.fromkeys(idx) if i < N_MAX_INDEX]
    x0, v0 = d['dx'][idx], d['dv_kms'][idx]
    out = {'indices': idx, 'times_myr': TIMES, 'positions_pc': [], 'sun_pc': []}
    for t in TIMES:
        out['positions_pc'].append(np.round(model.stars_at(t, x0, v0), 6).tolist())
        out['sun_pc'].append(np.round(model.sun_at(t)[0], 6).tolist())
    path = C.REPO / 'time' / 'test' / 'parity-fixture.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return path


if __name__ == '__main__':
    print(write())
