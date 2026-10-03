"""How far measurement errors alone let a star wander, as a function of time.

Monte Carlo: for a random sample of stars, draw clones from the Gaia/Hipparcos
uncertainties (parallax → distance, both proper motions, radial velocity;
independent Gaussians — correlations ignored), integrate every clone with the
reference integrator and record the clone scatter about the nominal orbit.
The page shows the median as "typical position uncertainty at this time".
The potential is held fixed, so this is a LOWER bound: model error is extra.
"""
import numpy as np

from . import config as C
from .catalogue import SRC_HIP, solar_state
from .orbit import Model

TIMES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 250.0]


def curve(cat, n_stars=400, n_clones=24, seed=5):
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Galactocentric, galactocentric_frame_defaults
    rng = np.random.default_rng(seed)
    sun = solar_state()
    model = Model(sun['pos_pc'], sun['vel_kms'])
    with galactocentric_frame_defaults.set(C.FRAME):
        gc = Galactocentric()
    out = {}
    for label, mask in (('local', (cat['sample'] == 0).to_numpy()), ('disk', (cat['sample'] == 1).to_numpy())):
        idx = rng.choice(np.flatnonzero(mask), n_stars, replace=False)
        s = cat.iloc[idx]
        k = n_stars * n_clones
        rep = lambda c: np.repeat(s[c].to_numpy(float), n_clones)
        plx = rep('plx') + rng.standard_normal(k) * rep('plx_err')
        # the nominal distance is Bailer-Jones; perturb it by the parallax error fraction
        dist = rep('dist') * rep('plx') / np.clip(plx, 0.05 * rep('plx'), None)
        pmra = rep('pmra') + rng.standard_normal(k) * rep('pmra_err')
        pmdec = rep('pmdec') + rng.standard_normal(k) * rep('pmdec_err')
        rv = rep('rv') + rng.standard_normal(k) * np.nan_to_num(rep('rv_err'), nan=1.0)
        c = SkyCoord(ra=rep('ra') * u.deg, dec=rep('dec') * u.deg, distance=dist * u.pc,
                     pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                     radial_velocity=rv * u.km / u.s, frame='icrs').transform_to(gc)
        x = np.stack([c.x.to(u.pc).value, c.y.to(u.pc).value, c.z.to(u.pc).value], -1) - sun['pos_pc']
        v = np.stack([c.v_x.to(u.km / u.s).value, c.v_y.to(u.km / u.s).value,
                      c.v_z.to(u.km / u.s).value], -1) - sun['vel_kms']
        res = []
        for t in TIMES:
            xt = model.stars_at(t, x, v).reshape(n_stars, n_clones, 3)
            spread = np.sqrt(((xt - xt.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))
            res.append(float(np.median(spread)))
        out[label] = res
    return {'times_myr': TIMES, 'median_spread_pc': out, 'n_stars': n_stars, 'n_clones': n_clones,
            'note': 'measurement errors only; the potential model adds more'}
