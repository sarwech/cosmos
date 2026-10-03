"""The five known-answer checks + engineering checks.

Pass criteria were fixed in the plan BEFORE any data was fetched and are not
tuned. Every check runs on the DECODED packed files (pack.read()) — the same
quantised initial conditions the browser integrates — with the reference
integrator (orbit.py), which the GPU kernel mirrors.
"""
import json
import time

import numpy as np

from . import config as C
from . import fetch_meta, pack
from .catalogue import solar_state
from .orbit import KV, Model, accel, sun_orbit

LY = 1 / C.PC_PER_LY                                  # ly per pc

# Fixed before seeing data (see plan): name → (low, high)
CRITERIA = {
    'cosmos_sky_max_sep_deg': 0.02,
    'barnard_min_ly': (3.70, 3.80), 'barnard_t_yr': (9_500, 10_000),
    'gl710_min_pc': (0.045, 0.070), 'gl710_t_myr': (1.25, 1.35),
    'dipper_member_max_kms': 4.0, 'dipper_outlier_min_kms': 10.0,
    'sun_period_myr': (220.0, 240.0),
}
BARNARD = 4472832130942575872
GL710 = 4270814637616488064
DIPPER_MEMBERS = {'Merak': 53910, 'Phecda': 58001, 'Megrez': 59774, 'Alioth': 62956, 'Mizar': 65378}
DIPPER_OUTLIERS = {'Dubhe': 54061, 'Alkaid': 67301}


def _load():
    d = pack.read()
    sun = solar_state()
    return d, sun


def _index(d, sid=None, hip=None):
    if sid is not None:
        k = np.flatnonzero(d['source_id'] == sid)
    else:
        k = np.flatnonzero(d['hip'] == hip)
    if not len(k):
        raise KeyError(sid or hip)
    return int(k[0])


def _closest(model, dx0, dv0, t_end, coarse=40, fine=4000):
    """Closest approach: coarse scan, then a fine scan of the bracketing step."""
    T, X = model.trajectory(dx0, dv0, t_end, samples_per_step=coarse)
    r = np.linalg.norm(X, axis=1)
    k = int(r.argmin())
    t0 = T[max(k - 1, 0)]
    ts = np.linspace(t0, T[min(k + 1, len(T) - 1)], fine)
    rs = np.array([np.linalg.norm(model.stars_at(t, dx0, dv0)) for t in ts])
    j = int(rs.argmin())
    return float(ts[j]), float(rs[j])


def check_cosmos_sky(d, sun):
    """1. t = 0 sky vs the ~90 Hipparcos-derived stars hard-coded in index.html."""
    cs = fetch_meta.cosmos_stars()
    ids = fetch_meta.simbad_ids(list(cs.name))
    ref = fetch_meta.simbad_coords(list(cs.name))          # independent referee

    def sep_deg(ra1, de1, ra2, de2):
        a = np.radians([ra1, de1, ra2, de2])
        c = np.sin(a[1]) * np.sin(a[3]) + np.cos(a[1]) * np.cos(a[3]) * np.cos(a[0] - a[2])
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    rot = sun['icrs_to_gc']                              # gc = rot @ icrs
    rows = []
    for _, s in cs.iterrows():
        r = ids[s['name']]
        try:
            i = _index(d, hip=r['hip']) if r['hip'] and np.any(d['hip'] == r['hip']) else _index(d, sid=r['gaia_dr3'])
        except (KeyError, TypeError):
            rows.append({'name': s['name'], 'found': False})
            continue
        x = rot.T @ d['dx'][i]                           # ICRS Cartesian at J2016
        v = rot.T @ d['dv_kms'][i] * KV
        x2000 = x - v * 16e-6                            # straight back to J2000 (16 yr)
        ra = np.degrees(np.arctan2(x2000[1], x2000[0])) % 360
        dec = np.degrees(np.arcsin(x2000[2] / np.linalg.norm(x2000)))
        sep = sep_deg(ra, dec, s['ra'], s['dec'])
        dist_ly = float(np.linalg.norm(x)) * LY
        src = 'Hipparcos' if d['flags'][i] & pack.FLAG_HIP else 'Gaia DR3'
        row = {'name': s['name'], 'found': True, 'source': src, 'sep_deg': round(sep, 5),
               'dist_ly': round(dist_ly, 2), 'cosmos_dist_ly': s['dist_ly'],
               'dist_ratio': round(dist_ly / s['dist_ly'], 3)}
        if ref.get(s['name']):
            row['ours_vs_simbad_deg'] = round(sep_deg(ra, dec, *ref[s['name']]), 5)
            row['cosmos_vs_simbad_deg'] = round(sep_deg(s['ra'], s['dec'], *ref[s['name']]), 5)
        rows.append(row)
    found = [r for r in rows if r['found']]
    worst = max(found, key=lambda r: r['sep_deg'])
    gaia = [r for r in found if r['source'] == 'Gaia DR3']
    lim = CRITERIA['cosmos_sky_max_sep_deg']
    ok = len(found) == len(rows) and worst['sep_deg'] <= lim
    # diagnosis only — does NOT change the verdict: who disagrees with SIMBAD?
    misses = [r for r in found if r['sep_deg'] > lim]
    cosmos_err = [r['name'] for r in misses if r.get('ours_vs_simbad_deg', 9) <= lim < r.get('cosmos_vs_simbad_deg', 0)]
    ours_err = [r['name'] for r in found if r.get('ours_vs_simbad_deg', 0) > lim]
    return {'pass': bool(ok), 'n': len(rows), 'found': len(found), 'n_gaia': len(gaia),
            'n_outside_tolerance': len(misses),
            'diagnosis': {'cosmos_coordinate_errors': cosmos_err, 'pipeline_disagrees_with_simbad': ours_err,
                          'median_sep_deg': round(float(np.median([r['sep_deg'] for r in found])), 5)},
            'max_sep_deg': worst['sep_deg'], 'worst': worst['name'],
            'max_sep_gaia_deg': max(r['sep_deg'] for r in gaia) if gaia else None,
            'criterion': f"all stars found; direction within {CRITERIA['cosmos_sky_max_sep_deg']}° "
                         '(distances reported, not gating)',
            'stars': rows}


def check_barnard(d, model):
    i = _index(d, sid=BARNARD)
    t, r = _closest(model, d['dx'][i], d['dv_kms'][i], 0.03, coarse=400, fine=2000)
    t_yr, r_ly = t * 1e6, r * LY
    lo, hi = CRITERIA['barnard_min_ly']
    tlo, thi = CRITERIA['barnard_t_yr']
    return {'pass': bool(lo <= r_ly <= hi and tlo <= t_yr <= thi), 'min_dist_ly': round(r_ly, 4),
            't_years': round(t_yr), 'calendar_year_CE': round(C.EPOCH + t_yr),
            'today_ly': round(float(np.linalg.norm(d['dx'][i])) * LY, 4),
            'criterion': f'{lo}–{hi} ly at +{tlo:,}–{thi:,} yr'}


def check_gl710(d, model):
    i = _index(d, sid=GL710)
    t, r = _closest(model, d['dx'][i], d['dv_kms'][i], 2.0, coarse=40, fine=4000)
    lo, hi = CRITERIA['gl710_min_pc']
    tlo, thi = CRITERIA['gl710_t_myr']
    # the same encounter with straight-line motion, for the record
    x0, v0 = d['dx'][i], d['dv_kms'][i] * KV
    tl = -np.dot(x0, v0) / np.dot(v0, v0)
    rl = np.linalg.norm(x0 + v0 * tl)
    return {'pass': bool(lo <= r <= hi and tlo <= t <= thi), 'min_dist_pc': round(r, 5),
            'min_dist_au': round(r * 206264.8), 't_myr': round(t, 4),
            'straight_line_min_pc': round(float(rl), 5), 'straight_line_t_myr': round(float(tl), 4),
            'today_pc': round(float(np.linalg.norm(x0)), 3),
            'criterion': f'{lo}–{hi} pc at +{tlo}–{thi} Myr'}


def check_dipper(d, model):
    vel = {}
    pos = {}
    src = {}
    for name, hip in {**DIPPER_MEMBERS, **DIPPER_OUTLIERS}.items():
        i = _index(d, hip=hip)
        vel[name] = d['dv_kms'][i]
        pos[name] = (d['dx'][i], d['dv_kms'][i])
        src[name] = 'Hipparcos+XHIP' if d['flags'][i] & pack.FLAG_HIP else 'Gaia DR3'
    mean = np.mean([vel[n] for n in DIPPER_MEMBERS], axis=0)
    dev = {n: round(float(np.linalg.norm(vel[n] - mean)), 2) for n in vel}
    member_max = max(dev[n] for n in DIPPER_MEMBERS)
    outlier_min = min(dev[n] for n in DIPPER_OUTLIERS)
    ok = member_max <= CRITERIA['dipper_member_max_kms'] and outlier_min >= CRITERIA['dipper_outlier_min_kms']
    # sky shape at 0, 50, 100 kyr (ICRS RA/Dec) for the report figure
    rot = solar_state()['icrs_to_gc']
    shape = {}
    for t in (0.0, 0.05, 0.1):
        sh = {}
        for n, (x0, v0) in pos.items():
            x = rot.T @ model.stars_at(t, x0, v0)
            sh[n] = [round(float(np.degrees(np.arctan2(x[1], x[0])) % 360), 3),
                     round(float(np.degrees(np.arcsin(x[2] / np.linalg.norm(x)))), 3)]
        shape[f'{t * 1e3:g} kyr'] = sh
    return {'pass': bool(ok), 'member_dev_kms': {n: dev[n] for n in DIPPER_MEMBERS},
            'outlier_dev_kms': {n: dev[n] for n in DIPPER_OUTLIERS}, 'sources': src,
            'group_velocity_kms': [round(float(c), 2) for c in mean],
            'criterion': f"members within {CRITERIA['dipper_member_max_kms']} km/s of their mean, "
                         f"Dubhe & Alkaid ≥ {CRITERIA['dipper_outlier_min_kms']} km/s away",
            'sky_radec_deg': shape}


def check_sun(sun):
    P, V, _ = sun_orbit(sun['pos_pc'], sun['vel_kms'], 30_000, C.DT_MYR)     # 3 Gyr
    phi = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    dphi = np.abs(phi - phi[0])
    k = int(np.argmax(dphi >= 2 * np.pi))
    first = k * C.DT_MYR
    mean = 2 * np.pi / (dphi[-1] / (len(P) - 1) / C.DT_MYR)
    z = P[:, 2]
    up = np.flatnonzero(np.diff(np.sign(z)) > 0)
    zper = float(np.mean(np.diff(up))) * C.DT_MYR if len(up) > 1 else None
    R = np.hypot(P[:, 0], P[:, 1])
    lo, hi = CRITERIA['sun_period_myr']
    seg = P[: int(250 / C.DT_MYR) + 1, 2]
    oscillates = (seg.max() > 20) and (seg.min() < -20)
    return {'pass': bool(lo <= first <= hi and oscillates), 'azimuthal_period_myr': round(first, 1),
            'mean_azimuthal_period_myr_3gyr': round(float(mean), 1),
            'vertical_amplitude_pc': round(float(np.abs(z).max()), 1),
            'vertical_period_myr': round(zper, 1) if zper else None,
            'R_min_kpc': round(float(R.min()) / 1e3, 3), 'R_max_kpc': round(float(R.max()) / 1e3, 3),
            'criterion': f'one revolution in {lo}–{hi} Myr, crossing the plane (|z| > 20 pc both sides)'}


# ------------------------------------------------------------ engineering ----
def check_energy(sun):
    from .potential import galpy_potential
    out = {}
    for sg in (1, -1):
        P, V, _ = sun_orbit(sun['pos_pc'], sun['vel_kms'], int(C.T_MAX_MYR / C.DT_MYR), sg * C.DT_MYR)
        idx = np.arange(0, len(P), 100)
        R = np.hypot(P[idx, 0], P[idx, 1])
        E = 0.5 * (V[idx] ** 2).sum(1) + galpy_potential(R, P[idx, 2])
        out['forward' if sg > 0 else 'backward'] = float(np.abs((E - E[0]) / E[0]).max())
    worst = max(out.values())
    return {'pass': worst < 1e-4, 'max_rel_energy_error': out, 'criterion': '|ΔE/E| < 1e-4 over ±250 Myr'}


def check_convergence(d, sun, n=1000, seed=2):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d['dx']), n, replace=False)
    out = {}
    for t in (1.3, 50.0, 250.0):
        a = Model(sun['pos_pc'], sun['vel_kms'], t_max=t + 0.2, dt=C.DT_MYR)
        b = Model(sun['pos_pc'], sun['vel_kms'], t_max=t + 0.2, dt=C.DT_MYR / 2)
        xa = a.stars_at(t, d['dx'][idx], d['dv_kms'][idx])
        xb = b.stars_at(t, d['dx'][idx], d['dv_kms'][idx])
        diff = np.linalg.norm(xa - xb, axis=1)
        sa, sb = a.sun_at(t)[0], b.sun_at(t)[0]
        out[f'{t:g} Myr'] = {'median_star_pc': float(np.median(diff)), 'p99_star_pc': float(np.percentile(diff, 99)),
                             'sun_pc': float(np.linalg.norm(sa - sb))}
    worst = out['250 Myr']
    ok = worst['sun_pc'] < 1.0 and worst['median_star_pc'] < 10.0
    return {'pass': bool(ok), 'dt_vs_dt_over_2': out,
            'criterion': 'halving dt moves the Sun < 1 pc and the median star < 10 pc at 250 Myr'}


def check_straight_line(d, model, n=2000, seed=3):
    """Short spans should agree with straight-line motion; the residual should be
    the size of the Galactic tide (~½ ν² r t², ν² ≈ 4πGρ₀ ≈ 5e-3 Myr⁻²)."""
    rng = np.random.default_rng(seed)
    loc = np.flatnonzero(np.linalg.norm(d['dx'], axis=1) < 200)
    idx = rng.choice(loc, n, replace=False)
    x0, v0 = d['dx'][idx], d['dv_kms'][idx] * KV
    out = {}
    for t in (0.01, 0.1, 1.0, 5.0):
        x = model.stars_at(t, x0, d['dv_kms'][idx])
        lin = x0 + v0 * t
        dev = np.linalg.norm(x - lin, axis=1)
        moved = np.linalg.norm(v0 * t, axis=1)
        tide = 0.5 * 5e-3 * np.linalg.norm(x0, axis=1) * t * t
        out[f'{t:g} Myr'] = {'median_dev_pc': float(np.median(dev)), 'median_rel_to_motion': float(np.median(dev / moved)),
                             'median_dev_over_tide_estimate': float(np.median(dev / tide))}
    ok = out['0.1 Myr']['median_rel_to_motion'] < 1e-3
    return {'pass': bool(ok), 'by_time': out,
            'criterion': 'at 0.1 Myr, integrated vs straight-line differ by < 0.1% of the distance moved'}


def check_reversibility(d, sun, n=500, seed=4):
    """Integrate 250 Myr out and back with the reverse step (the GPU's scrub path)."""
    from .orbit import step_rel
    m = Model(sun['pos_pc'], sun['vel_kms'])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d['dx']), n, replace=False)
    x, v = d['dx'][idx].copy(), d['dv_kms'][idx] * KV
    x0 = x.copy()
    N = int(C.T_MAX_MYR / C.DT_MYR)
    h = C.DT_MYR
    for k in range(N):
        p0, _, a0 = m.sun(k, 1); p1, _, a1 = m.sun(k + 1, 1)
        x, v = step_rel(x, v, (p0, a0), (p1, a1), h)
    for k in range(N, 0, -1):
        p0, _, a0 = m.sun(k, 1); p1, _, a1 = m.sun(k - 1, 1)
        x, v = step_rel(x, v, (p0, a0), (p1, a1), -h)
    err = np.linalg.norm(x - x0, axis=1)
    return {'pass': bool(err.max() < 1e-6), 'max_roundtrip_error_pc': float(err.max()),
            'criterion': 'float64 out-and-back over 250 Myr returns within 1e-6 pc'}


def run_all(write=True):
    t0 = time.time()
    d, sun = _load()
    model = Model(sun['pos_pc'], sun['vel_kms'])
    res = {
        '1_sky_matches_cosmos': check_cosmos_sky(d, sun),
        '2_barnards_star': check_barnard(d, model),
        '3_gliese_710': check_gl710(d, model),
        '4_big_dipper': check_dipper(d, model),
        '5_sun_orbit': check_sun(sun),
    }
    eng = {'force_table_accuracy': json.loads((C.REPORTS / 'table_accuracy.json').read_text())}
    eng['force_table_accuracy']['pass'] = eng['force_table_accuracy']['max_rel_err'] < 1e-3
    eng['energy'] = check_energy(sun)
    eng['convergence'] = check_convergence(d, sun)
    eng['straight_line'] = check_straight_line(d, model)
    eng['reversibility'] = check_reversibility(d, sun)
    res['engineering'] = eng
    res['runtime_s'] = round(time.time() - t0, 1)
    if write:
        C.REPORTS.mkdir(exist_ok=True)
        (C.REPORTS / 'validation.json').write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == '__main__':
    r = run_all()
    for k, v in r.items():
        if isinstance(v, dict) and 'pass' in v:
            print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}")
    for k, v in r['engineering'].items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  engineering/{k}")
