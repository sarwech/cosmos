"""Merge Gaia DR3 + Hipparcos into one 6D catalogue at epoch J2016.0.

Rules (documented in pipeline/README.md):
  * Gaia local sample: parallax > 4.4 mas, RV present, RUWE < 1.4, ϖ/σϖ ≥ 10.
  * Gaia disk sample: random_index draw beyond that, RUWE < 1.4, ϖ/σϖ ≥ 5.
  * "Bright & named" set H = Hipparcos stars with V < 6.5 ∪ constellation-figure
    stars ∪ named stars. A star in H uses its Gaia DR3 counterpart only if that
    solution passes the local cuts AND G > 3; otherwise Hipparcos-2 astrometry
    (propagated 1991.25 → 2016.0 with perspective acceleration) + XHIP RV.
  * Named stars outside Hipparcos (TRAPPIST-1, Wolf 359 …) always use Gaia,
    with SIMBAD's RV where Gaia has none.
  * Distances: Bailer-Jones et al. (2021) r_med_photogeo (r_med_geo, then
    1/ϖ as fallbacks); Hipparcos stars 1/ϖ.
  * Gaia RVs corrected per Katz et al. 2023 (cool, G_RVS ≥ 11) and
    Blomme et al. 2023 (8500–14500 K, 6 ≤ G_RVS ≤ 12).
  * Extinction: GSP-Phot A_G / E(BP−RP) applied to the disk sample only; the
    local sample sits inside the Local Bubble (E(B−V) ≲ 0.02).
"""
import re

import numpy as np
import pandas as pd

from . import colour
from . import config as C
from . import fetch_gaia, fetch_hip, fetch_meta

GREEK = re.compile(r'^(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|'
                   r'sigma|tau|upsilon|phi|chi|psi|omega)(-\d)?\s', re.I)
SRC_GAIA, SRC_HIP = 0, 1
RV_GAIA, RV_XHIP, RV_SIMBAD, RV_NONE = 0, 1, 2, 3
SAMPLE_LOCAL, SAMPLE_DISK, SAMPLE_BRIGHT = 0, 1, 2


def _gaia_distance(df):
    d = df['r_med_photogeo'].copy()
    d = d.fillna(df['r_med_geo'])
    d = d.fillna(1000.0 / df['parallax'].where(df['parallax'] > 0))
    return d


def _correct_gaia_rv(df):
    rv = df['radial_velocity'].to_numpy(float).copy()
    g = df['grvs_mag'].to_numpy(float)
    teff = df['rv_template_teff'].to_numpy(float)
    cool = (teff < 8500) & (g >= 11)
    rv[cool] -= 0.02755 * g[cool] ** 2 - 0.55863 * g[cool] + 2.81129        # Katz+2023 eq. 5
    hot = (teff >= 8500) & (teff <= 14500) & (g >= 6) & (g <= 12)
    rv[hot] += -7.98 + 1.135 * g[hot]                                        # Blomme+2023
    return rv


def _gaia_frame(df, sample):
    """Gaia rows → unified columns."""
    out = pd.DataFrame({
        'source': SRC_GAIA, 'source_id': df['source_id'].astype(np.int64),
        'hip': 0, 'ra': df['ra'], 'dec': df['dec'], 'dist': _gaia_distance(df),
        'pmra': df['pmra'], 'pmdec': df['pmdec'], 'rv': _correct_gaia_rv(df),
        'plx': df['parallax'], 'plx_err': df['parallax_error'],
        'pmra_err': df['pmra_error'], 'pmdec_err': df['pmdec_error'],
        'rv_err': df['radial_velocity_error'],
        'gmag': df['phot_g_mean_mag'], 'bp_rp_obs': df['bp_rp'],
        'ag': 0.0, 'ebprp': 0.0, 'rv_src': RV_GAIA, 'sample': sample,
        'ruwe': df['ruwe'],
    })
    if sample == SAMPLE_DISK:
        out['ag'] = df['ag_gspphot'].fillna(0.0).to_numpy()
        out['ebprp'] = df['ebpminrp_gspphot'].fillna(0.0).to_numpy()
    return out


def _hip_frame(hips, xh, h2, simbad_rv):
    """Hipparcos-2 astrometry + XHIP RV, propagated to J2016.0."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    h = h2.set_index('hip').loc[hips]
    x = xh.set_index('hip').reindex(hips)
    rv = x['rv'].to_numpy(float).copy()
    rv_err = x['e_rv'].to_numpy(float).copy()
    src = np.where(np.isfinite(rv), RV_XHIP, RV_NONE)
    for k, hip in enumerate(hips):
        if not np.isfinite(rv[k]) and hip in simbad_rv:
            rv[k], rv_err[k] = simbad_rv[hip]
            src[k] = RV_SIMBAD
    rv_filled = np.where(np.isfinite(rv), rv, 0.0)
    plx = h['plx'].to_numpy(float)
    # Hipparcos parallaxes below 0.5 mas (a handful of distant supergiants) are
    # floored at 0.5 mas (2 kpc) and flagged via plx_err.
    dist = 1000.0 / np.maximum(plx, 0.5)
    c = SkyCoord(ra=h['ra'].to_numpy() * u.deg, dec=h['dec'].to_numpy() * u.deg,
                 distance=dist * u.pc, pm_ra_cosdec=h['pmra'].to_numpy() * u.mas / u.yr,
                 pm_dec=h['pmde'].to_numpy() * u.mas / u.yr,
                 radial_velocity=rv_filled * u.km / u.s, frame='icrs',
                 obstime=Time(C.HIP_EPOCH, format='jyear'))
    c2 = c.apply_space_motion(new_obstime=Time(C.EPOCH, format='jyear'))
    vmag = x['vmag'].to_numpy(float)
    bv = x['b_v'].to_numpy(float)
    bv0 = np.nan_to_num(bv, nan=0.6)
    gmag = vmag + (-0.02704 + 0.01424 * bv0 - 0.2156 * bv0 ** 2 + 0.01426 * bv0 ** 3)  # Riello+2021
    gmag = np.where(np.isfinite(gmag), gmag, h['hpmag'].to_numpy(float))
    return pd.DataFrame({
        'source': SRC_HIP, 'source_id': np.zeros(len(hips), np.int64), 'hip': np.asarray(hips),
        'ra': c2.ra.deg, 'dec': c2.dec.deg, 'dist': c2.distance.to(u.pc).value,
        'pmra': c2.pm_ra_cosdec.to(u.mas / u.yr).value, 'pmdec': c2.pm_dec.to(u.mas / u.yr).value,
        'rv': rv_filled, 'plx': plx, 'plx_err': x['e_plx'].to_numpy(float),
        'pmra_err': x['e_pmra'].to_numpy(float), 'pmdec_err': x['e_pmde'].to_numpy(float),
        'rv_err': rv_err, 'gmag': gmag, 'bp_rp_obs': np.nan, 'b_v': bv,
        'ag': 0.0, 'ebprp': 0.0, 'rv_src': src, 'sample': SAMPLE_BRIGHT, 'ruwe': np.nan,
    })


def _unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)


def build():
    path = C.CACHE / 'catalogue.parquet'
    if path.exists():
        return pd.read_parquet(path)
    local = fetch_gaia.local_sample()
    disk = fetch_gaia.disk_sample()
    hipm = fetch_gaia.hipparcos_matches()
    xh, h2 = fetch_hip.xhip(), fetch_hip.hip2()
    cosmos = fetch_meta.cosmos_stars()
    named = fetch_meta.simbad_ids(list(cosmos.name) + fetch_meta.EXTRA_NAMED)
    figures = fetch_meta.constellations()
    common = fetch_meta.common_names()

    gaia = pd.concat([_gaia_frame(local, SAMPLE_LOCAL), _gaia_frame(disk, SAMPLE_DISK)],
                     ignore_index=True).drop_duplicates('source_id')
    hip_of = dict(zip(hipm['source_id'].astype(np.int64), hipm['hip'].astype(int)))

    # ---- the bright & named set H -------------------------------------------
    fig_hips = {h for f in figures for l in f['lines'] for h in l}
    named_hips = {r['hip'] for r in named.values() if r['hip']} | set(common)
    bright_hips = set(xh.loc[xh['vmag'] < C.NAKED_EYE_V, 'hip'].astype(int))
    H = sorted((fig_hips | named_hips | bright_hips) & set(h2['hip'].astype(int)))
    m = hipm.set_index('hip')
    use_gaia, use_hip = [], []
    for hip in H:
        if hip in m.index:
            r = m.loc[hip]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            ok = (r['phot_g_mean_mag'] > C.GAIA_BRIGHT_LIMIT and r['ruwe'] < C.MAX_RUWE
                  and np.isfinite(r['radial_velocity']) and r['parallax_over_error'] >= C.LOCAL_MIN_POE)
            (use_gaia if ok else use_hip).append(hip)
        else:
            use_hip.append(hip)
    # Gaia counterparts of H stars that use Gaia but are outside the samples
    gsel = hipm[hipm['hip'].isin(use_gaia) & ~hipm['source_id'].isin(gaia['source_id'])]
    gaia = pd.concat([gaia, _gaia_frame(gsel, SAMPLE_BRIGHT)], ignore_index=True)
    # Gaia counterparts of H stars that use Hipparcos must not appear twice
    drop = set(hipm.loc[hipm['hip'].isin(use_hip), 'source_id'].astype(np.int64))
    gaia = gaia[~gaia['source_id'].isin(drop)]

    # ---- named stars known only by Gaia id ------------------------------------
    simbad_rv_by_gaia, simbad_rv_by_hip = {}, {}
    extra_ids = []
    for r in named.values():
        if r['rv'] is not None:
            if r['gaia_dr3']:
                simbad_rv_by_gaia[r['gaia_dr3']] = (r['rv'], r['rv_err'] if r['rv_err'] else np.nan)
            if r['hip']:
                simbad_rv_by_hip[r['hip']] = (r['rv'], r['rv_err'] if r['rv_err'] else np.nan)
        if r['gaia_dr3'] and not (r['hip'] and r['hip'] in use_hip):
            if r['gaia_dr3'] not in set(gaia['source_id']):
                extra_ids.append(r['gaia_dr3'])
    if extra_ids:
        ex = fetch_gaia.by_source_id('gaia_named', extra_ids)
        exf = _gaia_frame(ex, SAMPLE_BRIGHT)
        for k, sid in enumerate(exf['source_id']):
            if not np.isfinite(exf['rv'].iloc[k]) and sid in simbad_rv_by_gaia:
                exf.loc[exf.index[k], ['rv', 'rv_err']] = simbad_rv_by_gaia[sid]
                exf.loc[exf.index[k], 'rv_src'] = RV_SIMBAD
        exf.loc[~np.isfinite(exf['rv']), 'rv_src'] = RV_NONE
        exf['rv'] = exf['rv'].fillna(0.0)
        gaia = pd.concat([gaia, exf], ignore_index=True)

    hipf = _hip_frame(use_hip, xh, h2, simbad_rv_by_hip)
    # Hipparcos stars whose Gaia twin escaped the HIP crossmatch: drop the
    # Gaia twin (same place within 3", parallax within 30%).
    from scipy.spatial import cKDTree
    tree = cKDTree(_unit(gaia['ra'].to_numpy(), gaia['dec'].to_numpy()))
    near = tree.query_ball_point(_unit(hipf['ra'].to_numpy(), hipf['dec'].to_numpy()),
                                 np.radians(3 / 3600))
    twins = set()
    gplx = gaia['plx'].to_numpy()
    for k, idx in enumerate(near):
        for j in idx:
            if abs(gplx[j] - hipf['plx'].iloc[k]) < 0.3 * max(hipf['plx'].iloc[k], 1):
                twins.add(j)
    gaia = gaia.drop(gaia.index[sorted(twins)])
    gaia['hip'] = gaia['source_id'].map(hip_of).fillna(0).astype(int)

    cat = pd.concat([gaia, hipf], ignore_index=True)
    cat = cat[np.isfinite(cat['dist']) & (cat['dist'] > 0) & np.isfinite(cat['rv'])
              & np.isfinite(cat['pmra']) & np.isfinite(cat['gmag'])].reset_index(drop=True)

    # ---- photometry -----------------------------------------------------------
    cat['abs_g'] = cat['gmag'] - 5 * np.log10(cat['dist'] / 10) - cat['ag']
    bprp0 = cat['bp_rp_obs'] - cat['ebprp']
    lt = np.where(np.isfinite(bprp0), colour.bprp_to_logteff(np.nan_to_num(bprp0, nan=0.8)),
                  colour.bv_to_logteff(np.nan_to_num(cat.get('b_v', np.nan), nan=0.6)))
    cat['logteff'] = lt
    cat['bp_rp'] = np.where(np.isfinite(bprp0), bprp0, colour.logteff_to_bprp(lt))

    # ---- 6D, galactocentric (astropy v4.0), Sun-relative ----------------------
    from astropy import units as u
    from astropy.coordinates import SkyCoord, Galactocentric, galactocentric_frame_defaults
    with galactocentric_frame_defaults.set(C.FRAME):
        gc_frame = Galactocentric()
    c = SkyCoord(ra=cat['ra'].to_numpy() * u.deg, dec=cat['dec'].to_numpy() * u.deg,
                 distance=cat['dist'].to_numpy() * u.pc,
                 pm_ra_cosdec=cat['pmra'].to_numpy() * u.mas / u.yr,
                 pm_dec=cat['pmdec'].to_numpy() * u.mas / u.yr,
                 radial_velocity=cat['rv'].to_numpy() * u.km / u.s, frame='icrs')
    g = c.transform_to(gc_frame)
    sun = solar_state()
    pos = np.stack([g.x.to(u.pc).value, g.y.to(u.pc).value, g.z.to(u.pc).value], -1)
    vel = np.stack([g.v_x.to(u.km / u.s).value, g.v_y.to(u.km / u.s).value,
                    g.v_z.to(u.km / u.s).value], -1)
    rel_x, rel_v = pos - sun['pos_pc'], vel - sun['vel_kms']
    for k, a in enumerate('xyz'):
        cat[f'dx_{a}'] = rel_x[:, k]
        cat[f'dv_{a}'] = rel_v[:, k]

    # ---- names ----------------------------------------------------------------
    cat['name'] = ''
    hip_index = {h: i for i, h in enumerate(cat['hip']) if h}
    sid_index = {s: i for i, s in enumerate(cat['source_id']) if s}
    for hip, nm in common.items():
        if hip in hip_index:
            cat.loc[hip_index[hip], 'name'] = nm
    cat['cosmos_name'] = ''
    for nm, r in named.items():
        i = hip_index.get(r['hip']) if r['hip'] else None
        if i is None and r['gaia_dr3']:
            i = sid_index.get(r['gaia_dr3'])
        if i is not None:
            cat.loc[i, 'cosmos_name'] = nm
            if not cat.loc[i, 'name']:
                cat.loc[i, 'name'] = nm
    cat['bayer'] = ''
    xn = xh.set_index('hip')['name']
    for hip, i in hip_index.items():
        s = xn.get(hip, '')
        s = re.sub(r'\(.*?\)', '', str(s)).strip()
        bare = re.sub(r'^\d+\s+', '', s)                    # drop the Flamsteed number …
        if GREEK.match(bare):                               # … only if a Bayer (Greek) letter remains
            s = bare
        if s and s != 'nan':
            cat.loc[i, 'bayer'] = s
    cat.to_parquet(path)
    return cat


def solar_state():
    """The Sun's galactocentric position (pc) and velocity (km/s), astropy v4.0,
    plus the ICRS → galactocentric rotation (columns = ICRS x, y, z axes)."""
    from astropy import units as u
    from astropy.coordinates import (ICRS, CartesianRepresentation, CartesianDifferential,
                                     Galactocentric, galactocentric_frame_defaults)
    with galactocentric_frame_defaults.set(C.FRAME):
        gc = Galactocentric()

    def to_gc(xyz_pc, v_kms=(0, 0, 0)):
        rep = CartesianRepresentation(np.array(xyz_pc, float).T * u.pc,
                                      differentials=CartesianDifferential(np.array(v_kms, float).T * u.km / u.s))
        g = ICRS(rep).transform_to(gc)
        return (np.stack([g.x.to(u.pc).value, g.y.to(u.pc).value, g.z.to(u.pc).value], -1),
                np.stack([g.v_x.to(u.km / u.s).value, g.v_y.to(u.km / u.s).value,
                          g.v_z.to(u.km / u.s).value], -1))
    p0, v0 = to_gc([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    axes, _ = to_gc(np.eye(3) * 1000.0, np.zeros((3, 3)))
    rot = ((axes - p0) / 1000.0).T                                   # gc = rot @ icrs
    return {'pos_pc': p0[0], 'vel_kms': v0[0], 'icrs_to_gc': rot}


if __name__ == '__main__':
    cat = build()
    print(len(cat), 'stars')
    print(cat.groupby(['source', 'sample']).size())
    print('rv sources:', cat['rv_src'].value_counts().to_dict())
