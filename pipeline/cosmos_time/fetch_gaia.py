"""Fetch Gaia DR3 6D astrometry from the ESA Gaia archive (anonymous access).

Anonymous limits (archive FAQ, 2026): 3,000,000 rows per query, 120 min async
timeout, results kept 3 days. Every query here is split so each job returns
well under 300k rows; nothing needs a registered account.

Each query result is cached in cache/<name>.parquet, so re-running the
pipeline never re-queries the archive unless the cache is deleted.
"""
import time

import numpy as np
import pandas as pd

from . import config as C

GAIA_COLS = """g.source_id, g.ra, g.dec, g.parallax, g.parallax_error,
  g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
  g.parallax_pmra_corr, g.parallax_pmdec_corr, g.pmra_pmdec_corr,
  g.radial_velocity, g.radial_velocity_error, g.rv_template_teff, g.grvs_mag,
  g.phot_g_mean_mag, g.bp_rp, g.ruwe, g.parallax_over_error,
  g.ag_gspphot, g.ebpminrp_gspphot,
  d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo, d.r_med_geo"""
GAIA_FROM = """gaiadr3.gaia_source AS g
  LEFT OUTER JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id"""
QUALITY = f"""g.radial_velocity IS NOT NULL AND g.ruwe < {C.MAX_RUWE}"""

# local sample, nearest first, split by parallax (the archive indexes parallax)
LOCAL_SLICES = [(12.0, None), (8.0, 12.0), (6.0, 8.0), (5.0, 6.0), (C.LOCAL_MIN_PARALLAX, 5.0)]


def _gaia():
    import warnings
    warnings.filterwarnings('ignore')
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    return Gaia


def run(name, query, retries=4):
    """Run an async ADQL job, cache as parquet, return a DataFrame."""
    path = C.CACHE / f'{name}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    Gaia = _gaia()
    for attempt in range(retries):
        t0 = time.time()
        try:
            tbl = Gaia.launch_job_async(query, verbose=False).get_results()
            break
        except Exception as e:                                   # archive hiccups
            print(f'  {name}: attempt {attempt + 1} failed after {time.time() - t0:.0f}s: {e}')
            if attempt == retries - 1:
                raise
            time.sleep(15 * 2 ** attempt)
    df = tbl.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    C.CACHE.mkdir(exist_ok=True)
    df.to_parquet(path)
    print(f'  {name}: {len(df):,} rows in {time.time() - t0:.0f}s')
    return df


def count(query):
    return int(run_uncached(query).iloc[0, 0])


def run_uncached(query):
    Gaia = _gaia()
    df = Gaia.launch_job_async(query, verbose=False).get_results().to_pandas()
    return df


def local_sample():
    parts = []
    for lo, hi in LOCAL_SLICES:
        cond = f'g.parallax > {lo}' + (f' AND g.parallax <= {hi}' if hi else '')
        q = f"""SELECT {GAIA_COLS} FROM {GAIA_FROM}
WHERE {cond} AND {QUALITY} AND g.parallax_over_error >= {C.LOCAL_MIN_POE}"""
        parts.append(run(f'gaia_local_{lo:g}_{hi or "inf"}', q))
    df = pd.concat(parts, ignore_index=True)
    df['sample'] = 'local'
    return df


def disk_sample():
    """A random draw (Gaia's random_index) of the 6D catalogue beyond the local
    cut. random_index is a random permutation of all 1.8 billion sources, so a
    cut random_index < K is an unbiased subsample; K is set from a count query
    so the result is ~DISK_TARGET rows."""
    path = C.CACHE / 'gaia_disk.parquet'
    if path.exists():
        return pd.read_parquet(path).assign(sample='disk')
    where = f"""g.parallax <= {C.LOCAL_MIN_PARALLAX} AND {QUALITY}
  AND g.parallax_over_error >= {C.DISK_MIN_POE}"""
    probe = 20_000_000
    n = count(f'SELECT COUNT(*) FROM gaiadr3.gaia_source AS g WHERE g.random_index < {probe} AND {where}')
    k = int(probe * C.DISK_TARGET / max(n, 1))
    print(f'  disk sample: {n:,} rows per {probe:,} random_index → K = {k:,}')
    parts = []
    edges = np.linspace(0, k, 5).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        q = f"""SELECT {GAIA_COLS}, g.random_index FROM {GAIA_FROM}
WHERE g.random_index >= {a} AND g.random_index < {b} AND {where}"""
        parts.append(run(f'gaia_disk_{a}_{b}', q))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(path)
    return df.assign(sample='disk')


def hipparcos_matches():
    """Gaia DR3 rows for every Hipparcos-2 star with a Gaia counterpart
    (gaiadr3.hipparcos2_best_neighbour), with no quality cut — the merge step
    decides per star whether Gaia or Hipparcos is better."""
    q = f"""SELECT h.original_ext_source_id AS hip, h.angular_distance, h.number_of_neighbours,
  {GAIA_COLS} FROM gaiadr3.hipparcos2_best_neighbour AS h
  JOIN gaiadr3.gaia_source AS g ON h.source_id = g.source_id
  LEFT OUTER JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id"""
    return run('gaia_hip_matches', q)


def by_source_id(name, ids):
    """Gaia rows for an explicit list of source_ids (named stars outside HIP)."""
    ids = sorted({int(i) for i in ids})
    q = f"""SELECT {GAIA_COLS} FROM {GAIA_FROM}
WHERE g.source_id IN ({', '.join(map(str, ids))})"""
    return run(name, q)


if __name__ == '__main__':
    local_sample()
    hipparcos_matches()
    disk_sample()
