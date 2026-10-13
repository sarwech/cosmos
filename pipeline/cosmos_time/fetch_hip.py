"""Hipparcos fallback data: XHIP (Anderson & Francis 2012, VizieR V/137D).

XHIP carries the Hipparcos new reduction (van Leeuwen 2007) astrometry at
epoch J1991.25 together with compiled literature radial velocities — exactly
what bright stars need where Gaia's astrometry is saturated or absent.
"""
import time

import pandas as pd

from . import config as C

XHIP_COLS = ['HIP', 'RAdeg', 'DEdeg', 'Plx', 'e_Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE',
             'RV', 'e_RV', 'q_RV', 'Vmag', 'B-V', 'SpType', 'Name', 'Cst']


def xhip():
    path = C.CACHE / 'xhip.parquet'
    if path.exists():
        return pd.read_parquet(path)
    import warnings
    warnings.filterwarnings('ignore')
    from astroquery.vizier import Vizier
    v = Vizier(columns=XHIP_COLS, row_limit=-1, timeout=600)
    for attempt in range(5):
        try:
            t0 = time.time()
            tbl = v.get_catalogs('V/137D/XHIP')[0]
            break
        except Exception as e:
            print(f'  xhip: attempt {attempt + 1} failed: {e}')
            if attempt == 4:
                raise
            time.sleep(10 * 2 ** attempt)
    df = tbl.to_pandas()
    df = df.rename(columns={'B-V': 'b_v'})
    df.columns = [c.lower() for c in df.columns]
    for c in ('sptype', 'name', 'cst', 'q_rv'):
        df[c] = df[c].astype(str).str.strip()
    df.to_parquet(path)
    print(f'  xhip: {len(df):,} rows in {time.time() - t0:.0f}s')
    return df


def hip2():
    """Hipparcos new reduction (van Leeuwen 2007, VizieR I/311): ICRS position
    at epoch J1991.25. XHIP shares this astrometry but VizieR only serves its
    positions re-computed to J2000, so positions come from here."""
    path = C.CACHE / 'hip2.parquet'
    if path.exists():
        return pd.read_parquet(path)
    import warnings
    warnings.filterwarnings('ignore')
    from astroquery.vizier import Vizier
    v = Vizier(columns=['HIP', 'RArad', 'DErad', 'Plx', 'pmRA', 'pmDE', 'Hpmag'],
               row_limit=-1, timeout=600)
    for attempt in range(5):
        try:
            t0 = time.time()
            tbl = v.get_catalogs('I/311/hip2')[0]
            break
        except Exception as e:
            print(f'  hip2: attempt {attempt + 1} failed: {e}')
            if attempt == 4:
                raise
            time.sleep(10 * 2 ** attempt)
    df = tbl.to_pandas()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={'rarad': 'ra', 'derad': 'dec'})
    df.to_parquet(path)
    print(f'  hip2: {len(df):,} rows in {time.time() - t0:.0f}s')
    return df


if __name__ == '__main__':
    print(xhip().head())
    print(hip2().head())
