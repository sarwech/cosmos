"""Names and constellation figures.

* Constellation figures and common star names: Stellarium "modern" sky
  culture (88 IAU constellations, figures by Hipparcos number).
  Licence: CC BY-SA 4.0 — time/data/constellations.json and the name list
  inherit it and carry the attribution.
* Cosmos's ~90 named stars: parsed straight out of ../index.html (so the
  t = 0 check compares against exactly what Cosmos shows) and resolved through
  SIMBAD to Hipparcos / Gaia DR3 identifiers, with SIMBAD's radial velocity as
  a last-resort RV for stars that have neither a Gaia nor an XHIP one.
"""
import json
import re
import time

import pandas as pd

from . import config as C

STELLARIUM_URL = ('https://raw.githubusercontent.com/Stellarium/stellarium/master/'
                  'skycultures/modern/index.json')
STELLARIUM_ATTRIBUTION = ('Constellation figures and star names: Stellarium "modern" sky culture, '
                          'CC BY-SA 4.0 (https://github.com/Stellarium/stellarium)')

# SIMBAD spellings for Cosmos names it does not resolve as written
SIMBAD_ALIASES = {
    'Alpha Centauri A': 'alf Cen A', 'Alpha Centauri B': 'alf Cen B',
    'Upsilon And': 'ups And', 'Mu Arae': 'mu Ara', 'Pi Mensae': 'pi Men',
    'Gliese 667': 'GJ 667 C', '40 Eridani': 'omi02 Eri', '70 Ophiuchi': '70 Oph',
    '61 Cygni A': '61 Cyg A', 'Kruger 60': 'HIP 110893', 'Castor': 'alf Gem A',
    'Mizar': 'zet UMa A', 'Capella': 'alf Aur', 'Acrux': 'alf Cru',
    'Gliese 581': 'GJ 581', 'Gliese 876': 'GJ 876', 'Gliese 832': 'GJ 832',
}
# stars the stories need that Cosmos does not list
EXTRA_NAMED = ['Gliese 710', 'Alcor']


def stellarium():
    path = C.CACHE / 'stellarium_modern.json'
    if not path.exists():
        import requests
        path.write_bytes(requests.get(STELLARIUM_URL, timeout=120).content)
    return json.loads(path.read_text())


def constellations():
    """[{abbr, name, lines: [[hip, hip, ...], ...]}] — polylines of HIP numbers."""
    out = []
    for c in stellarium()['constellations']:
        abbr = c['id'].split()[-1]
        lines = [[h for h in seg if isinstance(h, int)] for seg in c['lines']]
        out.append({'abbr': abbr, 'name': c['common_name'].get('native', abbr),
                    'english': c['common_name'].get('english', ''),
                    'lines': [l for l in lines if len(l) >= 2]})
    return out


def common_names():
    """HIP -> primary common name (the first, best-referenced Stellarium entry)."""
    names = {}
    for k, v in stellarium()['common_names'].items():
        if k.startswith('HIP '):
            names[int(k[4:])] = v[0]['english']
    return names


def cosmos_stars():
    """Parse the STARS table out of index.html: name, dist(ly), RA, Dec, class."""
    html = (C.REPO / 'index.html').read_text()
    block = html[html.index('const STARS = ['):]
    block = block[:block.index('];')]
    rows = []
    for m in re.finditer(r"\[\s*(['\"])(.+?)\1\s*,\s*([\d.]+)\s*,\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,\s*'(\w)'",
                         block):
        rows.append({'name': m.group(2).replace("\\'", "'"), 'dist_ly': float(m.group(3)),
                     'ra': float(m.group(4)), 'dec': float(m.group(5)), 'cls': m.group(6)})
    return pd.DataFrame(rows)


def simbad_ids(names):
    """name -> {hip, gaia_dr3, rv, rv_err} via SIMBAD. Cached."""
    path = C.CACHE / 'simbad_names.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [n for n in names if n not in cache]
    if todo:
        import warnings
        warnings.filterwarnings('ignore')
        from astroquery.simbad import Simbad
        s = Simbad()
        s.add_votable_fields('ids', 'rvz_radvel', 'rvz_err')
        for n in todo:
            q = SIMBAD_ALIASES.get(n, n)
            for attempt in range(4):
                try:
                    t = s.query_object(q)
                    break
                except Exception as e:
                    print(f'  simbad {q}: {e}')
                    time.sleep(5 * 2 ** attempt)
                    t = None
            rec = {'query': q, 'hip': None, 'gaia_dr3': None, 'rv': None, 'rv_err': None}
            if t is not None and len(t):
                cols = {c.lower(): c for c in t.colnames}
                ids = str(t[cols['ids']][0]).split('|')
                for i in ids:
                    m = re.fullmatch(r'HIP\s+(\d+)\w?', i.strip())
                    if m and rec['hip'] is None:
                        rec['hip'] = int(m.group(1))
                    m = re.fullmatch(r'Gaia DR3\s+(\d+)', i.strip())
                    if m:
                        rec['gaia_dr3'] = int(m.group(1))
                rv = t[cols['rvz_radvel']][0]
                if rv is not None and str(rv) not in ('--', 'nan'):
                    rec['rv'] = float(rv)
                    e = t[cols['rvz_err']][0]
                    rec['rv_err'] = float(e) if str(e) not in ('--', 'nan') else None
            else:
                print(f'  simbad: no match for {n!r} (queried {q!r})')
            cache[n] = rec
            time.sleep(0.3)
        path.write_text(json.dumps(cache, indent=1))
    return {n: cache[n] for n in names}


def simbad_coords(names):
    """name -> SIMBAD ICRS (ra, dec) at epoch J2000 — an independent referee for
    the t = 0 sky check. Cached separately from simbad_ids."""
    path = C.CACHE / 'simbad_coords.json'
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [n for n in names if n not in cache]
    if todo:
        import warnings
        warnings.filterwarnings('ignore')
        from astroquery.simbad import Simbad
        s = Simbad()
        for n in todo:
            t = None
            for attempt in range(4):
                try:
                    t = s.query_object(SIMBAD_ALIASES.get(n, n))
                    break
                except Exception as e:
                    print(f'  simbad {n}: {e}')
                    time.sleep(5 * 2 ** attempt)
            if t is not None and len(t):
                cols = {c.lower(): c for c in t.colnames}
                cache[n] = [float(t[cols['ra']][0]), float(t[cols['dec']][0])]
            else:
                cache[n] = None
            time.sleep(0.3)
        path.write_text(json.dumps(cache, indent=1))
    return {n: cache[n] for n in names}


if __name__ == '__main__':
    cs = cosmos_stars()
    print(len(cs), 'Cosmos stars')
    ids = simbad_ids(list(cs.name) + EXTRA_NAMED)
    missing = [n for n, r in ids.items() if not r['hip'] and not r['gaia_dr3']]
    print('unresolved:', missing)
    print(len(constellations()), 'constellations;', len(common_names()), 'named HIP stars')
