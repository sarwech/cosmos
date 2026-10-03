"""Write the binary files the Time page loads (time/data/).

Global star order: the PRECISION set first (float32 states — every star within
25 pc, every Hipparcos-sourced, named or constellation star, and anything too
fast for int16 velocities), then the BULK in order of apparent brightness from
Earth, split into tiers so the brightest render first.

stars-p.bin   32 B/star  f32 dx,dy,dz (pc) · f32 dvx,dvy,dvz (km/s) ·
                         u8 teff · u8 absmag · u16 flags · u32 hip
stars-N.bin   16 B/star  u16 oct_u, oct_v (direction) · u16 log-distance ·
                         i16 dvx,dvy,dvz (0.01 km/s) · u8 teff · u8 absmag · u16 flags
ids-NN.bin     8 B/star  i64 Gaia DR3 source_id (0 = Hipparcos only)
info-NN.bin   32 B/star  f32 parallax, σϖ, σμα*, σμδ, σRV, BP−RP · u32 hip · f32 RUWE
force-mcmillan17.bin     f32 [v][u][F_R, F_z]  (pc/Myr²)
All vectors are Sun-relative, in astropy Galactocentric (v4.0) axes; little-endian.
"""
import hashlib
import json

import numpy as np

from . import colour
from . import config as C
from . import fetch_meta
from .catalogue import RV_NONE, SRC_HIP, solar_state
from .potential import DU, DV, table_float32

FLAG_HIP = 1 << 0                 # astrometry from Hipparcos-2
RV_SHIFT = 1                      # bits 1-2: RV source (0 Gaia, 1 XHIP, 2 SIMBAD, 3 none → 0)
SAMPLE_SHIFT = 3                  # bits 3-4: 0 local, 1 disk, 2 bright/named
FLAG_NAMED = 1 << 5
FLAG_FIGURE = 1 << 6
FLAG_PRECISE = 1 << 7
FLAG_POOR_PLX = 1 << 8            # parallax / error < 5: distance poorly constrained

ABSMAG_MIN, ABSMAG_STEP = -10.0, 0.1


def oct_encode(n):
    n = n / np.abs(n).sum(-1, keepdims=True)
    p = n[:, :2].copy()
    neg = n[:, 2] < 0
    sx = np.where(p[neg, 0] >= 0, 1.0, -1.0)
    sy = np.where(p[neg, 1] >= 0, 1.0, -1.0)
    px, py = p[neg, 0].copy(), p[neg, 1].copy()
    p[neg, 0] = (1 - np.abs(py)) * sx
    p[neg, 1] = (1 - np.abs(px)) * sy
    return np.round((np.clip(p, -1, 1) + 1) * 0.5 * 65535).astype(np.uint16)


def oct_decode(q):
    p = q.astype(np.float64) / 65535 * 2 - 1
    n = np.stack([p[:, 0], p[:, 1], 1 - np.abs(p[:, 0]) - np.abs(p[:, 1])], -1)
    t = np.maximum(-n[:, 2], 0)
    n[:, 0] += np.where(n[:, 0] >= 0, -t, t)
    n[:, 1] += np.where(n[:, 1] >= 0, -t, t)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _flags(cat, figure_hips):
    f = np.zeros(len(cat), np.uint16)
    f |= np.where(cat['source'] == SRC_HIP, FLAG_HIP, 0).astype(np.uint16)
    f |= (cat['rv_src'].to_numpy().astype(np.uint16) & 3) << RV_SHIFT
    f |= (cat['sample'].to_numpy().astype(np.uint16) & 3) << SAMPLE_SHIFT
    f |= np.where(cat['name'] != '', FLAG_NAMED, 0).astype(np.uint16)
    f |= np.where(cat['hip'].isin(figure_hips), FLAG_FIGURE, 0).astype(np.uint16)
    poe = cat['plx'] / cat['plx_err']
    f |= np.where(poe < 5, FLAG_POOR_PLX, 0).astype(np.uint16)
    return f


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def order_and_tiers(cat, figure_hips):
    dv = cat[['dv_x', 'dv_y', 'dv_z']].to_numpy()
    precise = ((cat['dist'] < C.PRECISION_RADIUS_PC) | (cat['source'] == SRC_HIP)
               | (cat['name'] != '') | cat['hip'].isin(figure_hips)
               | (np.abs(dv).max(1) > 32767 * C.VEL_QUANT)).to_numpy()
    # tracked = named or constellation stars: first, so the page can read their
    # states back cheaply for labels and constellation lines
    tracked = ((cat['name'] != '') | cat['hip'].isin(figure_hips)).to_numpy()
    p_idx = np.flatnonzero(precise)
    b_idx = np.flatnonzero(~precise)
    p_idx = p_idx[np.lexsort((cat['gmag'].to_numpy()[p_idx], ~tracked[p_idx]))]
    b_idx = b_idx[np.argsort(cat['gmag'].to_numpy()[b_idx], kind='stable')]
    return p_idx, b_idx, int(tracked.sum())


def write(cat, uncertainty=None):
    C.OUT.mkdir(parents=True, exist_ok=True)
    for old in C.OUT.glob('*.bin'):
        old.unlink()
    figures = fetch_meta.constellations()
    figure_hips = {h for f in figures for l in f['lines'] for h in l}
    p_idx, b_idx, n_tracked = order_and_tiers(cat, figure_hips)
    order = np.concatenate([p_idx, b_idx])
    cat = cat.iloc[order].reset_index(drop=True)
    n_p = len(p_idx)
    flags = _flags(cat, figure_hips)
    flags[:n_p] |= FLAG_PRECISE
    teff = colour.teff_code(cat['logteff'].to_numpy())
    absm = np.clip(np.round((cat['abs_g'].to_numpy() - ABSMAG_MIN) / ABSMAG_STEP), 0, 255).astype(np.uint8)
    dx = cat[['dx_x', 'dx_y', 'dx_z']].to_numpy()
    dv = cat[['dv_x', 'dv_y', 'dv_z']].to_numpy()
    files = {}

    # precision tier
    rec = np.zeros(n_p, dtype=[('x', '<f4', 3), ('v', '<f4', 3), ('teff', 'u1'), ('absm', 'u1'),
                               ('flags', '<u2'), ('hip', '<u4')])
    rec['x'], rec['v'] = dx[:n_p], dv[:n_p]
    rec['teff'], rec['absm'], rec['flags'] = teff[:n_p], absm[:n_p], flags[:n_p]
    rec['hip'] = cat['hip'].to_numpy()[:n_p]
    (C.OUT / 'stars-p.bin').write_bytes(rec.tobytes())
    files['stars-p.bin'] = {'count': n_p, 'first': 0, 'record': 32}

    # bulk tiers
    nb = len(b_idx)
    d = np.linalg.norm(dx[n_p:], axis=1)
    rec = np.zeros(nb, dtype=[('oct', '<u2', 2), ('ld', '<u2'), ('v', '<i2', 3), ('teff', 'u1'),
                              ('absm', 'u1'), ('flags', '<u2')])
    rec['oct'] = oct_encode(dx[n_p:] / d[:, None])
    ld = (np.log10(d) - C.DIST_LOG_MIN) / (C.DIST_LOG_MAX - C.DIST_LOG_MIN)
    rec['ld'] = np.round(np.clip(ld, 0, 1) * 65535).astype(np.uint16)
    rec['v'] = np.round(dv[n_p:] / C.VEL_QUANT).astype(np.int16)
    rec['teff'], rec['absm'], rec['flags'] = teff[n_p:], absm[n_p:], flags[n_p:]
    sizes = [C.FIRST_TIER] + [C.TIER_SIZE] * ((max(nb - C.FIRST_TIER, 0) + C.TIER_SIZE - 1) // C.TIER_SIZE)
    start = 0
    tiers = []
    for k, s in enumerate(sizes):
        chunk = rec[start:start + s]
        if not len(chunk):
            break
        name = f'stars-{k}.bin'
        (C.OUT / name).write_bytes(chunk.tobytes())
        files[name] = {'count': len(chunk), 'first': n_p + start, 'record': 16}
        tiers.append(name)
        start += s

    # lazily-loaded ids and info shards
    sid = cat['source_id'].to_numpy().astype('<i8')
    info = np.zeros(len(cat), dtype=[('plx', '<f4'), ('plx_err', '<f4'), ('pmra_err', '<f4'),
                                     ('pmdec_err', '<f4'), ('rv_err', '<f4'), ('bp_rp', '<f4'),
                                     ('hip', '<u4'), ('ruwe', '<f4')])
    for c in ('plx', 'plx_err', 'pmra_err', 'pmdec_err', 'rv_err', 'bp_rp', 'ruwe'):
        info[c] = cat[c].to_numpy(float)
    info['hip'] = cat['hip'].to_numpy()
    n_sh = (len(cat) + C.ID_SHARD - 1) // C.ID_SHARD
    for k in range(n_sh):
        sl = slice(k * C.ID_SHARD, (k + 1) * C.ID_SHARD)
        (C.OUT / f'ids-{k:02d}.bin').write_bytes(sid[sl].tobytes())
        (C.OUT / f'info-{k:02d}.bin').write_bytes(info[sl].tobytes())

    # potential table
    (C.OUT / 'force-mcmillan17.bin').write_bytes(table_float32().tobytes())

    # names: every named star (+ aliases), and a HIP → index map
    names = []
    for i in np.flatnonzero((cat['name'] != '').to_numpy() | (cat['bayer'] != '').to_numpy()):
        alias = [a for a in {cat['cosmos_name'].iloc[i], cat['bayer'].iloc[i]}
                 if a and a != cat['name'].iloc[i]]
        names.append([int(i), cat['name'].iloc[i] or cat['bayer'].iloc[i], '|'.join(sorted(alias))])
    hip_pairs = np.array([(h, i) for i, h in enumerate(cat['hip'].to_numpy()) if h], dtype='<u4')
    hip_pairs = hip_pairs[np.argsort(hip_pairs[:, 0])]
    (C.OUT / 'hip-index.bin').write_bytes(hip_pairs.tobytes())
    (C.OUT / 'names.json').write_text(json.dumps(
        {'attribution': fetch_meta.STELLARIUM_ATTRIBUTION, 'stars': names}, separators=(',', ':')))

    # constellation figures as global indices
    hip_to_i = dict(zip(hip_pairs[:, 0].tolist(), hip_pairs[:, 1].tolist()))
    figs, dropped = [], 0
    for f in figures:
        lines = []
        for l in f['lines']:
            seg = []
            for h in l:
                if h in hip_to_i:
                    seg.append(hip_to_i[h])
                else:
                    dropped += 1
                    if len(seg) >= 2:
                        lines.append(seg)
                    seg = []
            if len(seg) >= 2:
                lines.append(seg)
        figs.append({'abbr': f['abbr'], 'name': f['name'], 'english': f['english'], 'lines': lines})
    (C.OUT / 'constellations.json').write_text(json.dumps(
        {'attribution': fetch_meta.STELLARIUM_ATTRIBUTION, 'figures': figs}, separators=(',', ':')))

    sun = solar_state()
    manifest = {
        'version': 1,
        'epoch': C.EPOCH,
        'count': int(len(cat)),
        'precise_count': int(n_p),
        'tracked_count': n_tracked,
        'tiers': ['stars-p.bin'] + tiers,
        'files': files,
        'shards': {'size': C.ID_SHARD, 'count': int(n_sh), 'ids': 'ids-{k}.bin', 'info': 'info-{k}.bin'},
        'frame': {'astropy_defaults': C.FRAME, 'R0_kpc': C.R0_KPC, 'z_sun_pc': C.Z_SUN_PC,
                  'v_sun_kms': list(C.V_SUN_KMS),
                  'sun_pos_pc': sun['pos_pc'].tolist(), 'sun_vel_kms': sun['vel_kms'].tolist(),
                  'icrs_to_gc': sun['icrs_to_gc'].tolist()},
        'potential': {'name': C.POTENTIAL, 'file': 'force-mcmillan17.bin', 'n': C.GRID_N,
                      'a_r': C.GRID_A_R, 'a_z': C.GRID_A_Z, 'du': float(DU), 'dv': float(DV),
                      'units': 'pc, Myr; table holds F_R, F_z in pc/Myr^2'},
        'integrator': {'scheme': 'kick-drift-kick leapfrog, Sun-relative, cubic Hermite in-step',
                       'dt_myr': C.DT_MYR, 't_max_myr': C.T_MAX_MYR,
                       'kms_to_pcmyr': C.KMS_TO_PCMYR},
        'quant': {'vel_kms': C.VEL_QUANT, 'logdist_min': C.DIST_LOG_MIN, 'logdist_max': C.DIST_LOG_MAX,
                  'absmag_min': ABSMAG_MIN, 'absmag_step': ABSMAG_STEP,
                  'logteff_min': float(colour.LOGT_MIN), 'logteff_max': float(colour.LOGT_MAX)},
        'flags': {'hip': FLAG_HIP, 'rv_shift': RV_SHIFT, 'sample_shift': SAMPLE_SHIFT,
                  'named': FLAG_NAMED, 'figure': FLAG_FIGURE, 'precise': FLAG_PRECISE,
                  'poor_parallax': FLAG_POOR_PLX,
                  'rv_sources': ['Gaia DR3', 'XHIP', 'SIMBAD', 'none (assumed 0)'],
                  'samples': ['local (nearest)', 'disk (random)', 'bright/named']},
        'palette': np.round(colour.palette(), 4).tolist(),
        'uncertainty': uncertainty,
        'stats': {
            'gaia': int((cat['source'] == 0).sum()), 'hipparcos': int((cat['source'] == SRC_HIP).sum()),
            'local': int((cat['sample'] == 0).sum()), 'disk': int((cat['sample'] == 1).sum()),
            'bright': int((cat['sample'] == 2).sum()),
            'rv_none': int((cat['rv_src'] == RV_NONE).sum()),
            'figure_stars_missing': int(dropped),
        },
        'attribution': ['ESA/Gaia/DPAC — Gaia DR3 (CC BY-SA 3.0 IGO)',
                        'Bailer-Jones et al. 2021, distances (Gaia archive external.gaiaedr3_distance)',
                        'van Leeuwen 2007, Hipparcos new reduction (VizieR I/311)',
                        'Anderson & Francis 2012, XHIP (VizieR V/137D)',
                        'SIMBAD (CDS)', fetch_meta.STELLARIUM_ATTRIBUTION,
                        'McMillan 2017 potential via galpy (Bovy 2015)'],
    }
    total = 0
    for p in sorted(C.OUT.iterdir()):
        if p.suffix in ('.bin', '.json') and p.name != 'manifest.json':
            manifest.setdefault('sizes', {})[p.name] = p.stat().st_size
            total += p.stat().st_size
    manifest['hash'] = {k: _sha(C.OUT / k) for k in manifest['tiers'] + ['force-mcmillan17.bin']}
    manifest['total_bytes'] = total
    (C.OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return cat


def read():
    """Decode the packed files exactly as the browser does → arrays for the checks."""
    m = json.loads((C.OUT / 'manifest.json').read_text())
    xs, vs, te, am, fl, hips = [], [], [], [], [], []
    for name in m['tiers']:
        raw = (C.OUT / name).read_bytes()
        if name == 'stars-p.bin':
            r = np.frombuffer(raw, dtype=[('x', '<f4', 3), ('v', '<f4', 3), ('teff', 'u1'), ('absm', 'u1'),
                                          ('flags', '<u2'), ('hip', '<u4')])
            xs.append(r['x'].astype(np.float64)); vs.append(r['v'].astype(np.float64)); hips.append(r['hip'])
        else:
            r = np.frombuffer(raw, dtype=[('oct', '<u2', 2), ('ld', '<u2'), ('v', '<i2', 3), ('teff', 'u1'),
                                          ('absm', 'u1'), ('flags', '<u2')])
            d = 10 ** (C.DIST_LOG_MIN + r['ld'].astype(np.float64) / 65535 * (C.DIST_LOG_MAX - C.DIST_LOG_MIN))
            xs.append(oct_decode(r['oct']) * d[:, None])
            vs.append(r['v'].astype(np.float64) * C.VEL_QUANT)
            hips.append(np.zeros(len(r), np.uint32))
        te.append(r['teff']); am.append(r['absm']); fl.append(r['flags'])
    n = m['count']
    sid = np.concatenate([np.frombuffer((C.OUT / f'ids-{k:02d}.bin').read_bytes(), '<i8')
                          for k in range(m['shards']['count'])])
    info = np.concatenate([np.frombuffer((C.OUT / f'info-{k:02d}.bin').read_bytes(),
                                         dtype=[('plx', '<f4'), ('plx_err', '<f4'), ('pmra_err', '<f4'),
                                                ('pmdec_err', '<f4'), ('rv_err', '<f4'), ('bp_rp', '<f4'),
                                                ('hip', '<u4'), ('ruwe', '<f4')])
                           for k in range(m['shards']['count'])])
    # round exactly as the browser holds them (Float32Array of pc and pc/Myr),
    # so checks and fixtures start from bit-identical initial conditions
    dx = np.concatenate(xs)[:n].astype(np.float32).astype(np.float64)
    dv = (np.concatenate(vs)[:n] * C.KMS_TO_PCMYR).astype(np.float32).astype(np.float64) / C.KMS_TO_PCMYR
    return {'manifest': m, 'dx': dx, 'dv_kms': dv,
            'teff': np.concatenate(te), 'absm': np.concatenate(am), 'flags': np.concatenate(fl),
            'source_id': sid, 'info': info, 'hip': info['hip']}
