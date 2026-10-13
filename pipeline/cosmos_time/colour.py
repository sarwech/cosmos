"""Star colour: Gaia BP−RP (or Hipparcos B−V) → Teff → blackbody sRGB.

Colour–temperature: Pecaut & Mamajek (2013) dwarf sequence, version
2022.04.16 (https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt).
Teff → RGB: a Planck spectrum integrated against the CIE 1931 2° colour
matching functions (Wyman, Sloan & Shirley 2013 analytic fit), converted to
linear sRGB (D65) and normalised to a peak channel of 1 — brightness comes
from the absolute magnitude, not from the colour.

Each star stores a one-byte Teff code; the page gets the 256-entry palette.
"""
import numpy as np

from . import config as C

MAMAJEK_URL = 'https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt'
LOGT_MIN, LOGT_MAX = np.log10(2300.0), np.log10(40000.0)


def _table():
    path = C.CACHE / 'mamajek_dwarfs.txt'
    if not path.exists():
        import requests
        path.write_text(requests.get(MAMAJEK_URL, timeout=120).text)
    rows, header = [], None
    for line in path.read_text().splitlines():
        if line.startswith('#SpT'):
            if header is not None:
                break                                    # only the first (main) table
            header = line[1:].split()
            continue
        if header is None or not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < len(header) - 1:
            continue
        rows.append(dict(zip(header, parts)))

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (ValueError, KeyError):
                out.append(np.nan)
        return np.array(out)
    return col('Teff'), col('B-V'), col('Bp-Rp')


def _inverse(colour, teff):
    """Monotone colour → log Teff interpolator (drops non-monotone tails)."""
    ok = np.isfinite(colour) & np.isfinite(teff)
    c, lt = colour[ok], np.log10(teff[ok])
    order = np.argsort(c)
    c, lt = c[order], lt[order]
    keep = np.concatenate([[True], np.diff(lt) < 0])            # Teff must fall as colour reddens
    c, lt = c[keep], lt[keep]
    # The BP−RP column stops at A0 (−0.12, 10,700 K). Hotter Gaia stars are
    # extrapolated along the bluest table segment, capped at 40,000 K.
    slope = (lt[3] - lt[0]) / (c[3] - c[0])

    def f(x):
        y = np.interp(x, c, lt)
        blue = x < c[0]
        return np.where(blue, np.minimum(lt[0] + slope * (x - c[0]), LOGT_MAX), y)
    return f


_TEFF, _BV, _BPRP = None, None, None


def _load():
    global _TEFF, _BV, _BPRP
    if _TEFF is None:
        _TEFF, _BV, _BPRP = _table()


def bprp_to_logteff(bprp):
    _load()
    return _inverse(_BPRP, _TEFF)(np.asarray(bprp, float))


def bv_to_logteff(bv):
    _load()
    return _inverse(_BV, _TEFF)(np.asarray(bv, float))


def logteff_to_bprp(lt):
    """For Hipparcos stars: the BP−RP a dwarf of this Teff has (for the card)."""
    _load()
    ok = np.isfinite(_BPRP)
    x, y = np.log10(_TEFF[ok])[::-1], _BPRP[ok][::-1]
    return np.interp(lt, x, y)


def teff_code(logt):
    """log10 Teff → uint8 (0 = 2,300 K … 255 = 40,000 K, log-spaced)."""
    f = (np.asarray(logt, float) - LOGT_MIN) / (LOGT_MAX - LOGT_MIN)
    return np.clip(np.round(np.nan_to_num(f, nan=0.6) * 255), 0, 255).astype(np.uint8)


def code_logteff(code):
    return LOGT_MIN + np.asarray(code, float) / 255 * (LOGT_MAX - LOGT_MIN)


def _cie_xyz(lam_nm):
    """CIE 1931 2° CMFs, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013)."""
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)
    x = (1.056 * g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * g(lam_nm, 442.0, 16.0, 26.7)
         - 0.065 * g(lam_nm, 501.1, 20.4, 26.2))
    y = 0.821 * g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * g(lam_nm, 459.0, 26.0, 13.8)
    return x, y, z


def blackbody_rgb(teff):
    """Linear sRGB (peak channel = 1) of a blackbody at teff kelvin."""
    lam = np.arange(380.0, 781.0, 2.0)
    xb, yb, zb = _cie_xyz(lam)
    teff = np.atleast_1d(np.asarray(teff, float))
    hc_k = 1.4387769e7                                         # nm·K
    planck = lam[None, :] ** -5 / np.expm1(hc_k / (lam[None, :] * teff[:, None]))
    X, Y, Z = (planck * xb).sum(1), (planck * yb).sum(1), (planck * zb).sum(1)
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = (m @ np.stack([X, Y, Z])).T
    rgb = np.clip(rgb, 0, None)
    return rgb / rgb.max(axis=1, keepdims=True)


def palette():
    """256 × RGB (linear, 0-1) indexed by teff_code."""
    return blackbody_rgb(10 ** code_logteff(np.arange(256)))


if __name__ == '__main__':
    for bprp in (-0.2, 0.0, 0.5, 0.82, 1.5, 2.5, 3.5):
        lt = bprp_to_logteff(bprp)
        print(f'BP-RP {bprp:5.2f} → {10**lt:7.0f} K → rgb {np.round(blackbody_rgb(10**lt)[0], 3)}')
    print('B-V 0.65 →', round(float(10 ** bv_to_logteff(0.65))), 'K')
