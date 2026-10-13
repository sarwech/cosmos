"""pytest entry point for the known-answer and engineering checks.

    cd pipeline && .venv/bin/python -m pytest -q

Runs on the packed files in time/data (what the browser loads) and writes
reports/validation.json. A failing check fails the test — thresholds live in
cosmos_time/validate.py:CRITERIA and were fixed before any data was fetched.
"""
import pytest

from cosmos_time import validate

_R = {}


def results():
    if not _R:
        _R.update(validate.run_all())
    return _R


@pytest.mark.parametrize('key', ['1_sky_matches_cosmos', '2_barnards_star', '3_gliese_710',
                                 '4_big_dipper', '5_sun_orbit'])
def test_known_answer(key):
    r = results()[key]
    assert r['pass'], {k: v for k, v in r.items() if k not in ('stars', 'sky_radec_deg')}


@pytest.mark.parametrize('key', ['force_table_accuracy', 'energy', 'convergence', 'straight_line',
                                 'reversibility'])
def test_engineering(key):
    r = results()['engineering'][key]
    assert r['pass'], r
