"""Run the whole pipeline: fetch → merge → pack → validate.

    python -m cosmos_time.build            # everything (cached steps are skipped)
    python -m cosmos_time.build --validate # only re-run the checks
"""
import json
import sys
import time

from . import config as C


def main():
    t0 = time.time()
    if '--validate' not in sys.argv:
        from . import catalogue, fetch_gaia, fetch_hip, fetch_meta, pack, potential, table_accuracy, uncertainty
        print('1/6 force table (McMillan17 via galpy)')
        potential.build()
        if not (C.REPORTS / 'table_accuracy.json').exists():
            table_accuracy.run()
        print('2/6 Gaia DR3 (ESA archive)')
        fetch_gaia.local_sample(); fetch_gaia.hipparcos_matches(); fetch_gaia.disk_sample()
        print('3/6 Hipparcos-2, XHIP, SIMBAD, Stellarium')
        fetch_hip.xhip(); fetch_hip.hip2()
        cs = fetch_meta.cosmos_stars()
        fetch_meta.simbad_ids(list(cs.name) + fetch_meta.EXTRA_NAMED)
        print('4/6 merge')
        cat = catalogue.build()
        print(f'    {len(cat):,} stars')
        print('5/6 uncertainty curve + pack')
        unc_path = C.CACHE / 'uncertainty.json'
        if unc_path.exists():
            unc = json.loads(unc_path.read_text())
        else:
            unc = uncertainty.curve(cat)
            unc_path.write_text(json.dumps(unc))
        pack.write(cat, uncertainty=unc)
    print('6/6 validation')
    from . import validate
    r = validate.run_all()
    for k, v in r.items():
        if isinstance(v, dict) and 'pass' in v:
            print(f"    {'PASS' if v['pass'] else 'FAIL'}  {k}")
    for k, v in r['engineering'].items():
        print(f"    {'PASS' if v['pass'] else 'FAIL'}  engineering/{k}")
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
