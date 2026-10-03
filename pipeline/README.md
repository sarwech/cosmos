# Cosmos · Time — data pipeline

Reproducible Python that turns Gaia DR3 (and Hipparcos for the brightest stars)
into the files `time/index.html` loads, and checks the physics against five
known answers before anything ships.

```
cd pipeline
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m cosmos_time.build          # fetch → merge → pack → validate
.venv/bin/python -m cosmos_time.report         # reports/validation.md + big_dipper.png
.venv/bin/python -m cosmos_time.fixtures       # reference positions for the browser tests
.venv/bin/python -m pytest -q                  # the checks as tests
```

Every network query is cached in `cache/` (gitignored), so re-runs are offline
and take about a minute. A cold run took about 25 minutes on 4 cores
(2026-10-02): Gaia ≈ 15 min, the force table ≈ 5 min, everything else ≈ 2 min.

**No Gaia account is needed.** Anonymous archive limits are 3,000,000 rows and
120 minutes per async query, with results kept 3 days. The largest single job
here returns 200k rows.

## What goes in

| Source | Use | Rows |
|---|---|---|
| Gaia DR3 `gaiadr3.gaia_source` + `external.gaiaedr3_distance` | local sample: ϖ > 4.4 mas (≈ 227 pc), RV present, RUWE < 1.4, ϖ/σϖ ≥ 10 | 801,806 |
| same | disk sample: random draw (`random_index` < 13,979,122) beyond 4.4 mas, RUWE < 1.4, ϖ/σϖ ≥ 5 | 200,062 |
| `gaiadr3.hipparcos2_best_neighbour` ⋈ `gaia_source` | Gaia rows of every Hipparcos star | 99,525 |
| Hipparcos-2 (van Leeuwen 2007, VizieR I/311) | positions at J1991.25 for the bright-star fallback | 117,955 |
| XHIP (Anderson & Francis 2012, VizieR V/137D) | literature radial velocities, V, B−V, Bayer names | 117,955 (46,392 with RV) |
| SIMBAD | identifiers for Cosmos's named stars; last-resort RVs | 91 names |
| Stellarium "modern" sky culture | 88 constellation figures (710 HIP stars), 659 common names | CC BY-SA 4.0 |
| Pecaut & Mamajek dwarf sequence (2022.04.16) | BP−RP / B−V → Teff | — |

The distance cut came from count queries on the archive before downloading
anything: ϖ > 4.5 mas → 766,330 stars; ϖ > 4.0 mas → 969,256.

## Decisions

* **Distances: Bailer-Jones et al. (2021) `r_med_photogeo`**, falling back to
  `r_med_geo`, then 1/ϖ. These include the parallax zero-point and handle the
  far disk sample. In the local sample, where ϖ/σϖ is mostly in the hundreds,
  they agree with 1/ϖ to about 1%. Hipparcos stars use 1/ϖ, with ϖ floored at
  0.5 mas.
* **Quality.** RUWE < 1.4 is the cut that bites locally: it removes about 23%
  of RV stars, mostly unresolved binaries. Bright ones come back through the
  Hipparcos path. ϖ/σϖ ≥ 10 barely bites inside 227 pc.
* **Radial velocities.** Gaia DR3 RVs are corrected as the DR3 papers
  recommend:
  * Katz et al. 2023 eq. 5 for rv_template_teff < 8500 K and G_RVS ≥ 11;
  * Blomme et al. 2023 for 8500–14500 K and 6 ≤ G_RVS ≤ 12.
* **Bright stars.** H = Hipparcos stars with V < 6.5, plus every
  constellation-figure star, plus every named star. A star in H uses Gaia only
  if its Gaia solution passes the local cuts and G > 3. Otherwise it uses
  Hipparcos-2 astrometry, propagated 1991.25 → 2016.0 with perspective
  acceleration, plus the XHIP radial velocity (or SIMBAD's). Result: 4,426
  Hipparcos stars, including all seven of the Big Dipper. 59 stars have no
  RV anywhere; they use RV = 0 and are flagged.
* **Extinction.** GSP-Phot A_G and E(BP−RP) are applied to the disk sample
  only. The local sample sits inside the Local Bubble.
* **Colour.** BP−RP (dereddened) or B−V → Teff via Pecaut & Mamajek → a Planck
  spectrum through the CIE 1931 colour-matching functions → linear sRGB with
  peak channel 1. Each star stores a 1-byte Teff code; the page gets a
  256-entry palette.
* **Frame.** astropy `Galactocentric` with the `v4.0` defaults:
  R0 = 8.122 kpc (26,490 ly), z☉ = 20.8 pc, v☉ = (12.9, 245.6, 7.78) km/s.
  Cosmos says "27,000 ly" (README) and draws the Sun at 26,700 ly. Those are
  within 2%, and Cosmos is left untouched.
* **Epoch.** t = 0 is J2016.0, the Gaia epoch. The page calls it "today".

## Physics

* **Potential: McMillan (2017)** via galpy (`ro` = 8.21 kpc, `vo` = 233.1 km/s).
  * It is consistent with the v4.0 Sun: v_c(8.122 kpc) = 233.2 km/s, so the
    implied peculiar V is 12.4 km/s.
  * MWPotential2014 has v_c = 220 km/s at 8 kpc, which would put this Sun
    25 km/s super-circular.
  * McMillan17 includes the HI and H₂ discs, which set the vertical force the
    Sun bobs in.
* **Force table.** galpy evaluates McMillan17 at ~10 ms per force, so
  `potential.py` tabulates it once:
  1. Evaluate Φ on a 512 × 512 grid uniform in asinh(R / 1 kpc) and
     asinh(|z| / 50 pc).
  2. Differentiate with a quintic spline to get F_R and F_z.
  3. Ship the forces as `force-mcmillan17.bin`.

  Everyone — this code, the JS twin and the GPU shader — does the same
  bilinear lookup. The maximum error against galpy's own forces is 5.5×10⁻⁵
  over 2,000 random points.
* **Integrator** (`orbit.py`, mirrored by `time/js/cpu.js` and the TSL kernel
  in `time/js/gpu.js`):
  * Kick-drift-kick leapfrog with dt = 0.1 Myr, 2,500 steps each way.
  * Stars are stored relative to the Sun, which is integrated separately in
    float64. The relative step is algebraically identical to integrating
    both and subtracting, and it keeps nearby stars exact in float32.
  * The state at time t is defined as ⌊|t|/dt⌋ steps from t = 0, then a
    cubic Hermite interpolation inside the step. Every deep link therefore
    shows the same sky.
  * dt was fixed by the convergence test: halving it moves the Sun 0.04 pc
    after 250 Myr.

## Validation

Run by `python -m cosmos_time.build` (or `pytest`). Every check uses the
decoded, quantised files, so it starts from bit-identical initial conditions
to the browser. Full results are in [`reports/validation.md`](reports/validation.md).

| # | Check | Result | |
|---|---|---|---|
| 1 | t = 0 sky vs Cosmos's 89 stars | 89/89 found, median 0.004°; **8 stars off by 0.02°–7.3°** | FAIL |
| 2 | Barnard's Star | 3.768 ly at +9,720 yr (11,736 CE) | PASS |
| 3 | Gliese 710 | 0.0516 pc (10,633 au) at +1.293 Myr | PASS |
| 4 | Big Dipper | the five moving-group stars within 3.4 km/s; Dubhe 36, Alkaid 33 km/s off | PASS |
| 5 | Sun | 221.6 Myr per revolution (226.6 averaged over 3 Gyr), ±113 pc every 88 Myr | PASS |

**Check 1 fails because of Cosmos's data, not this pipeline.** For all eight
outliers, SIMBAD agrees with our position to within 0.00001° and disagrees
with the coordinate hard-coded in `index.html`:
* LHS 1140 has RA 3.68° where it should be about 11.25°;
* GJ 504 is 1° off;
* Wolf 1061, TOI-700, Kruger 60, Kepler-452, YZ Ceti and Kepler-186 are off
  by 0.02°–0.47°.

The 0.02° threshold was not relaxed. Fixing those eight entries is a change
to `index.html`.

## Output (`time/data/`)

| File | Contents | Size |
|---|---|---|
| `manifest.json` | counts, frame, potential and integrator parameters, palette, uncertainty curve, hashes | 15 KB |
| `stars-p.bin` | 8,067 stars, 32 B each: float32 Δx (pc), Δv (km/s), Teff code, M_G, flags, HIP | 0.26 MB |
| `stars-0…4.bin` | 999,057 stars, 16 B each: octahedral direction, log-distance, int16 Δv at 0.01 km/s, Teff, M_G, flags | 16.0 MB |
| `ids-NN.bin` | Gaia DR3 source_id per star (loaded on click / search) | 8.1 MB |
| `info-NN.bin` | ϖ, σϖ, σμ, σRV, BP−RP, HIP, RUWE per star (loaded on click) | 32.2 MB |
| `force-mcmillan17.bin` | 512 × 512 × (F_R, F_z) float32 | 2.1 MB |
| `names.json`, `hip-index.bin`, `constellations.json` | search and figures | 0.7 MB |

That is 59.3 MB in total, of which 18.7 MB is needed to show everything. The
largest file is 4 MB, well inside GitHub's limits, so the data is committed
and served same-origin by GitHub Pages. GitHub Release assets can't be
fetched from a browser because they send no CORS headers.

The precision tier holds:
* every star within 25 pc;
* every Hipparcos, named or constellation star;
* anything faster than 327 km/s.

Gliese 710's sideways velocity is only 0.04 km/s, so int16 velocities would
have broken check 3.

## Licences

Gaia DR3 data: ESA/Gaia/DPAC, CC BY-SA 3.0 IGO. Constellation figures and names:
Stellarium, CC BY-SA 4.0. Hipparcos/XHIP/SIMBAD via CDS. Attribution lines are
in `manifest.json` and shown in the page.
