# Cosmos · Time

Scrub through ±250 million years and watch about a million real stars move.
Positions and velocities are measured by Gaia DR3, or Hipparcos for the
brightest stars, at epoch 2016.0. Every other moment is simulated: each star is
a test particle orbiting in the McMillan (2017) Milky Way potential, integrated
on the GPU.

**Moments** (⌘K → MOMENTS, or deep-link with `#moment=<name>`):

* `dipper` — the Big Dipper falls apart over 100,000 years. Five of its stars
  are the Ursa Major moving group; Dubhe and Alkaid are not.
* `barnard` — Barnard's Star comes to 3.77 ly around 11,700 CE. It does not
  become the nearest star: Proxima is closer then.
* `gl710` — Gliese 710 passes 10,600 au from the Sun in 1.29 million years,
  through the outer Oort cloud. At closest approach it shines at magnitude
  −3.7, brighter than Sirius.
* `galyear` — one galactic year (222 Myr) seen from above the disk.
* `ride` — riding with the Sun while the neighbourhood shears apart.
* `orion` — Orion, a million years ago.

## Run it

It has to be served over HTTP, because ES modules and `fetch` don't work from
`file://`. From the repository root:

```
python3 -m http.server 8000      # then open http://localhost:8000/time/
```

On GitHub Pages it works as-is at `<site>/time/`. Cosmos's `⏳ TIME` button
links here.

**Browsers.** WebGPU (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows /
145+ on macOS) gets all 1,007,124 stars and the full ±250 Myr. Without WebGPU
the page falls back to WebGL2 and says so in a notice: the 108,067 brightest
stars, moving in straight lines, limited to ±1 Myr.

## Deep links

The URL keeps itself up to date (`history.replaceState`, as in Cosmos), so
every view is shareable:

```
#t=-120000y&view=earth&target=barnards-star
#t=1.2934Myr&view=earth&target=gliese-710
#t=100Myr&view=disk
#t=0y&view=earth&cam=-1.571,0.980,60.0          (yaw, pitch, fov or distance)
```

* `t` accepts `y`, `kyr` and `Myr`.
* `view` is `earth`, `disk` or `ride`.
* `target` is a slug of any star name (apostrophes optional, so Cosmos's
  `barnard-s-star` works too), `gaia-dr3-<source_id>` or `hip-<number>`.

The state at a time is defined as a fixed number of leapfrog steps from t = 0,
so a link renders the same sky for everyone.

## Controls

Drag to look · scroll or pinch to zoom · click a star for its card.
SPACE play/pause · ← → step (⇧ for bigger steps) · [ ] speed (100 yr/s to
10 Myr/s) · 1 2 3 views · C constellation lines · L labels · G illustrated
galaxy · + − fainter/brighter stars · S photo · B bloom · ⌘K / CTRL+K search
(names, `HIP 54061`, Gaia DR3 source_id, moments) · 0 today · ESC close.

## What is measured and what is simulated

* A badge next to the time reads MEASURED at t = 0 and SIMULATED otherwise.
* Below it, a "typical position uncertainty" line comes from a Monte Carlo run
  over the catalogue errors (`manifest.uncertainty`).
* Each star's card separates the measured values from the simulated ones. It
  also shows the spread of 48 clones drawn from that star's own errors, both
  as a ± figure and as a cloud.

Limits, also stated in the UI:

* **Sample.** The sample holds stars near us today. Far from t = 0 the night
  sky thins: our real neighbours then are not in the data.
* **No stellar evolution.** Young blue giants appear before they were born.
* **Smooth model.** The potential is axisymmetric and static, with no bar,
  spiral arms or molecular clouds. Over tens of Myr, individual orbits are
  illustrative; shear and dispersal are the robust part.
* **Backdrop.** The spiral behind the stars above the disk is Cosmos's
  procedural galaxy, labelled as an illustration (G hides it).
* **Precession.** Earth's 26,000-year wobble isn't modelled. The Earth view
  is the sky from the Sun's position in a fixed (ICRS) orientation.

## How it works

```
index.html     page shell; CSS copied from ../index.html so both pages look alike
js/main.js     boot, WebGPU → WebGL2 fallback, frame loop
js/data.js     manifest, tiered .bin streaming and decoding, lazy id/info shards
js/cpu.js      float64 twin of the physics: force table, Sun orbit, leapfrog, Hermite
js/gpu.js      TSL: force lookup, out/in leapfrog kernels, star sprites
js/bloom.js    Cosmos's bloom, pass for pass
js/views.js    the three cameras and input
js/overlays.js constellation figures, labels, markers, the Sun's orbit, illustrated galaxy
js/ui.js       HUD, scrubber, play, keys, ⌘K, card, deep links, photo, moments
js/timefmt.js  time parsing and formatting, scrubber scale
data/          written by ../pipeline (see its README)
test/          parity and determinism tests
```

* **Integration.** Each star's state lives in GPU storage buffers relative to
  the Sun (pc, pc/Myr). Each dispatch runs K kick-drift-kick steps per star
  in registers.
  * Scrubbing outward continues from the current state.
  * Scrubbing back uses the time-reversed step. It is exact up to float32
    round-off: at most 0.08 pc after scrubbing from +240 back to +100 Myr.
  * Opening a link always integrates fresh from t = 0, so a shared moment is
    bit-identical for everyone.
  * Returning to "today" restores the exact measured state.
  * The step budget per frame adapts to the display's frame rate while
    integrating.
* **Rendering.** The vertex shader Hermite-interpolates between the two
  bracketing steps, so slow play needs no integration at all.
* **Brightness.** Stars are drawn from absolute G magnitude and distance to
  the camera.

## Tests

```
node time/test/cpu-parity.mjs     # JS float64 twin vs the Python integrator (≤ 1e-6 pc)
node time/test/gpu-parity.mjs     # WebGPU float32 kernel vs Python, ±250 Myr, + determinism
```

`gpu-parity.mjs` needs Playwright and a server on port 8800 (see the header of
the file).

Measured in headless Chromium with SwiftShader WebGPU, CPU-emulated:

| Time | GPU vs Python |
|---|---|
| +9,700 yr | 1×10⁻⁶ pc |
| +1.29 Myr | 1×10⁻⁴ pc |
| −250 Myr | 0.17 pc |

Scrubbing back from +240 to +100 Myr drifts at most 0.08 pc from the canonical
state, and the canonical recompute (what loading a link does) is bit-identical
to a fresh run.

**Test hooks:**
* `?n=<N>` loads only the first N stars.
* `?webgl` forces the fallback.
* `?budget=<steps>` sets the per-frame step budget.
* `?offscreen` renders to a texture and paints it into a 2D canvas.
  Headless SwiftShader can render but cannot present a WebGPU canvas.

**Performance.** I could only run in software emulation, not on real GPUs. The
60 fps target on an M1-class laptop with 1M stars is not yet measured.
