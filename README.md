# COSMOS — the Planck Length to the Edge of the Universe

A real-data *Powers of Ten*: one scroll wheel (or pinch), **61 orders of
magnitude**. Zoom out from Earth past real stars and galaxies to the cosmic
microwave background — then turn around and dive down through the ocean, into
a living cell, along a DNA helix, inside an atom, between three quarks, and
all the way to the quantum foam at the Planck length.

One self-contained HTML file, no build, no dependencies except three.js from
a CDN (pinned with an integrity hash, with a fallback CDN). Earth uses NASA's
Blue Marble when online and falls back to a built-in vector-coastline Earth
offline. Deep links work: try `#trappist-1`, `#orion-nebula`, or `#tour` —
and the URL updates as you explore, so **every view is shareable**.

**Built by Claude Fable 5 from a single prompt** (then 100×'d on request).

🔗 **Live:** *add your GitHub Pages URL here*

## What's real
- ~90 named stars at their actual positions (RA/Dec/distance, Hipparcos-derived)
- **Live NASA Exoplanet Archive data**: when online, the ~1,200 nearest
  confirmed-planet systems load into the sky — every one clickable and flyable.
  Offline, a curated set of 25 systems stands in
- **Planets where they really are today**: the solar system is drawn from J2000
  Keplerian elements — real eccentricities, real inclinations (watch Pluto cut
  across the ecliptic), real positions for the current date. The Moon shows
  today's phase angle too
- Nebulae and clusters at their real positions: Orion, Carina, the Pleiades,
  the Crab, Omega Centauri and more
- The Sun's real position 27,000 ly from the galactic centre, galaxy correctly
  oriented to the sky; Local Group galaxies at real positions and distances
- The descent uses real sizes: an 8 µm cell, 2 nm DNA, a 134 pm carbon atom,
  a 0.84 fm proton, the 1.6×10⁻³⁵ m Planck length

The Milky Way's 165,000-particle spiral is procedural (real orientation, real
scale). Rendering gets a custom bloom pass — toggle it with **B**. All three
music tracks are original and generated live with WebAudio — drop
`cosmos-epic.mp3` / `cosmos-drift.mp3` / `cosmos-tesseract.mp3` beside the
file to override them.

## Controls
Scroll / pinch / **↑ ↓** = zoom · Drag / **← →** = look ·
Click a named star or nebula = fly to it ·
**SPACE** = cinematic tour (universe to Planck length) ·
**⌘K / CTRL+K** = search everything · **S** = save a captioned photo ·
**B** = bloom · **M** = music · **P** = pause time · **R** = random ·
**ESC** = back / home

Respects `prefers-reduced-motion`. Photo mode (**S**) saves a PNG — commit one
as `preview.png` to light up the social-share card.

## Run
Open `index.html`. That's it.
