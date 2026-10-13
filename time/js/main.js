/* COSMOS · TIME — boot, state, frame loop.

   One time axis, three views. The stars' measured positions and velocities
   (Gaia DR3 / Hipparcos, epoch 2016.0) are integrated as test particles in the
   McMillan (2017) Milky Way potential: on the GPU when WebGPU is available,
   as straight lines (±1 Myr only) on the WebGL2 fallback. */

import * as THREE from 'three/webgpu';
import { ForceTable, SunOrbit, hermite, KMS_TO_PCMYR } from './cpu.js';
import { Catalogue, getJSON, getBin, displayPalette } from './data.js';
import { GPUIntegrator, starUniforms, starMaterial, makeStars } from './gpu.js';
import { T_MAX } from './timefmt.js';
import { Views } from './views.js';
import { Overlays } from './overlays.js';
import { UI } from './ui.js';
import { Bloom } from './bloom.js';

const qs = new URLSearchParams(location.search);
const loaderBar = document.getElementById('loaderBar'), loaderT = document.getElementById('loaderT');
const setLoad = (frac, txt) => { loaderBar.style.width = (frac * 100).toFixed(0) + '%'; if (txt) loaderT.textContent = txt; };

/* the shared state every module reads */
export const S = {
  t: 0,               // target time, Myr (0 = J2016.0)
  tShown: 0,          // time the stars currently show (lags while integrating)
  tMax: T_MAX,
  playing: false, speed: 2, dir: 1,
  view: 'earth',
  selected: -1,
  bloom: true, lines: true, labels: true,
  fallback: false,
  busy: false,
  lastInput: 0,
};

async function boot() {
  const canvas = document.getElementById('c');
  const forceWebGL = qs.has('webgl') || !navigator.gpu;
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, forceWebGL });
  THREE.ColorManagement.enabled = false;          // Cosmos (r128) draws in display space; match it
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  await renderer.init();
  S.fallback = renderer.backend.isWebGPUBackend !== true;
  if (!S.fallback) patchSwizzle(renderer.backend.device);
  if (S.fallback) S.tMax = 1;                     // straight lines are only honest for ~±1 Myr

  setLoad(0.05, 'LOADING THE GALAXY MODEL');
  const manifest = await getJSON('manifest.json');
  const limit = qs.has('n') ? +qs.get('n') : S.fallback ? manifest.precise_count + manifest.files['stars-0.bin'].count : Infinity;
  const [tableBuf, figures, names] = await Promise.all([
    getBin(manifest.potential.file), getJSON('constellations.json'), getJSON('names.json')]);
  const table = new ForceTable(manifest.potential, new Float32Array(tableBuf));
  const nMax = Math.ceil(manifest.integrator.t_max_myr / manifest.integrator.dt_myr) + 1;
  const sun = new SunOrbit(table, manifest.frame.sun_pos_pc, manifest.frame.sun_vel_kms, manifest.integrator.dt_myr, nMax);
  const cat = new Catalogue(manifest, limit);
  const palette = displayPalette(manifest.palette);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.001, 1e7);
  const u = starUniforms();
  let integ = null, stars = null;
  const starsFrame = new THREE.Group();           // its matrix maps Sun-relative gc pc → scene
  starsFrame.matrixAutoUpdate = false;
  scene.add(starsFrame);

  const bloom = new Bloom(renderer);
  const sizeBloom = () => { const v = renderer.getDrawingBufferSize(new THREE.Vector2()); bloom.setSize(v.x, v.y); };
  sizeBloom();

  const views = new Views(camera, canvas, manifest, sun);
  const overlays = new Overlays(scene, camera, manifest, sun, cat, figures, names, palette);
  const ui = new UI({ S, manifest, cat, names, views, overlays, sun, table, renderer, canvas });

  /* -------------------------------------------------------- frame loop --- */
  let budget = qs.has('budget') ? +qs.get('budget') : 32;        // leapfrog steps per frame
  let last = performance.now(), slow = 0;
  const _sp = [0, 0, 0], _sv = [0, 0, 0];

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    window.__loops = (window.__loops || 0) + 1;
    if (!stars) return;                                   // first tier still decoding
    ui.tick(dt);                                          // play, scrub easing → S.t
    const t = Math.max(-S.tMax, Math.min(S.tMax, S.t));
    const sg = t >= 0 ? 1 : -1, q = Math.abs(t) / sun.dt;
    const nT = Math.min(Math.floor(q), nMax - 2), f = q - nT;

    if (integ) {
      // Reverse steps are exact up to float32 round-off (≤ 0.1 pc after a
      // 240 → 100 Myr scrub), invisible at any scale shown, and a fresh load
      // of a link always integrates canonically from t = 0. Only "today" is
      // forced back to the bit-exact measured state, which costs one step.
      const left = integ.march(sg, nT, budget, nT === 0 && integ.dirty);
      S.busy = left > 0;
      // adapt the per-frame step budget while integrating, against the
      // frame rate this display actually runs at
      if (S.busy) {
        if (dt > 0.034) { if (++slow >= 3) { budget = Math.max(16, (budget * 0.7) | 0); slow = 0; } }
        else { slow = 0; if (dt < 0.02) budget = Math.min(1024, budget + 8); }
      }
      u.h.value = integ.sg * sun.dt;
      u.f.value = S.busy ? 0 : f;
      S.tShown = S.busy ? integ.sg * integ.n * sun.dt : t;
      overlays.requestTracked(integ);
    } else {
      u.t.value = t;
      S.tShown = t;
    }

    sun.at(S.tShown, _sp, _sv);
    views.update(dt, S, _sp, starsFrame);                 // sets camera + frame matrix
    u.mLim.value = views.mLim; u.gain.value = views.gain; u.sizeBase.value = views.sizeBase;
    overlays.update(dt, S, views, _sp, integ, u);
    ui.frame();
    if (offscreen) { offscreen.frame(); return; }
    bloom.render(scene, camera, S.bloom);
    ui.afterRender();                                     // photo capture lives here (same task)
  }
  /* test hook (?offscreen): render into a texture and paint it into a 2D canvas.
     Headless SwiftShader can render and compute but cannot present a WebGPU
     canvas; this lets the automated tests take real screenshots. */
  const offscreen = qs.has('offscreen') ? (() => {
    const w = 1280, h = 760, rt = new THREE.RenderTarget(w, h);
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
    cv.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:1;pointer-events:none';
    document.body.appendChild(cv);
    const g = cv.getContext('2d'); let reading = false;
    return { frame() {
      if (reading) return;                        // one render per readback: no GPU backlog
      bloom.render(scene, camera, S.bloom, rt);
      renderer.setRenderTarget(null);
      reading = true;
      renderer.readRenderTargetPixelsAsync(rt, 0, 0, w, h).then(px => {
        const img = g.createImageData(w, h);
        img.data.set(new Uint8ClampedArray(px.buffer, px.byteOffset, w * h * 4));
        g.putImageData(img, 0, 0); reading = false; window.__frames = (window.__frames || 0) + 1;
      });
    } };
  })() : null;
  renderer.setAnimationLoop(frame);
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    sizeBloom();
  });

  /* stream the tiers; the sky appears with the first one and fills in */
  let first = true;
  const t0 = performance.now();
  await cat.load((loaded, name) => {
    setLoad(0.1 + 0.9 * loaded / cat.N, `LOADING ${loaded.toLocaleString()} STARS`);
    S.loading = loaded < cat.N ? `LOADING STARS · ${loaded.toLocaleString()} OF ${cat.N.toLocaleString()}` : '';
    if (first) {
      first = false;
      if (S.fallback) {
        const src = { P0: new THREE.InstancedBufferAttribute(cat.pos, 4), V0: new THREE.InstancedBufferAttribute(cat.vel, 4) };
        stars = makeStars(starMaterial(src, cat.phot, palette, u), loaded);
        stars.userData.attrs = [src.P0, src.V0];
      } else {
        integ = new GPUIntegrator(renderer, manifest, table.data, sun, { pos: cat.pos, vel: cat.vel });
        stars = makeStars(starMaterial(integ, cat.phot, palette, u), loaded);
      }
      starsFrame.add(stars);
      overlays.setLoaded(loaded);
      document.getElementById('loader').style.opacity = 0;
      ui.ready({ integ });
    } else {
      stars.count = loaded;
      if (integ) integ.reupload();
      else for (const a of stars.userData.attrs) a.needsUpdate = true;
      stars.material.userData.photAttr.needsUpdate = true;
      overlays.setLoaded(loaded);
    }
  });
  S.loading = '';
  if (!location.hash) ui.caption('A million real stars, as Gaia measured them',
    'DRAG THE TIME SLIDER · SPACE TO PLAY · ⌘K FOR A GUIDED MOMENT', 6000);
  console.info(`[time] ${cat.loaded.toLocaleString()} stars in ${((performance.now() - t0) / 1000).toFixed(1)} s`,
    S.fallback ? '(WebGL2 fallback)' : '(WebGPU)');
  if (S.fallback) ui.notice('WebGPU is not available in this browser, so this is the WebGL2 fallback: the ' +
    `${cat.loaded.toLocaleString()} brightest stars, moving in straight lines, limited to ±1 million years ` +
    '(beyond that straight lines would be wrong). Open in a WebGPU browser for the full million stars and ±250 million years.');
  window.__time = { S, cat, sun, table, integ: () => integ, renderer, manifest, views, overlays, THREE, scene, bloom };
}

/* three r186 always passes the identity swizzle 'rgba' as a string (the final
   WebGPU spec). Some 2025 Chromium builds implement an older form and throw on
   it; for those, drop the (no-op) identity swizzle. */
function patchSwizzle(device) {
  try {
    const tx = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
    tx.createView({ swizzle: 'rgba' });
    tx.destroy();
  } catch (e) {
    const orig = GPUTexture.prototype.createView;
    GPUTexture.prototype.createView = function (d) {
      if (d && d.swizzle === 'rgba') { d = { ...d }; delete d.swizzle; }
      return orig.call(this, d);
    };
  }
}

boot().catch(e => {
  console.error(e);
  document.getElementById('loaderT').textContent = 'COULD NOT START: ' + (e.message || e);
});
