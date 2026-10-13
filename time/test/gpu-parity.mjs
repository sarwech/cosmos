/* GPU (float32 WebGPU compute) vs the Python reference integrator.

   Needs a static server on the repo root and Playwright's Chromium:
     python3 -m http.server 8800 &      (from the repo root)
     node time/test/gpu-parity.mjs      (playwright must be resolvable)
   THREE_LOCAL=/path/to/node_modules/three serves three.js from disk instead of
   the CDN; CHROME=/path/to/chrome picks the browser binary.
   Criteria (fixed before the first run): |t| ≤ 2 Myr → max error < 0.001 pc;
   otherwise max error < max(0.5 pc, 1e-4·|x|). */
import { chromium } from 'playwright';
import { readFileSync, existsSync } from 'fs';

const base = process.env.BASE || 'http://localhost:8800';
const b = await chromium.launch({
  executablePath: process.env.CHROME || undefined, headless: true,
  args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'],
});
const p = await b.newPage({ viewport: { width: 640, height: 400 } });
if (process.env.THREE_LOCAL) await p.route('https://cdn.jsdelivr.net/npm/three@0.186.1/**', r => {
  const f = process.env.THREE_LOCAL + '/' + r.request().url().split('three@0.186.1/')[1];
  if (!existsSync(f)) return r.fulfill({ status: 404, body: '' });
  r.fulfill({ status: 200, contentType: 'application/javascript', body: readFileSync(f) });
});
p.on('pageerror', e => console.log('[pageerror]', e.message));
await p.goto(base + '/time/?n=20000&offscreen');
await p.waitForFunction(() => window.__time && window.__time.integ() && window.__time.cat.loaded >= 20000, null, { timeout: 120000 });
const res = await p.evaluate(async () => {
  const T = window.__time, integ = T.integ();
  T.renderer.setAnimationLoop(null);
  const { hermite } = await import('/time/js/cpu.js');
  const fx = await (await fetch('/time/test/parity-fixture.json')).json();
  const out = [];
  for (const [it, t] of fx.times_myr.entries()) {
    const sg = t >= 0 ? 1 : -1, dt = T.sun.dt, q = Math.abs(t) / dt;
    const n = Math.min(Math.floor(q), T.sun.nMax - 2), f = q - n;
    const t0 = performance.now();
    let left = integ.march(sg, n, 250, true);
    while (left > 0) { left = integ.march(sg, n, 250, false); await new Promise(r => setTimeout(r, 0)); }
    const r = await integ.readback(Math.max(...fx.indices) + 1);
    let worst = 0, worstI = -1, worstX = 0;
    fx.indices.forEach((i, k) => {
      const x = [0, 0, 0];
      hermite(r.pa, r.va, i * 4, r.pb, r.vb, i * 4, sg * dt, f, x, null);
      const ref = fx.positions_pc[it][k];
      const e = Math.hypot(x[0] - ref[0], x[1] - ref[1], x[2] - ref[2]);
      if (e > worst) { worst = e; worstI = i; worstX = Math.hypot(...ref); }
    });
    out.push({ t, worst, worstI, worstX, ms: Math.round(performance.now() - t0) });
  }
  // determinism: the same t reached by a different path, then the idle "settle"
  const run = async (fn) => { await fn(); const r = await integ.readback(20000); return r; };
  const go = async (n, canonical) => { let left = integ.march(1, n, 250, canonical); while (left > 0) { left = integ.march(1, n, 250, false); await new Promise(r => setTimeout(r, 0)); } };
  const ref = await run(() => go(1000, true));
  await go(2400, false);                                   // out to +240 Myr …
  const back = await run(() => go(1000, false));           // … and scrub back with reverse steps
  const settled = await run(() => go(1000, true));         // what the idle page does
  let drift = 0, same = true;
  for (let k = 0; k < ref.pa.length; k++) {
    drift = Math.max(drift, Math.abs(back.pa[k] - ref.pa[k]));
    if (settled.pa[k] !== ref.pa[k] || settled.pb[k] !== ref.pb[k]) same = false;
  }
  out.determinism = { drift_pc: drift, settled_bitwise_equal: same, dirtyAfterBack: true };
  return { rows: out, determinism: out.determinism };
});
let fails = 0;
for (const r of res.rows) {
  const tol = Math.abs(r.t) <= 2 ? 1e-3 : Math.max(0.5, 1e-4 * r.worstX);
  const ok = r.worst < tol;
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  t=${r.t} Myr  max |gpu−ref| = ${r.worst.toExponential(2)} pc (star ${r.worstI}, |x| ${r.worstX.toFixed(1)} pc)  tol ${tol.toExponential(1)}  [${r.ms} ms]`);
}
const d = res.determinism;
console.log(`${d.settled_bitwise_equal ? 'PASS' : 'FAIL'}  determinism: scrubbing back 240 → 100 Myr drifts ≤ ${d.drift_pc.toExponential(2)} pc; settled state bit-identical to a fresh run: ${d.settled_bitwise_equal}`);
if (!d.settled_bitwise_equal) fails++;
await b.close();
process.exit(fails ? 1 : 0);
