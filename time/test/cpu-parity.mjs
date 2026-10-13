/* JS float64 twin (time/js/cpu.js) vs the Python reference integrator.
   Run from the repo root:  node time/test/cpu-parity.mjs
   Exits non-zero if any position differs by more than 1e-6 pc + 1e-9·|x|. */
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';

// data.js fetches relative to its own URL; serve file: URLs from disk in Node
const realFetch = globalThis.fetch;
globalThis.fetch = async (u, o) => {
  const s = String(u);
  if (!s.startsWith('file:')) return realFetch(u, o);
  const buf = readFileSync(fileURLToPath(s));
  return { ok: true, status: 200, json: async () => JSON.parse(buf.toString()), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};

const { ForceTable, SunOrbit, starAt } = await import('../js/cpu.js');
const { Catalogue, getJSON, getBin } = await import('../js/data.js');

const fx = JSON.parse(readFileSync(new URL('./parity-fixture.json', import.meta.url)));
const m = await getJSON('manifest.json');
const table = new ForceTable(m.potential, new Float32Array(await getBin(m.potential.file)));
const nMax = Math.ceil(m.integrator.t_max_myr / m.integrator.dt_myr) + 1;
const sun = new SunOrbit(table, m.frame.sun_pos_pc, m.frame.sun_vel_kms, m.integrator.dt_myr, nMax);
const cat = new Catalogue(m, Math.max(...fx.indices) + 1);
await cat.load();

let worst = 0, worstAt = '', fails = 0;
for (const [it, t] of fx.times_myr.entries()) {
  const ref = fx.positions_pc[it];
  const sp = [0, 0, 0]; sun.at(t, sp, null);
  const se = Math.hypot(...sp.map((x, k) => x - fx.sun_pc[it][k]));
  if (se > 1e-6) { fails++; console.log(`sun t=${t}: ${se.toExponential(2)} pc`); }
  fx.indices.forEach((i, k) => {
    const x0 = [cat.pos[i * 4], cat.pos[i * 4 + 1], cat.pos[i * 4 + 2]];
    const v0 = [cat.vel[i * 4], cat.vel[i * 4 + 1], cat.vel[i * 4 + 2]];
    const [x] = starAt(table, sun, x0, v0, t);
    const e = Math.hypot(...x.map((q, c) => q - ref[k][c]));
    const tol = 1e-6 + 1e-9 * Math.hypot(...ref[k]);
    if (e > worst) { worst = e; worstAt = `star ${i} t=${t}`; }
    if (e > tol) fails++;
  });
}
console.log(`cpu parity: ${fx.indices.length} stars × ${fx.times_myr.length} times; worst ${worst.toExponential(2)} pc (${worstAt}); ${fails} over tolerance`);
process.exit(fails ? 1 : 0);
