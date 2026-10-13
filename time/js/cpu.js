/* CPU (float64) side of the physics — no three.js, so Node tests can import it.

   Mirrors pipeline/cosmos_time/orbit.py exactly:
   * ForceTable.accel  ≡ potential.interp_accel  (bilinear in asinh-stretched R, |z|)
   * SunOrbit          ≡ orbit.sun_orbit          (KDK leapfrog, both directions)
   * stepRel / hermite ≡ orbit.step_rel / hermite (Sun-relative stars)
   Units: pc, Myr, pc/Myr. */

export const KMS_TO_PCMYR = 1.0227121650537077;
export const PC_TO_LY = 3.2615637771674333;

export class ForceTable {
  constructor(p, data) {           // p = manifest.potential, data = Float32Array [v][u][F_R, F_z]
    this.n = p.n; this.aR = p.a_r; this.aZ = p.a_z; this.du = p.du; this.dv = p.dv;
    this.data = data;
  }
  accel(x, y, z, out) {
    const n = this.n, d = this.data;
    const R = Math.sqrt(x * x + y * y);
    let fu = Math.asinh(R / this.aR) / this.du;
    let fv = Math.asinh(Math.abs(z) / this.aZ) / this.dv;
    fu = Math.min(Math.max(fu, 0), n - 1.000001);
    fv = Math.min(Math.max(fv, 0), n - 1.000001);
    const i = Math.floor(fu), j = Math.floor(fv), a = fu - i, b = fv - j;
    const k00 = (j * n + i) * 2, k10 = k00 + 2, k01 = k00 + 2 * n, k11 = k01 + 2;
    const fr = (d[k00] * (1 - a) + d[k10] * a) * (1 - b) + (d[k01] * (1 - a) + d[k11] * a) * b;
    const fz = ((d[k00 + 1] * (1 - a) + d[k10 + 1] * a) * (1 - b)
              + (d[k01 + 1] * (1 - a) + d[k11 + 1] * a) * b) * Math.sign(z);
    const inv = R > 0 ? 1 / Math.max(R, 1e-12) : 0;
    out[0] = fr * x * inv; out[1] = fr * y * inv; out[2] = fz;
    return out;
  }
}

/* The Sun's grid states for k = 0 … nMax in both directions, stored in one
   array indexed by (nMax + σk): pos, vel (pc/Myr), acc. */
export class SunOrbit {
  constructor(table, pos0, velKms0, dt, nMax) {
    this.dt = dt; this.nMax = nMax;
    const L = 2 * nMax + 1;
    this.pos = new Float64Array(L * 3); this.vel = new Float64Array(L * 3); this.acc = new Float64Array(L * 3);
    const a = [0, 0, 0];
    for (const sg of [1, -1]) {
      let x = pos0.slice(), v = velKms0.map(c => c * KMS_TO_PCMYR);
      table.accel(x[0], x[1], x[2], a);
      const h = sg * dt;
      for (let k = 0; k <= nMax; k++) {
        const o = (nMax + sg * k) * 3;
        for (let c = 0; c < 3; c++) { this.pos[o + c] = x[c]; this.vel[o + c] = v[c]; this.acc[o + c] = a[c]; }
        if (k === nMax) break;
        for (let c = 0; c < 3; c++) { v[c] += 0.5 * h * a[c]; x[c] += h * v[c]; }
        table.accel(x[0], x[1], x[2], a);
        for (let c = 0; c < 3; c++) v[c] += 0.5 * h * a[c];
      }
    }
  }
  /* galactocentric position (pc) and velocity (pc/Myr) at time t (Myr) */
  at(t, outPos, outVel) {
    const sg = t >= 0 ? 1 : -1, h = sg * this.dt;
    const q = Math.abs(t) / this.dt;
    const n = Math.min(Math.floor(q), this.nMax - 1), f = q - n;
    const ia = (this.nMax + sg * n) * 3, ib = (this.nMax + sg * (n + 1)) * 3;
    hermite(this.pos, this.vel, ia, this.pos, this.vel, ib, h, f, outPos, outVel);
  }
}

/* Cubic Hermite at fraction f of a step of length h between states (xa, va) and (xb, vb). */
export function hermite(XA, VA, ia, XB, VB, ib, h, f, outX, outV) {
  const f2 = f * f, f3 = f2 * f;
  const h00 = 2 * f3 - 3 * f2 + 1, h10 = f3 - 2 * f2 + f, h01 = -2 * f3 + 3 * f2, h11 = f3 - f2;
  const d00 = 6 * f2 - 6 * f, d10 = 3 * f2 - 4 * f + 1, d01 = -6 * f2 + 6 * f, d11 = 3 * f2 - 2 * f;
  for (let c = 0; c < 3; c++) {
    const xa = XA[ia + c], va = VA[ia + c], xb = XB[ib + c], vb = VB[ib + c];
    outX[c] = h00 * xa + h10 * h * va + h01 * xb + h11 * h * vb;
    if (outV) outV[c] = (d00 * xa + d01 * xb) / h + d10 * va + d11 * vb;
  }
}

/* One KDK step of a Sun-relative state (dx, dv: arrays of 3), in place, from
   Sun grid index i0 to i1 with signed step h. */
const _a = [0, 0, 0];
export function stepRel(table, sun, dx, dv, i0, i1, h) {
  const P = sun.pos, A = sun.acc;
  let o = i0 * 3;
  table.accel(P[o] + dx[0], P[o + 1] + dx[1], P[o + 2] + dx[2], _a);
  for (let c = 0; c < 3; c++) dv[c] += 0.5 * h * (_a[c] - A[o + c]);
  for (let c = 0; c < 3; c++) dx[c] += h * dv[c];
  o = i1 * 3;
  table.accel(P[o] + dx[0], P[o + 1] + dx[1], P[o + 2] + dx[2], _a);
  for (let c = 0; c < 3; c++) dv[c] += 0.5 * h * (_a[c] - A[o + c]);
}

/* Canonical Sun-relative state of one star at time t: ⌊|t|/dt⌋ steps from
   t = 0, then Hermite. dv0 in pc/Myr. Returns [x, v] (pc, pc/Myr). */
export function starAt(table, sun, dx0, dv0, t) {
  const sg = t >= 0 ? 1 : -1, h = sg * sun.dt;
  const q = Math.abs(t) / sun.dt;
  const n = Math.min(Math.floor(q), sun.nMax - 1), f = q - n;
  const x = dx0.slice(), v = dv0.slice();
  for (let k = 0; k < n; k++) stepRel(table, sun, x, v, sun.nMax + sg * k, sun.nMax + sg * (k + 1), h);
  const xb = x.slice(), vb = v.slice();
  stepRel(table, sun, xb, vb, sun.nMax + sg * n, sun.nMax + sg * (n + 1), h);
  const ox = [0, 0, 0], ov = [0, 0, 0];
  hermite(x, v, 0, xb, vb, 0, h, f, ox, ov);
  return [ox, ov];
}

/* Gaussian deviate (Box–Muller) from a seeded PRNG, for reproducible clouds. */
export function rng(seed) {
  let s = seed >>> 0 || 1;
  const u = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; };
  return () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
}
