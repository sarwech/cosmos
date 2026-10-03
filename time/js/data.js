/* Loading and decoding the files written by pipeline/cosmos_time/pack.py.
   Decoding must match pack.read() — the validation checks run on the same
   quantised numbers the browser integrates. */

import { KMS_TO_PCMYR } from './cpu.js';

export const BASE = new URL('../data/', import.meta.url);

export async function getJSON(name) {
  const r = await fetch(new URL(name, BASE));
  if (!r.ok) throw new Error(name + ': HTTP ' + r.status);
  return r.json();
}
export async function getBin(name) {
  const r = await fetch(new URL(name, BASE));
  if (!r.ok) throw new Error(name + ': HTTP ' + r.status);
  return r.arrayBuffer();
}

/* Octahedral unit-vector decode (pack.oct_decode) */
function octDecode(u, v, out) {
  let x = u / 65535 * 2 - 1, y = v / 65535 * 2 - 1;
  let z = 1 - Math.abs(x) - Math.abs(y);
  const t = Math.max(-z, 0);
  x += x >= 0 ? -t : t; y += y >= 0 ? -t : t;
  const l = Math.hypot(x, y, z);
  out[0] = x / l; out[1] = y / l; out[2] = z / l;
}

export class Catalogue {
  constructor(manifest, limit = Infinity) {
    this.m = manifest;
    this.N = Math.min(manifest.count, limit);
    this.pos = new Float32Array(this.N * 4);           // Δx (pc), gc axes
    this.vel = new Float32Array(this.N * 4);           // Δv (pc/Myr)
    this.phot = new Float32Array(this.N * 2);          // absolute G, Teff code
    this.flags = new Uint16Array(this.N);
    this.hip = new Uint32Array(this.N);
    this.loaded = 0;                                    // stars [0, loaded) are decoded
    this._waiting = [];
  }

  /* resolves true once star i is decoded, false if it is not in this view
     (the WebGL2 fallback holds only the brightest ~108k) */
  whenLoaded(i) {
    if (!(i >= 0) || i >= this.N) return Promise.resolve(false);
    if (i < this.loaded) return Promise.resolve(true);
    return new Promise(res => this._waiting.push([i, res]));
  }

  /* Stream tiers in order; onTier(loadedCount) after each. */
  async load(onTier) {
    const q = this.m.quant;
    const ld0 = q.logdist_min, ldr = q.logdist_max - q.logdist_min;
    const dir = [0, 0, 0];
    for (const name of this.m.tiers) {
      const info = this.m.files[name];
      if (info.first >= this.N) break;
      const buf = await getBin(name);
      const dv = new DataView(buf);
      const n = Math.min(info.count, this.N - info.first);
      for (let k = 0; k < n; k++) {
        const i = info.first + k, o4 = i * 4, o = k * info.record;
        if (info.record === 32) {
          for (let c = 0; c < 3; c++) {
            this.pos[o4 + c] = dv.getFloat32(o + c * 4, true);
            this.vel[o4 + c] = dv.getFloat32(o + 12 + c * 4, true) * KMS_TO_PCMYR;
          }
          this.phot[i * 2 + 1] = dv.getUint8(o + 24);
          this.phot[i * 2] = q.absmag_min + dv.getUint8(o + 25) * q.absmag_step;
          this.flags[i] = dv.getUint16(o + 26, true);
          this.hip[i] = dv.getUint32(o + 28, true);
        } else {
          octDecode(dv.getUint16(o, true), dv.getUint16(o + 2, true), dir);
          const d = Math.pow(10, ld0 + dv.getUint16(o + 4, true) / 65535 * ldr);
          for (let c = 0; c < 3; c++) {
            this.pos[o4 + c] = dir[c] * d;
            this.vel[o4 + c] = dv.getInt16(o + 6 + c * 2, true) * q.vel_kms * KMS_TO_PCMYR;
          }
          this.phot[i * 2 + 1] = dv.getUint8(o + 12);
          this.phot[i * 2] = q.absmag_min + dv.getUint8(o + 13) * q.absmag_step;
          this.flags[i] = dv.getUint16(o + 14, true);
        }
      }
      this.loaded = info.first + n;
      onTier && onTier(this.loaded, name);
      this._waiting = this._waiting.filter(([i, res]) => (i < this.loaded ? (res(true), false) : true));
    }
  }

  /* lazily-loaded per-star details */
  async details(i) {
    const S = this.m.shards.size, k = Math.floor(i / S), j = i - k * S;
    const pad = String(k).padStart(2, '0');
    this._ids = this._ids || {}; this._info = this._info || {};
    if (!this._ids[k]) this._ids[k] = getBin(`ids-${pad}.bin`);
    if (!this._info[k]) this._info[k] = getBin(`info-${pad}.bin`);
    const [ids, info] = await Promise.all([this._ids[k], this._info[k]]);
    const sid = new DataView(ids).getBigInt64(j * 8, true);
    const d = new DataView(info), o = j * 32;
    const f = c => d.getFloat32(o + c * 4, true);
    return { sourceId: sid > 0n ? sid.toString() : null, plx: f(0), plxErr: f(1), pmraErr: f(2),
      pmdecErr: f(3), rvErr: f(4), bpRp: f(5), hip: d.getUint32(o + 24, true), ruwe: f(7) };
  }

  /* Gaia source_id → index (scans the id shards; rare, so loaded on demand) */
  async findSourceId(str) {
    const want = BigInt(str);
    for (let k = 0; k < this.m.shards.count; k++) {
      const pad = String(k).padStart(2, '0');
      this._ids = this._ids || {};
      if (!this._ids[k]) this._ids[k] = getBin(`ids-${pad}.bin`);
      const a = new BigInt64Array(await this._ids[k]);
      const j = a.indexOf(want);
      if (j >= 0) return k * this.m.shards.size + j;
    }
    return -1;
  }

  async findHip(hip) {
    if (!this._hipIdx) this._hipIdx = getBin('hip-index.bin').then(b => new Uint32Array(b));
    const a = await this._hipIdx;
    let lo = 0, hi = a.length / 2 - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1, v = a[mid * 2];
      if (v === hip) return a[mid * 2 + 1];
      if (v < hip) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
  }
}

/* manifest palette is linear sRGB; this page draws in display space like
   Cosmos (no colour management), so encode once here. */
export function displayPalette(lin) {
  const enc = c => c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return lin.map(([r, g, b]) => [enc(r), enc(g), enc(b)]);
}
