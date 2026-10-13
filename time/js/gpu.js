/* GPU side: TSL force lookup, leapfrog compute kernels, star sprites.

   State per star (storage buffers, vec4, Sun-relative galactocentric axes):
     P0, V0  measured state at t = 0 (pc, pc/Myr)
     PA, VA  state at grid step n          ┐ the vertex shader Hermite-
     PB, VB  state at grid step n + 1      ┘ interpolates between them
   n counts steps outward from t = 0 in the direction of sign(t).
   "out" kernel:  A ← s(n+K),  B ← s(n+K+1)   (reads B = s(n+1))
   "in"  kernel:  A ← s(n−K),  B ← s(n−K+1)   (reads A = s(n); the same KDK
                                               step with h negated — leapfrog
                                               is time-reversible)
   Each dispatch loops K steps per thread entirely in registers. */

import * as THREE from 'three/webgpu';
import {
  Fn, Loop, float, int, vec2, vec3, vec4, sqrt, asinh, abs, floor, clamp, mix, sign, select,
  uniform, uniformArray, instancedArray, instanceIndex, instancedBufferAttribute,
  modelWorldMatrix, cameraPosition, positionLocal, varying, uv, smoothstep, pow, log, min, max, length,
} from 'three/tsl';

export function tslAccel(table, p) {
  const n = p.n;
  return Fn(([pos]) => {
    const R = sqrt(pos.x.mul(pos.x).add(pos.y.mul(pos.y)));
    const fu = clamp(asinh(R.div(p.a_r)).div(p.du), 0.0, n - 1.000001);
    const fv = clamp(asinh(abs(pos.z).div(p.a_z)).div(p.dv), 0.0, n - 1.000001);
    const fi = floor(fu), fj = floor(fv);
    const a = fu.sub(fi), b = fv.sub(fj);
    const k00 = int(fj).mul(n).add(int(fi));
    const f = mix(mix(table.element(k00), table.element(k00.add(1)), a),
                  mix(table.element(k00.add(n)), table.element(k00.add(n + 1)), a), b);
    const inv = select(R.greaterThan(0.0), float(1.0).div(R.max(1e-12)), float(0.0));
    const fr = f.x.mul(inv);
    return vec3(fr.mul(pos.x), fr.mul(pos.y), f.y.mul(sign(pos.z)));
  });
}

export class GPUIntegrator {
  /* init: {pos: Float32Array N*4, vel: Float32Array N*4 (pc/Myr)} */
  constructor(renderer, manifest, tableData, sun, init) {
    this.renderer = renderer;
    this.N = init.pos.length / 4;
    this.dt = sun.dt; this.nMax = sun.nMax;
    this.P0 = instancedArray(init.pos, 'vec4');
    this.V0 = instancedArray(init.vel, 'vec4');
    this.PA = instancedArray(this.N, 'vec4'); this.VA = instancedArray(this.N, 'vec4');
    this.PB = instancedArray(this.N, 'vec4'); this.VB = instancedArray(this.N, 'vec4');
    this.table = instancedArray(tableData, 'vec2');
    const L = 2 * sun.nMax + 1;
    const sp = new Float32Array(L * 4), sa = new Float32Array(L * 4);
    for (let i = 0; i < L; i++) for (let c = 0; c < 3; c++) {
      sp[i * 4 + c] = sun.pos[i * 3 + c]; sa[i * 4 + c] = sun.acc[i * 3 + c];
    }
    this.sunP = instancedArray(sp, 'vec4');
    this.sunA = instancedArray(sa, 'vec4');
    this.uKm1 = uniform(0, 'int'); this.uBase = uniform(0, 'int'); this.uStride = uniform(1, 'int');
    this.uH = uniform(0.1);
    const accel = tslAccel(this.table, manifest.potential);
    const { sunP, sunA, uBase, uStride, uH, uKm1 } = this;

    const step = (x, v, j) => {                      // KDK from sun index i0 → i0 + stride
      const i0 = uBase.add(uStride.mul(j)), i1 = i0.add(uStride);
      const s0 = sunP.element(i0).xyz, s1 = sunP.element(i1).xyz;
      v.addAssign(accel(s0.add(x)).sub(sunA.element(i0).xyz).mul(uH.mul(0.5)));
      x.addAssign(v.mul(uH));
      v.addAssign(accel(s1.add(x)).sub(sunA.element(i1).xyz).mul(uH.mul(0.5)));
    };
    const kernel = (from, toLast, toPrev) => Fn(() => {
      const i = instanceIndex;
      const x = from[0].element(i).xyz.toVar(), v = from[1].element(i).xyz.toVar();
      Loop({ start: int(0), end: uKm1, type: 'int', condition: '<' }, ({ i: j }) => { step(x, v, j); });
      toPrev[0].element(i).assign(vec4(x, 0)); toPrev[1].element(i).assign(vec4(v, 0));
      step(x, v, uKm1);
      toLast[0].element(i).assign(vec4(x, 0)); toLast[1].element(i).assign(vec4(v, 0));
    })().compute(this.N);
    // out: start from B, after K−1 steps → A, after K → B
    this.kOut = kernel([this.PB, this.VB], [this.PB, this.VB], [this.PA, this.VA]);
    // in: start from A, after K−1 steps → B, after K → A
    this.kIn = kernel([this.PA, this.VA], [this.PA, this.VA], [this.PB, this.VB]);
    this.kReset = Fn(() => {
      const i = instanceIndex;
      this.PB.element(i).assign(this.P0.element(i)); this.VB.element(i).assign(this.V0.element(i));
      this.PA.element(i).assign(this.P0.element(i)); this.VA.element(i).assign(this.V0.element(i));
    })().compute(this.N);
    this.sg = 1; this.n = -1;                         // −1: nothing integrated yet
  }

  /* new measured data streamed in → canonical state must be rebuilt */
  invalidate() { this.n = -1; }

  reupload() {
    this.P0.value.needsUpdate = true; this.V0.value.needsUpdate = true;
    this.invalidate();
  }

  /* Advance toward (sg, nTarget) by at most `budget` steps. Returns the number
     of steps still outstanding (0 = A/B now bracket the target). */
  march(sg, nTarget, budget, canonical) {
    const r = this.renderer, nm = this.nMax;
    // canonical = the state must equal "n steps outward from t = 0" bit for
    // bit (deep links, photos, the settled view). Reverse steps are exact only
    // up to float32 round-off, so a canonical request after any reverse step,
    // or one that would need reverse steps, restarts from the measured state.
    if (this.n < 0 || sg !== this.sg || (canonical && (this.dirty || nTarget < this.n))) {
      r.compute(this.kReset);
      this.sg = sg; this.n = -1; this.dirty = false;  // B = s0 (virtual n = −1)
    }
    if (nTarget > this.n) {
      const K = Math.min(nTarget - this.n, budget);
      this.uKm1.value = K - 1; this.uBase.value = nm + sg * (this.n + 1);
      this.uStride.value = sg; this.uH.value = sg * this.dt;
      r.compute(this.kOut);
      this.n += K;
    } else if (nTarget < this.n) {
      const K = Math.min(this.n - nTarget, budget);
      this.uKm1.value = K - 1; this.uBase.value = nm + sg * this.n;
      this.uStride.value = -sg; this.uH.value = -sg * this.dt;
      r.compute(this.kIn);
      this.n -= K;
      this.dirty = true;
    }
    return Math.abs(nTarget - this.n);
  }

  get canonical() { return this.n >= 0 && !this.dirty; }

  /* read back states [0, count) of A and B: Float32Array views (vec4) */
  async readback(count) {
    const bytes = Math.ceil(count * 16 / 4) * 4;
    const r = this.renderer;
    const [pa, va, pb, vb] = await Promise.all([this.PA, this.VA, this.PB, this.VB]
      .map(b => r.getArrayBufferAsync(b.value, null, 0, bytes)));
    return { pa: new Float32Array(pa), va: new Float32Array(va), pb: new Float32Array(pb), vb: new Float32Array(vb) };
  }
}

/* Uniforms shared by every star material. */
export function starUniforms() {
  return {
    f: uniform(0), h: uniform(0.1), t: uniform(0),
    mLim: uniform(6.5), sizeBase: uniform(2.2), sizeMax: uniform(22), gain: uniform(1),
    pxScale: uniform(1),
  };
}

/* Star sprites. src: either the GPU integrator (Hermite between A and B) or
   {P0, V0} instanced attributes for the WebGL2 straight-line fallback.
   phot: Float32Array N*2 (absolute G mag, Teff code). palette: 256 RGB (display space). */
export function starMaterial(src, phot, palette, u) {
  const m = new THREE.PointsNodeMaterial({ transparent: true, depthWrite: false, depthTest: false,
    blending: THREE.AdditiveBlending, sizeAttenuation: false });
  let pos;
  if (src.PA) {
    const xa = src.PA.toAttribute().xyz, va = src.VA.toAttribute().xyz;
    const xb = src.PB.toAttribute().xyz, vb = src.VB.toAttribute().xyz;
    pos = Fn(() => {
      const f = u.f, f2 = f.mul(f), f3 = f2.mul(f);
      const h00 = f3.mul(2).sub(f2.mul(3)).add(1), h10 = f3.sub(f2.mul(2)).add(f);
      const h01 = f3.mul(-2).add(f2.mul(3)), h11 = f3.sub(f2);
      return xa.mul(h00).add(va.mul(h10.mul(u.h))).add(xb.mul(h01)).add(vb.mul(h11.mul(u.h)));
    })();
  } else {                                          // WebGL2 fallback: straight lines
    pos = instancedBufferAttribute(src.P0).xyz.add(instancedBufferAttribute(src.V0).xyz.mul(u.t));
  }
  const photAttr = new THREE.InstancedBufferAttribute(phot, 2);
  m.userData.photAttr = photAttr;
  const ph = instancedBufferAttribute(photAttr);
  const pal = uniformArray(palette.map(c => new THREE.Color(c[0], c[1], c[2])), 'color');
  const world = modelWorldMatrix.mul(vec4(pos, 1)).xyz;
  const d = length(world.sub(cameraPosition)).max(1e-6);
  // apparent magnitude from the camera; L = 1 at the limiting magnitude
  const mApp = ph.x.add(log(d.div(10)).mul(5 / Math.LN10));
  const L = pow(10, mApp.sub(u.mLim).mul(-0.4)).mul(u.gain);
  const alpha = min(L, 1.0);
  const size = clamp(u.sizeBase.mul(pow(max(L, 1e-6), 0.27)), 0.0, u.sizeMax).mul(u.pxScale);
  m.positionNode = pos;
  m.sizeNode = select(alpha.greaterThan(0.003), size.max(1.0), float(0.0));
  const col = pal.element(int(ph.y));
  const vCol = varying(vec4(col, alpha));
  m.colorNode = Fn(() => {
    const r = length(uv().sub(0.5)).mul(2);
    const a = smoothstep(1.0, 0.15, r);
    return vec4(vCol.rgb.mul(a.mul(a)).mul(vCol.a), 1.0);
  })();
  return m;
}

export function makeStars(material, count) {
  const s = new THREE.Sprite(material);
  s.count = count;
  s.frustumCulled = false;
  return s;
}
