/* Everything drawn on top of the stars: constellation figures (today's, and
   the same figures following their stars), labels, the Sun and Galactic
   Centre markers, the Sun's orbit, the selection and its uncertainty cloud,
   and — above the disk — an ILLUSTRATIVE Milky Way (Cosmos's procedural
   galaxy, clearly labelled as not data). */

import * as THREE from 'three/webgpu';
import { Fn, vec4, uv, length, smoothstep, instancedBufferAttribute, varying, float, abs, select } from 'three/tsl';
import { hermite, starAt } from './cpu.js';

const LY = 3.2615637771674333;
const FLAG_NAMED = 1 << 5;

export class Overlays {
  constructor(scene, camera, manifest, sun, cat, figures, names, palette) {
    this.scene = scene; this.camera = camera; this.m = manifest; this.sun = sun; this.cat = cat;
    this.palette = palette;
    this.T = manifest.tracked_count;
    this.trPos = new Float64Array(this.T * 3);
    this.tr = null; this.pending = false;
    this.loaded = 0;
    this.nameOf = new Map(names.stars.map(([i, n]) => [i, n]));
    this.cloud = null;                         // Sun-relative gc positions of the selected star's clones

    // ---- constellation figures: pairs of tracked indices
    const seg = [];
    this.figures = figures.figures.map(f => {
      const ids = new Set();
      for (const line of f.lines) for (let k = 0; k + 1 < line.length; k++) {
        seg.push(line[k], line[k + 1]); ids.add(line[k]); ids.add(line[k + 1]);
      }
      return { name: f.name, ids: [...ids] };
    }).filter(f => f.ids.length);
    this.seg = Int32Array.from(seg);
    const mkLines = (opacity, color) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.seg.length * 3), 3));
      // per-vertex fade: a figure segment stretched across the sky is no longer a figure
      const c = new THREE.Color(color).multiplyScalar(opacity), col = new Float32Array(this.seg.length * 3);
      for (let k = 0; k < this.seg.length; k++) col.set([c.r, c.g, c.b], k * 3);
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      g.userData.base = c;
      const mat = new THREE.LineBasicNodeMaterial({ vertexColors: true, transparent: true, depthTest: false, depthWrite: false,
        blending: THREE.AdditiveBlending });
      const l = new THREE.LineSegments(g, mat);
      l.frustumCulled = false; l.matrixAutoUpdate = false;
      scene.add(l);
      return l;
    };
    this.lines = mkLines(0.30, 0x6f8cb4);
    this.ghost = mkLines(0.11, 0x6f8cb4);
    this.ghostFilled = false;

    // ---- the Sun's orbit, absolute galactocentric, ±250 Myr
    {
      const N = 1001, pos = new Float32Array(N * 3), col = new Float32Array(N * 3), p = [0, 0, 0];
      for (let k = 0; k < N; k++) {
        const t = -250 + k * 0.5;
        sun.at(t, p, null);
        pos.set(p, k * 3);
        const past = t < 0, a = 0.25 + 0.75 * (1 - Math.abs(t) / 250);
        col.set(past ? [0.35 * a, 0.55 * a, 0.9 * a] : [0.95 * a, 0.7 * a, 0.35 * a], k * 3);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.BufferAttribute(col, 3));
      this.trail = new THREE.Line(g, new THREE.LineBasicNodeMaterial({ vertexColors: true, transparent: true,
        opacity: 0.55, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending }));
      this.trail.frustumCulled = false; this.trail.matrixAutoUpdate = false;
      scene.add(this.trail);
    }

    // ---- markers: Sun, Galactic Centre, selection ring, uncertainty cloud
    this.MK = 3 + 64;
    this.mkPos = new THREE.InstancedBufferAttribute(new Float32Array(this.MK * 3), 3);
    this.mkCol = new THREE.InstancedBufferAttribute(new Float32Array(this.MK * 4), 4);
    this.mkSize = new THREE.InstancedBufferAttribute(new Float32Array(this.MK * 2), 2);
    {
      const m = new THREE.PointsNodeMaterial({ transparent: true, depthTest: false, depthWrite: false,
        blending: THREE.AdditiveBlending, sizeAttenuation: false });
      const col = instancedBufferAttribute(this.mkCol), sz = instancedBufferAttribute(this.mkSize);
      m.positionNode = instancedBufferAttribute(this.mkPos);
      m.sizeNode = sz.x;
      const vc = varying(col), ring = varying(sz.y);
      m.colorNode = Fn(() => {
        const r = length(uv().sub(0.5)).mul(2);
        const dot = smoothstep(1.0, 0.1, r).pow(2);
        const rim = smoothstep(0.16, 0.0, abs(r.sub(0.82)));
        const a = select(ring.greaterThan(0.5), rim, dot);
        return vec4(vc.rgb.mul(a).mul(vc.a), 1);
      })();
      this.markers = new THREE.Sprite(m);
      this.markers.count = this.MK; this.markers.frustumCulled = false;
      scene.add(this.markers);
    }

    // ---- illustrative galaxy (disk & ride views)
    this.galaxy = makeGalaxy();
    this.galaxy.matrixAutoUpdate = false;
    scene.add(this.galaxy);

    // ---- labels (DOM pool, as in Cosmos)
    this.pool = [];
    for (let i = 0; i < 30; i++) {
      const d = document.createElement('div');
      d.className = 'olabel'; d.style.display = 'none';
      d.addEventListener('pointerdown', e => {
        e.stopPropagation(); e.preventDefault();
        if (d._idx >= 0 && this.onSelect) this.onSelect(d._idx);
      });
      document.body.appendChild(d); this.pool.push(d);
    }
    this._v = new THREE.Vector3(); this._f = new THREE.Vector3();
  }

  setLoaded(n) { this.loaded = n; this._sel = null; }

  /* ask the GPU for the tracked stars' bracketing states (small async readback) */
  requestTracked(integ) {
    if (this.pending || integ.n < 0) return;
    const key = integ.sg + ':' + integ.n + ':' + (integ.dirty ? 1 : 0);
    if (this.tr && this.tr.key === key) return;
    this.pending = true;
    const sg = integ.sg, n = integ.n;
    integ.readback(Math.min(this.T, this.loaded)).then(r => {
      this.tr = { ...r, sg, n, key }; this.pending = false;
    }).catch(e => { this.pending = false; if (!this._warned) { this._warned = true; console.warn('[time] tracked readback failed', e); } });
  }

  /* Sun-relative gc positions of tracked stars at the shown time */
  _tracked(S, integ, u) {
    const T = Math.min(this.T, this.loaded), out = this.trPos, cat = this.cat;
    if (!integ) {
      const t = S.tShown;
      for (let i = 0; i < T; i++) for (let c = 0; c < 3; c++) out[i * 3 + c] = cat.pos[i * 4 + c] + cat.vel[i * 4 + c] * t;
      return true;
    }
    const tr = this.tr;
    if (!tr) return false;
    // The readback can trail the GPU by a step or two; interpolate within the
    // bracket we have (clamped to its ends) instead of snapping back to A.
    const h = tr.sg * this.sun.dt, x = [0, 0, 0];
    const f = tr.sg === Math.sign(S.tShown || tr.sg)
      ? Math.min(1, Math.max(0, Math.abs(S.tShown) / this.sun.dt - tr.n)) : 0;
    for (let i = 0; i < T; i++) { hermite(tr.pa, tr.va, i * 4, tr.pb, tr.vb, i * 4, h, f, x, null); out.set(x, i * 3); }
    return true;
  }

  /* Sun-relative gc position of any star at the shown time (float64 CPU twin) */
  starPos(i, S, integ) {
    if (i < 0) return null;
    if (i < this.T && this.tr) return [this.trPos[i * 3], this.trPos[i * 3 + 1], this.trPos[i * 3 + 2]];
    const c = this.cat;
    const x0 = [c.pos[i * 4], c.pos[i * 4 + 1], c.pos[i * 4 + 2]], v0 = [c.vel[i * 4], c.vel[i * 4 + 1], c.vel[i * 4 + 2]];
    if (!integ) return x0.map((x, k) => x + v0[k] * S.tShown);
    if (this._sel && this._sel.i === i && this._sel.t === S.tShown) return this._sel.x;
    const [x] = starAt(this.table, this.sun, x0, v0, S.tShown);
    this._sel = { i, t: S.tShown, x };
    return x;
  }

  update(dt, S, views, sunPos, integ, u) {
    const v = views.view, earth = v === 'earth';
    const have = this._tracked(S, integ, u);
    const cam = this.camera;

    // constellation lines (Earth view)
    const showLines = earth && S.lines && have;
    this.lines.visible = showLines; this.ghost.visible = showLines && Math.abs(S.tShown) > 1e-6;
    if (showLines) {
      const g = this.lines.geometry, p = g.attributes.position.array, col = g.attributes.color.array, base = g.userData.base;
      const s = this.seg, tp = this.trPos;
      for (let k = 0; k < s.length; k++) { p[k * 3] = tp[s[k] * 3]; p[k * 3 + 1] = tp[s[k] * 3 + 1]; p[k * 3 + 2] = tp[s[k] * 3 + 2]; }
      for (let k = 0; k < s.length; k += 2) {
        const a = k * 3, b = a + 3;
        const la = Math.hypot(p[a], p[a + 1], p[a + 2]), lb = Math.hypot(p[b], p[b + 1], p[b + 2]);
        const cos = (p[a] * p[b] + p[a + 1] * p[b + 1] + p[a + 2] * p[b + 2]) / (la * lb);
        const deg = Math.acos(Math.max(-1, Math.min(1, cos))) * 57.29578;
        const f = deg < 25 ? 1 : deg > 45 ? 0 : 1 - (deg - 25) / 20;
        for (const o of [a, b]) { col[o] = base.r * f; col[o + 1] = base.g * f; col[o + 2] = base.b * f; }
      }
      g.attributes.position.needsUpdate = true; g.attributes.color.needsUpdate = true;
      if (!this.ghostFilled && this.loaded >= this.T) {
        const g = this.ghost.geometry.attributes.position.array, cp = this.cat.pos;
        for (let k = 0; k < s.length; k++) for (let c = 0; c < 3; c++) g[k * 3 + c] = cp[s[k] * 4 + c];
        this.ghost.geometry.attributes.position.needsUpdate = true;
        this.ghostFilled = true;
      }
      this.lines.matrix.copy(views.frame); this.ghost.matrix.copy(views.frame);
      this.lines.matrixWorldNeedsUpdate = this.ghost.matrixWorldNeedsUpdate = true;
    }

    // trail + galaxy (above the disk / riding)
    this.trail.visible = !earth;
    this.galaxy.visible = !earth && S.backdrop !== false;
    if (!earth) {
      this.trail.matrix.copy(views.gcFrame); this.trail.matrixWorldNeedsUpdate = true;
      this.galaxy.matrix.copy(views.gcFrame).multiply(GINV); this.galaxy.matrixWorldNeedsUpdate = true;
    }

    // markers
    const P = this.mkPos.array, Cc = this.mkCol.array, Z = this.mkSize.array;
    P.fill(0); Cc.fill(0); Z.fill(0);
    const put = (k, sp, rgba, size, ring) => { P.set([sp.x, sp.y, sp.z], k * 3); Cc.set(rgba, k * 4); Z.set([size, ring ? 1 : 0], k * 2); };
    const toScene = (x, m, out) => out.set(x[0], x[1], x[2]).applyMatrix4(m);
    if (!earth) {
      put(0, toScene([0, 0, 0], views.frame, this._v), [1, 0.86, 0.45, 1], 9, false);              // the Sun
      put(1, toScene([0, 0, 0], views.gcFrame, this._f), [1, 0.75, 0.55, 0.8], 7, false);          // Sgr A*
    }
    const sel = this.starPos(S.selected, S, integ);
    this.selScene = null;
    if (sel) {
      this.selScene = toScene(sel, views.frame, new THREE.Vector3());
      put(2, this.selScene, [1, 0.88, 0.62, 0.9], earth ? 26 : 22, true);
      if (this.cloud) {
        for (let k = 0; k < Math.min(64, this.cloud.length); k++)
          put(3 + k, toScene(this.cloud[k], views.frame, this._v), [1, 0.8, 0.5, 0.5], 3, false);
      }
    }
    views.followPos = views.follow >= 0 ? (views.follow === S.selected ? this.selScene : null) : null;
    this.mkPos.needsUpdate = this.mkCol.needsUpdate = this.mkSize.needsUpdate = true;

    this._labels(S, views, sunPos);
  }

  _labels(S, views, sunPos) {
    const cam = this.camera, W = innerWidth, H = innerHeight, cands = [];
    const camDir = cam.getWorldDirection(new THREE.Vector3());
    const v = this._v, earth = views.view === 'earth';
    const proj = (sp, item) => {
      v.copy(sp);
      if (this._f.subVectors(v, cam.position).dot(camDir) <= 0) return;
      v.project(cam);
      if (v.z > 1) return;
      const x = (v.x * 0.5 + 0.5) * W, y = (-v.y * 0.5 + 0.5) * H;
      if (x < -40 || x > W + 40 || y < -30 || y > H + 30) return;
      cands.push({ ...item, x, y });
    };
    const T = Math.min(this.T, this.loaded), tp = this.trPos, ph = this.cat.phot, fl = this.cat.flags;
    if (S.labels && this.tr !== undefined) {
      if (earth) {
        const lim = views.mLim - 3.8;
        for (let i = 0; i < T; i++) {
          if (!(fl[i] & FLAG_NAMED)) continue;
          const d = Math.hypot(tp[i * 3], tp[i * 3 + 1], tp[i * 3 + 2]);
          const mApp = ph[i * 2] + 5 * Math.log10(d / 10);
          if (mApp > lim && i !== S.selected) continue;
          const sp = new THREE.Vector3(tp[i * 3], tp[i * 3 + 1], tp[i * 3 + 2]).applyMatrix4(views.frame);
          proj(sp, { i, name: this.nameOf.get(i) || '', pri: -mApp, cls: mApp < lim - 2 ? '' : 'dim' });
        }
        if (S.lines) for (const f of this.figures) {
          let x = 0, y = 0, z = 0;
          for (const i of f.ids) { const d = Math.hypot(tp[i * 3], tp[i * 3 + 1], tp[i * 3 + 2]) || 1;
            x += tp[i * 3] / d; y += tp[i * 3 + 1] / d; z += tp[i * 3 + 2] / d; }
          const sp = new THREE.Vector3(x, y, z).normalize().multiplyScalar(1000).applyMatrix4(views.frame);
          proj(sp, { i: -1, name: f.name.toUpperCase(), pri: -50, cls: 'const' });
        }
      } else {
        proj(new THREE.Vector3(0, 0, 0).applyMatrix4(views.frame), { i: -1, name: 'THE SUN', pri: 100, cls: 'big' });
        proj(new THREE.Vector3(0, 0, 0).applyMatrix4(views.gcFrame), { i: -1, name: 'SAGITTARIUS A★', pri: 90, cls: 'dim' });
      }
    }
    if (this.selScene && S.selected >= 0) {
      proj(this.selScene, { i: S.selected, name: this.nameOf.get(S.selected) || this.selLabel || 'SELECTED STAR', pri: 1000, cls: 'sel' });
    }
    // de-duplicate the selected star, sort by priority, place
    const seen = new Set();
    const list = cands.sort((a, b) => b.pri - a.pri).filter(c => {
      if (c.i >= 0) { if (seen.has(c.i)) return false; seen.add(c.i); }
      return true;
    });
    const n = Math.min(list.length, this.pool.length);
    for (let k = 0; k < n; k++) {
      const c = list[k], d = this.pool[k];
      d.style.display = 'block';
      d.style.left = c.x + 'px'; d.style.top = c.y + 'px';
      d.className = 'olabel ' + c.cls;
      if (d.textContent !== c.name.toUpperCase()) d.textContent = c.name.toUpperCase();
      d._idx = c.i;
    }
    for (let k = n; k < this.pool.length; k++) this.pool[k].style.display = 'none';
  }
}

/* disk scene ← gc is G; the galaxy is generated directly in disk-scene axes */
const GINV = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1);

/* Cosmos's procedural Milky Way (index.html makeGalaxy), rebuilt for
   three/webgpu in parsecs: exponential disk with inter-arm fill, central
   bar, two major + two minor arms, warm bulge, pink HII knots. Arms trail
   the rotation the orbits actually have. Illustration only. */
function makeGalaxy() {
  const U = 30.66;                                   // Cosmos's unit: 100 ly in pc
  const R = 500 * U, N = 110000, arms = 4, pitch = 0.23, barR = 0.26 * R, minor = 0.55;
  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) * 0.85;
  const lerp = (a, b, t) => a + (b - a) * t;
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 4), sz = new Float32Array(N);
  const cCore = [1, 0.886, 0.69], cArm = [0.784, 0.847, 1], cFill = [0.937, 0.902, 0.847], cPink = [1, 0.66, 0.745];
  const weights = []; let wSum = 0;
  for (let k = 0; k < arms; k++) { const w = k % 2 === 0 ? 1 : minor; weights.push(w); wSum += w; }
  const r0 = Math.max(barR * 0.85, R * 0.05), NB = Math.floor(N * 0.22);
  const setC = (i, c, s) => { col[i * 4] = c[0] * s; col[i * 4 + 1] = c[1] * s; col[i * 4 + 2] = c[2] * s; col[i * 4 + 3] = 1; };
  const mixC = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  const barAngle = -0.5;                             // bar points ~30° from the Sun–Centre line
  for (let i = 0; i < N; i++) {
    let x, y, z;
    if (i < NB) {
      if (Math.random() < 0.52) { x = gauss() * barR * 0.6; z = gauss() * barR * 0.17; y = gauss() * barR * 0.11;
        const c = Math.cos(barAngle), s = Math.sin(barAngle); [x, z] = [x * c - z * s, x * s + z * c]; }
      else { const rr = Math.abs(gauss()) * R * 0.13, a = Math.random() * 6.283; x = Math.cos(a) * rr; z = Math.sin(a) * rr * 0.85; y = gauss() * R * 0.05; }
      setC(i, cCore, 0.6 + Math.random() * 0.8); sz[i] = 1.2 + Math.random() * 1.5;
    } else {
      let r = -0.30 * R * Math.log(1 - Math.random());
      r = Math.min(Math.max(r, R * 0.04), R * 1.08);
      const inArm = r > r0 * 0.85 && Math.random() < lerp(0.74, 0.52, r / R);
      let a;
      if (inArm) {
        let pick = Math.random() * wSum, k = 0;
        while (pick > weights[k]) { pick -= weights[k]; k++; }
        a = barAngle - Math.log(Math.max(r, r0) / r0) / pitch + k * (6.283 / arms) + gauss() * (0.14 + 0.11 * r / R);
      } else a = Math.random() * 6.283;
      x = Math.cos(a) * r; z = Math.sin(a) * r;
      y = gauss() * (R * 0.011 + R * 0.02 * Math.pow(r / R, 2));
      const t = r / R;
      if (inArm) {
        const pink = Math.random() < 0.045;
        setC(i, pink ? cPink : mixC(cArm, cCore, Math.max(0, 0.7 - t * 1.5)), 0.5 + Math.random() * 0.7);
        sz[i] = (pink ? 1.5 : 1.05) + Math.random() * 1.35;
      } else {
        setC(i, mixC(cFill, cCore, Math.max(0, 0.6 - t * 1.2)), 0.22 + Math.random() * 0.33);
        sz[i] = 0.9 + Math.random() * 0.9;
      }
    }
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
  }
  const m = new THREE.PointsNodeMaterial({ transparent: true, depthTest: false, depthWrite: false,
    blending: THREE.AdditiveBlending, sizeAttenuation: false });
  const p = instancedBufferAttribute(new THREE.InstancedBufferAttribute(pos, 3));
  const c = varying(instancedBufferAttribute(new THREE.InstancedBufferAttribute(col, 4)));
  m.positionNode = p;
  m.sizeNode = instancedBufferAttribute(new THREE.InstancedBufferAttribute(sz, 1));
  m.colorNode = Fn(() => {
    const r = length(uv().sub(0.5)).mul(2);
    const a = smoothstep(1.0, 0.15, r);
    return vec4(c.rgb.mul(a.mul(a)).mul(0.16), 1);
  })();
  const s = new THREE.Sprite(m);
  s.count = N; s.frustumCulled = false;
  return s;
}
