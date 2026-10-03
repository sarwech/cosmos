/* HUD, scrubber, play controls, keys, ⌘K, the star card, deep links, photo
   mode and the guided "moments". */

import * as THREE from 'three/webgpu';
import { sToT, tToS, parseTime, linkTime, describe, shortTime, T_MAX } from './timefmt.js';
import { hermite, stepRel, starAt, rng, KMS_TO_PCMYR, PC_TO_LY } from './cpu.js';
import { VIEWS, VIEW_LABEL } from './views.js';

const $ = id => document.getElementById(id);
const IS_MAC = /Mac|iPhone|iPad|iPod/.test(navigator.platform || '');
const MODK = IS_MAC ? '⌘K' : 'CTRL+K';
const SPEEDS = [1e-4, 1e-3, 1e-2, 0.1, 1, 10];                // Myr per second
const SPEED_LABEL = ['100 YR / S', '1,000 YR / S', '10,000 YR / S', '100,000 YR / S', '1 MILLION YR / S', '10 MILLION YR / S'];
const FLAG = { hip: 1, named: 32 };
const RV_SRC = ['Gaia DR3', 'XHIP (literature)', 'SIMBAD (literature)', 'none — assumed 0'];
const nf = (x, d = 0) => x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
export const slug = s => String(s).toLowerCase().replace(/['’]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const loose = s => slug(s).replace(/-/g, '');

export class UI {
  constructor(ctx) {
    Object.assign(this, ctx);
    const { S } = ctx;
    this.views.onPick = (x, y) => this.pick(x, y);
    this.views.onInput = () => { S.lastInput = performance.now(); };
    this.overlays.onSelect = i => this.select(i, true);
    this.overlays.table = ctx.table;
    this.dragging = false; this.playTo = null;
    this.wantShot = false; this.lastHash = ''; this.hashT = 0;
    this.slugIdx = new Map();
    for (const [i, n, alias] of ctx.names.stars) {
      for (const s of [n, ...(alias ? alias.split('|') : [])]) if (s && !this.slugIdx.has(loose(s))) this.slugIdx.set(loose(s), i);
    }
    this._hud(); this._scrubber(); this._buttons(); this._keys(); this._palette();
    $('modkHint').textContent = MODK; $('palBtn').textContent = MODK + ' EXPLORE';
    addEventListener('hashchange', () => this.applyHash(location.hash));
  }

  ready({ integ }) {
    this.integ = integ;
    this.applyHash(location.hash, true);
    this.clones = null;
  }

  notice(text) { const n = $('notice'); n.textContent = text; n.style.display = 'block'; }

  /* ---------------------------------------------------------------- HUD -- */
  _hud() {
    const tr = $('ladderTrack');
    this.ticks = VIEWS.map((v, k) => {
      const d = document.createElement('div');
      d.className = 'tick';
      d.innerHTML = VIEW_LABEL[v] + `<small>${['MEASURED SKY, THEN SIMULATED', 'THE SUN’S ORBIT', 'A CO-MOVING FRAME'][k]}</small>`;
      d.style.top = (10 + k * 40) + '%';
      d.onclick = () => this.setView(v);
      tr.appendChild(d);
      return d;
    });
  }

  setView(v) {
    if (!VIEW_LABEL[v]) return;
    this.views.set(v); this.S.view = v;
    this.ticks.forEach((d, k) => d.classList.toggle('on', VIEWS[k] === v));
    $('ladderDot').style.top = (10 + VIEWS.indexOf(v) * 40) + '%';
    $('viewLbl').textContent = VIEW_LABEL[v];
    if (this.S.selected >= 0 && this.views.follow === this.S.selected && v === 'earth') this.views.lookAt(this.overlays.selScene || new THREE.Vector3(0, 0, -1));
  }

  frame() {
    const S = this.S, t = S.tShown;
    const d = describe(t);
    $('timeVal').textContent = d.main;
    $('timeYr').textContent = d.sub;
    const measured = Math.abs(t) * 1e6 < 0.5;
    const b = $('badge');
    b.className = measured ? 'measured' : 'simulated';
    b.textContent = measured ? 'MEASURED · GAIA DR3 & HIPPARCOS' : (S.fallback ? 'SIMULATED · STRAIGHT-LINE MOTION' : 'SIMULATED · McMILLAN17 POTENTIAL');
    $('honest').innerHTML = this._honest(t);
    const busy = $('busy');
    if (S.loading) { busy.textContent = S.loading; busy.style.opacity = 1; }
    else if (S.busy && this.integ) {
      const tot = Math.max(1, Math.floor(Math.abs(S.t) / this.sun.dt));
      busy.textContent = `INTEGRATING ${this.cat.loaded.toLocaleString()} ORBITS → ${shortTime(S.t)}`;
      busy.style.opacity = 1;
    } else busy.style.opacity = 0;
    this._placeDot();
    this._hashTick();
    if (this.S.selected >= 0) this._cardTick();
  }

  _honest(t) {
    const a = Math.abs(t), S = this.S;
    if (a * 1e6 < 0.5) return 'Positions and velocities as <b>measured</b>, epoch 2016.0. Move the time slider to simulate.' +
      (S.view !== 'earth' && S.backdrop !== false ? '<br>The spiral behind the real stars is an illustration, not data (G hides it).' : '');
    const u = this.manifest.uncertainty, which = S.view === 'disk' ? 'disk' : 'local';
    let txt = '';
    if (u) {
      const ts = u.times_myr, ys = u.median_spread_pc[which];
      const lt = Math.log10(Math.max(a, ts[0])), k = Math.min(ts.length - 2, Math.max(0, ts.findIndex(x => Math.log10(x) >= lt) - 1));
      const f = Math.min(1, Math.max(0, (lt - Math.log10(ts[k])) / (Math.log10(ts[k + 1]) - Math.log10(ts[k]))));
      const pc = ys[k] + (ys[k + 1] - ys[k]) * f;
      const ly = pc * PC_TO_LY;
      txt = `Typical position uncertainty from the measurements alone: <b>±${ly < 10 ? ly.toFixed(1) : nf(Math.round(ly))} ly</b>`;
    }
    const notes = [];
    if (S.view !== 'earth' && S.backdrop !== false) notes.push('the spiral behind is an illustration, not data (G hides it)');
    if (S.view === 'earth' && a > 2) notes.push('the sky is thinning: only stars near us <i>today</i> are in the data');
    if (a > 10) notes.push('no stellar evolution — young blue giants appear before they were born');
    if (a > 30 && !S.fallback) notes.push('no spiral arms or bar in the model, so single orbits are illustrative');
    return txt + (notes.length ? '<br>' + notes.join(' · ') : '');
  }

  /* ----------------------------------------------------------- scrubber -- */
  _scrubber() {
    const track = $('scrubTrack'), wrap = $('scrub');
    const marks = [[-250, '−250 MYR'], [-100, '−100 MYR', 1], [-10, '−10 MYR', 1], [-1, '−1 MYR'], [-0.1, '−100,000 YR', 1],
      [-0.01, '−10,000 YR', 1], [0, 'TODAY'], [0.01, '+10,000 YR', 1], [0.1, '+100,000 YR', 1], [1, '+1 MYR'],
      [10, '+10 MYR', 1], [100, '+100 MYR', 1], [250, '+250 MYR']];
    this.stickEls = marks.map(([t, label, minor]) => {
      const d = document.createElement('div');
      d.className = 'stick' + (t === 0 ? ' zero' : '') + (minor ? ' minor' : '');
      d.textContent = label;
      d.style.left = ((tToS(t) + 1) / 2 * 100) + '%';
      d.onclick = () => { this.stopPlay(); this.setT(t); };
      wrap.appendChild(d);
      return { d, t };
    });
    const setFromX = x => {
      const r = track.getBoundingClientRect();
      let s = Math.max(-1, Math.min(1, ((x - r.left) / r.width) * 2 - 1));
      let t = sToT(s);
      if (Math.abs(s) < 0.004) t = 0;                             // a detent at today
      this.setT(t);
    };
    track.addEventListener('pointerdown', e => {
      track.setPointerCapture(e.pointerId); this.dragging = true; this.stopPlay(); setFromX(e.clientX);
    });
    track.addEventListener('pointermove', e => { if (this.dragging) setFromX(e.clientX); });
    const end = () => { this.dragging = false; };
    track.addEventListener('pointerup', end); track.addEventListener('pointercancel', end);
  }
  _placeDot() {
    const s = tToS(this.S.t);
    $('scrubDot').style.left = ((s + 1) / 2 * 100) + '%';
    if (this.S.fallback) for (const { d, t } of this.stickEls) d.style.opacity = Math.abs(t) > 1 ? 0.25 : 1;
  }

  /* ---------------------------------------------------------- playback -- */
  _buttons() {
    $('playBtn').onclick = () => this.togglePlay(1);
    $('revBtn').onclick = () => this.togglePlay(-1);
    $('fwdBtn').onclick = () => this.setSpeed(this.S.speed + 1);
    $('todayBtn').onclick = () => { this.stopPlay(); this.setT(0); };
    $('constBtn').onclick = () => this.toggleLines();
    $('palBtn').onclick = () => this.togglePal();
    $('cardX').onclick = () => this.select(-1);
    this.setSpeed(this.S.speed);
  }
  /* every jump of the time target goes through here, so it is always in range */
  setT(t) {
    clearTimeout(this._momentT);
    const S = this.S;
    S.t = Math.max(-S.tMax, Math.min(S.tMax, t));
    S.lastInput = performance.now();
  }
  /* the play speed that covers |Δt| in roughly 6 seconds (speeds step by 10×) */
  speedFor(dt) {
    const want = Math.max(Math.abs(dt), 1e-6) / 6;
    let best = 0;
    SPEEDS.forEach((v, k) => { if (Math.abs(Math.log(v / want)) < Math.abs(Math.log(SPEEDS[best] / want))) best = k; });
    return best;
  }
  setSpeed(i) {
    this.S.speed = Math.max(0, Math.min(SPEEDS.length - 1, i));
    $('speed').textContent = SPEED_LABEL[this.S.speed];
  }
  togglePlay(dir) {
    clearTimeout(this._momentT);
    const S = this.S;
    if (S.playing && S.dir === dir) { this.stopPlay(); return; }
    S.playing = true; S.dir = dir; this.playTo = null;
    if (Math.abs(S.t) >= S.tMax - 1e-9 && Math.sign(S.t) === dir) S.t = 0;
    this._playBtns();
  }
  stopPlay() { this.S.playing = false; this.playTo = null; this._playBtns(); }
  _playBtns() {
    const S = this.S;
    $('playBtn').textContent = S.playing && S.dir > 0 ? '❚❚' : '▶';
    $('revBtn').textContent = S.playing && S.dir < 0 ? '❚❚' : '◀◀';
    $('playBtn').classList.toggle('on', S.playing && S.dir > 0);
    $('revBtn').classList.toggle('on', S.playing && S.dir < 0);
  }
  /* animate to a time at a given speed, then stop */
  playToTime(t, speed) {
    const S = this.S;
    t = Math.max(-S.tMax, Math.min(S.tMax, t));
    if (speed !== undefined) this.setSpeed(speed);
    S.dir = t >= S.t ? 1 : -1; S.playing = true; this.playTo = t; this._playBtns();
  }
  tick(dt) {
    const S = this.S;
    if (!S.playing) return;
    if (S.busy && this.integ) return;               // let the integrator catch up
    let t = S.t + S.dir * SPEEDS[S.speed] * dt;
    if (this.playTo !== null && (t - this.playTo) * S.dir >= 0) { t = this.playTo; this.stopPlay(); }
    if (Math.abs(t) >= S.tMax) { t = Math.sign(t) * S.tMax; this.stopPlay(); }
    S.t = t;
  }

  toggleLines() { this.S.lines = !this.S.lines; $('constBtn').classList.toggle('on', this.S.lines); }

  /* -------------------------------------------------------------- keys -- */
  _keys() {
    $('constBtn').classList.toggle('on', this.S.lines);
    addEventListener('keydown', e => {
      if ($('palette').classList.contains('open')) return;
      const S = this.S;
      if ((e.metaKey || e.ctrlKey) && e.code === 'KeyK') { e.preventDefault(); this.togglePal(); return; }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const step = e.shiftKey ? 0.05 : 0.005;
      switch (e.code) {
        case 'Space': e.preventDefault(); this.togglePlay(S.playing ? S.dir : 1); break;
        case 'ArrowRight': this.stopPlay(); this.setT(sToT(tToS(S.t) + step)); break;
        case 'ArrowLeft': this.stopPlay(); this.setT(sToT(tToS(S.t) - step)); break;
        case 'BracketRight': this.setSpeed(S.speed + 1); break;
        case 'BracketLeft': this.setSpeed(S.speed - 1); break;
        case 'Digit1': this.setView('earth'); break;
        case 'Digit2': this.setView('disk'); break;
        case 'Digit3': this.setView('ride'); break;
        case 'KeyC': this.toggleLines(); break;
        case 'KeyL': S.labels = !S.labels; break;
        case 'KeyB': S.bloom = !S.bloom; this.caption('Bloom ' + (S.bloom ? 'on' : 'off'), ''); break;
        case 'KeyG': S.backdrop = S.backdrop === false; this.caption('Illustrated galaxy ' + (S.backdrop === false ? 'off' : 'on'), 'ABOVE-THE-DISK VIEWS'); break;
        case 'KeyS': this.wantShot = true; break;
        case 'Equal': case 'NumpadAdd': this.views.depth = Math.min(14, this.views.depth + 1); this.caption('Fainter stars', `LIMITING MAGNITUDE ${this.views.depth.toFixed(1)}`); break;
        case 'Minus': case 'NumpadSubtract': this.views.depth = Math.max(2, this.views.depth - 1); this.caption('Brighter stars only', `LIMITING MAGNITUDE ${this.views.depth.toFixed(1)}`); break;
        case 'Digit0': case 'Home': this.stopPlay(); this.setT(0); break;
        case 'Escape': if (S.selected >= 0) this.select(-1); break;
        default: return;
      }
    });
  }

  caption(main, sub, ms = 2200) {
    const c = $('caption');
    c.innerHTML = main + (sub ? `<small>${sub}</small>` : '');
    c.style.opacity = 1;
    clearTimeout(this._capT);
    this._capT = setTimeout(() => { c.style.opacity = 0; }, ms);
  }

  /* --------------------------------------------------------- selection -- */
  select(i, fly) {
    const S = this.S;
    if (i >= this.cat.loaded) i = -1;               // never touch a star whose tier isn't decoded
    S.selected = i;
    this.clones = null; this.overlays.cloud = null; this.closest = null;
    this.overlays.selLabel = ''; this.selIds = null;
    if (i < 0) { $('card').classList.remove('show'); this.views.follow = -1; return; }
    if (fly || this.views.follow >= 0) this.views.follow = i;
    if (fly && this.S.view !== 'earth') this.views.snapTarget = true;
    this._card(i);
  }

  async _card(i) {
    const c = this.cat, S = this.S;
    const name = this.overlays.nameOf.get(i);
    const flags = c.flags[i], hip = !!(flags & FLAG.hip), rvSrc = (flags >> 1) & 3, sample = (flags >> 3) & 3;
    const x = [c.pos[i * 4], c.pos[i * 4 + 1], c.pos[i * 4 + 2]], v = [c.vel[i * 4], c.vel[i * 4 + 1], c.vel[i * 4 + 2]];
    const d = Math.hypot(...x), rv = (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]) / d / KMS_TO_PCMYR;
    const sp = Math.hypot(...v) / KMS_TO_PCMYR, vt = Math.sqrt(Math.max(0, sp * sp - rv * rv));
    const det = await c.details(i).catch(() => null);
    if (S.selected !== i) return;
    const id = det && det.sourceId ? `GAIA DR3 ${det.sourceId}` : det && det.hip ? `HIP ${det.hip}` : 'STAR #' + i;
    this.selIds = det ? { sourceId: det.sourceId, hip: det.hip } : null;
    this.overlays.selLabel = name || id;
    $('cardName').textContent = (name || id).toUpperCase();
    const teff = Math.round(Math.pow(10, this.manifest.quant.logteff_min + c.phot[i * 2 + 1] / 255 *
      (this.manifest.quant.logteff_max - this.manifest.quant.logteff_min)) / 100) * 100;
    $('cardSub').textContent = [name ? id : null, `≈${nf(teff)} K`, ['NEAREST-FIRST SAMPLE', 'DISK SAMPLE', 'BRIGHT/NAMED STAR'][sample]].filter(Boolean).join(' · ');
    const row = (k, val) => `<div class="row"><span>${k}</span><b>${val}</b></div>`;
    let h = `<div class="ph m">MEASURED · TODAY (EPOCH 2016.0)</div>`;
    h += row('ASTROMETRY', hip ? 'Hipparcos-2 (van Leeuwen 2007)' : 'Gaia DR3');
    if (det && det.hip && det.sourceId) h += row('HIPPARCOS', 'HIP ' + det.hip);
    const dly = d * PC_TO_LY;
    const derr = det && det.plx > 0 ? dly * det.plxErr / det.plx : null;
    h += row('DISTANCE', `${dly < 100 ? dly.toFixed(2) : nf(Math.round(dly))} ly` + (derr ? ` ± ${sig(derr)}` : ''));
    if (det && det.plx) h += row('PARALLAX', `${det.plx.toFixed(3)} ± ${det.plxErr.toFixed(3)} mas`);
    h += row('RADIAL VELOCITY', `${rv.toFixed(2)} km/s` + (det && isFinite(det.rvErr) && det.rvErr > 0 ? ` ± ${det.rvErr.toFixed(2)}` : ''));
    h += row('RV SOURCE', RV_SRC[rvSrc]);
    h += row('TANGENTIAL VELOCITY', `${vt.toFixed(2)} km/s`);
    h += row('SPEED RELATIVE TO SUN', `${sp.toFixed(1)} km/s`);
    if (det && isFinite(det.ruwe) && det.ruwe > 0) h += row('RUWE', det.ruwe.toFixed(2));
    h += `<div class="ph s">SIMULATED · AT THE SHOWN TIME</div><div id="cardSim"></div>`;
    h += `<div class="acts"><div class="btn" id="cardFly">◎ CENTRE</div><div class="btn" id="cardNear">⟷ CLOSEST APPROACH</div></div>`;
    h += `<div class="note">Distance and velocity today are measured. Every other time is a test-particle orbit in the McMillan (2017) potential${S.fallback ? ' (here: a straight line)' : ''}; the ± cloud shows only the spread from the measurement errors.</div>`;
    $('cardRows').innerHTML = h;
    $('card').classList.add('show');
    $('cardFly').onclick = () => { this.views.follow = i; if (this.S.view === 'earth' && this.overlays.selScene) this.views.lookAt(this.overlays.selScene); };
    $('cardNear').onclick = () => {
      const ca = this.closestApproach(i);
      if (ca) { this.playToTime(ca.t, this.speedFor(ca.t - this.S.t));
        this.caption(`Closest approach: ${fmtDist(ca.d)}`, describe(ca.t).main.toUpperCase()); }
    };
    if (det) this._makeClones(i, det);
    this._cardSimT = null;
  }

  _cardTick() {
    const S = this.S, i = S.selected, el = $('cardSim');
    if (!el) return;
    if (this._cardSimT === S.tShown && !this._cloneDirty) return;
    this._cardSimT = S.tShown; this._cloneDirty = false;
    const x = this.overlays.starPos(i, S, this.integ);
    if (!x) return;
    const sunP = [0, 0, 0]; this.sun.at(S.tShown, sunP, null);
    const g = [x[0] + sunP[0], x[1] + sunP[1], x[2] + sunP[2]];
    const row = (k, v) => `<div class="row"><span>${k}</span><b>${v}</b></div>`;
    let h = row('TIME', describe(S.tShown).main);
    h += row('DISTANCE FROM THE SUN', fmtDist(Math.hypot(...x)));
    h += row('FROM GALACTIC CENTRE', `${(Math.hypot(g[0], g[1]) * PC_TO_LY / 1000).toFixed(2)} kly · z ${nf(Math.round(g[2] * PC_TO_LY))} ly`);
    if (this.clones) {
      this._marchClones();
      const pts = this.overlays.cloud;
      if (pts && pts.length) {
        const m = [0, 0, 0]; for (const p of pts) for (let k = 0; k < 3; k++) m[k] += p[k] / pts.length;
        let s2 = 0; for (const p of pts) s2 += (p[0] - m[0]) ** 2 + (p[1] - m[1]) ** 2 + (p[2] - m[2]) ** 2;
        h += row('± FROM MEASUREMENT ERRORS', fmtDist(Math.sqrt(s2 / pts.length)));
      }
    }
    if (!this.closest || this.closest.i !== i) this.closest = { i, ...(this.closestApproach(i) || {}) };
    if (this.closest.t !== undefined) h += row('CLOSEST APPROACH', `${fmtDist(this.closest.d)} · ${shortTime(this.closest.t)}`);
    el.innerHTML = h;
  }

  /* closest approach to the Sun within the allowed range (CPU float64 twin) */
  closestApproach(i) {
    const c = this.cat, S = this.S;
    const x0 = [c.pos[i * 4], c.pos[i * 4 + 1], c.pos[i * 4 + 2]], v0 = [c.vel[i * 4], c.vel[i * 4 + 1], c.vel[i * 4 + 2]];
    if (S.fallback) {
      const vv = v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2;
      let t = -(x0[0] * v0[0] + x0[1] * v0[1] + x0[2] * v0[2]) / vv;
      t = Math.max(-S.tMax, Math.min(S.tMax, t));
      return { t, d: Math.hypot(...x0.map((x, k) => x + v0[k] * t)) };
    }
    const sun = this.sun, tab = this.table, N = Math.floor(S.tMax / sun.dt);
    let best = { t: 0, d: Math.hypot(...x0) };
    for (const sg of [1, -1]) {
      const x = x0.slice(), v = v0.slice(), h = sg * sun.dt;
      let prev = { x: x.slice(), v: v.slice() };
      for (let k = 0; k < N; k++) {
        stepRel(tab, sun, x, v, sun.nMax + sg * k, sun.nMax + sg * (k + 1), h);
        // sample the Hermite curve inside the step
        for (let j = 1; j <= 8; j++) {
          const o = [0, 0, 0]; hermite(prev.x, prev.v, 0, x, v, 0, h, j / 8, o, null);
          const d = Math.hypot(...o);
          if (d < best.d) best = { t: sg * (k + j / 8) * sun.dt, d, k, sg };
        }
        prev = { x: x.slice(), v: v.slice() };
        if (Math.hypot(...x) > 4 * best.d + 2000) break;          // receding for good
      }
    }
    if (best.t === 0) return null;
    // refine with a fine scan of the canonical state around the coarse minimum
    let lo = best.t - sun.dt / 8, hi = best.t + sun.dt / 8;
    for (let it = 0; it < 5; it++) {
      let bt = best.t, bd = best.d;
      for (let j = 0; j <= 20; j++) {
        const t = lo + (hi - lo) * j / 20;
        if (Math.sign(t) !== Math.sign(best.t)) continue;
        const d = Math.hypot(...starAt(tab, sun, x0, v0, t)[0]);
        if (d < bd) { bd = d; bt = t; }
      }
      best = { t: bt, d: bd }; const w = (hi - lo) / 10; lo = bt - w; hi = bt + w;
    }
    return best;
  }

  /* 48 clones drawn from the measurement errors, integrated on the CPU */
  _makeClones(i, det) {
    const c = this.cat, R = this.manifest.frame.icrs_to_gc, r = rng(i * 2654435761 + 7);
    const x = [c.pos[i * 4], c.pos[i * 4 + 1], c.pos[i * 4 + 2]], v = [c.vel[i * 4], c.vel[i * 4 + 1], c.vel[i * 4 + 2]];
    const tGc = (a) => [R[0][0] * a[0] + R[0][1] * a[1] + R[0][2] * a[2], R[1][0] * a[0] + R[1][1] * a[1] + R[1][2] * a[2], R[2][0] * a[0] + R[2][1] * a[1] + R[2][2] * a[2]];
    const tIcrs = (a) => [R[0][0] * a[0] + R[1][0] * a[1] + R[2][0] * a[2], R[0][1] * a[0] + R[1][1] * a[1] + R[2][1] * a[2], R[0][2] * a[0] + R[1][2] * a[1] + R[2][2] * a[2]];
    const xi = tIcrs(x), vi = tIcrs(v), d = Math.hypot(...xi);
    const n = xi.map(q => q / d), ra = Math.atan2(n[1], n[0]), de = Math.asin(n[2]);
    const ea = [-Math.sin(ra), Math.cos(ra), 0], ed = [-Math.sin(de) * Math.cos(ra), -Math.sin(de) * Math.sin(ra), Math.cos(de)];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const K = 4.740470463533348 * KMS_TO_PCMYR / 1000;            // pc/Myr per (mas/yr · pc)
    const rv = dot(vi, n), mua = dot(vi, ea) / (K * d), mud = dot(vi, ed) / (K * d);
    const plx = det.plx > 0 ? det.plx : 1000 / d;
    const clones = [];
    for (let k = 0; k < 48; k++) {
      const p2 = Math.max(plx * 0.05, plx + r() * (det.plxErr || 0));
      const d2 = d * plx / p2;
      const a2 = mua + r() * (det.pmraErr || 0), b2 = mud + r() * (det.pmdecErr || 0);
      const rv2 = rv + r() * (isFinite(det.rvErr) && det.rvErr > 0 ? det.rvErr : 0.5) * KMS_TO_PCMYR;
      const xx = n.map(q => q * d2);
      const vv = [0, 1, 2].map(j => rv2 * n[j] + K * d2 * (a2 * ea[j] + b2 * ed[j]));
      clones.push({ x0: tGc(xx), v0: tGc(vv), x: null, v: null });
    }
    this.clones = { list: clones, sg: 1, n: -1 };
    this._cloneDirty = true;
  }

  _marchClones() {
    const S = this.S, sun = this.sun, cl = this.clones;
    const t = S.tShown, sg = t >= 0 ? 1 : -1, q = Math.abs(t) / sun.dt;
    const nT = Math.min(Math.floor(q), sun.nMax - 2), f = q - nT, h = sg * sun.dt;
    if (S.fallback) { this.overlays.cloud = cl.list.map(c => c.x0.map((x, k) => x + c.v0[k] * t)); return; }
    if (cl.n < 0 || cl.sg !== sg || nT < cl.n) {
      for (const c of cl.list) { c.x = c.x0.slice(); c.v = c.v0.slice(); }
      cl.n = 0; cl.sg = sg;
    }
    const budget = 400;                       // steps per update; enough to keep up with play
    let k = 0;
    while (cl.n < nT && k < budget) {
      for (const c of cl.list) stepRel(this.table, sun, c.x, c.v, sun.nMax + sg * cl.n, sun.nMax + sg * (cl.n + 1), h);
      cl.n++; k++;
    }
    const out = [];
    for (const c of cl.list) {
      if (cl.n === nT) {
        const xb = c.x.slice(), vb = c.v.slice();
        stepRel(this.table, sun, xb, vb, sun.nMax + sg * nT, sun.nMax + sg * (nT + 1), h);
        const o = [0, 0, 0]; hermite(c.x, c.v, 0, xb, vb, 0, h, f, o, null); out.push(o);
      } else out.push(c.x.slice());
    }
    if (cl.n < nT) this._cloneDirty = true;
    this.overlays.cloud = out;
  }

  /* ------------------------------------------------------------ picking -- */
  async pick(px, py) {
    const S = this.S, cat = this.cat, cam = this.views.camera, M = this.views.frame;
    const W = innerWidth, H = innerHeight;
    const e = M.elements, pe = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).multiply(M).elements;
    const camP = cam.position;
    const mLim = this.views.mLim, gain = this.views.gain;
    let best = -1, bestScore = -Infinity;
    const consider = (i, x, y, z) => {
      const cw = pe[3] * x + pe[7] * y + pe[11] * z + pe[15];
      if (cw <= 0) return;
      const sx = ((pe[0] * x + pe[4] * y + pe[8] * z + pe[12]) / cw * 0.5 + 0.5) * W;
      const sy = (-(pe[1] * x + pe[5] * y + pe[9] * z + pe[13]) / cw * 0.5 + 0.5) * H;
      const dpx = Math.hypot(sx - px, sy - py);
      if (dpx > 18) return;
      const wx = e[0] * x + e[4] * y + e[8] * z + e[12] - camP.x, wy = e[1] * x + e[5] * y + e[9] * z + e[13] - camP.y,
            wz = e[2] * x + e[6] * y + e[10] * z + e[14] - camP.z;
      const mApp = cat.phot[i * 2] + 5 * Math.log10(Math.max(Math.hypot(wx, wy, wz), 1e-6) / 10);
      const L = Math.pow(10, -0.4 * (mApp - mLim)) * gain;
      if (L < 0.05) return;
      const score = Math.log10(L) * 2 - dpx / 5;
      if (score > bestScore) { bestScore = score; best = i; }
    };
    const n = cat.loaded;
    if (!this.integ) {
      const t = S.tShown;
      for (let i = 0; i < n; i++) consider(i, cat.pos[i * 4] + cat.vel[i * 4] * t, cat.pos[i * 4 + 1] + cat.vel[i * 4 + 1] * t, cat.pos[i * 4 + 2] + cat.vel[i * 4 + 2] * t);
    } else {
      // tracked stars first (cheap), then everything from a GPU readback
      const tp = this.overlays.trPos, T = Math.min(this.overlays.T, n);
      for (let i = 0; i < T; i++) consider(i, tp[i * 3], tp[i * 3 + 1], tp[i * 3 + 2]);
      if (best < 0) {
        const integ = this.integ, r = this.renderer;
        const [pa, pb] = await Promise.all([r.getArrayBufferAsync(integ.PA.value, null, 0, n * 16), r.getArrayBufferAsync(integ.PB.value, null, 0, n * 16)]);
        const A = new Float32Array(pa), B = new Float32Array(pb), f = S.busy ? 0 : Math.min(1, Math.max(0, (Math.abs(S.tShown) / this.sun.dt) % 1));
        for (let i = T; i < n; i++) {
          const o = i * 4;
          consider(i, A[o] + (B[o] - A[o]) * f, A[o + 1] + (B[o + 1] - A[o + 1]) * f, A[o + 2] + (B[o + 2] - A[o + 2]) * f);
        }
      }
    }
    if (best >= 0) this.select(best, false);
    else if (S.selected >= 0) this.select(-1);
  }

  /* ----------------------------------------------------------- ⌘K ------- */
  _palette() {
    const pal = $('palette'), inp = $('palInput'), list = $('palList');
    const S = this.S;
    const P = this.PAL = [];
    const moment = (t, s, fn) => P.push({ t, s, g: 'MOMENTS', fn });
    moment('The Big Dipper falls apart', '0 → +100,000 YEARS · NIGHT SKY', () => this.moment('dipper'));
    moment('Barnard’s Star comes closest', '≈ +9,700 YEARS · 3.77 LIGHT-YEARS', () => this.moment('barnard'));
    moment('Gliese 710 crosses the Oort cloud', '≈ +1.29 MILLION YEARS · 10,600 AU', () => this.moment('gl710'));
    moment('One galactic year', 'THE SUN’S ORBIT · ABOVE THE DISK', () => this.moment('galyear'));
    moment('Ride with the Sun', '250 MILLION YEARS IN A CO-MOVING FRAME', () => this.moment('ride'));
    moment('Orion, a million years ago', 'NIGHT SKY · −1 MYR', () => this.moment('orion'));
    for (const v of VIEWS) P.push({ t: VIEW_LABEL[v].toLowerCase().replace(/^./, c => c.toUpperCase()), s: 'VIEW · ' + (VIEWS.indexOf(v) + 1), g: 'VIEWS', fn: () => this.setView(v) });
    P.push({ t: '⌂ Today', s: 'BACK TO THE MEASURED SKY', g: 'ACTIONS', fn: () => { this.stopPlay(); this.setT(0); } });
    P.push({ t: '✦ Constellation lines', s: 'TOGGLE (C)', g: 'ACTIONS', fn: () => this.toggleLines() });
    P.push({ t: '📷 Save a photo', s: 'PNG WITH CAPTION (S)', g: 'ACTIONS', fn: () => { this.wantShot = true; } });
    P.push({ t: '✨ Bloom', s: 'TOGGLE THE GLOW (B)', g: 'ACTIONS', fn: () => { S.bloom = !S.bloom; } });
    P.push({ t: '← Back to Cosmos', s: 'THE PLANCK LENGTH TO THE COSMIC HORIZON', g: 'ACTIONS', fn: () => { location.href = '../'; } });
    for (const [i, n, alias] of this.names.stars) P.push({ t: n, s: (alias ? alias.split('|')[0].toUpperCase() : 'STAR'), g: 'STARS', i, fn: () => this.goStar(i) });
    let sel = 0, items = [];
    const render = async () => {
      const q = inp.value.trim().toLowerCase();
      if (!q) items = P.filter(e => e.g !== 'STARS').slice(0, 24);
      else {
        const score = e => { const t = e.t.toLowerCase(); return t === q ? 0 : t.startsWith(q) ? 1 : t.includes(q) ? 2 : e.s.toLowerCase().includes(q) ? 3 : 9; };
        items = P.map(e => [score(e), e]).filter(x => x[0] < 9).sort((a, b) => a[0] - b[0]).slice(0, 18).map(x => x[1]);
        const hip = q.match(/^hip\s*(\d+)$/), gid = q.match(/^(?:gaia\s*(?:dr3)?\s*)?(\d{6,20})$/);
        if (hip) items.unshift({ t: 'HIP ' + hip[1], s: 'HIPPARCOS NUMBER', g: 'LOOKUP', fn: async () => {
          const i = await this.cat.findHip(+hip[1]); if (i >= 0) this.goStar(i); else this.caption('Not in this catalogue', 'HIP ' + hip[1]); } });
        if (gid) items.unshift({ t: 'Gaia DR3 ' + gid[1], s: 'SOURCE_ID · SEARCHES 1M IDS', g: 'LOOKUP', fn: async () => {
          this.caption('Searching…', 'GAIA DR3 ' + gid[1], 6000);
          const i = await this.cat.findSourceId(gid[1]); if (i >= 0) { this.goStar(i); this.caption('Found', 'GAIA DR3 ' + gid[1]); } else this.caption('Not in this catalogue', 'GAIA DR3 ' + gid[1]); } });
      }
      sel = Math.max(0, Math.min(sel, items.length - 1));
      let html = '', lastG = '';
      items.forEach((e, k) => {
        if (e.g !== lastG) { html += `<div class="pgroup">${e.g}</div>`; lastG = e.g; }
        html += `<div class="prow${k === sel ? ' sel' : ''}" data-i="${k}"><span class="pt">${esc(e.t.toUpperCase())}</span><span class="ps">${esc(e.s)}</span></div>`;
      });
      list.innerHTML = html || '<div class="pgroup">NO MATCHES</div>';
      const s = list.querySelector('.prow.sel'); if (s) s.scrollIntoView({ block: 'nearest' });
    };
    const exec = k => { const e = items[k]; if (!e) return; this.closePal(); e.fn(); };
    this.openPal = () => { pal.classList.add('open'); inp.value = ''; sel = 0; render(); inp.focus(); setTimeout(() => inp.focus(), 30); };
    this.closePal = () => { pal.classList.remove('open'); inp.blur(); };
    this.togglePal = () => pal.classList.contains('open') ? this.closePal() : this.openPal();
    inp.addEventListener('input', () => { sel = 0; render(); });
    inp.addEventListener('keydown', e => {
      e.stopPropagation();
      if (e.code === 'ArrowDown') { e.preventDefault(); sel++; render(); }
      else if (e.code === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, sel - 1); render(); }
      else if (e.code === 'Enter') exec(sel);
      else if (e.code === 'Escape') this.closePal();
      else if ((e.metaKey || e.ctrlKey) && e.code === 'KeyK') { e.preventDefault(); this.closePal(); }
    });
    list.addEventListener('pointerdown', e => { const r = e.target.closest('.prow'); if (r) exec(+r.dataset.i); });
    pal.addEventListener('pointerdown', e => { if (e.target === pal) this.closePal(); });
  }

  async goStar(i) {
    if (!(await this.cat.whenLoaded(i))) {
      this.caption('Not in this view', this.S.fallback ? 'THE WebGL2 FALLBACK HOLDS ONLY THE BRIGHTEST STARS' : 'STAR NOT FOUND');
      return;
    }
    this.select(i, true);
    if (this.S.view === 'earth') {
      const x = this.overlays.starPos(i, this.S, this.integ);
      if (x) this.views.lookAt(new THREE.Vector3(x[0], x[1], x[2]).applyMatrix4(this.views.frame));
    }
  }

  findSlug(s) {
    const l = loose(s);
    if (this.slugIdx.has(l)) return Promise.resolve(this.slugIdx.get(l));
    let m = String(s).match(/^gaia-?dr3-?(\d+)$/i);
    if (m) return this.cat.findSourceId(m[1]);
    m = String(s).match(/^hip-?(\d+)$/i);
    if (m) return this.cat.findHip(+m[1]);
    return Promise.resolve(-1);
  }

  /* the guided moments — times come from the same integrator, not from text */
  _later(fn, ms) { clearTimeout(this._momentT); this._momentT = setTimeout(fn, ms); }
  async moment(k) {
    const S = this.S;
    clearTimeout(this._momentT);
    this.stopPlay();
    const star = async name => {
      const i = await this.findSlug(name);
      if (i >= 0 && await this.cat.whenLoaded(i)) { this.select(i, true); return i; }
      return -1;
    };
    if (k === 'dipper') {
      this.setView('earth'); this.select(-1); S.t = 0; S.lines = true; $('constBtn').classList.add('on');
      Object.assign(this.views.st.earth, { yaw: -Math.PI / 2 + 0.05, pitch: 0.95, fov: 58 });
      this.caption('The Big Dipper', 'FIVE OF ITS STARS TRAVEL TOGETHER · DUBHE AND ALKAID DO NOT', 4200);
      this._later(() => this.playToTime(0.1, 2), 1500);
    } else if (k === 'barnard') {
      this.setView('earth'); const i = await star('barnards-star');
      if (i < 0) return;
      const ca = this.closestApproach(i);
      S.t = 0; this.caption('Barnard’s Star', 'THE FASTEST-MOVING STAR IN OUR SKY, HEADING OUR WAY', 3600);
      this._later(() => this.playToTime(ca ? ca.t : 0.0097, this.speedFor(ca ? ca.t : 0.0097)), 1800);
    } else if (k === 'gl710') {
      this.setView('earth'); const i = await star('gliese-710');
      if (i < 0) return;
      const ca = this.closestApproach(i);
      this.caption('Gliese 710', 'A DIM ORANGE DWARF, 62 LIGHT-YEARS AWAY · FOR NOW', 3600);
      S.t = 0; this._later(() => this.playToTime(ca ? ca.t : 1.29, this.speedFor(ca ? ca.t : 1.29)), 1800);
    } else if (k === 'galyear') {
      this.setView('disk'); this.select(-1); S.t = 0;
      Object.assign(this.views.st.disk, { yaw: -Math.PI / 2, pitch: 1.2, dist: 38000 });
      this.caption('One galactic year', 'THE SUN AND A MILLION NEIGHBOURS, ONCE AROUND THE CENTRE', 4200);
      this._later(() => this.playToTime(Math.min(S.tMax, 222), 5), 1500);
    } else if (k === 'ride') {
      this.setView('ride'); this.select(-1); S.t = 0;
      this.caption('Riding with the Sun', 'THE NEIGHBOURHOOD SHEARS APART AS THE GALAXY TURNS', 4200);
      this._later(() => this.playToTime(S.tMax, 5), 1500);
    } else if (k === 'orion') {
      this.setView('earth'); this.select(-1); S.lines = true; $('constBtn').classList.add('on');
      Object.assign(this.views.st.earth, { yaw: Math.atan2(0.08, 0.99) + Math.PI, pitch: 0.0, fov: 64 });
      const betel = await this.findSlug('betelgeuse');
      if (betel >= 0 && await this.cat.whenLoaded(betel)) {      // aim at where Orion is at t = 0
        const c = this.cat.pos;
        this.views.lookAt(new THREE.Vector3(c[betel * 4], c[betel * 4 + 1], c[betel * 4 + 2]).applyMatrix4(this.views.E));
      }
      S.t = 0; this.caption('Orion', 'ITS STARS ARE HUNDREDS OF LIGHT-YEARS APART, SO IT HOLDS ITS SHAPE', 4200);
      this._later(() => this.playToTime(-1, 4), 1500);
    }
  }

  /* --------------------------------------------------------- deep links -- */
  async applyHash(hash, first) {
    const p = new URLSearchParams(String(hash || '').replace(/^#/, ''));
    if (!p.toString()) return;
    const S = this.S;
    clearTimeout(this._momentT);
    if (p.has('view')) this.setView(p.get('view'));
    if (p.has('cam')) this.views.setCam(p.get('cam'), first);
    if (p.has('t')) { const t = parseTime(p.get('t')); if (t !== null) { this.stopPlay(); S.t = Math.max(-S.tMax, Math.min(S.tMax, t)); } }
    if (p.has('target')) {
      const i = await this.findSlug(p.get('target'));
      if (i >= 0 && await this.cat.whenLoaded(i)) {
        const earth = this.S.view === 'earth';
        // in the Earth view a camera in the link wins; above the disk the
        // camera orbits the target, which is what the link was looking at
        this.select(i, !earth || !p.has('cam'));
        if (earth && !p.has('cam')) this.goStar(i);
      } else if (i >= this.cat.N) this.caption('Not in this view', 'THE WebGL2 FALLBACK HOLDS ONLY THE BRIGHTEST STARS');
    }
    if (p.has('moment')) this.moment(p.get('moment'));
    this.lastHash = location.hash;
  }

  _hashTick() {
    const now = performance.now();
    if (now - this.hashT < 1000 || this.dragging) return;
    this.hashT = now;
    const S = this.S;
    let h = '#t=' + linkTime(S.t) + '&view=' + S.view;
    if (S.selected >= 0) {
      const n = this.overlays.nameOf.get(S.selected), ids = this.selIds;
      if (n && this.slugIdx.get(loose(n)) === S.selected) h += '&target=' + slug(n);
      else if (ids && ids.sourceId) h += '&target=gaia-dr3-' + ids.sourceId;
      else if (ids && ids.hip) h += '&target=hip-' + ids.hip;
    }
    h += '&cam=' + this.views.camString();
    if (h !== this.lastHash) { this.lastHash = h; try { history.replaceState(null, '', h); } catch (e) {} }
  }

  /* -------------------------------------------------------------- photo -- */
  afterRender() {
    if (!this.wantShot) return;
    this.wantShot = false;
    try {
      const cv = this.canvas, w = cv.width, h = cv.height;
      const out = document.createElement('canvas'); out.width = w; out.height = h;
      const g = out.getContext('2d');
      g.drawImage(cv, 0, 0);
      const s = Math.max(1, h / 900), S = this.S, d = describe(S.tShown);
      g.fillStyle = 'rgba(232,238,247,.95)';
      g.font = `600 ${13 * s | 0}px Helvetica, Arial, sans-serif`;
      g.fillText('C O S M O S  ·  T I M E', 26 * s, 36 * s);
      g.font = `300 ${21 * s | 0}px Helvetica, Arial, sans-serif`;
      g.fillText(d.main + (d.sub ? '  ·  ' + d.sub : ''), 26 * s, h - 52 * s);
      g.font = `400 ${10.5 * s | 0}px Helvetica, Arial, sans-serif`;
      g.fillStyle = 'rgba(150,170,200,.9)';
      const measured = Math.abs(S.tShown) * 1e6 < 0.5;
      const target = S.selected >= 0 ? ' · ' + (this.overlays.nameOf.get(S.selected) || this.overlays.selLabel || '').toUpperCase() : '';
      g.fillText(VIEW_LABEL[S.view] + target, 26 * s, h - 32 * s);
      g.fillText(measured ? 'MEASURED · GAIA DR3 & HIPPARCOS, EPOCH 2016.0' :
        `SIMULATED FROM MEASURED GAIA DR3 MOTIONS · ${S.fallback ? 'STRAIGHT-LINE MOTION' : 'McMILLAN (2017) POTENTIAL'}`, 26 * s, h - 14 * s);
      out.toBlob(b => {
        if (!b) return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(b);
        a.download = 'cosmos-time-' + linkTime(S.tShown) + '-' + S.view + '.png';
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      });
      this.caption('Photo saved', 'THE TIME IS IN THE CAPTION', 2400);
    } catch (e) { console.warn(e); }
  }
}

function fmtDist(pc) {
  const ly = pc * PC_TO_LY;
  if (ly < 0.5) return `${nf(Math.round(pc * 206264.8))} au (${ly.toFixed(3)} ly)`;
  if (ly < 100) return `${ly.toFixed(2)} ly`;
  return `${nf(Math.round(ly))} ly`;
}
function sig(x) { return x >= 100 ? nf(Math.round(x)) : x >= 1 ? x.toFixed(1) : x.toPrecision(1); }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
