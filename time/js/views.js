/* The three views and their cameras.

   Scene frames (all in parsecs):
     earth — equatorial, y = celestial north (same as Cosmos's radecToVec);
             the Sun at the origin, the camera at the Sun.
     disk  — galactocentric, the Galactic plane horizontal (y = north
             Galactic pole), the Galactic Centre at the origin.
     ride  — like disk, but centred on the Sun at time t and rotated with
             its galactocentric azimuth, so the Centre stays put. */

import * as THREE from 'three/webgpu';

export const VIEWS = ['earth', 'disk', 'ride'];
export const VIEW_LABEL = { earth: 'THE NIGHT SKY FROM EARTH', disk: 'ABOVE THE GALACTIC DISK', ride: 'RIDING WITH THE SUN' };
const DEG = Math.PI / 180;

export class Views {
  constructor(camera, canvas, manifest, sun) {
    this.camera = camera; this.canvas = canvas; this.sun = sun;
    const R = manifest.frame.icrs_to_gc;                      // gc = R · icrs
    // gc → earth scene: icrs = Rᵀ gc, then (x, z, −y)
    const rt = new THREE.Matrix4().set(
      R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, 0, 0, 0, 1);
    this.EQ = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);  // icrs → scene
    this.E = this.EQ.clone().multiply(rt);
    this.G = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);   // gc → disk scene
    this.phi0 = Math.atan2(manifest.frame.sun_pos_pc[1], manifest.frame.sun_pos_pc[0]);
    this.frame = new THREE.Matrix4();        // Sun-relative gc → scene (stars)
    this.gcFrame = new THREE.Matrix4();      // absolute gc → scene (trail, backdrop)
    this.st = {                              // per-view camera state (targets)
      earth: { yaw: -Math.PI / 2, pitch: 0.98, fov: 72 },
      disk: { yaw: -Math.PI / 2, pitch: 1.05, dist: 36000 },
      ride: { yaw: -Math.PI / 2 - 0.35, pitch: 0.42, dist: 1400 },
    };
    this.cur = JSON.parse(JSON.stringify(this.st));        // smoothed
    this.view = 'earth';
    this.follow = -1;                       // star index the camera tracks
    this.followPos = null;                  // its scene position, set by overlays
    this.target = new THREE.Vector3();
    this.mLim = 6.5; this.gain = 1; this.sizeBase = 2.2;
    this.depth = 6.5;                       // Earth-view limiting magnitude (+/− keys)
    this.onPick = null; this.onInput = null;
    this._input();
  }

  set(view) {
    if (!VIEW_LABEL[view]) return;
    this.view = view;
  }

  /* point the Earth-view camera at a scene direction / follow a star */
  lookAt(dir) {
    const l = dir.length();
    if (!(l > 0) || !isFinite(l)) return;          // never let NaN into the camera
    const s = this.st.earth;
    s.yaw = Math.atan2(dir.x, dir.z);
    s.pitch = Math.asin(Math.max(-1, Math.min(1, dir.y / dir.length())));
    this._unwrapYaw('earth');
  }
  _unwrapYaw(v) {
    const c = this.cur[v], s = this.st[v];
    while (s.yaw - c.yaw > Math.PI) s.yaw -= 2 * Math.PI;
    while (s.yaw - c.yaw < -Math.PI) s.yaw += 2 * Math.PI;
  }

  update(dt, S, sunPos, starsFrame) {
    const v = this.view, s = this.st[v], c = this.cur[v];
    const k = Math.min(1, dt * 6);
    // a followed star in the Earth view: keep it centred (until the user drags)
    if (this.follow >= 0 && this.followPos && v === 'earth') {
      this.lookAt(this.followPos);
    }
    c.yaw += (s.yaw - c.yaw) * k; c.pitch += (s.pitch - c.pitch) * k;
    if (s.fov) c.fov += (s.fov - c.fov) * Math.min(1, dt * 5);
    if (s.dist) c.dist *= Math.pow(s.dist / c.dist, Math.min(1, dt * 4));

    const cam = this.camera;
    const dir = new THREE.Vector3(Math.cos(c.pitch) * Math.sin(c.yaw), Math.sin(c.pitch), Math.cos(c.pitch) * Math.cos(c.yaw));
    if (v === 'earth') {
      this.frame.copy(this.E);
      this.gcFrame.copy(this.E).multiply(new THREE.Matrix4().makeTranslation(-sunPos[0], -sunPos[1], -sunPos[2]));
      cam.position.set(0, 0, 0);
      cam.up.set(0, 1, 0);
      cam.lookAt(dir);
      cam.fov = c.fov; cam.near = 1e-4; cam.far = 1e7;
      this.mLim = this.depth + 2.5 * Math.log10(72 / c.fov);      // zooming in reaches fainter, like optics
      this.gain = 1; this.sizeBase = 2.4;
    } else {
      if (v === 'disk') {
        this.gcFrame.copy(this.G);
        this.frame.copy(this.G).multiply(new THREE.Matrix4().makeTranslation(sunPos[0], sunPos[1], sunPos[2]));
      } else {
        const phi = Math.atan2(sunPos[1], sunPos[0]);
        const rot = new THREE.Matrix4().makeRotationZ(-(phi - this.phi0));
        this.frame.copy(this.G).multiply(rot);
        this.gcFrame.copy(this.frame).multiply(new THREE.Matrix4().makeTranslation(-sunPos[0], -sunPos[1], -sunPos[2]));
      }
      // orbit target: a followed star, else the Galactic Centre (disk) / the Sun (ride)
      if (this.follow >= 0 && this.followPos && this.snapTarget) { this.target.copy(this.followPos); this.snapTarget = false; }
      else if (this.follow >= 0 && this.followPos) this.target.lerp(this.followPos, Math.min(1, dt * 5));
      else this.target.lerp(new THREE.Vector3(0, 0, 0), Math.min(1, dt * 3));
      cam.position.copy(this.target).addScaledVector(dir, c.dist);
      cam.up.set(0, 1, 0);
      cam.lookAt(this.target);
      cam.fov = 50; cam.near = Math.max(0.01, c.dist * 1e-4); cam.far = c.dist * 50 + 2e5;
      // brightness: everything is ~dist away, so set the limit relative to it
      const dm = 5 * Math.log10(c.dist / 10);
      this.mLim = dm + (v === 'disk' ? -0.5 : 1.5);
      this.gain = v === 'disk' ? 0.6 : 0.8; this.sizeBase = v === 'disk' ? 1.6 : 1.9;
    }
    cam.aspect = innerWidth / innerHeight;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
    starsFrame.matrix.copy(this.frame);
    starsFrame.matrixWorldNeedsUpdate = true;
  }

  /* ---------------------------------------------------------------- input */
  _input() {
    const cv = this.canvas, ptrs = new Map();
    let downX = 0, downY = 0, moved = 99, pinch0 = 0;
    const touch = () => { this.follow = this.view === 'earth' ? -1 : this.follow; this.onInput && this.onInput(); };
    cv.addEventListener('pointerdown', e => {
      cv.setPointerCapture(e.pointerId);
      ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      downX = e.clientX; downY = e.clientY; moved = 0;
      if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch0 = Math.hypot(a.x - b.x, a.y - b.y); }
      cv.classList.add('dragging');
    });
    cv.addEventListener('pointermove', e => {
      const p = ptrs.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);
      const s = this.st[this.view];
      if (ptrs.size === 2) {
        const [a, b] = [...ptrs.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch0 > 0) this.zoom(pinch0 / d);
        pinch0 = d;
        return;
      }
      if (moved < 4) return;
      touch();
      const sens = this.view === 'earth' ? (this.cur.earth.fov / 72) * 0.0045 : 0.006;
      s.yaw += (this.view === 'earth' ? dx : -dx) * sens;
      s.pitch = Math.max(-1.5, Math.min(1.5, s.pitch + (this.view === 'earth' ? dy : dy) * sens));
    });
    const up = e => {
      if (!ptrs.has(e.pointerId)) return;            // the press began elsewhere (e.g. the ⌘K list)
      ptrs.delete(e.pointerId);
      cv.classList.remove('dragging');
      if (moved < 5 && e.type === 'pointerup' && this.onPick) this.onPick(e.clientX, e.clientY);
      moved = 99;
    };
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', up);
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoom(Math.exp(e.deltaY * 0.0012));
      this.onInput && this.onInput();
    }, { passive: false });
  }

  zoom(f) {
    const s = this.st[this.view];
    if (this.view === 'earth') s.fov = Math.max(3, Math.min(115, s.fov * f));
    else if (this.view === 'disk') s.dist = Math.max(300, Math.min(90000, s.dist * f));
    else s.dist = Math.max(15, Math.min(30000, s.dist * f));
  }

  /* camera state for deep links: yaw,pitch,zoom */
  camString() {
    const s = this.st[this.view];
    return [s.yaw.toFixed(3), s.pitch.toFixed(3), (s.fov || s.dist).toFixed(s.fov ? 1 : 0)].join(',');
  }
  setCam(str, snap) {
    const p = String(str).split(',').map(Number);
    const s = this.st[this.view];
    if (isFinite(p[0])) s.yaw = p[0];
    if (isFinite(p[1])) s.pitch = Math.max(-1.5, Math.min(1.5, p[1]));
    if (isFinite(p[2])) { if (s.fov) s.fov = p[2]; else s.dist = p[2]; this.zoom(1); }
    if (snap) Object.assign(this.cur[this.view], s);
  }
  snap() { Object.assign(this.cur[this.view], this.st[this.view]); }
}
