/* Cosmos's bloom (index.html, "BLOOM"), ported line for line to three/webgpu:
   scene → bright-pass (threshold 0.30, +0.5 soft knee) → three widening
   separable 9-tap Gaussian passes at half resolution → additive composite at
   0.85. Same render targets, same taps, same weights. */

import * as THREE from 'three/webgpu';
import { Fn, texture, uv, vec2, vec4, max, smoothstep, uniform } from 'three/tsl';

export class Bloom {
  constructor(renderer) {
    this.r = renderer;
    const opts = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: true };
    this.rtScene = new THREE.RenderTarget(2, 2, opts);
    this.rtA = new THREE.RenderTarget(2, 2, { ...opts, depthBuffer: false });
    this.rtB = new THREE.RenderTarget(2, 2, { ...opts, depthBuffer: false });
    this.res = uniform(new THREE.Vector2(1, 1));
    this.strength = uniform(0.85);
    const mat = node => { const m = new THREE.NodeMaterial(); m.fragmentNode = node; m.depthTest = false; m.depthWrite = false; return m; };
    this.bright = mat(Fn(() => {
      const c = texture(this.rtScene.texture, uv()).rgb;
      const l = max(c.r, max(c.g, c.b));
      return vec4(c.mul(smoothstep(0.30, 0.80, l)), 1);
    })());
    const blur = (src, dx, dy) => mat(Fn(() => {
      const px = vec2(dx, dy).div(this.res), t = q => texture(src, q).rgb, u = uv();
      const s = t(u).mul(0.227)
        .add(t(u.add(px.mul(1.385))).add(t(u.sub(px.mul(1.385)))).mul(0.316))
        .add(t(u.add(px.mul(3.231))).add(t(u.sub(px.mul(3.231)))).mul(0.070));
      return vec4(s, 1);
    })());
    this.blurs = [];
    for (let i = 0; i < 3; i++) {                          // widening ping-pong, as in Cosmos
      this.blurs.push([blur(this.rtA.texture, 1 + i * 0.9, 0), this.rtB]);
      this.blurs.push([blur(this.rtB.texture, 0, 1 + i * 0.9), this.rtA]);
    }
    this.comp = mat(Fn(() => vec4(texture(this.rtScene.texture, uv()).rgb
      .add(texture(this.rtA.texture, uv()).rgb.mul(this.strength)), 1))());
    this.plain = mat(Fn(() => vec4(texture(this.rtScene.texture, uv()).rgb, 1))());
    this.quad = new THREE.QuadMesh(this.bright);
  }

  setSize(w, h) {
    this.rtScene.setSize(w, h);
    const bw = Math.max(1, w >> 1), bh = Math.max(1, h >> 1);
    this.rtA.setSize(bw, bh); this.rtB.setSize(bw, bh);
    this.res.value.set(bw, bh);
  }

  /* target: null = the canvas */
  render(scene, camera, on, target = null) {
    const r = this.r, q = this.quad;
    r.setRenderTarget(this.rtScene); r.render(scene, camera);
    if (on) {
      q.material = this.bright; r.setRenderTarget(this.rtA); q.render(r);
      for (const [m, rt] of this.blurs) { q.material = m; r.setRenderTarget(rt); q.render(r); }
      q.material = this.comp;
    } else q.material = this.plain;
    r.setRenderTarget(target); q.render(r);
  }
}
