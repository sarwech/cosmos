/* Time: parsing, formatting and the non-linear scrubber scale.
   Internal unit is Myr; t = 0 is the Gaia DR3 epoch, J2016.0. */

export const EPOCH_YEAR = 2016;
export const T_MAX = 250;                         // Myr each way
export const FINE = 1;                            // inner half of the track: ±1 Myr
const K = 6;                                      // asinh stretch of the inner half

/* scrubber position s ∈ [-1, 1] ↔ t (Myr). Inner half: asinh-stretched ±1 Myr
   (≈100-year resolution at the centre); outer quarters: log 1 → 250 Myr. */
export function sToT(s) {
  const a = Math.abs(s), sg = Math.sign(s);
  if (a <= 0.5) return sg * FINE * Math.sinh(K * a / 0.5) / Math.sinh(K);
  return sg * FINE * Math.pow(T_MAX / FINE, (a - 0.5) / 0.5);
}
export function tToS(t) {
  const a = Math.abs(t), sg = Math.sign(t);
  if (a <= FINE) return sg * 0.5 * Math.asinh(a / FINE * Math.sinh(K)) / K;
  return sg * (0.5 + 0.5 * Math.log(a / FINE) / Math.log(T_MAX / FINE));
}

/* "#t=-120000y", "t=1.3Myr", "250 My", "-9.8kyr" → Myr (null if unparseable) */
export function parseTime(str) {
  const m = String(str).trim().replace(/,/g, '').match(/^([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s*(y|yr|yrs|years?|ky|kyr|kyrs|my|myr|myrs|gy|gyr)?$/i);
  if (!m) return null;
  const v = parseFloat(m[1]);
  const u = (m[2] || 'y').toLowerCase();
  const myr = u.startsWith('g') ? v * 1e3 : u.startsWith('m') ? v : u.startsWith('k') ? v * 1e-3 : v * 1e-6;
  return Math.max(-T_MAX, Math.min(T_MAX, myr));
}

/* canonical deep-link form: whole years below 1 Myr, Myr above */
export function linkTime(t) {
  if (Math.abs(t) < 1) return Math.round(t * 1e6) + 'y';
  return (Math.round(t * 1000) / 1000) + 'Myr';
}

const nf = new Intl.NumberFormat('en-US');
/* readout: "112,400 years from now" / "1.29 million years ago" / "today" */
export function describe(t) {
  const yrs = t * 1e6, a = Math.abs(yrs);
  if (a < 0.5) return { main: 'Today', sub: 'GAIA EPOCH ' + EPOCH_YEAR + '.0' };
  const dir = yrs > 0 ? 'from now' : 'ago';
  let main;
  if (a < 1e6) {
    const r = a < 1e3 ? Math.round(a) : a < 1e4 ? Math.round(a / 10) * 10 : Math.round(a / 100) * 100;
    main = nf.format(r) + (r === 1 ? ' year ' : ' years ') + dir;
  } else {
    const my = a / 1e6;
    main = (my < 10 ? my.toFixed(2) : my < 100 ? my.toFixed(1) : my.toFixed(0)) + ' million years ' + dir;
  }
  let sub = '';
  if (a < 2e5) {
    const year = Math.round(EPOCH_YEAR + yrs);
    sub = year > 0 ? 'YEAR ' + nf.format(year) + ' CE' : 'YEAR ' + nf.format(1 - year) + ' BCE';
  }
  return { main, sub };
}

export function shortTime(t) {
  const a = Math.abs(t * 1e6);
  if (a < 0.5) return 'today';
  const s = t > 0 ? '+' : '−';
  if (a < 1e6) return s + nf.format(Math.round(a)) + ' yr';
  return s + (a / 1e6).toFixed(a < 1e7 ? 2 : 1) + ' Myr';
}
