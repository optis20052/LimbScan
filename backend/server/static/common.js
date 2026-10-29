export const $ = s => document.querySelector(s);
export const api = (p, o) => fetch(p, o).then(async r => { if (!r.ok) throw new Error(await r.text()); return r.json(); });
export const fmt = (v, d = 1) => v == null || isNaN(v) ? '–' : (+v).toFixed(d);
export const ui = { pollTimer: null };
