export const $ = s => document.querySelector(s);
export const api = (p, o) => fetch(p, o).then(async r => { if (!r.ok) throw new Error(await r.text()); return r.json(); });
export const fmt = (v, d = 1) => v == null || isNaN(v) ? '–' : (+v).toFixed(d);
export const ui = { pollTimer: null };
export const esc = t => String(t ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
export async function renameScan(id, old) {
  const n = prompt('Scan name', old || '');
  if (n == null || !n.trim() || n.trim() === old) return null;
  const fd = new FormData(); fd.append('name', n);
  return (await api(`/api/scans/${id}/name`, { method: 'POST', body: fd })).name;
}
