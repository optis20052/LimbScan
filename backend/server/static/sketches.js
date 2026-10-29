// Inline SVG illustrations for the guided setup and capture. Colours follow the CSS theme variables.
const W = (body, vb = '0 0 320 200') => `<svg class="sk" viewBox="${vb}" xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;

const sheet = (x, y, w, h, rot = 0) => {
  let m = '';
  const cw = w / 3, ch = h / 4;
  for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++)
    m += `<rect x="${x + cw * c + cw * .22}" y="${y + ch * r + ch * .2}" width="${cw * .56}" height="${cw * .56}" fill="var(--fg)" opacity=".85"/>`;
  return `<g transform="rotate(${rot} ${x + w / 2} ${y + h / 2})"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="2" fill="#f4f1ea" stroke="var(--mut)"/>${m.replaceAll('var(--fg)', '#111')}</g>`;
};

const phone = (x, y, a = 0, s = 1, hl = false) => `<g transform="translate(${x} ${y}) rotate(${a}) scale(${s})">
  <rect x="-9" y="-16" width="18" height="32" rx="4" fill="${hl ? 'var(--acc)' : '#21262d'}" stroke="${hl ? '#9cc3ff' : 'var(--mut)'}" stroke-width="1.5"/>
  <circle cx="0" cy="-10" r="2.2" fill="#0d1117"/></g>`;

// forearm + hand, side view, pointing right
const armSide = (x, y, s = 1) => `<g transform="translate(${x} ${y}) scale(${s})" stroke="#c98d6e" stroke-width="2" fill="#e8b89a">
  <path d="M0 -12 C40 -16 90 -14 130 -10 C142 -9 150 -12 162 -10 C176 -8 190 -9 200 -6 C206 -4 206 2 200 3 C190 5 176 6 162 7 C150 9 142 8 130 9 C90 12 40 14 0 12 Z"/>
  <path d="M168 -8 C176 -14 186 -15 190 -12" fill="none"/></g>`;

const card = (x, y, w, rot = 0) => { const h = w / 1.586; return `<g transform="rotate(${rot} ${x + w / 2} ${y + h / 2})"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${w * .04}" fill="#1f3f94" stroke="#9cc3ff" stroke-width="1.5"/>
  <rect x="${x}" y="${y + h * .55}" width="${w}" height="${h * .12}" fill="#dbe4f5"/><rect x="${x + w * .1}" y="${y + h * .22}" width="${w * .14}" height="${h * .2}" rx="2" fill="#d8b54a"/></g>`; };

// upright forearm (elbow on table, fingers up), side view, with the camera ring at a given height
const upright = (ring = null) => {
  const ry = { hand: 62, forearm: 118, top: 18 }[ring];
  const ringSvg = ring ? `<ellipse cx="160" cy="${ry}" rx="${ring === 'top' ? 46 : 74}" ry="${ring === 'top' ? 12 : 18}" stroke="var(--acc)" stroke-width="5" class="flow"/>
    ${ring === 'top' ? `${phone(160, -6, 180, 1.05, true)}` : `${phone(86, ry, 90, 1.05, true)}${phone(234, ry, -90, 1.05, true)}`}` : '';
  return `<line x1="10" y1="178" x2="310" y2="178" stroke="#5b4636" stroke-width="6"/>
    <ellipse cx="160" cy="170" rx="34" ry="7" fill="#6e7fa8" opacity=".8"/>
    <path d="M148 170 C146 140 144 110 146 84 L174 84 C176 110 176 140 172 170 Z" fill="#e8b89a" stroke="#c98d6e" stroke-width="2"/>
    <path d="M144 86 C140 70 136 58 140 46 L144 20 C145 14 151 14 151 20 L152 40 L154 12 C155 6 161 6 161 12 L162 40 L165 14 C166 8 172 8 172 14 L172 44 L177 26 C179 21 185 22 184 28 L180 58 C179 72 176 80 174 86 Z" fill="#e8b89a" stroke="#c98d6e" stroke-width="2"/>
    <path d="M144 70 C132 62 124 56 120 50 C117 46 121 41 126 44 L142 56" fill="#e8b89a" stroke="#c98d6e" stroke-width="2"/>
    ${ringSvg}`;
};

export const S = {
  calib: () => W(`
    <path d="M10 120 C70 110 150 108 220 112 C240 113 262 106 290 110 C304 112 304 124 290 126 C262 130 240 126 220 128 C150 132 70 134 10 132 Z" fill="#e8b89a" stroke="#c98d6e" stroke-width="2"/>
    ${card(110, 86, 70, -4)}
    ${phone(70, 40, -20, 1.2, true)}${phone(240, 40, 20, 1.2, true)}
    <path d="M92 34 C140 14 180 14 222 34" stroke="var(--acc)" stroke-width="4" class="flow"/><path d="M214 26 l9 8 l-11 4" stroke="var(--acc)" stroke-width="3"/>
    <text x="160" y="160" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">card held flat on the arm · phone moves left ↔ right</text>`, '0 0 320 170'),
  poseUpright: () => W(`${upright()}<text x="160" y="196" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">elbow on the table, fingers up</text>`, '0 0 320 200'),
  poseResting: () => W(`
    <line x1="5" y1="150" x2="315" y2="150" stroke="#5b4636" stroke-width="6"/>
    <ellipse cx="90" cy="134" rx="62" ry="13" fill="#6e7fa8" stroke="#9aa9cc" stroke-width="2"/>
    ${armSide(30, 115, 1.2)}
    <text x="160" y="176" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">arm resting on a pillow / the table</text>`, '0 0 320 185'),
  uprightPass: k => W(upright(k), '0 -20 320 205'),

  card: () => W(`
    <rect x="10" y="12" width="300" height="150" rx="10" fill="#3a2c22" stroke="#5b4636"/>
    <path d="M0 70 C70 60 150 60 220 64 C240 65 262 58 290 62 C304 64 304 76 290 78 C262 82 240 78 220 80 C150 84 70 86 0 84 Z" fill="#e8b89a" stroke="#c98d6e" stroke-width="2"/>
    ${card(150, 104, 62, -8)}
    <g stroke="var(--ok)" stroke-width="1.5"><path d="M222 120 q 24 -6 30 -30" /><path d="M247 92 l5 -3 l1 6"/></g>
    <text x="258" y="86" fill="var(--ok)" font-size="11" stroke="none" font-weight="700">flat, fully</text><text x="258" y="99" fill="var(--ok)" font-size="11" stroke="none" font-weight="700">visible</text>
    <g transform="translate(40 180)">${card(0, 0, 96)}
      <line x1="0" y1="70" x2="96" y2="70" stroke="var(--mut)"/><text x="48" y="84" font-size="11" fill="var(--mut)" text-anchor="middle" stroke="none">85.6 mm</text>
      <line x1="104" y1="0" x2="104" y2="60.5" stroke="var(--mut)"/><text x="110" y="34" font-size="11" fill="var(--mut)" stroke="none">54 mm</text>
      <text x="150" y="16" font-size="12" fill="var(--fg)" stroke="none" font-weight="700">Any standard card:</text>
      <text x="150" y="34" font-size="11" fill="var(--mut)" stroke="none">bank · ID · health card</text>
      <text x="150" y="50" font-size="11" fill="var(--mut)" stroke="none">Carte Vitale · licence</text>
      <text x="150" y="66" font-size="11" fill="var(--bad)" stroke="none">✗ not a business card</text></g>`, '0 0 320 275'),

  support: () => W(`
    <line x1="5" y1="170" x2="315" y2="170" stroke="#5b4636" stroke-width="6"/>
    <ellipse cx="80" cy="152" rx="62" ry="16" fill="#6e7fa8" stroke="#9aa9cc" stroke-width="2"/>
    <path d="M-10 80 C10 100 30 118 40 132" stroke="#c98d6e" stroke-width="22" opacity=".55"/>
    ${armSide(40, 130, 1.25)}
    <text x="80" y="192" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">forearm rests on a pillow / towel</text>
    <text x="230" y="105" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">hand free if possible</text>`, '0 0 320 200'),

  print: () => W(`
    <rect x="40" y="60" width="110" height="60" rx="8" fill="#21262d" stroke="var(--mut)" stroke-width="2"/>
    <rect x="55" y="45" width="80" height="20" rx="3" fill="#161b22" stroke="var(--mut)"/>
    ${sheet(62, 96, 66, 88)}
    <text x="95" y="80" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">100%</text>
    <g transform="translate(170 70)">
      ${sheet(0, 0, 100, 120)}
      <rect x="0" y="128" width="100" height="14" fill="#ffd86b" stroke="#a5852f"/>
      ${[...Array(11)].map((_, i) => `<line x1="${i * 10}" y1="128" x2="${i * 10}" y2="${i % 5 ? 133 : 137}" stroke="#6b5617"/>`).join('')}
      <text x="50" y="160" fill="var(--ok)" font-size="13" text-anchor="middle" stroke="none" font-weight="700">100 mm ✓</text>
    </g>`, '0 0 320 240'),

  layout: () => W(`
    <rect x="10" y="20" width="300" height="170" rx="10" fill="#3a2c22" stroke="#5b4636"/>
    ${sheet(30, 45, 80, 112)}${sheet(120, 45, 80, 112)}${sheet(210, 45, 80, 112)}
    ${['30,45', '110,45', '30,157', '110,157', '120,45', '200,45', '120,157', '200,157', '210,45', '290,45', '210,157', '290,157']
      .map(p => { const [a, b] = p.split(','); return `<rect x="${+a - 5}" y="${+b - 3}" width="10" height="6" fill="#d9c27a" opacity=".85" transform="rotate(30 ${a} ${b})"/>`; }).join('')}
    <path d="M0 108 C70 96 150 96 220 100 C240 101 262 94 290 98 C304 100 304 112 290 114 C262 118 240 114 220 116 C150 120 70 122 0 120 Z" fill="#e8b89a" stroke="#c98d6e" stroke-width="2" opacity=".55"/>
    <text x="160" y="205" fill="var(--mut)" font-size="12" text-anchor="middle" stroke="none">top view · 3 sheets flat, taped, arm floating above</text>`, '0 0 320 215'),

  pose: () => W(`
    <line x1="5" y1="170" x2="315" y2="170" stroke="#5b4636" stroke-width="6"/>
    <rect x="70" y="163" width="230" height="4" fill="#f4f1ea"/>
    <g fill="#2f81f7" opacity=".9"><rect x="10" y="130" width="60" height="12" rx="2"/><rect x="14" y="142" width="54" height="13" rx="2" fill="#d29922"/><rect x="8" y="155" width="62" height="13" rx="2" fill="#3fb950"/></g>
    <path d="M-10 70 C10 90 30 110 40 122" stroke="#c98d6e" stroke-width="22" opacity=".55"/>
    ${armSide(40, 120, 1.25)}
    <g stroke="var(--warn)" stroke-width="1.5"><line x1="190" y1="133" x2="190" y2="161"/><path d="M186 137 l4 -5 l4 5 M186 157 l4 5 l4 -5"/></g>
    <text x="198" y="152" fill="var(--warn)" font-size="12" stroke="none" font-weight="700">10–25 cm</text>
    <text x="40" y="190" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">elbow on support</text>
    <text x="240" y="100" fill="var(--mut)" font-size="11" text-anchor="middle" stroke="none">hand relaxed, still</text>`, '0 0 320 200'),

  light: () => W(`
    <g stroke="var(--warn)" stroke-width="2"><path d="M60 30 h60 l-12 30 h-36 z" fill="#2b2410"/><line x1="90" y1="60" x2="90" y2="70"/></g>
    ${[...Array(7)].map((_, i) => `<line x1="${70 + i * 7}" y1="72" x2="${40 + i * 20}" y2="150" stroke="#ffd86b" stroke-width="1" opacity=".5"/>`).join('')}
    ${armSide(30, 150, .9)}
    <text x="100" y="185" fill="var(--ok)" font-size="12" text-anchor="middle" stroke="none" font-weight="700">✓ bright &amp; even</text>
    <g transform="translate(205 40)">
      <circle cx="40" cy="20" r="14" fill="#ffd86b"/><line x1="40" y1="36" x2="70" y2="105" stroke="#ffd86b"/>
      <path d="M0 115 h90" stroke="#000" stroke-width="10" opacity=".55"/>
      <rect x="10" y="60" width="22" height="34" rx="4" fill="#21262d" stroke="var(--mut)"/><text x="21" y="82" font-size="14" text-anchor="middle" fill="#fff" stroke="none">⚡</text>
      <line x1="0" y1="45" x2="95" y2="140" stroke="var(--bad)" stroke-width="4"/><line x1="95" y1="45" x2="0" y2="140" stroke="var(--bad)" stroke-width="4"/>
      <text x="48" y="160" fill="var(--bad)" font-size="12" text-anchor="middle" stroke="none" font-weight="700">✗ flash · hard shadow</text>
    </g>`, '0 0 320 210'),

  // phase: 'level' | 'high' | 'low' | 'hand' — end-on view of the arm with the camera path
  orbit: phase => {
    const cx = 160, cy = 105, R = 72;
    const ring = { level: [-25, 25], high: [-80, -30], low: [30, 70] };
    const arc = (a0, a1, side) => {
      const p = a => [cx + side * R * Math.cos(a * Math.PI / 180), cy + R * Math.sin(a * Math.PI / 180)];
      const [x0, y0] = p(a0), [x1, y1] = p(a1);
      return `M${x0} ${y0} A${R} ${R} 0 0 ${side > 0 ? 1 : 0} ${x1} ${y1}`;
    };
    let paths = '';
    for (const [k, [a0, a1]] of Object.entries(ring)) {
      const on = k === phase;
      for (const side of [1, -1]) paths += `<path d="${arc(a0, a1, side)}" stroke="${on ? 'var(--acc)' : 'var(--line)'}" stroke-width="${on ? 7 : 4}" ${on ? 'class="flow"' : ''}/>`;
    }
    const ph = { level: [[cx + R, cy, -90], [cx - R, cy, 90]], high: [[cx + R * .6, cy - R * .8, -140], [cx - R * .6, cy - R * .8, 140]], low: [[cx + R * .75, cy + R * .65, -40], [cx - R * .75, cy + R * .65, 40]] }[phase] || [];
    const hand = phase === 'hand' ? `<g transform="translate(160 105)"><path d="M-40 -14 C-10 -18 20 -16 40 -20 l26 -4 c6 0 6 6 0 7 l-22 6 l26 0 c6 0 6 7 0 7 l-26 2 l22 6 c5 2 3 8 -2 7 l-24 -5 c-14 10 -30 12 -40 12 C-30 14 -40 12 -40 12 Z" fill="#e8b89a" stroke="#c98d6e" stroke-width="2"/>
      ${phone(60, -55, -150, 1.2, true)}${phone(70, 50, -30, 1.2, true)}<path d="M40 -60 C80 -30 90 20 60 60" stroke="var(--acc)" stroke-width="4" class="flow"/></g>` : '';
    return W(`
      <line x1="10" y1="195" x2="310" y2="195" stroke="#5b4636" stroke-width="5"/><rect x="60" y="190" width="200" height="3" fill="#f4f1ea"/>
      ${phase === 'hand' ? hand : `${paths}
      <circle cx="${cx}" cy="${cy}" r="24" fill="#e8b89a" stroke="#c98d6e" stroke-width="2.5"/>
      <text x="${cx}" y="${cy + 4}" font-size="10" fill="#7a4b33" text-anchor="middle" stroke="none">arm</text>
      ${ph.map(([x, y, a]) => phone(x, y, a, 1.1, true)).join('')}
      <path d="M${cx} ${cy - R - 16} a14 6 0 1 1 -1 0" stroke="var(--mut)" stroke-width="1.5"/>
      <text x="${cx + 22}" y="${cy - R - 12}" font-size="10" fill="var(--mut)" stroke="none">walk all the way round</text>`}`, '0 0 320 205');
  },
};
