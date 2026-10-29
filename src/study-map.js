import './study-map.css';

export { paths as studyPaths };

const escape = (value = '') => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));

const paths = [
  {
    id: 'attention',
    name: 'Attention 基础',
    short: '表示与结构',
    question: '上下文如何进入表示？',
    summary: '从 Q、K、V 的相关性计算出发，理解多头注意力与位置编码如何组合成完整的 Transformer 架构。',
    diagramLabel: 'Attention：输入序列通过注意力计算，得到包含上下文的表示。',
    diagramNote: 'Self-Attention → Multi-Head → Transformer',
    nodes: ['输入序列', '注意力加权', '上下文表示'],
    concepts: ['Q / K / V', 'Multi-Head', 'Positional Encoding', 'Encoder / Decoder'],
    notes: ['self-attention-1', 'self-attention-2', 'transformer-1', 'transformer-2', 'attention-is-all-you-need'],
  },
  {
    id: 'architecture',
    name: '生成架构',
    short: '空间与主干',
    question: '生成过程在哪里计算？',
    summary: '从像素空间的逐步加噪与去噪（DDPM），演进到潜空间建模（LDM）与可扩展的 Transformer 主干（DiT）。',
    diagramLabel: '潜空间扩散：噪声潜变量经过条件去噪和解码，得到图像。',
    diagramNote: 'DDPM 像素扩散 → LDM 潜空间 → DiT 主干',
    nodes: ['噪声潜变量', '条件去噪', '解码图像'],
    concepts: ['DDPM', 'Latent Space', 'Cross-Attention', 'DiT Patchify'],
    notes: ['ddpm', 'latent-diffusion', 'dit', 'stable-diffusion-lab'],
  },
  {
    id: 'flow',
    name: 'Flow 方法',
    short: '路径与速度',
    question: '如何走直线连接噪声与图像？',
    summary: '通过常微分方程（ODE）学习分布之间的直线运输速度场，并串联起 MM-DiT 架构与免反演编辑。',
    diagramLabel: 'Rectified Flow：从噪声分布出发，沿学习的速度场到达数据分布。',
    diagramNote: 'Rectified Flow：直线插值与速度场匹配',
    nodes: ['噪声分布 z₀', '速度场 v_θ', '数据分布 z₁'],
    concepts: ['Rectified Flow', 'ODE 轨迹', 'MM-DiT / SD3', 'Flow Editing'],
    notes: ['rectified-flow', 'mmdit', 'flowedit', 'stable-flow'],
  },
  {
    id: 'editing',
    name: '编辑与控制',
    short: '条件与保留',
    question: '编辑时改变什么，保留什么？',
    summary: '系统对比加噪重绘强度、潜空间掩码融合、交叉注意力替换与旁路结构控制（ControlNet）的约束机制。',
    diagramLabel: '可控图像编辑：源图结合文本、掩码或结构条件，得到编辑后的图像。',
    diagramNote: '在不同表征层级施加结构与语义约束',
    nodes: ['源图与先验', '条件与约束', '可控编辑结果'],
    concepts: ['SDEdit 加噪点', 'Latent Mask', 'Cross-Attention 控制', 'ControlNet 零卷积'],
    notes: ['sdedit', 'blended-latent-diffusion', 'prompt-to-prompt', 'controlnet'],
  },
];

const notePathLookup = new Map();
for (const path of paths) {
  for (const id of path.notes) notePathLookup.set(id, path.name);
}
notePathLookup.set('generative-models-map', '知识地图');

export function getNoteCategory(note) {
  if (!note) return '研究笔记';
  return notePathLookup.get(note.id) || note.category || '研究笔记';
}

const arrow = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 12h16m-6-6 6 6-6 6"/></svg>';

function readProgress(noteId) {
  try {
    const val = Number(localStorage.getItem(`mufc-reader-${noteId}`));
    return Number.isFinite(val) ? Math.max(0, Math.min(100, val)) : 0;
  } catch {
    return 0;
  }
}

function diagram(path) {
  const prefix = `study-diagram-${path.id}`;
  const base = `<path d="M121 96H154M267 96h33" class="study-diagram-connector"/><path d="m149 92 5 4-5 4m146-8 5 4-5 4" class="study-diagram-arrow"/>`;
  let symbols;
  if (path.id === 'attention') {
    symbols = `<g class="study-diagram-symbol"><rect x="48" y="53" width="10" height="10" rx="2"/><rect x="64" y="53" width="10" height="10" rx="2"/><rect x="80" y="53" width="10" height="10" rx="2"/><path d="M183 50h54m-54 8h54m-54 8h54"/><path d="M54 69c20 14 30-8 45 2M67 69c10 10 15 4 22 2" class="study-diagram-soft"/><rect x="333" y="51" width="10" height="16" rx="2"/><rect x="348" y="47" width="10" height="20" rx="2"/><rect x="363" y="53" width="10" height="14" rx="2"/></g>`;
  } else if (path.id === 'architecture') {
    symbols = `<g class="study-diagram-symbol">${Array.from({ length: 16 }, (_, index) => `<circle cx="${50 + (index % 4) * 12}" cy="${43 + Math.floor(index / 4) * 9}" r="${index % 3 === 0 ? 2 : 1.1}"/>`).join('')}<rect x="186" y="44" width="17" height="24" rx="2"/><rect x="207" y="49" width="11" height="14" rx="2"/><path d="m224 49 12 7-12 7Z"/><rect x="334" y="44" width="37" height="25" rx="2"/><circle cx="361" cy="51" r="2"/><path d="m337 65 10-10 8 7 5-5 9 8"/></g>`;
  } else if (path.id === 'flow') {
    symbols = `<g class="study-diagram-symbol"><circle cx="54" cy="46" r="1.5"/><circle cx="82" cy="66" r="2"/><circle cx="65" cy="55" r="2"/><circle cx="87" cy="43" r="1"/><circle cx="49" cy="67" r="1"/><circle cx="78" cy="51" r="1"/><circle cx="63" cy="70" r="1"/><path d="M181 64c18 0 22-20 52-20m-52 12c22 0 28-7 52-7m-52 19c22 0 27-12 52-12"/><path d="m229 40 5 4-5 4m0-3 5 4-5 4m0-1 5 4-5 4"/><circle cx="350" cy="55" r="12"/><circle cx="350" cy="55" r="5"/><circle cx="350" cy="55" r="1"/></g>`;
  } else {
    symbols = `<g class="study-diagram-symbol"><rect x="49" y="44" width="37" height="25" rx="2"/><path d="m52 65 10-10 8 7 5-5 9 8"/><rect x="186" y="43" width="18" height="25" rx="2"/><path d="M209 47h26m-26 8h20m-20 8h26"/><rect x="334" y="44" width="37" height="25" rx="2"/><path d="m337 65 10-10 8 7 5-5 9 8"/><path d="M350 39v9m-5-4h10" class="study-diagram-highlight"/></g>`;
  }
  return `<svg class="study-diagram" viewBox="0 0 420 149" role="img" aria-labelledby="${prefix}-title"><title id="${prefix}-title">${escape(path.diagramLabel)}</title><g class="study-diagram-box"><rect x="18" y="26" width="103" height="91" rx="8"/><rect x="154" y="26" width="113" height="91" rx="8"/><rect x="300" y="26" width="103" height="91" rx="8"/></g>${base}${symbols}${path.nodes.map((node, index) => `<text x="${[69.5, 210.5, 351.5][index]}" y="95" text-anchor="middle">${escape(node)}</text>`).join('')}<circle class="study-diagram-highlight" cx="210" cy="138" r="2.2"/></svg>`;
}

function panel(path, index, catalogue) {
  const entries = path.notes.map(id => catalogue.get(id)).filter(Boolean);
  const totalMinutes = entries.reduce((sum, n) => sum + (Number(n.minutes) || 0), 0);
  return `<div class="study-panel" id="study-panel-${path.id}" role="tabpanel" aria-labelledby="study-tab-${path.id}" tabindex="0"${index ? ' hidden' : ''}>
    <div class="study-explanation">
      <div class="study-kicker-row">
        <span class="study-kicker">PATH ${String(index + 1).padStart(2, '0')} · ${escape(path.short)}</span>
        <span class="study-duration">共 ${entries.length} 篇 · 约 ${totalMinutes} 分钟</span>
      </div>
      <h3>${escape(path.question)}</h3>
      <p class="study-summary">${escape(path.summary)}</p>
      <div class="study-figure">${diagram(path)}<span class="study-figure-note">${escape(path.diagramNote)}</span></div>
      <div class="study-concepts">${path.concepts.map(concept => `<span>${escape(concept)}</span>`).join('')}</div>
    </div>
    <div class="study-reading">
      <div class="study-reading-heading"><span>专题阅读顺序</span><a class="study-filter-link" href="#/notes?cat=${encodeURIComponent(path.name)}">在列表中筛选 →</a></div>
      <ol class="study-note-list">${entries.map((note, noteIndex) => {
        const prog = readProgress(note.id);
        const statusBadge = prog >= 95
          ? '<span class="study-note-badge is-done">已读</span>'
          : prog >= 2
            ? `<span class="study-note-badge">${Math.round(prog)}%</span>`
            : '';
        return `<li><a href="#/notes/${escape(note.id)}"><span class="study-note-number">${String(noteIndex + 1).padStart(2, '0')}</span><span class="study-note-copy"><strong>${escape(note.title)}${statusBadge}</strong><span>${escape(note.subtitle)}</span></span><span class="study-note-end"><span>${Number(note.minutes) || 1} 分钟</span>${arrow}</span></a></li>`;
      }).join('')}</ol>
    </div>
  </div>`;
}

export function renderStudyMap(notes = []) {
  const catalogue = new Map((Array.isArray(notes) ? notes : []).filter(note => note && /^[a-z0-9-]+$/.test(note.id)).map(note => [note.id, note]));
  const overview = catalogue.get('generative-models-map');
  return `<div class="study-map">
    <div class="study-tabs" role="tablist" aria-label="研究路径" aria-orientation="horizontal">${paths.map((path, index) => `<button class="study-tab" id="study-tab-${path.id}" type="button" role="tab" aria-controls="study-panel-${path.id}" aria-selected="${index === 0}" tabindex="${index === 0 ? '0' : '-1'}"><span class="study-tab-number">${String(index + 1).padStart(2, '0')}</span><span>${escape(path.name)}</span><span class="study-tab-count">${path.notes.filter(id => catalogue.has(id)).length}</span></button>`).join('')}</div>
    ${paths.map((path, index) => panel(path, index, catalogue)).join('')}
    ${overview ? `<a class="study-overview" href="#/notes/${overview.id}"><span class="study-overview-label">知识体系总览</span><div class="study-overview-body"><strong>${escape(overview.title)}</strong><span>${escape(overview.subtitle)}</span></div><span class="study-overview-meta">约 ${overview.minutes} 分钟 ${arrow}</span></a>` : ''}
  </div>`;
}

export function initStudyMap() {
  const removers = [];
  document.querySelectorAll('.study-map').forEach(root => {
    const tabs = [...root.querySelectorAll('[role="tab"]')];
    const panels = [...root.querySelectorAll('[role="tabpanel"]')];
    const activate = (tab, moveFocus = false) => {
      if (!tabs.includes(tab)) return;
      tabs.forEach(button => {
        const selected = button === tab;
        button.setAttribute('aria-selected', String(selected));
        button.tabIndex = selected ? 0 : -1;
      });
      panels.forEach(item => { item.hidden = item.id !== tab.getAttribute('aria-controls'); });
      if (moveFocus) tab.focus({ preventScroll: true });
    };
    const click = event => {
      const tab = event.target.closest('[role="tab"]');
      if (tab && root.contains(tab)) activate(tab);
    };
    const keydown = event => {
      const tab = event.target.closest('[role="tab"]');
      if (!tabs.includes(tab)) return;
      const index = tabs.indexOf(tab);
      let next;
      if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
      else if (event.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length];
      else if (event.key === 'Home') next = tabs[0];
      else if (event.key === 'End') next = tabs.at(-1);
      else return;
      event.preventDefault();
      activate(next, true);
      next.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'nearest' });
    };
    root.addEventListener('click', click);
    root.addEventListener('keydown', keydown);
    removers.push(() => { root.removeEventListener('click', click); root.removeEventListener('keydown', keydown); });
  });
  return () => removers.forEach(remove => remove());
}
