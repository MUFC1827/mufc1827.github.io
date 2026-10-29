import MarkdownIt from 'markdown-it';
import texmath from 'markdown-it-texmath';
import katex from 'katex';
import DOMPurify from 'dompurify';
import notes from './data/notes.json';
import { getNoteCategory } from './study-map.js';
import 'katex/dist/katex.min.css';
import './reader.css';

const escapeHtml = (value = '') => String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
const clamp = (value, low = 0, high = 100) => Math.min(high, Math.max(low, value));
const slugify = (value) => value.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-+|-+$/g, '');
const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* Reading works when storage is unavailable. */ } },
};

const markdown = new MarkdownIt({ html: false, linkify: true, typographer: false, breaks: false })
  .use(texmath, {
    engine: katex,
    delimiters: ['dollars', 'brackets'],
    katexOptions: { throwOnError: false, strict: 'ignore', trust: false, maxExpand: 1000, maxSize: 20, output: 'htmlAndMathml' },
  });

markdown.core.ruler.after('inline', 'reader-structure', (state) => {
  const { tokens, env } = state;
  const used = new Map();
  let firstHeadingRemoved = false;
  env.headings = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token.type === 'heading_open') {
      // The page header already presents the note title. Remove exactly the
      // first document H1; all subsequent content remains untouched.
      if (token.tag === 'h1' && !firstHeadingRemoved) {
        firstHeadingRemoved = true;
        token.hidden = true;
        tokens[index + 1].hidden = true;
        tokens[index + 1].content = '';
        tokens[index + 1].children = [];
        tokens[index + 2].hidden = true;
        continue;
      }
      const inline = tokens[index + 1];
      const text = inline.children?.map((child) => child.content).join('') || inline.content;
      const slug = slugify(text) || `section-${env.headings.length + 1}`;
      const duplicate = used.get(slug) || 0;
      used.set(slug, duplicate + 1);
      const id = `reader-${env.noteId}-${slug}${duplicate ? `-${duplicate + 1}` : ''}`;
      token.attrSet('id', id);
      token.attrSet('tabindex', '-1');
      if (token.tag === 'h2' || token.tag === 'h3') env.headings.push({ id, text, slug, level: Number(token.tag.slice(1)) });
    }
    if (token.type === 'blockquote_open') {
      const inline = tokens[index + 2];
      const first = inline?.children?.[0];
      const callout = first?.type === 'text' && first.content.match(/^\[!(\w+)\]\s*(.*)$/);
      if (callout) {
        const kind = callout[1].toLowerCase();
        token.attrSet('class', `reader-callout reader-callout-${kind}`);
        first.content = callout[2] || ({ note: '笔记', tip: '提示', important: '重点', warning: '注意', abstract: '导读' }[kind] || '提示');
        const open = new state.Token('strong_open', 'strong', 1);
        const close = new state.Token('strong_close', 'strong', -1);
        inline.children.splice(0, 1, open, first, close);
      }
    }
    if (token.type === 'list_item_open') {
      const inline = tokens[index + 2];
      const first = inline?.children?.[0];
      const task = first?.type === 'text' && first.content.match(/^\[([ xX])\]\s+/);
      if (task) {
        const checked = task[1].toLowerCase() === 'x';
        first.content = first.content.slice(task[0].length);
        token.attrSet('class', 'reader-task');
        const checkbox = new state.Token('html_inline', '', 0);
        checkbox.content = `<input type="checkbox" disabled tabindex="-1" aria-label="${checked ? '已完成' : '未完成'}"${checked ? ' checked' : ''}> `;
        inline.children.unshift(checkbox);
      }
    }
  }
});

const defaultLinkOpen = markdown.renderer.rules.link_open || ((tokens, index, options, env, renderer) => renderer.renderToken(tokens, index, options));
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  const href = tokens[index].attrGet('href') || '';
  if (/^https?:\/\//i.test(href)) {
    tokens[index].attrSet('target', '_blank');
    tokens[index].attrSet('rel', 'noopener noreferrer');
  }
  return defaultLinkOpen(tokens, index, options, env, renderer);
};

function noteUrl(note) {
  const url = new URL(note.path, document.baseURI);
  if (url.origin !== location.origin || !url.pathname.endsWith('.md')) throw new Error('Invalid note resource');
  return url;
}

function failure() {
  return {
    html: `<section class="reader reader-error"><a class="reader-back" href="#/notes">← 返回研究笔记</a><div class="reader-error-card"><span class="reader-eyebrow">RESEARCH NOTES</span><h1>这篇笔记暂时没能打开</h1><p>请检查网络连接后重试。</p><button class="reader-button" type="button" data-reader-retry>重新加载</button></div></section>`,
    init() {
      const button = document.querySelector('[data-reader-retry]');
      const retry = () => location.reload();
      button?.addEventListener('click', retry);
      return () => button?.removeEventListener('click', retry);
    },
  };
}

/** Fetch and render a public note. Call init only after placing html in the DOM. */
export async function loadReader(note) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    if (!note || !/^[a-z0-9-]+$/.test(note.id)) return failure(note);
    const resource = noteUrl(note);
    const response = await fetch(resource, { signal: controller.signal });
    if (!response.ok) return failure(note);
    // One source note contains an escaped-backspace typo before "igl". Restore
    // its intended TeX command for display without changing the source file.
    const raw = (await response.text()).replace(/\u0008igl/g, '\\bigl');
    if (raw.length > 1024 * 1024 || /^\s*<!doctype html/i.test(raw)) return failure(note);
    const env = { noteId: note.id, headings: [] };
    const prose = DOMPurify.sanitize(markdown.render(raw, env), {
      USE_PROFILES: { html: true, svg: true, mathMl: true },
      ADD_ATTR: ['target', 'tabindex'],
      FORBID_TAGS: ['style', 'script', 'iframe', 'form', 'object', 'embed'],
    });
    const stored = clamp(Number(storage.get(`mufc-reader-${note.id}`)) || 0);
    const categoryName = getNoteCategory(note);
    const noteIndex = notes.findIndex(item => item.id === note.id);
    const prevNote = noteIndex > 0 ? notes[noteIndex - 1] : null;
    const nextNote = noteIndex >= 0 && noteIndex < notes.length - 1 ? notes[noteIndex + 1] : null;
    const sourceLink = /^https?:\/\//i.test(note.sourceUrl || '')
      ? `<a href="${escapeHtml(note.sourceUrl)}" target="_blank" rel="noopener noreferrer">阅读原始来源 ↗</a>` : '';
    const toc = env.headings.map((heading) => `<button type="button" class="reader-toc-item reader-toc-level-${heading.level}" data-reader-section="${escapeHtml(heading.id)}">${escapeHtml(heading.text)}</button>`).join('');
    const html = `<section class="reader" data-reader-id="${note.id}">
      <div class="reader-progress" role="progressbar" aria-label="阅读进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span></span></div>
      <a class="reader-back" href="#/notes">← 返回研究笔记</a>
      <header class="reader-header">
        <div class="reader-breadcrumb"><a href="#/">主页</a><span>/</span><a href="#/notes">研究笔记</a><span>/</span><a href="#/notes?cat=${encodeURIComponent(categoryName)}">${escapeHtml(categoryName)}</a></div>
        <div class="reader-meta"><span>${escapeHtml(categoryName)}</span><span>第 ${String(noteIndex + 1).padStart(2, '0')} / ${String(notes.length).padStart(2, '0')} 篇</span><span>约 ${Number(note.minutes) || 1} 分钟阅读</span>${note.date ? `<time datetime="${escapeHtml(note.date)}">${escapeHtml(note.date)}</time>` : ''}</div>
        <h1>${escapeHtml(note.title)}</h1>
        <p class="reader-subtitle">${escapeHtml(note.subtitle)}</p>
        <div class="reader-actions"><button class="reader-button" type="button" data-save-note="${note.id}" aria-pressed="false">收藏笔记</button><a href="${escapeHtml(resource.pathname)}" download="${note.id}.md">下载 Markdown ↓</a>${sourceLink}</div>
        ${stored > 1 && stored < 99 ? `<button class="reader-resume" type="button" data-reader-resume>继续上次的阅读 · ${Math.round(stored)}% <span aria-hidden="true">↓</span></button>` : ''}
      </header>
      <div class="reader-layout"><article class="reader-prose" aria-label="笔记正文">${prose}</article>
        <aside class="reader-sidebar" aria-label="文章目录"><div class="reader-toc"><div class="reader-toc-heading">本文目录 <span data-reader-percent>0%</span></div><nav>${toc || '<p class="reader-toc-empty">暂无目录</p>'}</nav></div></aside></div>
      <footer class="reader-footer">
        <div class="reader-pager">
          ${prevNote ? `<a class="reader-pager-card" href="#/notes/${escapeHtml(prevNote.id)}"><span class="reader-pager-label">← 上一篇</span><strong>${escapeHtml(prevNote.title)}</strong><small>${escapeHtml(prevNote.subtitle)}</small></a>` : `<a class="reader-pager-card" href="#/notes"><span class="reader-pager-label">← 笔记目录</span><strong>浏览全部 ${notes.length} 篇研究笔记</strong><small>按研究路径与关键词筛选</small></a>`}
          ${nextNote ? `<a class="reader-pager-card is-next" href="#/notes/${escapeHtml(nextNote.id)}"><span class="reader-pager-label">下一篇 →</span><strong>${escapeHtml(nextNote.title)}</strong><small>${escapeHtml(nextNote.subtitle)}</small></a>` : `<a class="reader-pager-card is-next" href="#/notes"><span class="reader-pager-label">已完成最后一篇 →</span><strong>返回研究笔记列表</strong><small>回顾 Attention、扩散与流模型全线</small></a>`}
        </div>
      </footer>
    </section>`;
    return { html, init: () => initReader(note, stored, env.headings) };
  } catch {
    return failure(note);
  } finally {
    clearTimeout(timeout);
  }
}

function initReader(note, previous, toc) {
  const root = document.querySelector(`[data-reader-id="${note.id}"]`);
  if (!root) return () => {};
  const prose = root.querySelector('.reader-prose');
  const progressBar = root.querySelector('.reader-progress');
  const progressFill = progressBar.querySelector('span');
  const percentLabel = root.querySelector('[data-reader-percent]');
  const tocButtons = [...root.querySelectorAll('[data-reader-section]')];
  const headings = toc.map((entry) => document.getElementById(entry.id)).filter(Boolean);
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const key = `mufc-reader-${note.id}`;
  let frame = 0;
  let lastSaved = 0;
  let fraction = 0;
  let hasScrolled = false;
  let lastScrollY = window.scrollY;
  let activeId = '';
  storage.set('mufc-last-read', note.id);

  const persist = () => {
    if (hasScrolled) storage.set(key, String(Math.round(fraction * 1000) / 10));
  };
  const update = () => {
    frame = 0;
    const bounds = prose.getBoundingClientRect();
    const distance = Math.max(1, prose.scrollHeight - window.innerHeight + 120);
    fraction = clamp((120 - bounds.top) / distance, 0, 1);
    const percent = Math.round(fraction * 100);
    progressFill.style.transform = `scaleX(${fraction})`;
    progressBar.setAttribute('aria-valuenow', String(percent));
    percentLabel.textContent = `${percent}%`;
    let current = headings[0]?.id || '';
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top > 170) break;
      current = heading.id;
    }
    if (current !== activeId) {
      activeId = current;
      for (const button of tocButtons) {
        if (button.dataset.readerSection === current) button.setAttribute('aria-current', 'location');
        else button.removeAttribute('aria-current');
      }
    }
    if (hasScrolled && Date.now() - lastSaved > 1000) {
      persist();
      lastSaved = Date.now();
    }
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
  const onScroll = () => {
    if (window.scrollY !== lastScrollY) {
      hasScrolled = true;
      lastScrollY = window.scrollY;
    }
    schedule();
  };
  const jump = (element) => {
    element.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
    element.focus({ preventScroll: true });
  };
  const onClick = (event) => {
    const button = event.target.closest('[data-reader-section]');
    if (button) {
      const heading = document.getElementById(button.dataset.readerSection);
      if (heading) jump(heading);
      return;
    }
    if (event.target.closest('[data-reader-resume]')) {
      const top = prose.getBoundingClientRect().top + window.scrollY;
      const distance = Math.max(1, prose.scrollHeight - window.innerHeight + 120);
      window.scrollTo({ top: top - 120 + distance * previous / 100, behavior: reducedMotion ? 'auto' : 'smooth' });
      return;
    }
    const link = event.target.closest('.reader-prose a[href^="#"]');
    if (link && !link.getAttribute('href').startsWith('#/')) {
      // Hash fragments inside an article must not replace the application route.
      event.preventDefault();
      let anchor;
      try { anchor = decodeURIComponent(link.getAttribute('href').slice(1)); } catch { return; }
      const entry = toc.find((heading) => heading.id === anchor || heading.slug === anchor || heading.text === anchor);
      const target = document.getElementById(entry?.id || `reader-${note.id}-${anchor}`);
      if (target && prose.contains(target)) jump(target);
    }
  };

  root.addEventListener('click', onClick);
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', schedule, { passive: true });
  window.addEventListener('pagehide', persist);
  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(schedule) : null;
  observer?.observe(prose);
  update();
  return () => {
    persist();
    if (frame) cancelAnimationFrame(frame);
    observer?.disconnect();
    root.removeEventListener('click', onClick);
    window.removeEventListener('scroll', onScroll);
    window.removeEventListener('resize', schedule);
    window.removeEventListener('pagehide', persist);
  };
}
