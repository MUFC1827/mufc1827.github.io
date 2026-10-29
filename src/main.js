import './style.css';
import './lab.css';
import './play.css';
import notes from './data/notes.json';
import { icon, escapeHTML as esc } from './icons.js';
import { projects } from './projects.js';
import { renderHome, initHome, renderProjectPreview } from './home.js';
import { studyPaths, getNoteCategory } from './study-map.js';
import { renderLab, initLab } from './lab.js';
import { brandSymbol } from './brand.js';

const read = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
const write = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} };
const readNoteProgress = id => {
  try {
    const value = Number(localStorage.getItem(`mufc-reader-${id}`));
    return Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
  } catch {
    return 0;
  }
};

let saved = read('mufc-saved-notes', []);
if (!Array.isArray(saved)) saved = [];
saved = saved.filter(id => notes.some(n => n.id === id));

let cleanup = () => {};
let revealCleanup = () => {};
let navigation = 0;
let lastFocus = null;
let searchIndex = -1;

const categories = ['全部', ...studyPaths.map(p => p.name), '知识地图'];
const categoryCounts = new Map([['全部', notes.length]]);
for (const cat of categories.slice(1)) {
  categoryCounts.set(cat, notes.filter(n => getNoteCategory(n) === cat).length);
}

const routeHref = id => `#/notes/${id}`;
const external = 'target="_blank" rel="noopener noreferrer"';
const totalWords = notes.reduce((sum, n) => sum + (Number(n.words) || 0), 0);

document.querySelector('#app').innerHTML = `
  <header class="site-header"><div class="header-inner container">
    <a class="brand" href="#/" aria-label="MUFC 主页">${brandSymbol}<span>MUFC<span class="brand-handle">1827</span></span></a>
    <nav class="desktop-nav" aria-label="主导航">
      <a href="#/" data-nav="home">主页</a>
      <a href="#/works" data-nav="works">作品</a>
      <a href="#/notes" data-nav="notes">研究笔记</a>
      <a href="#/play/penalty" data-nav="play">点球</a>
      <a href="#/lab" data-nav="lab">实验室</a>
      <span class="nav-indicator" aria-hidden="true"></span>
    </nav>
    <div class="header-actions">
      <button class="search-trigger" data-open-search aria-label="搜索网站（Ctrl 或 Command 加 K）">${icon('search', 16)}<span>搜索</span><kbd>⌘K</kbd></button>
      <button class="icon-button theme-toggle" aria-label="切换深色模式"></button>
      <button class="icon-button menu-toggle" aria-label="打开导航菜单" aria-expanded="false">${icon('menu')}</button>
    </div>
  </div>
  <nav class="mobile-nav" aria-label="移动端导航" hidden>
    <a href="#/" data-nav="home">主页</a>
    <a href="#/works" data-nav="works">作品集</a>
    <a href="#/notes" data-nav="notes">研究笔记</a>
    <a href="#/play/penalty" data-nav="play">点球模拟</a>
    <a href="#/lab" data-nav="lab">实验室 · 专注计时</a>
  </nav></header>
  <main id="main" class="container" tabindex="-1"></main>
  <footer class="site-footer container">
    <div class="footer-top">
      <div class="footer-intro">
        <a class="footer-brand" href="#/">${brandSymbol}<span>MUFC<span class="brand-handle">1827</span></span></a>
        <p>生成式 AI 研究笔记与独立开发作品集。数据与阅读进度保存在本地浏览器。</p>
      </div>
      <div class="footer-columns">
        <div>
          <h2>浏览</h2>
          <ul>
            <li><a href="#/">个人主页</a></li>
            <li><a href="#/works">个人项目 (${projects.length})</a></li>
            <li><a href="#/notes">研究笔记 (${notes.length})</a></li>
          </ul>
        </div>
        <div>
          <h2>互动与外链</h2>
          <ul>
            <li><a href="#/play/penalty">点球物理模拟</a></li>
            <li><a href="#/lab">点云与专注计时</a></li>
            <li><a href="https://github.com/MUFC1827" ${external}>GitHub ${icon('external', 13)}</a></li>
          </ul>
        </div>
      </div>
    </div>
    <div class="footer-bottom">
      <span>© ${new Date().getFullYear()} MUFC1827 · Built with Vanilla ES Modules & Vite</span>
      <span>快捷键 <kbd>⌘K</kbd> 全局搜索</span>
    </div>
  </footer>
  <button class="back-top icon-button" aria-label="返回顶部" hidden>${icon('arrow', 18)}</button>
  <dialog class="search-dialog" aria-labelledby="search-title">
    <div class="dialog-search">${icon('search', 18)}<label id="search-title" class="sr-only" for="global-search">搜索笔记、作品与页面</label><input id="global-search" type="search" placeholder="搜索笔记、作品、标签或页面…" autocomplete="off"><button class="icon-button" data-close-search aria-label="关闭搜索">${icon('close', 18)}</button></div>
    <div id="search-results"></div>
    <div class="search-help"><span><kbd>↑</kbd> <kbd>↓</kbd> 选择 · <kbd>Enter</kbd> 打开</span><span><kbd>Esc</kbd> 关闭</span></div>
  </dialog>
  <div class="toast" role="status" aria-live="polite"></div>`;

const main = document.querySelector('main');
const siteHeader = document.querySelector('.site-header');
const dialog = document.querySelector('.search-dialog');
const searchInput = document.querySelector('#global-search');
let toastTimer;

function toast(message) {
  const el = document.querySelector('.toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
}

function noteRow(note, index = 0) {
  const cat = getNoteCategory(note);
  const progress = readNoteProgress(note.id);
  const isSaved = saved.includes(note.id);
  const tagsHtml = (note.tags || []).slice(0, 2).map(t => `<span class="note-tag">${esc(t)}</span>`).join('');
  let progressHtml = '';
  if (progress >= 95) {
    progressHtml = `<span class="note-done">${icon('check', 12)} 已读</span>`;
  } else if (progress >= 2) {
    const pct = Math.round(progress);
    progressHtml = `<span class="note-progress" title="已读 ${pct}%"><i style="--p:${pct}%"></i><span>${pct}%</span></span>`;
  }
  return `<a class="note-row" href="${routeHref(note.id)}">
    <span class="note-number">${String(index + 1).padStart(2, '0')}</span>
    <div class="note-row-body">
      <div class="note-meta"><span class="note-category">${esc(cat)}</span>${tagsHtml}</div>
      <h3>${esc(note.title)}</h3>
      <p>${esc(note.subtitle)}</p>
    </div>
    <div class="note-side">
      ${isSaved ? `<span class="note-saved" title="已收藏">${icon('bookmark', 15)}</span>` : ''}
      ${progressHtml}
      <span class="note-time">${note.minutes} 分钟</span>
      ${icon('arrow', 16)}
    </div>
  </a>`;
}

function worksPage() {
  return `<section class="page-intro" data-reveal>
    <div class="eyebrow"><span class="accent">01 — 03</span> / SELECTED PROJECTS</div>
    <h1>个人项目<span class="accent">.</span></h1>
    <p>围绕声音交互、线下桌游聚会与本地优先的桌面效率工具，保持克制的视觉与即开即用的体验。</p>
    <div class="page-meta">
      <span><strong>${projects.length}</strong> 个独立项目</span>
      <span>Web 应用与跨平台桌面软件</span>
    </div>
  </section>
  <div class="project-list">${projects.map((p, idx) => `<article class="project-feature" data-reveal style="--i:${idx}">
    <a class="project-feature-art stage-${p.id}" href="#/work/${p.id}" aria-label="查看 ${p.name}">${renderProjectPreview(p)}</a>
    <div class="project-feature-copy">
      <div class="eyebrow"><span class="accent">${p.number}</span> / ${p.type}</div>
      <h2><a href="#/work/${p.id}">${p.name}</a></h2>
      <p>${p.description}${p.detail ? ` ${p.detail}` : ''}</p>
      <div class="project-tags">${p.features.map(f => `<span class="chip">${f}</span>`).join('')}</div>
      <div class="project-actions">
        <a class="button primary" href="${p.url}" ${external}>${p.action} ${icon('external', 15)}</a>
        ${p.download ? `<a class="button secondary" href="${p.download}" ${external}>下载最新版 ${icon('external', 15)}</a>` : ''}
        <a class="text-link muted-link" href="#/work/${p.id}">项目详情 ${icon('arrow', 15)}</a>
      </div>
    </div>
  </article>`).join('')}</div>`;
}

function projectPage(project) {
  const next = projects[(projects.indexOf(project) + 1) % projects.length];
  return `<div class="project-detail">
    <a class="back-link" href="#/works">${icon('back', 16)} 返回作品集</a>
    <div class="project-detail-heading" data-reveal>
      <div>
        <div class="eyebrow"><span class="accent">PROJECT ${project.number}</span> / ${project.type}</div>
        <h1>${project.name}<span class="accent">.</span></h1>
        <p>${project.label}</p>
      </div>
      <div class="detail-actions">
        ${project.download ? `<a class="button secondary" href="${project.download}" ${external}>下载应用 ${icon('external', 15)}</a>` : ''}
        <a class="button primary" href="${project.url}" ${external}>${project.action} ${icon('external', 16)}</a>
      </div>
    </div>
    <div class="detail-art stage-${project.id}" data-reveal>${renderProjectPreview(project, { eager: true })}</div>
    <div class="project-detail-body" data-reveal>
      <div>
        <span class="eyebrow">设计初衷与简介</span>
        <p>${project.description}</p>
        <p>${project.detail}</p>
      </div>
      <aside>
        <span class="eyebrow">核心特性</span>
        <ul>${project.features.map(f => `<li>${icon('check', 16)} <span>${f}</span></li>`).join('')}</ul>
        <a class="button primary" href="${project.download || project.url}" ${external}>${project.download ? '下载 MemoUnited 最新版' : project.action} ${icon('external', 15)}</a>
        <a class="project-live-url" href="${project.url}" ${external}>${new URL(project.url).hostname} ${icon('external', 13)}</a>
      </aside>
    </div>
    <a class="next-project" href="#/work/${next.id}" data-reveal>
      <div class="next-project-thumb stage-${next.id}">${renderProjectPreview(next)}</div>
      <div class="next-project-copy">
        <span class="eyebrow">下一个作品 · ${next.number}</span>
        <strong>${next.name}</strong>
        <span>${next.description}</span>
      </div>
      ${icon('arrow', 20)}
    </a>
  </div>`;
}

function notesPage() {
  return `<section class="page-intro notes-intro" data-reveal>
    <div class="eyebrow"><span class="accent">RESEARCH NOTES</span> / AIGC & GENERATIVE MODELS</div>
    <h1>研究笔记<span class="accent">.</span></h1>
    <p>系统梳理从 Self-Attention、Transformer 到 DDPM、Latent Diffusion、Rectified Flow 及可控图像编辑的论文推导与实验记录。</p>
    <div class="page-meta">
      <span><strong>${notes.length}</strong> 篇长文笔记</span>
      <span>约 <strong>${(totalWords / 1000).toFixed(1)}k</strong> 字</span>
      <span>4 条专题路径 + 1 篇总览地图</span>
    </div>
  </section>
  <div class="notes-toolbar">
    <div class="category-tabs" role="group" aria-label="笔记分类">
      ${categories.map((cat, i) => `<button type="button" data-category="${esc(cat)}" aria-pressed="${i === 0}">${esc(cat)}<span>${categoryCounts.get(cat) || 0}</span></button>`).join('')}
    </div>
    <button type="button" class="filter-toggle" id="only-saved" aria-pressed="false">${icon('bookmark', 14)}<span>我的收藏</span></button>
    <div class="notes-search">
      ${icon('search', 16)}
      <label class="sr-only" for="notes-query">搜索笔记标题与关键词</label>
      <input id="notes-query" type="search" placeholder="搜索标题、模型或标签…" autocomplete="off">
      <kbd>/</kbd>
    </div>
    <button type="button" class="button small secondary random-note" title="随机打开一篇笔记">${icon('shuffle', 15)} 随机阅读</button>
  </div>
  <div class="notes-list-header">
    <span id="notes-result-count" aria-live="polite">${notes.length} 篇笔记 · 按推荐学习顺序排列</span>
    <span>状态 / 预计阅读</span>
  </div>
  <div id="notes-list" class="notes-list"></div>
  <p class="local-storage-note">${icon('bookmark', 14)} 收藏夹与每篇笔记的阅读进度均自动保存在当前浏览器中。</p>`;
}

function initNotes(initialCategory = '全部') {
  let category = categories.includes(initialCategory) ? initialCategory : '全部';
  let onlySaved = false;
  const input = document.querySelector('#notes-query');
  const savedBtn = document.querySelector('#only-saved');
  const catButtons = [...document.querySelectorAll('[data-category]')];

  const syncTabs = () => {
    catButtons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.category === category)));
    savedBtn.setAttribute('aria-pressed', String(onlySaved));
  };

  const update = () => {
    const q = input.value.trim().toLowerCase();
    const filtered = notes.filter(n => {
      const nCat = getNoteCategory(n);
      return (category === '全部' || nCat === category)
        && (!onlySaved || saved.includes(n.id))
        && `${n.title} ${n.subtitle} ${nCat} ${n.tags.join(' ')}`.toLowerCase().includes(q);
    });
    const suffix = category === '全部' ? '按推荐学习顺序排列' : `专题：${category}`;
    document.querySelector('#notes-result-count').textContent = `${filtered.length} 篇笔记 · ${suffix}`;
    document.querySelector('#notes-list').innerHTML = filtered.length
      ? filtered.map(n => noteRow(n, notes.indexOf(n))).join('')
      : `<div class="empty-state">${icon('search', 32)}<h3>没有匹配的笔记</h3><p>${onlySaved ? '你还没有收藏该分类下的笔记，可以在阅读笔记时点击「收藏笔记」。' : '请尝试调整搜索关键词或切换其他研究专题。'}</p><button class="button secondary" id="reset-notes">显示全部笔记</button></div>`;
    document.querySelector('#reset-notes')?.addEventListener('click', () => {
      input.value = '';
      onlySaved = false;
      category = '全部';
      syncTabs();
      update();
    });
  };

  catButtons.forEach(button => button.addEventListener('click', () => {
    category = button.dataset.category;
    syncTabs();
    update();
  }));
  savedBtn.addEventListener('click', () => {
    onlySaved = !onlySaved;
    syncTabs();
    update();
  });
  input.addEventListener('input', update);
  syncTabs();
  update();

  const focusSearch = event => {
    if (event.key === '/' && !/INPUT|TEXTAREA/.test(document.activeElement.tagName) && !dialog.open) {
      event.preventDefault();
      input.focus();
    }
  };
  document.addEventListener('keydown', focusSearch);
  const refreshSaved = event => {
    if (!event.key || event.key === 'mufc-saved-notes' || event.key.startsWith('mufc-reader-')) update();
  };
  window.addEventListener('storage', refreshSaved);
  return () => {
    document.removeEventListener('keydown', focusSearch);
    window.removeEventListener('storage', refreshSaved);
  };
}

function notFound() {
  return `<section class="not-found"><div class="eyebrow">PAGE NOT FOUND</div><span class="error-number">404<span class="accent">.</span></span><h1>页面不存在</h1><p>请检查访问地址，或返回个人主页继续浏览。</p><a class="button primary" href="#/">返回主页 ${icon('arrow', 16)}</a></section>`;
}

function syncSavedButtons() {
  document.querySelectorAll('[data-save-note]').forEach(button => {
    const active = saved.includes(button.dataset.saveNote);
    button.setAttribute('aria-pressed', String(active));
    button.innerHTML = `${icon('bookmark', 15)} ${active ? '已收藏' : '收藏笔记'}`;
  });
}

function setupReveal() {
  revealCleanup();
  const items = [...main.querySelectorAll('[data-reveal]')];
  if (!items.length) return;
  if (matchMedia('(prefers-reduced-motion: reduce)').matches || typeof IntersectionObserver !== 'function') {
    items.forEach(el => el.classList.add('is-visible'));
    return;
  }
  document.documentElement.classList.add('reveal-ready');
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { rootMargin: '0px 0px -40px 0px', threshold: 0.06 });
  items.forEach(el => observer.observe(el));
  revealCleanup = () => observer.disconnect();
}

const desktopNav = document.querySelector('.desktop-nav');
const navIndicator = desktopNav.querySelector('.nav-indicator');
let pendingRoute = null;
let renderedRoute = null;
let contentMotion = null;

function positionNavIndicator() {
  const active = desktopNav.querySelector('[aria-current="page"]');
  if (!active || desktopNav.offsetWidth === 0) {
    navIndicator.style.opacity = '0';
    return;
  }
  navIndicator.style.width = `${active.offsetWidth}px`;
  navIndicator.style.transform = `translateX(${active.offsetLeft}px)`;
  navIndicator.style.opacity = '1';
}

function updateNavigation(page) {
  const activePage = page === 'work' ? 'works' : page;
  document.querySelectorAll('[data-nav]').forEach(link => {
    if (link.dataset.nav === activePage) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  positionNavIndicator();
}

function closeMenu() {
  document.querySelector('.mobile-nav').hidden = true;
  const toggle = document.querySelector('.menu-toggle');
  toggle.setAttribute('aria-expanded', 'false');
  toggle.setAttribute('aria-label', '打开导航菜单');
}

function stopContentMotion() {
  contentMotion?.cancel();
  contentMotion = null;
}

const loadingFailure = () => ({
  title: '加载失败',
  html: '<div class="empty-state"><h1>加载失败</h1><p>请检查网络连接后重试。</p><button class="button primary" data-retry-reader>重新加载</button></div>',
});

async function prepareView(parts, queryParams) {
  const page = parts[0] || 'home';
  if (page === 'home' && parts.length === 0) return { title: '个人主页', html: renderHome(notes, projects), init: initHome };
  if (page === 'works' && parts.length === 1) return { title: '作品集', html: worksPage() };
  if (page === 'work' && parts.length === 2) {
    const project = projects.find(item => item.id === parts[1]);
    if (project) return { title: project.name, html: projectPage(project) };
  }
  if (page === 'notes' && parts.length === 1) {
    const initialCat = queryParams.get('cat') || '全部';
    return { title: '研究笔记', html: notesPage(), init: () => initNotes(initialCat) };
  }
  if (page === 'notes' && parts.length === 2) {
    const note = notes.find(item => item.id === parts[1]);
    if (note) {
      const { loadReader } = await import('./reader.js');
      return { title: note.title, ...await loadReader(note) };
    }
  }
  if (page === 'play' && parts.length === 2 && parts[1] === 'penalty') {
    const [game] = await Promise.all([import('./penalty.js'), import('./penalty.css')]);
    const rawHtml = game.renderPenalty();
    const wrappedHtml = rawHtml.includes('play-page')
      ? rawHtml
      : `<section class="play-page"><a class="play-back back-link" href="#/">${icon('back', 16)} 返回主页</a><div class="play-heading"><h1>点球<span>.</span></h1><span>INTERACTIVE / 01 · STRETFORD END</span></div>${rawHtml}</section>`;
    return { title: '点球模拟', html: wrappedHtml, init: game.initPenalty };
  }
  if (page === 'lab' && parts.length === 1) return { title: '实验室', html: renderLab(), init: initLab };
  return { title: '页面不存在', html: notFound() };
}

async function navigate({ force = false } = {}) {
  const rawHash = location.hash.replace(/^#/, '') || '/';
  const [pathPart, queryString = ''] = rawHash.split('?');
  const parts = pathPart.split('/').filter(Boolean);
  const queryParams = new URLSearchParams(queryString);
  const key = `/${parts.join('/')}${queryString ? `?${queryString}` : ''}`;

  if (!force && pendingRoute === key) return;
  const current = ++navigation;
  pendingRoute = key;
  stopContentMotion();
  closeMenu();
  updateNavigation(parts[0] || 'home');

  if (!force && renderedRoute === key) {
    pendingRoute = null;
    main.setAttribute('aria-busy', 'false');
    return;
  }
  main.setAttribute('aria-busy', 'true');
  if (!renderedRoute) main.innerHTML = '<div class="reader-loading">正在加载内容…</div>';

  let view;
  try { view = await prepareView(parts, queryParams); }
  catch { view = loadingFailure(); }
  if (current !== navigation) return;

  const animate = renderedRoute !== null && !matchMedia('(prefers-reduced-motion: reduce)').matches && typeof main.animate === 'function';
  if (animate) {
    contentMotion = main.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 70, easing: 'ease-out', fill: 'forwards' });
    try { await contentMotion.finished; } catch { /* Cancelled by newer navigation */ }
  }
  if (current !== navigation) return;

  cleanup();
  cleanup = () => {};
  stopContentMotion();
  main.innerHTML = view.html;
  window.scrollTo({ top: 0, behavior: 'instant' });
  document.title = `${view.title} · MUFC`;
  cleanup = view.init?.() || (() => {});
  setupReveal();
  syncSavedButtons();
  renderedRoute = key;
  pendingRoute = null;
  main.setAttribute('aria-busy', 'false');
  if (current > 1) main.focus({ preventScroll: true });

  if (animate) {
    const entering = main.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, easing: 'cubic-bezier(.2,.65,.3,1)' });
    contentMotion = entering;
    entering.finished.then(() => {
      if (contentMotion === entering) { entering.cancel(); contentMotion = null; }
    }).catch(() => {});
  }
}

window.addEventListener('resize', positionNavIndicator, { passive: true });
if (typeof ResizeObserver === 'function') new ResizeObserver(positionNavIndicator).observe(desktopNav);
document.fonts?.ready.then(positionNavIndicator);
requestAnimationFrame(() => desktopNav.classList.add('nav-ready'));

function themeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  document.querySelector('.theme-toggle').innerHTML = icon(dark ? 'sun' : 'moon', 18);
  document.querySelector('.theme-toggle').setAttribute('aria-label', dark ? '切换浅色模式' : '切换深色模式');
  document.querySelector('meta[name="theme-color"]').content = dark ? '#121212' : '#f7f6f2';
}
themeButton();

document.querySelector('.theme-toggle').addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('mufc-theme', theme); } catch {}
  themeButton();
});

document.querySelector('.menu-toggle').addEventListener('click', () => {
  const nav = document.querySelector('.mobile-nav');
  nav.hidden = !nav.hidden;
  const b = document.querySelector('.menu-toggle');
  b.setAttribute('aria-expanded', String(!nav.hidden));
  b.setAttribute('aria-label', nav.hidden ? '打开导航菜单' : '关闭导航菜单');
});

document.addEventListener('click', event => {
  const target = event.target.closest('button,a');
  if (!target) return;
  if (target.matches('.random-note')) location.hash = routeHref(notes[Math.floor(Math.random() * notes.length)].id);
  if (target.matches('[data-save-note]')) {
    const id = target.dataset.saveNote;
    if (!notes.some(n => n.id === id)) return;
    saved = saved.includes(id) ? saved.filter(x => x !== id) : [...saved, id];
    write('mufc-saved-notes', saved);
    syncSavedButtons();
    toast(saved.includes(id) ? '已加入收藏夹' : '已从收藏夹移除');
  }
  if (target.matches('[data-retry-reader]')) navigate({ force: true });
  if (target.matches('.skip-link')) {
    event.preventDefault();
    main.focus();
    main.scrollIntoView();
  }
  if (target.matches('a[href^="#/"]')) {
    closeMenu();
    if (target.getAttribute('href') === (location.hash || '#/')) {
      event.preventDefault();
      dialog.close();
      window.scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
    }
  }
});

window.addEventListener('hashchange', () => {
  dialog.close();
  navigate();
});

const topButton = document.querySelector('.back-top');
window.addEventListener('scroll', () => {
  const y = window.scrollY;
  topButton.hidden = y < 600;
  siteHeader.classList.toggle('is-scrolled', y > 12);
}, { passive: true });
topButton.addEventListener('click', () => window.scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' }));

window.addEventListener('storage', event => {
  if (event.key === 'mufc-saved-notes' || event.key === null) {
    const value = read('mufc-saved-notes', []);
    saved = Array.isArray(value) ? value.filter(id => notes.some(n => n.id === id)) : [];
    syncSavedButtons();
  }
});

const searchEntries = [
  ...[
    { title: '主页', url: '#/', subtitle: '个人项目与生成式 AI 研究笔记' },
    { title: '作品集', url: '#/works', subtitle: '个人独立开发项目展示' },
    { title: '研究笔记', url: '#/notes', subtitle: '18 篇图像生成与扩散模型笔记' },
    { title: '点球模拟', url: '#/play/penalty', subtitle: '120 Hz 物理轨迹与守门员扑救博弈' },
    { title: '实验室 · 专注计时', url: '#/lab', subtitle: '交互式点云生成艺术与 25 分钟番茄钟' },
  ].map(n => ({ ...n, type: '页面', terms: '' })),
  ...projects.map(p => ({ title: p.name, url: `#/work/${p.id}`, subtitle: p.label, type: '作品', terms: `${p.type} ${p.detail}` })),
  ...notes.map(n => ({ title: n.title, url: routeHref(n.id), subtitle: n.subtitle, type: getNoteCategory(n), terms: n.tags.join(' ') })),
];

function renderSearch() {
  const query = searchInput.value.trim().toLowerCase();
  const results = searchEntries.filter(x => `${x.title} ${x.subtitle} ${x.type} ${x.terms}`.toLowerCase().includes(query)).slice(0, 10);
  searchIndex = -1;
  document.querySelector('#search-results').innerHTML = `<div class="search-result-label">${query ? `找到 ${results.length}${results.length === 10 ? ' +' : ''} 个匹配项` : '快速跳转 · 页面、作品与笔记'}</div>${results.length
    ? results.map(r => `<a class="search-result" href="${r.url}"><span class="search-result-icon">${icon(r.type === '作品' ? 'code' : r.type === '页面' ? 'spark' : 'book', 17)}</span><span><strong>${esc(r.title)}</strong><small>${esc(r.subtitle)}</small></span><em>${esc(r.type)}</em>${icon('arrow', 16)}</a>`).join('')
    : `<div class="search-empty">没有找到“${esc(searchInput.value.trim())}”<br><small>试试搜索 Diffusion、Flow、ControlNet、钢琴 或 点球。</small></div>`}`;
}

function openSearch() {
  if (dialog.open) return;
  lastFocus = document.activeElement;
  searchInput.value = '';
  renderSearch();
  dialog.showModal();
  searchInput.focus();
}

document.querySelector('[data-open-search]').addEventListener('click', openSearch);
document.querySelector('[data-close-search]').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => lastFocus?.focus({ preventScroll: true }));
dialog.addEventListener('click', event => {
  if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }
  if (event.target.closest('.search-result')) dialog.close();
});
searchInput.addEventListener('input', renderSearch);
dialog.addEventListener('keydown', event => {
  const links = [...dialog.querySelectorAll('.search-result')];
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    if (!links.length) return;
    searchIndex = searchIndex < 0 ? (event.key === 'ArrowDown' ? 0 : links.length - 1) : (searchIndex + (event.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length;
    links.forEach((l, i) => l.classList.toggle('selected', i === searchIndex));
    links[searchIndex].focus({ preventScroll: true });
    links[searchIndex].scrollIntoView({ block: 'nearest' });
  }
  if (event.key === 'Enter' && document.activeElement === searchInput) {
    event.preventDefault();
    links[Math.max(0, searchIndex)]?.click();
  }
});
document.querySelector('#search-results').addEventListener('focusin', event => {
  const link = event.target.closest('.search-result');
  if (link) {
    const links = [...dialog.querySelectorAll('.search-result')];
    searchIndex = links.indexOf(link);
    links.forEach(l => l.classList.toggle('selected', l === link));
  }
});
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    dialog.open ? dialog.close() : openSearch();
  }
  if (event.key === 'Escape') {
    if (dialog.open) {
      event.preventDefault();
      dialog.close();
    }
    closeMenu();
  }
});

navigate();
