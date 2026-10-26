import './style.css';
import './redesign.css';
import './lab.css';
import './play.css';
import './navigation.css';
import notes from './data/notes.json';
import { icon, escapeHTML as esc } from './icons.js';
import { projects } from './projects.js';
import { renderHome, initHome, renderProjectPreview } from './home.js';
import { renderLab, initLab } from './lab.js';
import { brandSymbol } from './brand.js';

const read = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
const write = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} };
let saved = read('mufc-saved-notes', []);
if (!Array.isArray(saved)) saved = [];
saved = saved.filter(id => notes.some(n => n.id === id));
let cleanup = () => {};
let navigation = 0;
let lastFocus = null;
let searchIndex = -1;
const categories = ['全部', ...new Set(notes.map(n => n.category))];
const routeHref = id => `#/notes/${id}`;
const external = 'target="_blank" rel="noopener noreferrer"';

document.querySelector('#app').innerHTML = `
  <header class="site-header"><div class="header-inner container">
    <a class="brand" href="#/" aria-label="MUFC 主页">${brandSymbol}<span>MUFC<span class="brand-handle">1827</span></span></a>
    <nav class="desktop-nav" aria-label="主导航"><a href="#/" data-nav="home">主页</a><a href="#/works" data-nav="works">作品</a><a href="#/notes" data-nav="notes">研究笔记</a><span class="nav-indicator" aria-hidden="true"></span></nav>
    <div class="header-actions"><button class="search-trigger" data-open-search aria-label="搜索网站（Ctrl 或 Command 加 K）">${icon('search',17)}<span>搜索</span><kbd>⌘ K</kbd></button><button class="icon-button theme-toggle" aria-label="切换深色模式"></button><button class="icon-button menu-toggle" aria-label="打开导航菜单" aria-expanded="false">${icon('menu')}</button></div>
  </div><nav class="mobile-nav" aria-label="移动端导航" hidden><a href="#/" data-nav="home">主页</a><a href="#/works" data-nav="works">作品</a><a href="#/notes" data-nav="notes">研究笔记</a></nav></header>
  <main id="main" class="container" tabindex="-1"></main>
  <footer class="site-footer container"><div><a class="footer-brand" href="#/">MUFC<span>·</span></a></div><div class="footer-links"><a href="#/play/penalty">点球</a><a href="#/lab">专注计时</a><a href="https://github.com/MUFC1827" ${external}>GitHub ${icon('external',14)}</a><span>© ${new Date().getFullYear()} MUFC</span></div></footer>
  <button class="back-top icon-button" aria-label="返回顶部" hidden>${icon('arrow')}</button>
  <dialog class="search-dialog" aria-labelledby="search-title"><div class="dialog-search">${icon('search')}<label id="search-title" class="sr-only" for="global-search">搜索笔记、作品与页面</label><input id="global-search" type="search" placeholder="搜索笔记、作品与页面…" autocomplete="off"><button class="icon-button" data-close-search aria-label="关闭搜索">${icon('close',18)}</button></div><div id="search-results"></div><div class="search-help"><span>↑ ↓ 选择 · Enter 打开</span><span>Esc 关闭</span></div></dialog>
  <div class="toast" role="status" aria-live="polite"></div>`;

const main = document.querySelector('main');
const dialog = document.querySelector('.search-dialog');
const searchInput = document.querySelector('#global-search');
let toastTimer;
function toast(message) {
  const el = document.querySelector('.toast');
  el.textContent = message; el.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
}

function noteRow(note, index = 0, compact = false) {
  return `<a class="note-row ${compact?'compact':''}" href="${routeHref(note.id)}"><span class="note-number">${String(index+1).padStart(2,'0')}</span><div class="note-row-body"><span class="note-category">${esc(note.category)}</span><h3>${esc(note.title)}</h3>${compact?'':`<p>${esc(note.subtitle)}</p>`}</div><span class="note-time">${note.minutes} 分钟 ${icon('arrow',17)}</span></a>`;
}
function worksPage() {
  return `<section class="page-intro"><div class="eyebrow">PROJECTS</div><h1>个人项目<span class="accent">.</span></h1><p>音乐、桌游与桌面效率工具。</p><span class="page-number">01 — 03</span></section><div class="project-list">${projects.map(p=>`<article class="project-feature"><a class="project-feature-art" href="#/work/${p.id}" aria-label="查看 ${p.name}">${renderProjectPreview(p)}</a><div class="project-feature-copy"><div class="eyebrow"><span class="accent">${p.number}</span> / ${p.type}</div><h2><a href="#/work/${p.id}">${p.name}</a></h2><p>${p.description}</p><div class="project-tags">${p.features.slice(0,2).map(f=>`<span>${f}</span>`).join('')}</div><a class="text-link" href="#/work/${p.id}">项目介绍 ${icon('arrow',17)}</a><a class="text-link muted-link" href="${p.url}" ${external}>${p.action} ${icon('external',16)}</a></div></article>`).join('')}</div><div class="projects-bottom-space"></div>`;
}
function projectPage(project) {
  return `<div class="project-detail"><a class="back-link" href="#/works">${icon('back',17)} 返回作品集</a><div class="project-detail-heading"><div><div class="eyebrow">PROJECT ${project.number} / ${project.type}</div><h1>${project.name}<span class="accent">.</span></h1><p>${project.label}</p></div><a class="button primary" href="${project.url}" ${external}>${project.action} ${icon('external',18)}</a></div><div class="detail-art">${renderProjectPreview(project,{eager:true})}</div><div class="project-detail-body"><div><span class="eyebrow">项目简介</span><p>${project.description}</p><p>${project.detail}</p></div><aside><span class="eyebrow">功能</span><ul>${project.features.map(f=>`<li>${icon('check',17)} ${f}</li>`).join('')}</ul>${project.download?`<a class="button secondary" href="${project.download}" ${external}>下载 MemoUnited ${icon('external',17)}</a>`:''}<a class="project-live-url" href="${project.url}" ${external}>${new URL(project.url).hostname} ${icon('external',14)}</a></aside></div><div class="next-project"><span class="eyebrow">其他项目</span><a href="#/work/${projects[(projects.indexOf(project)+1)%3].id}">下一个作品：${projects[(projects.indexOf(project)+1)%3].name} ${icon('arrow',22)}</a></div></div>`;
}
function notesPage() {
  return `<section class="page-intro notes-intro"><div class="eyebrow">RESEARCH NOTES</div><h1>研究笔记<span class="accent">.</span></h1><p>图像生成相关的论文阅读与实验记录。</p><div class="notes-count"><strong>${notes.length}</strong><span>篇笔记</span></div></section><div class="notes-toolbar"><div class="notes-search">${icon('search',19)}<label class="sr-only" for="notes-query">搜索笔记标题与关键词</label><input id="notes-query" type="search" placeholder="搜索标题、模型或关键词…" autocomplete="off"><kbd>/</kbd></div><button class="button secondary random-note">${icon('shuffle',16)} 随机阅读</button></div><div class="notes-filter"><div class="category-tabs" role="group" aria-label="笔记分类">${categories.map((cat,i)=>`<button data-category="${cat}" aria-pressed="${i===0}">${cat}${i===0?`<span>${notes.length}</span>`:''}</button>`).join('')}</div><label class="saved-filter"><input type="checkbox" id="only-saved">${icon('bookmark',15)} 我的收藏</label></div><div class="notes-list-header"><span id="notes-result-count" aria-live="polite">${notes.length} 篇笔记 · 按学习顺序排列</span><span>预计阅读</span></div><div id="notes-list" class="notes-list"></div><p class="local-storage-note">${icon('bookmark',14)} 收藏与阅读进度保存在当前浏览器。</p>`;
}
function initNotes() {
  let category = '全部';
  const input = document.querySelector('#notes-query');
  const onlySaved = document.querySelector('#only-saved');
  const update = () => {
    const q = input.value.trim().toLowerCase();
    const filtered = notes.filter(n => (category==='全部'||n.category===category) && (!onlySaved.checked||saved.includes(n.id)) && `${n.title} ${n.subtitle} ${n.tags.join(' ')}`.toLowerCase().includes(q));
    document.querySelector('#notes-result-count').textContent = `${filtered.length} 篇笔记 · 按学习顺序排列`;
    document.querySelector('#notes-list').innerHTML = filtered.length ? filtered.map((n)=>noteRow(n,notes.indexOf(n))).join('') : `<div class="empty-state">${icon('search',32)}<h3>没有匹配的笔记</h3><p>${onlySaved.checked?'暂无收藏。':'可调整关键词或分类。'}</p><button class="button secondary" id="reset-notes">显示全部笔记</button></div>`;
    document.querySelector('#reset-notes')?.addEventListener('click',()=>{input.value='';onlySaved.checked=false;category='全部';document.querySelectorAll('[data-category]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.category==='全部')));update();});
  };
  document.querySelectorAll('[data-category]').forEach(button=>button.addEventListener('click',()=>{category=button.dataset.category;document.querySelectorAll('[data-category]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));update();}));
  input.addEventListener('input',update);onlySaved.addEventListener('change',update);update();
  const focusSearch = event => {if(event.key==='/'&&!/INPUT|TEXTAREA/.test(document.activeElement.tagName)&&!dialog.open){event.preventDefault();input.focus();}};
  document.addEventListener('keydown',focusSearch);
  const refreshSaved = event => { if (event.key === 'mufc-saved-notes' || event.key === null) update(); };
  window.addEventListener('storage', refreshSaved);
  return ()=>{document.removeEventListener('keydown',focusSearch);window.removeEventListener('storage',refreshSaved);};
}
function notFound() {
  return `<section class="not-found"><div class="eyebrow">PAGE NOT FOUND</div><span class="error-number">404<span class="accent">.</span></span><h1>页面不存在</h1><p>请检查地址或返回主页。</p><a class="button primary" href="#/">返回主页 ${icon('arrow',18)}</a></section>`;
}
function syncSavedButtons() {
  document.querySelectorAll('[data-save-note]').forEach(button=>{
    const active=saved.includes(button.dataset.saveNote);
    button.setAttribute('aria-pressed',String(active));
    button.innerHTML=`${icon('bookmark',17)} ${active?'已收藏':'收藏笔记'}`;
  });
}

const desktopNav = document.querySelector('.desktop-nav');
const navIndicator = desktopNav.querySelector('.nav-indicator');
let pendingRoute = null;
let renderedRoute = null;
let contentMotion = null;

function positionNavIndicator() {
  const active = desktopNav.querySelector('[aria-current="page"]');
  if (!active || desktopNav.offsetWidth === 0) { navIndicator.style.opacity = '0'; return; }
  navIndicator.style.width = active.offsetWidth + 'px';
  navIndicator.style.transform = 'translateX(' + active.offsetLeft + 'px)';
  navIndicator.style.opacity = '1';
}
function updateNavigation(page) {
  const activePage = page === 'work' ? 'works' : page;
  document.querySelectorAll('[data-nav]').forEach(link => {
    if (link.dataset.nav === activePage) link.setAttribute('aria-current','page');
    else link.removeAttribute('aria-current');
  });
  positionNavIndicator();
}
function closeMenu() {
  document.querySelector('.mobile-nav').hidden = true;
  const toggle = document.querySelector('.menu-toggle');
  toggle.setAttribute('aria-expanded','false');
  toggle.setAttribute('aria-label','打开导航菜单');
}
function stopContentMotion() {
  contentMotion?.cancel();
  contentMotion = null;
}
const loadingFailure = () => ({
  title: '加载失败',
  html: '<div class="empty-state"><h1>加载失败</h1><p>请检查网络连接后重试。</p><button class="button primary" data-retry-reader>重新加载</button></div>',
});
async function prepareView(parts) {
  const page = parts[0] || 'home';
  if (page === 'home' && parts.length === 0) return { title:'个人主页', html:renderHome(notes,projects), init:initHome };
  if (page === 'works' && parts.length === 1) return { title:'作品集', html:worksPage() };
  if (page === 'work' && parts.length === 2) {
    const project = projects.find(item => item.id === parts[1]);
    if (project) return { title:project.name, html:projectPage(project) };
  }
  if (page === 'notes' && parts.length === 1) return { title:'研究笔记', html:notesPage(), init:initNotes };
  if (page === 'notes' && parts.length === 2) {
    const note = notes.find(item => item.id === parts[1]);
    if (note) {
      const { loadReader } = await import('./reader.js');
      return { title:note.title, ...await loadReader(note) };
    }
  }
  if (page === 'play' && parts.length === 2 && parts[1] === 'penalty') {
    const [game] = await Promise.all([import('./penalty.js'),import('./penalty.css')]);
    return { title:'点球', html:'<section class="play-page"><a class="play-back" href="#/">← 返回主页</a><div class="play-heading"><h1>点球<span>.</span></h1><span>INTERACTIVE / 01</span></div>'+game.renderPenalty()+'</section>', init:game.initPenalty };
  }
  if (page === 'lab' && parts.length === 1) return { title:'实验室', html:renderLab(), init:initLab };
  return { title:'页面不存在', html:notFound() };
}
async function navigate({ force = false } = {}) {
  const parts = (location.hash.replace(/^#/, '') || '/').split('?')[0].split('/').filter(Boolean);
  const key = '/' + parts.join('/');
  if (!force && pendingRoute === key) return;
  const current = ++navigation;
  pendingRoute = key;
  stopContentMotion();
  closeMenu();
  updateNavigation(parts[0] || 'home');
  if (!force && renderedRoute === key) {
    pendingRoute = null;
    main.setAttribute('aria-busy','false');
    return;
  }
  main.setAttribute('aria-busy','true');
  if (!renderedRoute) main.innerHTML = '<div class="reader-loading">正在加载…</div>';
  let view;
  try { view = await prepareView(parts); }
  catch { view = loadingFailure(); }
  if (current !== navigation) return;
  const animate = renderedRoute !== null && !matchMedia('(prefers-reduced-motion: reduce)').matches && typeof main.animate === 'function';
  if (animate) {
    contentMotion = main.animate([{opacity:1},{opacity:0}], {duration:70,easing:'ease-out',fill:'forwards'});
    try { await contentMotion.finished; } catch { /* A newer navigation cancels this transition. */ }
  }
  if (current !== navigation) return;
  cleanup();
  cleanup = () => {};
  stopContentMotion();
  main.innerHTML = view.html;
  window.scrollTo({top:0,behavior:'instant'});
  document.title = view.title + ' · MUFC';
  cleanup = view.init?.() || (() => {});
  syncSavedButtons();
  renderedRoute = key;
  pendingRoute = null;
  main.setAttribute('aria-busy','false');
  if (current > 1) main.focus({preventScroll:true});
  if (animate) {
    const entering = main.animate([{opacity:0},{opacity:1}], {duration:160,easing:'cubic-bezier(.2,.65,.3,1)'});
    contentMotion = entering;
    entering.finished.then(() => {
      if (contentMotion === entering) { entering.cancel(); contentMotion = null; }
    }).catch(() => {});
  }
}
window.addEventListener('resize',positionNavIndicator,{passive:true});
if (typeof ResizeObserver === 'function') new ResizeObserver(positionNavIndicator).observe(desktopNav);
document.fonts?.ready.then(positionNavIndicator);
requestAnimationFrame(() => desktopNav.classList.add('nav-ready'));

function themeButton() {
  const dark=document.documentElement.dataset.theme==='dark';
  document.querySelector('.theme-toggle').innerHTML=icon(dark?'sun':'moon',19);
  document.querySelector('.theme-toggle').setAttribute('aria-label',dark?'切换浅色模式':'切换深色模式');
  document.querySelector('meta[name="theme-color"]').content=dark?'#191b1a':'#f6f5f2';
}
themeButton();
document.querySelector('.theme-toggle').addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;try{localStorage.setItem('mufc-theme',theme);}catch{}themeButton();});
document.querySelector('.menu-toggle').addEventListener('click',()=>{const nav=document.querySelector('.mobile-nav');nav.hidden=!nav.hidden;const b=document.querySelector('.menu-toggle');b.setAttribute('aria-expanded',String(!nav.hidden));b.setAttribute('aria-label',nav.hidden?'打开导航菜单':'关闭导航菜单');});
document.addEventListener('click',event=>{
  const target=event.target.closest('button,a'); if(!target)return;
  if(target.matches('.random-note')) location.hash=routeHref(notes[Math.floor(Math.random()*notes.length)].id);
  if(target.matches('[data-save-note]')) {const id=target.dataset.saveNote;if(!notes.some(n=>n.id===id))return;saved=saved.includes(id)?saved.filter(x=>x!==id):[...saved,id];write('mufc-saved-notes',saved);syncSavedButtons();toast(saved.includes(id)?'已收藏':'已取消收藏');}
  if(target.matches('[data-retry-reader]'))navigate({force:true});
  if(target.matches('.skip-link')) {event.preventDefault();main.focus();main.scrollIntoView();}
  if(target.matches('a[href^="#/"]')){document.querySelector('.mobile-nav').hidden=true;document.querySelector('.menu-toggle').setAttribute('aria-expanded','false');document.querySelector('.menu-toggle').setAttribute('aria-label','打开导航菜单');if(target.getAttribute('href')===(location.hash||'#/')){event.preventDefault();dialog.close();window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}}
});
window.addEventListener('hashchange',()=>{dialog.close();navigate();});
const topButton=document.querySelector('.back-top');
window.addEventListener('scroll',()=>{topButton.hidden=window.scrollY<650;},{passive:true});
topButton.addEventListener('click',()=>window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'}));
window.addEventListener('storage',event=>{if(event.key==='mufc-saved-notes'||event.key===null){const value=read('mufc-saved-notes',[]);saved=Array.isArray(value)?value.filter(id=>notes.some(n=>n.id===id)):[];syncSavedButtons();}});

const searchEntries=[
  ...[{title:'主页',url:'#/',subtitle:'个人主页'},{title:'作品集',url:'#/works',subtitle:'个人项目'},{title:'研究笔记',url:'#/notes',subtitle:'图像生成学习笔记'},{title:'专注计时',url:'#/lab',subtitle:'25分钟专注计时与点云工具'},{title:'点球',url:'#/play/penalty',subtitle:'点球与守门员扑救模拟'}].map(n=>({...n,type:'页面',terms:''})),
  ...projects.map(p=>({title:p.name,url:`#/work/${p.id}`,subtitle:p.label,type:'作品',terms:p.type+' '+p.detail})),
  ...notes.map(n=>({title:n.title,url:routeHref(n.id),subtitle:n.subtitle,type:n.category,terms:n.tags.join(' ')})),
];
function renderSearch() {
  const query=searchInput.value.trim().toLowerCase();
  const results=searchEntries.filter(x=>`${x.title} ${x.subtitle} ${x.terms}`.toLowerCase().includes(query)).slice(0,10);
  searchIndex=-1;
  document.querySelector('#search-results').innerHTML=`<div class="search-result-label">${query?`找到 ${results.length}${results.length===10?' +':''} 个结果`:'页面与笔记'}</div>${results.length?results.map(r=>`<a class="search-result" href="${r.url}"><span class="search-result-icon">${icon(r.type==='作品'?'code':r.type==='页面'?'arrow':'book',18)}</span><span><strong>${esc(r.title)}</strong><small>${esc(r.subtitle)}</small></span><em>${esc(r.type)}</em>${icon('arrow',17)}</a>`).join(''):`<div class="search-empty">没有找到“${esc(searchInput.value.trim())}”<br><small>试试 Diffusion、钢琴、Flow 或专注。</small></div>`}`;
}
function openSearch() {if(dialog.open)return;lastFocus=document.activeElement;searchInput.value='';renderSearch();dialog.showModal();searchInput.focus();}
document.querySelector('[data-open-search]').addEventListener('click',openSearch);
document.querySelector('[data-close-search]').addEventListener('click',()=>dialog.close());
dialog.addEventListener('close',()=>lastFocus?.focus({preventScroll:true}));
dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();}if(event.target.closest('.search-result'))dialog.close();});
searchInput.addEventListener('input',renderSearch);
dialog.addEventListener('keydown',event=>{
  const links=[...dialog.querySelectorAll('.search-result')];
  if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();if(!links.length)return;searchIndex=searchIndex<0?(event.key==='ArrowDown'?0:links.length-1):(searchIndex+(event.key==='ArrowDown'?1:-1)+links.length)%links.length;links.forEach((l,i)=>l.classList.toggle('selected',i===searchIndex));links[searchIndex].focus({preventScroll:true});links[searchIndex].scrollIntoView({block:'nearest'});}
  if(event.key==='Enter'&&document.activeElement===searchInput){event.preventDefault();links[Math.max(0,searchIndex)]?.click();}
});
document.querySelector('#search-results').addEventListener('focusin',event=>{const link=event.target.closest('.search-result');if(link){const links=[...dialog.querySelectorAll('.search-result')];searchIndex=links.indexOf(link);links.forEach(l=>l.classList.toggle('selected',l===link));}});
document.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'){event.preventDefault();dialog.open?dialog.close():openSearch();}if(event.key==='Escape'){if(dialog.open){event.preventDefault();dialog.close();}document.querySelector('.mobile-nav').hidden=true;document.querySelector('.menu-toggle').setAttribute('aria-expanded','false');document.querySelector('.menu-toggle').setAttribute('aria-label','打开导航菜单');}});
navigate();
