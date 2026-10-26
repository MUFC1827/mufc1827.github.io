import { icon, escapeHTML as esc } from './icons.js';
import { renderStudyMap, initStudyMap } from './study-map.js';
import './study-map.css';
import './home.css';

const external = 'target="_blank" rel="noopener noreferrer"';

const projectCopy = {
  piano: { category: '音乐', description: '88 键、真实钢琴采样与电脑键盘演奏。', detail: '延音 / 和弦识别 / 独立八度', image: './images/piano-preview.png', alt: 'Piano MUFC 在线钢琴的实际界面' },
  revolver: { category: '桌游', description: '为线下桌游制作的虚拟轮盘。', detail: '自定义槽位 / 随机装填 / 音效', image: './images/revolver-preview.png', alt: 'REVOLVER 虚拟轮盘的实际界面' },
  memo: { category: '桌面工具', description: '文字、待办与附件，保存在本机。', detail: 'Windows / macOS' },
};

export function renderProjectPreview(project, { eager = false } = {}) {
  const copy = projectCopy[project.id];
  if (project.id === 'memo') return `<div class="project-preview project-preview-memo"><div class="preview-browser"><span></span><span></span><span></span><small>MemoUnited</small><em>界面示意</em></div><div class="memo-demo" aria-label="MemoUnited 功能示意"><div class="memo-demo-rail"><b>m<span>·</span></b><i class="active"></i><i></i><i></i></div><div class="memo-demo-content"><div class="memo-demo-tabs"><strong>Memo</strong><span>Muse</span><span>＋</span></div><div class="memo-demo-note"><span>文字</span><i></i><i></i><i></i></div><div class="memo-demo-tasks"><span>事项</span><div><b></b><i></i></div><div><b></b><i></i></div></div><div class="memo-demo-files"><span>${icon('book',14)} 附件</span><span>${icon('check',13)} 本地保存</span></div></div></div></div>`;
  return `<div class="project-preview project-preview-${project.id}"><div class="preview-browser" aria-hidden="true"><span></span><span></span><span></span><small>${project.id === 'memo' ? 'MemoUnited' : new URL(project.url).hostname}</small>${icon('external',12)}</div><img src="${copy.image}" alt="${copy.alt}" loading="${eager ? 'eager' : 'lazy'}" decoding="async" width="1440" height="900"></div>`;
}

function projectCard(project) {
  const copy = projectCopy[project.id];
  return `<article class="home-project home-project-${project.id}">
    <a class="home-project-visual" href="#/work/${project.id}" aria-label="查看 ${project.name} 项目介绍">${renderProjectPreview(project)}</a>
    <div class="home-project-body"><div class="home-project-meta"><span>${project.number} / ${copy.category}</span><span>${project.id === 'memo' ? 'DESKTOP' : 'WEB APP'}</span></div>
      <div class="home-project-name"><h3><a href="#/work/${project.id}">${project.name}</a></h3><a class="home-project-open" href="${project.url}" ${external} aria-label="${project.action}，新标签页">${icon('external',22)}</a></div>
      <p>${copy.description}</p><div class="home-project-details">${copy.detail}</div>
      ${project.download ? `<a class="home-download" href="${project.download}" ${external}>下载 ${icon('external',15)}</a>` : ''}
    </div>
  </article>`;
}

export function renderHome(notes, projects) {
  let lastNote;
  try { lastNote = notes.find(note => note.id === localStorage.getItem('mufc-last-read')); } catch {}
  return `<div class="home-page">
    <section class="landing-hero" aria-labelledby="home-title">
      <div class="landing-intro"><div class="landing-eyebrow"><span class="landing-rule" aria-hidden="true"></span>个人主页</div>
        <h1 id="home-title">MUFC<span>1827</span><b>.</b></h1>
        <p class="landing-subtitle">个人项目与研究笔记</p>
        <p class="landing-description">图像生成、独立开发与实验记录。</p>
        <div class="landing-links"><a class="button primary" href="#/works">查看作品 ${icon('arrow',17)}</a><a class="landing-notes-link" href="#/notes">阅读笔记 ${icon('external',16)}</a></div>
        <div class="landing-stats"><button data-home-scroll="home-projects" aria-label="浏览 ${projects.length} 个项目"><strong>${String(projects.length).padStart(2,'0')}</strong><span>个人项目 ${icon('arrow',14)}</span></button><button data-home-scroll="home-research" aria-label="浏览 ${notes.length} 篇研究笔记"><strong>${String(notes.length).padStart(2,'0')}</strong><span>研究笔记 ${icon('arrow',14)}</span></button></div>
      </div><button class="landing-scroll" data-home-scroll="home-projects" aria-label="向下浏览项目与研究内容">向下浏览 ${icon('arrow',14)}</button>
    </section>
    <div class="home-section-nav"><button data-home-scroll="home-projects"><span>01</span> 项目 ${icon('arrow',15)}</button><button data-home-scroll="home-research"><span>02</span> 研究 ${icon('arrow',15)}</button><a href="#/notes"><span>03</span> 全部笔记 ${icon('external',15)}</a><span class="home-section-note">PERSONAL INDEX</span></div>
    <section id="home-projects" class="home-section" aria-labelledby="home-project-title"><div class="home-section-heading"><div><span class="home-section-index">01 — PROJECTS</span><h2 id="home-project-title">项目选集<span>.</span></h2></div><a class="home-section-more" href="#/works">全部项目 ${icon('arrow',17)}</a></div>
      <div class="home-projects-grid">${projects.map(projectCard).join('')}</div>
    </section>
    <section id="home-research" class="home-section home-research" aria-labelledby="home-research-title"><div class="home-section-heading"><div><span class="home-section-index">02 — RESEARCH</span><h2 id="home-research-title">研究路径<span>.</span></h2></div><a class="home-section-more" href="#/notes">${notes.length} 篇笔记 ${icon('arrow',17)}</a></div><p class="home-section-description">从注意力机制到生成模型，再到图像编辑。</p>
      ${renderStudyMap(notes)}
    </section>
    ${lastNote ? `<a class="home-resume" href="#/notes/${lastNote.id}"><span class="home-resume-label">${icon('book',18)} 上次阅读</span><strong>${esc(lastNote.title)}</strong><span>继续 ${icon('arrow',17)}</span></a>` : ''}
    <div class="home-bottom-meta"><span>MUFC1827</span><span>PROJECTS / RESEARCH / FOOTBALL</span><a href="https://github.com/MUFC1827/mufc1827.github.io" ${external}>网站源码 ${icon('external',14)}</a></div>
  </div>`;
}

export function initHome() {
  const root = document.querySelector('.home-page');
  const cleanups = [initStudyMap()];
  const onClick = event => {
    const trigger = event.target.closest('[data-home-scroll]');
    if (!trigger) return;
    const section = document.getElementById(trigger.dataset.homeScroll);
    section?.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
  };
  root.addEventListener('click', onClick);
  return () => { root.removeEventListener('click', onClick); cleanups.forEach(cleanup => cleanup?.()); };
}
