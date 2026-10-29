import { icon, escapeHTML as esc } from './icons.js';
import { renderStudyMap, initStudyMap } from './study-map.js';
import './study-map.css';
import './home.css';

const external = 'target="_blank" rel="noopener noreferrer"';

export function renderProjectPreview(project, { eager = false } = {}) {
  if (project.id === 'memo') {
    return `<div class="project-preview project-preview-memo"><div class="preview-browser"><span></span><span></span><span></span><small>MemoUnited</small><em>界面示意</em></div><div class="memo-demo" aria-label="MemoUnited 功能示意"><div class="memo-demo-rail"><b>m<span>·</span></b><i class="active"></i><i></i><i></i></div><div class="memo-demo-content"><div class="memo-demo-tabs"><strong>Memo</strong><span>Muse</span><span>＋</span></div><div class="memo-demo-note"><span>文字速记</span><i></i><i></i><i></i></div><div class="memo-demo-tasks"><span>待办事项</span><div><b></b><i></i></div><div><b></b><i></i></div></div><div class="memo-demo-files"><span>${icon('book', 13)} 附件内联预览</span><span>${icon('check', 13)} 本地离线存储</span></div></div></div></div>`;
  }
  return `<div class="project-preview project-preview-${project.id}"><div class="preview-browser" aria-hidden="true"><span></span><span></span><span></span><small>${new URL(project.url).hostname}</small>${icon('external', 12)}</div><img src="${project.image}" alt="${project.alt}" loading="${eager ? 'eager' : 'lazy'}" decoding="async" width="1440" height="900"></div>`;
}

function renderHeroDiagram() {
  return `<svg class="hero-flow-svg" viewBox="0 0 460 210" role="img" aria-label="从随机噪声分布经速度场流向结构化表征的示意图">
    <defs>
      <linearGradient id="flowLineGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="var(--muted)" stop-opacity=".28"/>
        <stop offset="52%" stop-color="var(--accent)" stop-opacity=".75"/>
        <stop offset="100%" stop-color="var(--accent)" stop-opacity="1"/>
      </linearGradient>
      <radialGradient id="flowSpot" cx="78%" cy="50%" r="36%">
        <stop offset="0%" stop-color="var(--accent)" stop-opacity=".14"/>
        <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <rect x="0" y="0" width="460" height="210" fill="url(#flowSpot)"/>
    <!-- Source distribution z0 -->
    <g class="hero-flow-source">
      <circle cx="58" cy="105" r="42" fill="none" stroke="var(--border-strong)" stroke-dasharray="3 4"/>
      <circle cx="44" cy="84" r="2.2" fill="var(--muted)"/>
      <circle cx="71" cy="79" r="1.8" fill="var(--muted)"/>
      <circle cx="38" cy="112" r="2.5" fill="var(--muted)"/>
      <circle cx="62" cy="102" r="2.2" fill="var(--text-2)"/>
      <circle cx="78" cy="116" r="1.9" fill="var(--muted)"/>
      <circle cx="51" cy="129" r="2.1" fill="var(--muted)"/>
      <circle cx="54" cy="95" r="1.6" fill="var(--accent)"/>
      <text x="58" y="166" text-anchor="middle" class="hero-svg-caption">NOISE z₀ ~ N(0, I)</text>
    </g>
    <!-- Flow trajectories -->
    <g fill="none" stroke="url(#flowLineGrad)" stroke-width="1.35" stroke-linecap="round">
      <path d="M74 80 C 165 62, 255 68, 356 76"/>
      <path d="M64 96 C 168 92, 260 96, 356 96" stroke-width="1.7"/>
      <path d="M80 116 C 175 124, 260 118, 356 116"/>
      <path d="M54 128 C 162 152, 258 140, 356 134" stroke-dasharray="4 4"/>
    </g>
    <!-- Midpoint velocity vectors -->
    <g class="hero-flow-mid">
      <rect x="192" y="48" width="78" height="106" rx="8" fill="var(--surface)" fill-opacity=".86" stroke="var(--border)"/>
      <line x1="208" y1="74" x2="248" y2="74" stroke="var(--accent)" stroke-width="1.5"/>
      <path d="m244 70 5 4-5 4" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
      <line x1="208" y1="101" x2="252" y2="101" stroke="var(--text)" stroke-width="1.5"/>
      <path d="m248 97 5 4-5 4" fill="none" stroke="var(--text)" stroke-width="1.5"/>
      <line x1="208" y1="128" x2="246" y2="128" stroke="var(--accent)" stroke-width="1.5"/>
      <path d="m242 124 5 4-5 4" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
      <text x="231" y="170" text-anchor="middle" class="hero-svg-caption">VELOCITY v_θ(z_t, t)</text>
    </g>
    <!-- Target structured representation z1 -->
    <g class="hero-flow-target">
      <rect x="356" y="58" width="68" height="94" rx="10" fill="var(--surface)" stroke="var(--accent)" stroke-width="1.4"/>
      <rect x="368" y="72" width="44" height="32" rx="4" fill="var(--accent-soft)" stroke="var(--accent)" stroke-opacity=".35"/>
      <circle cx="381" cy="83" r="3" fill="var(--accent)"/>
      <path d="m372 99 11-10 8 7 6-5 11 8" fill="none" stroke="var(--accent)" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
      <line x1="368" y1="118" x2="412" y2="118" stroke="var(--border-strong)" stroke-width="2" stroke-linecap="round"/>
      <line x1="368" y1="128" x2="398" y2="128" stroke="var(--border)" stroke-width="2" stroke-linecap="round"/>
      <line x1="368" y1="138" x2="405" y2="138" stroke="var(--border)" stroke-width="2" stroke-linecap="round"/>
      <text x="390" y="168" text-anchor="middle" class="hero-svg-caption">DATA / IMAGE z₁</text>
    </g>
  </svg>`;
}

function projectCard(project, index) {
  return `<article class="home-project home-project-${project.id}" data-reveal style="--i:${index}">
    <a class="home-project-visual stage-${project.id}" href="#/work/${project.id}" aria-label="查看 ${project.name} 项目详情">${renderProjectPreview(project)}</a>
    <div class="home-project-body">
      <div class="home-project-meta"><span><strong class="accent">${project.number}</strong> / ${project.category}</span><span class="home-project-badge">${project.badge}</span></div>
      <div class="home-project-name"><h3><a href="#/work/${project.id}">${project.name}</a></h3></div>
      <p>${project.description}</p>
      <div class="home-project-footer">
        <span class="home-project-details">${project.spec}</span>
        <div class="home-project-links">
          <a class="text-link" href="#/work/${project.id}">项目详情 ${icon('arrow', 15)}</a>
          <a class="button small secondary" href="${project.download || project.url}" ${external}>${project.download ? '下载应用' : project.action} ${icon('external', 14)}</a>
        </div>
      </div>
    </div>
  </article>`;
}

export function renderHome(notes, projects) {
  let lastNote;
  try { lastNote = notes.find(note => note.id === localStorage.getItem('mufc-last-read')); } catch {}
  const featuredNote = lastNote || notes.find(n => n.id === 'generative-models-map') || notes[0];
  const isResume = Boolean(lastNote);

  return `<div class="home-page">
    <section class="landing-hero" aria-labelledby="home-title">
      <div class="landing-intro" data-reveal style="--i:0">
        <span class="hero-kicker"><i class="hero-dot" aria-hidden="true"></i>INDEPENDENT DEVELOPER · AIGC RESEARCH</span>
        <h1 id="home-title">MUFC<span>1827</span><b>.</b></h1>
        <p class="landing-subtitle">个人项目与生成式 AI 研究笔记</p>
        <p class="landing-lead">围绕 <strong>Attention、Diffusion 与 Rectified Flow</strong> 整理系统化的论文推导与实验记录；同时独立开发轻量、趁手的音乐、桌游与桌面效率工具。</p>
        <div class="landing-links">
          <a class="button primary" href="#/works">浏览作品集 ${icon('arrow', 16)}</a>
          <a class="button secondary" href="#/notes">阅读研究笔记 ${icon('arrow', 16)}</a>
        </div>
        <dl class="landing-stats" aria-label="站点内容概览">
          <div><dt>个人项目</dt><dd>${String(projects.length).padStart(2, '0')}</dd></div>
          <div><dt>研究笔记</dt><dd>${String(notes.length).padStart(2, '0')}</dd></div>
          <div><dt>核心专题</dt><dd>04</dd></div>
          <div><dt>交互实验</dt><dd>02</dd></div>
        </dl>
      </div>

      <aside class="landing-visual" aria-label="研究主线与快速入口" data-reveal style="--i:1">
        <div class="hero-card">
          <div class="hero-card-header">
            <span class="hero-card-tag">FIG. 01 — GENERATIVE FLOW</span>
            <span class="hero-card-seed">SEED · 1827</span>
          </div>
          <div class="hero-card-stage">
            ${renderHeroDiagram()}
          </div>
          <div class="hero-card-paths" aria-label="四大研究专题">
            <a href="#home-research" data-jump-tab="attention"><span>01</span><strong>Attention 基础</strong><small>5 篇</small></a>
            <a href="#home-research" data-jump-tab="architecture"><span>02</span><strong>生成架构</strong><small>4 篇</small></a>
            <a href="#home-research" data-jump-tab="flow"><span>03</span><strong>Flow 方法</strong><small>4 篇</small></a>
            <a href="#home-research" data-jump-tab="editing"><span>04</span><strong>编辑与控制</strong><small>4 篇</small></a>
          </div>
          ${featuredNote ? `<a class="hero-card-resume" href="#/notes/${featuredNote.id}">
            <span class="hero-resume-badge">${icon('book', 14)} ${isResume ? '继续上次阅读' : '核心知识总览'}</span>
            <strong>${esc(featuredNote.title)}</strong>
            <span class="hero-resume-go">${icon('arrow', 15)}</span>
          </a>` : ''}
        </div>
      </aside>
    </section>

    <section id="home-projects" class="home-section" aria-labelledby="home-project-title">
      <div class="home-section-heading" data-reveal>
        <div>
          <span class="eyebrow"><span class="accent">01</span> — SELECTED WORKS</span>
          <h2 id="home-project-title">项目选集<span>.</span></h2>
          <p class="home-section-description">独立设计与实现的 Web 交互应用与桌面效率工具。</p>
        </div>
        <a class="text-link" href="#/works">查看全部作品 ${icon('arrow', 16)}</a>
      </div>
      <div class="home-projects-grid">${projects.map(projectCard).join('')}</div>
    </section>

    <section id="home-research" class="home-section home-research" aria-labelledby="home-research-title">
      <div class="home-section-heading" data-reveal>
        <div>
          <span class="eyebrow"><span class="accent">02</span> — RESEARCH PATHS</span>
          <h2 id="home-research-title">研究路径<span>.</span></h2>
          <p class="home-section-description">从注意力机制到潜空间扩散、Rectified Flow，再到可控图像编辑。</p>
        </div>
        <a class="text-link" href="#/notes">全部 ${notes.length} 篇笔记 ${icon('arrow', 16)}</a>
      </div>
      <div data-reveal>${renderStudyMap(notes)}</div>
    </section>

    <section id="home-interactive" class="home-section home-interactive" aria-labelledby="home-interactive-title">
      <div class="home-section-heading" data-reveal>
        <div>
          <span class="eyebrow"><span class="accent">03</span> — INTERACTIVE & LAB</span>
          <h2 id="home-interactive-title">互动与实验<span>.</span></h2>
          <p class="home-section-description">物理模拟小游戏、生成式点云演化与专注计时工具。</p>
        </div>
      </div>
      <div class="home-interactive-grid">
        <a class="interactive-card interactive-card-penalty" href="#/play/penalty" data-reveal style="--i:0">
          <div class="interactive-card-top">
            <span class="eyebrow"><span class="accent">PLAY</span> / STRETFORD END</span>
            <span class="chip">120 Hz 物理模拟</span>
          </div>
          <h3>点球模拟<span>.</span></h3>
          <p>支持推射、大力与弧线三种脚法，实时计算重力、空气阻力、马格努斯旋转、门框圆柱碰撞与守门员延迟预判扑救。</p>
          <span class="text-link">进入球场挑战 ${icon('arrow', 16)}</span>
        </a>
        <a class="interactive-card interactive-card-lab" href="#/lab" data-reveal style="--i:1">
          <div class="interactive-card-top">
            <span class="eyebrow"><span class="accent">LAB</span> / GENERATIVE & FOCUS</span>
            <span class="chip">点云 · 番茄钟</span>
          </div>
          <h3>实验室与专注计时<span>.</span></h3>
          <p>交互式观察随机散点向有序波形的连续过渡（呼应扩散与流模型的生成直觉），并内置 25 分钟沉浸式专注计时器。</p>
          <span class="text-link">打开实验室 ${icon('arrow', 16)}</span>
        </a>
      </div>
    </section>
  </div>`;
}

export function initHome() {
  const cleanStudy = initStudyMap();
  const jumpLinks = [...document.querySelectorAll('[data-jump-tab]')];
  const onJump = event => {
    const link = event.currentTarget;
    const tabId = link.dataset.jumpTab;
    const targetTab = document.getElementById(`study-tab-${tabId}`);
    const researchSection = document.getElementById('home-research');
    if (targetTab && researchSection) {
      event.preventDefault();
      targetTab.click();
      researchSection.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    }
  };
  jumpLinks.forEach(link => link.addEventListener('click', onJump));
  return () => {
    cleanStudy?.();
    jumpLinks.forEach(link => link.removeEventListener('click', onJump));
  };
}
