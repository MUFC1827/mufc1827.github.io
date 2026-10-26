const TIMER_KEY = 'mufc:focus:v1';
const DURATIONS = { focus: 25 * 60, break: 5 * 60 };

const arrow = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M7 17 17 7M7 7h10v10"/></svg>';

export function renderLab() {
  return `<div class="lab-page">
    <header class="lab-heading">
      <p class="lab-eyebrow">EXPERIMENTS</p>
      <h1>实验室<span>。</span></h1>
      <p class="lab-intro">生成艺术与专注计时。</p>
    </header>

    <div class="lab-grid">
      <section class="lab-art" aria-labelledby="lab-art-title">
        <div class="lab-section-top"><span class="lab-eyebrow">01 / GENERATIVE ART</span><span class="lab-dot-tag">可交互</span></div>
        <h2 id="lab-art-title">点云</h2>
        <p class="lab-description">拖动滑块调整有序程度。</p>
        <div class="lab-canvas-wrap">
          <canvas id="lab-field" role="img" aria-label="交互式点云：拖动下方滑块，观察随机散点逐渐组成有序波形。">你的浏览器暂不支持画布，可使用下方专注计时器。</canvas>
          <div class="lab-canvas-caption" aria-hidden="true"><span>MUFC / FIELD STUDY</span><span id="lab-seed-label">SEED 1827</span></div>
        </div>
        <div class="lab-slider-label"><label for="lab-order">随机 → 有序</label><output id="lab-order-value" for="lab-order">45%</output></div>
        <input id="lab-order" class="lab-range" type="range" min="0" max="100" value="45" step="1" aria-label="点云有序程度">
        <div class="lab-art-actions">
          <button class="lab-button lab-button-primary" id="lab-play" type="button"><span class="lab-play-icon" aria-hidden="true">▷</span><span>播放变化</span></button>
          <button class="lab-button" id="lab-reseed" type="button"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 7a7 7 0 0 1 11.5-1L20 9M4 15l2.4 3A7 7 0 0 0 18 17"/></svg>重新生成</button>
          <button class="lab-icon-button" id="lab-export" type="button" aria-label="将当前点云保存为 PNG 图片" title="保存图片"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M5 15v5h14v-5"/></svg></button>
        </div>
        <p id="lab-art-status" class="lab-sr-only" role="status"></p>
      </section>

      <section class="lab-focus" aria-labelledby="lab-focus-title">
        <div class="lab-section-top"><span class="lab-eyebrow">02 / FOCUS TIMER</span><svg class="lab-spark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.25" aria-hidden="true"><path d="M12 2v20M2 12h20M5 5l14 14M5 19 19 5"/></svg></div>
        <h2 id="lab-focus-title">专注计时</h2>
        <div class="lab-timer-modes" role="group" aria-label="选择计时模式">
          <button type="button" id="lab-mode-focus" aria-pressed="true">专注 25 分钟</button>
          <button type="button" id="lab-mode-break" aria-pressed="false">休息 5 分钟</button>
        </div>
        <div class="lab-clock">
          <svg class="lab-timer-ring" viewBox="0 0 220 220" aria-hidden="true"><circle class="lab-ring-track" cx="110" cy="110" r="100"/><circle id="lab-ring-progress" cx="110" cy="110" r="100" pathLength="100" stroke-dasharray="100" stroke-dashoffset="0"/></svg>
          <div class="lab-clock-content"><span class="lab-clock-mode" id="lab-clock-mode">专注时间</span><span class="lab-clock-time" id="lab-time" role="timer" aria-label="剩余 25 分钟" aria-live="off">25:00</span><span class="lab-clock-hint" id="lab-clock-hint">未开始</span></div>
        </div>
        <div class="lab-timer-actions"><button id="lab-timer-toggle" class="lab-button lab-button-primary" type="button">开始专注 ${arrow}</button><button id="lab-timer-reset" class="lab-icon-button" type="button" aria-label="重置当前计时" title="重置计时"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M4 4v6h6M4.5 10A8 8 0 1 1 4 15"/></svg></button></div>
        <p class="lab-timer-note">离开页面后继续计时。</p>
        <p id="lab-timer-status" class="lab-sr-only" role="status"></p>
      </section>
    </div>
  </div>`;
}

function defaultTimer(mode = 'focus') {
  return { mode, remaining: DURATIONS[mode], endAt: null, running: false, finished: false };
}

function loadTimer() {
  try {
    const saved = JSON.parse(localStorage.getItem(TIMER_KEY));
    if (!saved || !Object.hasOwn(DURATIONS, saved.mode)) return defaultTimer();
    if (typeof saved.running !== 'boolean' || !Number.isFinite(saved.remaining) || saved.remaining < 0 || saved.remaining > DURATIONS[saved.mode]) return defaultTimer();
    if (saved.running && (!Number.isFinite(saved.endAt) || saved.endAt > Date.now() + DURATIONS[saved.mode] * 1000 + 1000)) return defaultTimer();
    return { mode: saved.mode, remaining: saved.remaining, endAt: saved.running ? saved.endAt : null, running: saved.running, finished: Boolean(saved.finished) };
  } catch {
    return defaultTimer();
  }
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function initLab() {
  const page = document.querySelector('.lab-page');
  if (!page) return () => {};
  const $ = (selector) => page.querySelector(selector);
  const removers = [];
  const listen = (target, event, handler, options) => {
    target.addEventListener(event, handler, options);
    removers.push(() => target.removeEventListener(event, handler, options));
  };
  let disposed = false;

  // The deadline, rather than interval ticks, is the source of timer truth.
  let timer = loadTimer();
  const time = $('#lab-time');
  const timerStatus = $('#lab-timer-status');
  const toggle = $('#lab-timer-toggle');
  function saveTimer() {
    try { localStorage.setItem(TIMER_KEY, JSON.stringify(timer)); } catch { /* Works for this visit when storage is unavailable. */ }
  }
  function remainingSeconds() {
    return timer.running ? Math.max(0, Math.ceil((timer.endAt - Date.now()) / 1000)) : timer.remaining;
  }
  function updateTimer() {
    const remaining = remainingSeconds();
    if (timer.running && remaining <= 0) {
      timer = { ...timer, running: false, endAt: null, remaining: 0, finished: true };
      saveTimer();
      timerStatus.textContent = timer.mode === 'focus' ? '专注完成。' : '休息结束。';
    }
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;
    const label = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    if (time.textContent !== label) time.textContent = label;
    time.setAttribute('aria-label', `剩余 ${minutes} 分 ${seconds} 秒`);
    $('#lab-clock-mode').textContent = timer.mode === 'focus' ? '专注时间' : '休息时间';
    $('#lab-clock-hint').textContent = timer.finished ? (timer.mode === 'focus' ? '专注完成' : '休息结束') : timer.running ? (timer.mode === 'focus' ? '专注中' : '休息中') : remaining < DURATIONS[timer.mode] ? '已暂停' : '未开始';
    $('#lab-ring-progress').style.strokeDashoffset = String(100 * (1 - remaining / DURATIONS[timer.mode]));
    $('#lab-mode-focus').setAttribute('aria-pressed', String(timer.mode === 'focus'));
    $('#lab-mode-break').setAttribute('aria-pressed', String(timer.mode === 'break'));
    const buttonText = timer.running ? '暂停计时' : timer.finished ? '重新开始' : remaining < DURATIONS[timer.mode] ? '继续计时' : timer.mode === 'focus' ? '开始专注' : '开始休息';
    if (toggle.dataset.label !== buttonText) {
      toggle.dataset.label = buttonText;
      toggle.innerHTML = `${buttonText} ${arrow}`;
    }
  }
  listen(toggle, 'click', () => {
    if (timer.running) {
      timer.remaining = remainingSeconds();
      timer.running = false;
      timer.endAt = null;
      if (timer.remaining === 0) timer.finished = true;
      timerStatus.textContent = '计时已暂停。';
    } else {
      if (timer.finished || timer.remaining <= 0) timer = defaultTimer(timer.mode);
      timer.endAt = Date.now() + timer.remaining * 1000;
      timer.running = true;
      timerStatus.textContent = timer.mode === 'focus' ? '专注计时开始。' : '休息计时开始。';
    }
    saveTimer();
    updateTimer();
  });
  listen($('#lab-timer-reset'), 'click', () => {
    timer = defaultTimer(timer.mode);
    saveTimer();
    updateTimer();
    timerStatus.textContent = '计时已重置。';
  });
  for (const mode of ['focus', 'break']) {
    listen($(`#lab-mode-${mode}`), 'click', () => {
      if (timer.mode === mode) return;
      timer = defaultTimer(mode);
      saveTimer();
      updateTimer();
      timerStatus.textContent = `已切换到${mode === 'focus' ? '专注' : '休息'}模式。`;
    });
  }
  listen(window, 'storage', (event) => {
    if (event.key === TIMER_KEY || event.key === null) { timer = loadTimer(); updateTimer(); }
  });
  listen(document, 'visibilitychange', () => { if (!document.hidden) updateTimer(); });
  const interval = window.setInterval(updateTimer, 250);
  updateTimer();

  const canvas = $('#lab-field');
  const context = canvas.getContext('2d');
  const slider = $('#lab-order');
  const play = $('#lab-play');
  const artStatus = $('#lab-art-status');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let order = 45;
  let seed = 1827;
  let points = [];
  let animation = 0;
  let playing = false;
  let startAt = 0;
  let startOrder = 0;

  function makePoints() {
    const random = seededRandom(seed);
    points = Array.from({ length: 1080 }, (_, index) => ({
      x: random(), y: random(), u: (index % 90) / 89,
      v: Math.floor(index / 90) / 11,
      radius: 0.65 + random() * 1.0, phase: random() * Math.PI * 2,
      color: index % 4, opacity: 0.35 + random() * 0.6,
    }));
  }
  function draw() {
    if (!context || disposed) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const styles = getComputedStyle(canvas);
    const background = styles.getPropertyValue('--surface').trim() || '#faf9f6';
    const ink = styles.getPropertyValue('--muted').trim() || '#77736e';
    const accent = styles.getPropertyValue('--accent').trim() || '#c52d36';
    context.globalAlpha = 1;
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);
    context.strokeStyle = ink;
    context.lineWidth = 0.5;
    context.globalAlpha = 0.12;
    for (let x = 24; x < width; x += 32) {
      for (let y = 24; y < height; y += 32) {
        context.beginPath(); context.moveTo(x - 2, y); context.lineTo(x + 2, y); context.moveTo(x, y - 2); context.lineTo(x, y + 2); context.stroke();
      }
    }
    const t = order / 100;
    const ease = t * t * (3 - 2 * t);
    const colors = [accent, accent, '#bc826a', ink];
    for (const point of points) {
      const targetX = width * (0.10 + point.u * 0.8);
      const wave = Math.sin(point.u * Math.PI * 2.8 + point.v * 1.8);
      const targetY = height * (0.50 + wave * 0.20 + (point.v - 0.5) * 0.29);
      const x = point.x * width * (1 - ease) + targetX * ease;
      const y = point.y * height * (1 - ease) + targetY * ease;
      context.fillStyle = colors[point.color];
      context.globalAlpha = point.opacity;
      context.beginPath(); context.arc(x, y, point.radius * Math.max(0.8, width / 650), 0, Math.PI * 2); context.fill();
    }
    context.globalAlpha = 1;
  }
  function setOrder(value) {
    order = Math.max(0, Math.min(100, value));
    slider.value = String(Math.round(order));
    $('#lab-order-value').textContent = `${Math.round(order)}%`;
    draw();
  }
  function updatePlayButton() {
    play.innerHTML = reducedMotion.matches ? '<span aria-hidden="true">↗</span><span>显示结果</span>' : playing ? '<span aria-hidden="true">Ⅱ</span><span>暂停变化</span>' : '<span class="lab-play-icon" aria-hidden="true">▷</span><span>播放变化</span>';
    play.setAttribute('aria-pressed', String(playing));
  }
  function stopAnimation() {
    window.cancelAnimationFrame(animation);
    animation = 0;
    playing = false;
    updatePlayButton();
  }
  function animate(now) {
    if (disposed || !playing) return;
    const progress = Math.min(1, (now - startAt) / 6000);
    setOrder(startOrder + (100 - startOrder) * progress);
    if (progress < 1) animation = window.requestAnimationFrame(animate);
    else { stopAnimation(); artStatus.textContent = '点云已变为有序波形。'; }
  }
  listen(slider, 'input', () => { stopAnimation(); setOrder(Number(slider.value)); });
  listen(play, 'click', () => {
    if (reducedMotion.matches) { stopAnimation(); setOrder(100); artStatus.textContent = '已显示有序波形。'; return; }
    if (playing) { stopAnimation(); return; }
    if (order >= 99.5) setOrder(0);
    playing = true;
    startAt = performance.now();
    startOrder = order;
    updatePlayButton();
    animation = window.requestAnimationFrame(animate);
  });
  listen($('#lab-reseed'), 'click', () => {
    stopAnimation();
    seed = Math.floor(Math.random() * 90000) + 10000;
    $('#lab-seed-label').textContent = `SEED ${seed}`;
    makePoints(); draw();
    artStatus.textContent = '已重新生成点云。';
  });
  listen($('#lab-export'), 'click', () => {
    if (!context) return;
    try {
      const link = document.createElement('a');
      link.download = `mufc-field-${seed}-${Math.round(order)}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
      artStatus.textContent = '图片已准备好下载。';
    } catch { artStatus.textContent = '暂时无法保存图片，请再试一次。'; }
  });
  listen(reducedMotion, 'change', () => { stopAnimation(); draw(); });
  let resizeObserver;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(draw);
    resizeObserver.observe(canvas);
  } else listen(window, 'resize', draw);
  const themeObserver = new MutationObserver(draw);
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'style', 'data-theme'] });
  if (document.body) themeObserver.observe(document.body, { attributes: true, attributeFilter: ['class', 'style', 'data-theme'] });
  makePoints(); updatePlayButton(); draw();
  if (!context) {
    artStatus.textContent = '浏览器暂不支持画布。专注计时器仍可使用。';
    for (const element of [slider, play, $('#lab-reseed'), $('#lab-export')]) element.disabled = true;
  }

  return () => {
    disposed = true;
    window.clearInterval(interval);
    window.cancelAnimationFrame(animation);
    resizeObserver?.disconnect();
    themeObserver.disconnect();
    for (const remove of removers) remove();
  };
}
