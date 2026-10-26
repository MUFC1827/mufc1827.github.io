export const projects = [
  { id:'piano', number:'01', name:'Piano MUFC', label:'浏览器钢琴', type:'音乐 · 交互体验', url:'https://piano-mufc.vercel.app/', action:'开始弹奏', description:'使用电脑键盘演奏的在线钢琴，覆盖 A0–C8 共 88 键。', detail:'支持真实钢琴采样、左右手独立八度、延音、和弦识别与入门教学。', features:['88 键与真实钢琴采样','电脑键盘演奏 · 独立八度','延音 · 和弦识别 · 入门教学'], color:'piano' },
  { id:'revolver', number:'02', name:'REVOLVER', label:'桌游虚拟轮盘', type:'桌游 · 聚会工具', url:'https://revolver-mufc.vercel.app/', action:'打开虚拟轮盘', description:'为《骗子酒馆》等线下桌游制作的虚拟轮盘。', detail:'支持 2–12 个槽位、自定义随机装填数量、音效与触感。', features:['2–12 槽位 · 自定义装填','随机轮盘 · 音效与触感','为线下聚会设计'], color:'revolver' },
  { id:'memo', number:'03', name:'MemoUnited', label:'桌面记录工具', type:'效率 · 桌面工具', url:'https://github.com/MUFC1827/MemoUnited-Updates', download:'https://github.com/MUFC1827/MemoUnited-Updates/releases/latest', action:'查看项目', description:'用于收集文字、待办与附件的桌面工具，支持 Windows 和 macOS。', detail:'Memo / Muse 双面板，支持快捷键呼出、附件预览与整体备份，数据保存在本机。', features:['Memo / Muse 双面板','快捷键呼出 · 附件预览','本地存储 · 整体备份'], color:'memo' },
];

export function projectArt(id) {
  if (id === 'piano') return `<div class="project-art piano-art" aria-hidden="true"><span class="art-caption">PIANO / 88 KEYS</span><div class="piano-keys">${Array.from({length:12},(_,i)=>`<i class="white-key ${i===5?'pressed':''}">${[0,1,3,4,5,7,8,10,11].includes(i)?'<b></b>':''}</i>`).join('')}</div><span class="piano-note">♪</span></div>`;
  if (id === 'revolver') return `<div class="project-art revolver-art" aria-hidden="true"><span class="art-caption">REVOLVER / TABLETOP</span><div class="cylinder"><div class="cylinder-core"></div>${Array.from({length:6},(_,i)=>`<i style="--i:${i}" class="${i===1?'red-slot':''}"></i>`).join('')}</div><span class="art-corner">MUFC / 02</span></div>`;
  return `<div class="project-art memo-art" aria-hidden="true"><span class="art-caption">MEMOUNITED / DESKTOP</span><div class="memo-paper back-paper"></div><div class="memo-paper front-paper"><span class="memo-symbol">m<span>·</span></span><i></i><i></i><i></i><b>notes & ideas</b></div><span class="memo-star">✳</span></div>`;
}

export function sphereArt() {
  const dots = [];
  for (let i=0;i<900;i++) {
    const y = 1 - i / 899 * 2, r = Math.sqrt(1-y*y), theta = i*2.39996323;
    const x = Math.cos(theta)*r, z = Math.sin(theta)*r;
    const xx = x*.88+z*.475, zz = -x*.475+z*.88;
    if (zz < -.2) continue;
    const red = Math.sin(y*7+xx*3) > -.18;
    dots.push(`<circle cx="${(210+xx*135).toFixed(1)}" cy="${(188+y*135).toFixed(1)}" r="${(1.1+(zz+1)*.65).toFixed(1)}" fill="${red?'var(--accent)':'var(--sphere-dot)'}" opacity="${(.27+(zz+1)*.33).toFixed(2)}"/>`);
  }
  return `<svg class="sphere-svg" viewBox="0 0 440 390" role="img" aria-label="由红与灰的点构成的球体，象征探索与创造"><defs><radialGradient id="sphereglow"><stop stop-color="var(--accent)" stop-opacity=".06"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></radialGradient></defs><circle cx="210" cy="188" r="187" fill="url(#sphereglow)"/><g class="sphere-dots">${dots.join('')}</g><ellipse cx="210" cy="188" rx="190" ry="48" fill="none" stroke="var(--accent)" stroke-opacity=".28" stroke-width=".8" transform="rotate(-27 210 188)"/><circle cx="44" cy="277" r="4" fill="var(--accent)"/><path d="M335 62h27m-13-13v27M61 98h13m-6-6v13" stroke="var(--muted)" stroke-width=".7"/><text x="272" y="352" fill="var(--muted)" font-family="monospace" font-size="9" letter-spacing="2">MUFC / 1827</text><text x="33" y="56" fill="var(--muted)" font-family="monospace" font-size="9">FIG. 01 — CURIOSITY</text></svg>`;
}
