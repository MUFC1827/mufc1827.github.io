/**
 * Publish a read-only copy of the owner's research notes.
 * Usage: node scripts/import-notes.mjs [source-directory]
 * Existing public copies are committed so builds never require the source disk.
 */
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(process.argv[2] || 'D:\\Code\\Academic Reserch Project\\Notes');
const destination = path.join(root, 'public', 'notes');
const dataDirectory = path.join(root, 'src', 'data');

// Deliberately stable ASCII identifiers: URLs remain readable when a title changes.
const catalog = [
  ['01-', 'self-attention-1', 'Self-Attention · 上', '从 Q、K、V 出发，理解上下文如何进入每一个位置。', '基础架构', ['Attention', 'QKV']],
  ['02-', 'self-attention-2', 'Self-Attention · 下', '多头注意力、位置编码，以及不同 Attention 之间的联系。', '基础架构', ['Attention', 'Multi-Head']],
  ['03-', 'transformer-1', 'Transformer · 上', 'Encoder、归一化与前馈网络：把注意力组成完整架构。', '基础架构', ['Transformer', 'Encoder']],
  ['04-', 'transformer-2', 'Transformer · 下', '从 Decoder 到自回归生成，理解训练与推理之间的区别。', '基础架构', ['Transformer', 'Decoder']],
  ['05-', 'attention-is-all-you-need', 'Attention Is All You Need', '回到 Transformer 原论文，串起架构、公式与设计取舍。', '基础架构', ['Transformer', '论文阅读']],
  ['06-', 'ddpm', 'DDPM', '一步步加入噪声，再学习一步步去噪：扩散模型的起点。', '生成模型', ['Diffusion', 'DDPM']],
  ['07-', 'rectified-flow', 'Rectified Flow', '用速度场连接噪声与数据，理解直线插值和生成轨迹。', '生成模型', ['Flow Matching', 'ODE']],
  ['08-Experiment', 'stable-diffusion-lab', 'Stable Diffusion 实验', '文生图、DDIM 与图像编辑的实验记录和理解。', '实践记录', ['Stable Diffusion', 'DDIM', '实验']],
  ['08-High', 'latent-diffusion', 'Latent Diffusion Models', '把扩散移到潜空间，用 Cross-Attention 接入文本条件。', '生成模型', ['LDM', 'Latent', 'Stable Diffusion']],
  ['09-', 'dit', 'Diffusion Transformer', '从 U-Net 到 Transformer：在潜空间中用 patch 组织图像。', '生成模型', ['DiT', 'Transformer', 'Scaling']],
  ['10-', 'mmdit', 'MM-DiT / Stable Diffusion 3', 'Rectified Flow 与多模态 Transformer 如何组成生成系统。', '生成模型', ['MM-DiT', 'Rectified Flow', 'SD3']],
  ['11-', 'sdedit', 'SDEdit', '先加噪、再去噪：让生成先验参与引导式图像编辑。', '编辑与控制', ['Image Editing', 'Diffusion']],
  ['12-', 'blended-latent-diffusion', 'Blended Latent Diffusion', '用 Mask 在潜空间融合，在局部编辑中保留源图。', '编辑与控制', ['Image Editing', 'Mask', 'Latent']],
  ['13-', 'prompt-to-prompt', 'Prompt-to-Prompt', '通过控制 Cross-Attention，让文字编辑与图像结构相连。', '编辑与控制', ['Attention Control', 'Image Editing']],
  ['14-', 'flowedit', 'FlowEdit', '无需反演的 Flow 编辑：从源速度与目标速度的差异出发。', '编辑与控制', ['Flow', 'Inversion-Free', 'Image Editing']],
  ['15-', 'stable-flow', 'Stable Flow', '寻找影响图像形成的关键层，在注意力内部控制编辑。', '编辑与控制', ['Vital Layers', 'Flow', 'Image Editing']],
  ['16-', 'controlnet', 'ControlNet', '冻结生成主干，用可训练分支接入边缘、姿态与深度等条件。', '编辑与控制', ['ControlNet', 'Conditional Generation']],
  ['17-', 'generative-models-map', '生成式模型知识地图', '把基础、生成架构与可控编辑放进同一个系统，建立完整学习主线。', '知识体系', ['知识地图', '复习', '生成模型']],
];

const files = (await readdir(source)).filter((name) => name.endsWith('.md')).sort();
if (files.length !== catalog.length) {
  throw new Error(`Expected ${catalog.length} catalog entries; found ${files.length} Markdown files. Update the catalog before publishing new notes.`);
}
const records = [];
const stems = new Map();
const byFile = new Map();
for (const entry of catalog) {
  const matches = files.filter((file) => file.startsWith(entry[0]));
  if (matches.length !== 1) throw new Error(`Ambiguous or missing note: ${entry[0]}`);
  byFile.set(matches[0], entry);
  stems.set(matches[0].replace(/\.md$/, ''), entry[1]);
}

let localLinesRemoved = 0;
let crossLinksConverted = 0;
let attachmentLinksFlattened = 0;

await Promise.all([mkdir(destination, { recursive: true }), mkdir(dataDirectory, { recursive: true })]);

for (const [index, entry] of catalog.entries()) {
  const [prefix, id, title, subtitle, category, tags] = entry;
  const sourceFile = files.find((file) => file.startsWith(prefix));
  const original = (await readFile(path.join(source, sourceFile), 'utf8')).replace(/^\uFEFF/, '').replace(/\r\n/g, '\n');
  const frontmatter = original.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
  let content = frontmatter ? original.slice(frontmatter[0].length) : original;
  const metadata = frontmatter?.[1] || '';
  const metadataValue = (key) => metadata.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'))?.[1]?.replace(/^['"]|['"]$/g, '') || '';

  // Local papers, experiment folders and private report attachments are not web
  // resources. Omit those directory-only lines while preserving academic text.
  content = content.split('\n').filter((line) => {
    const directoryLine = /^(?:\s*-\s*)?(?:本地\s*(?:PDF|论文)|主\s*Notebook|实验\s*Notebook|实验输出|实验日志|老师原手册|现有汇报文件)\s*[:：]/i.test(line);
    if (directoryLine) localLinesRemoved += 1;
    return !directoryLine;
  }).join('\n');

  // Convert Obsidian wikilinks into working, client-side article URLs.
  // Unknown local attachments are readable text, never broken download links.
  content = content.replace(/\[\[([^\]\n]+)\]\]/g, (_whole, inner) => {
    const [target, alias] = inner.split('|');
    const [stem, anchor] = target.split('#');
    const noteId = stems.get(stem.replace(/\.md$/, ''));
    const label = (alias || stem).replace(/\\/g, '/').split('/').pop();
    if (noteId) {
      crossLinksConverted += 1;
      // Article heading anchors are generated by the UI; note-level links are
      // intentionally stable regardless of heading punctuation or translation.
      return `[${label}](#/notes/${noteId})`;
    }
    attachmentLinksFlattened += 1;
    return label || anchor || '相关资料';
  });

  // Guard against accidental disclosure when the source collection evolves.
  // The backslash must begin a path, not a TeX expression such as T:\mathbb.
  if (/(?:[A-Za-z]:\\(?:Code|Users|Documents|Project|Academic)|file:\/\/|\\\\[A-Za-z][A-Za-z0-9.-]{1,}\\)/i.test(content)) {
    throw new Error(`Unreviewed local path remains in ${sourceFile}`);
  }
  if (/\b(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b/.test(content)) {
    throw new Error(`Potential credential remains in ${sourceFile}`);
  }

  content = `${content.trim()}\n`;
  await writeFile(path.join(destination, `${id}.md`), content, 'utf8');
  // Approximate reading length: CJK characters and Latin words. Math and code
  // take time too, so the estimate is intentionally a rough reading aid.
  const readable = content.replace(/```[\s\S]*?```/g, '').replace(/https?:\/\/\S+/g, '');
  const words = (readable.match(/[\u3400-\u9fff]/g) || []).length + (readable.match(/[A-Za-z]+(?:['’-][A-Za-z]+)*/g) || []).length;
  records.push({
    id, title, subtitle, category, tags, words,
    minutes: Math.max(1, Math.ceil(words / 300)),
    sourceFile,
    path: `./notes/${id}.md`,
    order: index + 1,
    date: metadataValue('学习日期') || metadataValue('完成日期') || null,
    sourceName: metadataValue('来源名称') || null,
    sourceUrl: metadataValue('来源地址') || null,
  });
}

await writeFile(path.join(dataDirectory, 'notes.json'), `${JSON.stringify(records, null, 2)}\n`, 'utf8');
await writeFile(path.join(destination, 'README.txt'), [
  'MUFC / Research Notes',
  '',
  '本目录是站点主人提供的个人科研学习笔记的公开副本，包含 18 篇 Markdown 笔记。',
  '它们是对原论文、课程和实验的个人理解与整理；引用的论文与课程归原作者所有。',
  '正文保留来源链接、公式、学习记录及原有解释，未虚构科研成果或学习日期。',
  '公开版本去掉了 Obsidian frontmatter、本地磁盘路径、未上传的本地论文与汇报附件说明。',
  '已存在笔记之间的 Obsidian 双链转换为站内链接。未上传附件仅保留名称文字。',
  '本次导入未发现依赖本地图片的 Markdown 图像引用。',
  '阅读时长根据中英文长度估算，仅作参考。',
  '',
  `整理记录：移除 ${localLinesRemoved} 行本地资源说明，转换 ${crossLinksConverted} 处笔记双链，将 ${attachmentLinksFlattened} 处未发布附件链接转为文字。`,
  '重新导入：node scripts/import-notes.mjs "你的笔记目录"',
  '源文件未被修改；常规构建使用本目录中的副本，不要求访问源磁盘。',
  '',
].join('\n'), 'utf8');

console.log(JSON.stringify({ notes: records.length, localLinesRemoved, crossLinksConverted, attachmentLinksFlattened, totalWords: records.reduce((sum, note) => sum + note.words, 0), ids: records.map((note) => note.id) }, null, 2));
