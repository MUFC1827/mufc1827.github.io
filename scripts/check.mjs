import { readFile, access } from 'node:fs/promises';
import assert from 'node:assert/strict';
import './check-penalty.mjs';

const notes = JSON.parse(await readFile(new URL('../src/data/notes.json', import.meta.url), 'utf8'));
assert(notes.length > 0, '笔记目录不能为空');
const ids = new Set(notes.map(note => note.id));
assert.equal(ids.size, notes.length, '笔记 ID 重复');
let links = 0;
for (const note of notes) {
  assert(/^[a-z0-9-]+$/.test(note.id), `不安全的笔记 ID: ${note.id}`);
  assert.equal(note.path, `./notes/${note.id}.md`, `笔记路径不一致: ${note.id}`);
  assert(note.title && note.subtitle && note.category && Array.isArray(note.tags), `笔记元数据缺失: ${note.id}`);
  const markdown = await readFile(new URL(`../public/notes/${note.id}.md`, import.meta.url), 'utf8');
  assert(!/\b[A-Za-z]:\\[^\\\s]+\\|file:\/\/|\[\[/.test(markdown), `仍有本地路径或未转换双链: ${note.id}`);
  for (const link of markdown.matchAll(/#\/notes\/([a-z0-9-]+)/g)) {
    assert(ids.has(link[1]), `${note.id} 存在无效笔记链接: ${link[1]}`);
    links++;
  }
}
for (const file of ['index.html','src/main.js','src/style.css','src/redesign.css','src/home.js','src/home.css','src/navigation.css','src/brand.js','src/study-map.js','src/study-map.css','src/penalty.js','src/penalty.css','src/play.css','src/lab.js','src/reader.js','public/favicon.svg','public/images/piano-preview.png','public/images/revolver-preview.png','.github/workflows/deploy.yml']) {
  await access(new URL(`../${file}`, import.meta.url));
}
console.log(`内容检查通过：${notes.length} 篇笔记，${links} 处文章链接，发布资源完整。`);
