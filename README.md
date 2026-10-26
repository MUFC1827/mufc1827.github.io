# MUFC · 个人网站

MUFC1827 的个人网站，展示个人项目与图像生成研究笔记。使用原生 JavaScript ES modules 与 Vite 构建，通过 GitHub Pages 发布。

目标网址：[https://mufc1827.github.io/](https://mufc1827.github.io/)

## 本地运行

使用 Node.js 24 LTS，在项目目录执行：

```sh
npm ci
npm run dev
```

构建及预览发布版本：

```sh
npm run build
npm run preview
```

`build` 先检查站点内容，再生成 `dist/`。构建不需要访问原始笔记所在的磁盘，也不依赖后端服务。

## 内容与笔记

- `src/`：页面、样式和交互逻辑。
- `src/data/notes.json`：笔记标题、摘要、分类、标签与阅读时长。
- `public/notes/`：公开的 Markdown 笔记副本，公式由 KaTeX 显示，正文使用 markdown-it 渲染。
- `.github/workflows/deploy.yml`：GitHub Pages 自动部署。

首页首屏采用左侧介绍、右侧留白的布局，保留“查看作品”“阅读笔记”两个入口；项目和笔记数量仅作静态信息展示。下滑后展示项目预览与研究路径；项目卡片直接打开作品。18 篇笔记分为 Attention 基础、生成架构、Flow 方法、编辑与控制四组，以及一篇知识地图总览。

导航在目标内容准备好后切换，支持短淡入淡出及选中线移动；连续点击时会取消旧过渡，减少动态效果偏好下直接切换。站点图标与 favicon 使用同一枚红色三叉形 SVG。

点球游戏独立于首页，通过页尾或搜索进入 `#/play/penalty`，按需加载。截图素材位于 `public/images/`；MemoUnited 暂用明确标注的界面示意。

点球支持蓄力、推射/大力/弧线、旋转调整与慢速回放。球路按固定时间步长计算重力、阻力、旋转和碰撞，守门员有反应延迟；这是简化物理模拟。关键射门与门框反弹情形会随构建执行回归检查。

重新导入原始笔记：

```sh
npm run notes:import
```

默认读取 `D:\Code\Academic Reserch Project\Notes`，也可以指定其他目录：

```sh
npm run notes:import -- "D:\你的笔记目录"
```

导入过程不修改源笔记；它会更新公开副本和目录数据，移除本地资源路径，将已有笔记的 Obsidian 双链转换为站内链接。新增文章时，先更新 `scripts/import-notes.mjs` 中的文章目录。重新导入后执行 `npm run build`，再提交变更。

笔记是个人阅读、课程与实验记录，原论文和课程的权利归各自作者。每篇保留已有来源信息；公开副本的处理说明见 `public/notes/README.txt`。

## 页面与本地数据

站内使用 `#/...` 形式的 hash 路由，作品、笔记与互动页面链接可直接分享；刷新子页面不会触发 GitHub Pages 的路径 404。

收藏、阅读位置、专注计时、游戏纪录与主题偏好只保存在当前浏览器，不会上传到服务器，也不会在设备之间同步。清除浏览器网站数据会同时清除这些记录。

## 作品入口

- [Piano MUFC](https://piano-mufc.vercel.app/)：用电脑键盘演奏的钢琴。
- [REVOLVER](https://revolver-mufc.vercel.app/)：用于线下桌游的虚拟轮盘。
- [MemoUnited](https://github.com/MUFC1827/MemoUnited-Updates)：桌面工具的说明与更新；[下载最新版](https://github.com/MUFC1827/MemoUnited-Updates/releases/latest)。

## 部署到 GitHub Pages

1. 在 GitHub 账号 `MUFC1827` 下使用名为 `mufc1827.github.io` 的仓库，默认分支设为 `main`。
2. 在仓库 **Settings → Pages → Build and deployment** 中，将 **Source** 设为 **GitHub Actions**。
3. 将本项目推送到 `main`。工作流会执行 `npm ci` 和 `npm run build`，上传 `dist/` 并部署到 `github-pages` 环境。
4. 在仓库 **Actions** 中确认 `Deploy GitHub Pages` 完成，然后访问目标网址。也可通过 **Run workflow** 手动重新部署。

这个 `github.io` 地址由 GitHub 的账号与仓库命名规则提供，无需额外购买域名。部署方式参考 [GitHub Pages 自定义工作流文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
