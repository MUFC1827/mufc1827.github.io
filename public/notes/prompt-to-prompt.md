# 13 Prompt-to-Prompt Image Editing with Cross-Attention Control

## 来源与定位

- 论文：*Prompt-to-Prompt Image Editing with Cross-Attention Control*
- 作者：Amir Hertz、Ron Mokady、Jay Tenenbaum、Kfir Aberman、Yael Pritch、Daniel Cohen-Or
- 会议：ICLR 2023
- [官方项目页](https://prompt-to-prompt.github.io/)
- [arXiv](https://arxiv.org/abs/2208.01626)
- [官方代码](https://github.com/google/prompt-to-prompt)

$$
\boxed{
\text{Prompt-to-Prompt 不训练新模型，而是在采样时修改 Cross-Attention Map，复用原图布局并改变文字语义。}
}
$$

它回答的问题不是“怎样训练扩散模型”，而是：**一个已经训练好的文生图扩散模型，内部究竟在哪里决定某个词应该作用于图像的哪个位置，以及能否直接操控这个内部变量来编辑图像？**

## 一、与前文的知识链

```text
Attention Is All You Need：Q、K、V 和 Attention 的一般规则
→ LDM / Stable Diffusion：图像特征作 Q，文本特征作 K、V
→ SDEdit：从扩散中途开始，控制整图改动强度
→ BLD：用用户 Mask 在每一步限制“哪里可以改”
→ Prompt-to-Prompt：直接修改词与空间位置之间的 Cross-Attention 对应关系
```

最重要的对比：

| 方法 | 主要控制量 | Mask 从哪里来 | 擅长什么 |
|---|---|---|---|
| SDEdit | 起始噪声等级 | 无 | 整体图生图、真实性与忠实度折中 |
| BLD | 每一步 latent 的空间融合 | 用户手工给出 | 强保护 Mask 外背景的局部编辑 |
| Prompt-to-Prompt | Cross-Attention Map | 核心方法不需要用户 Mask；可选 LocalBlend 从 Attention 自动估计 | 只改 prompt，同时尽量保持构图 |

一句话：

> BLD 用外部空间 Mask 管理“哪些 latent 位置能被改”；Prompt-to-Prompt 管理模型内部“哪些图像位置读取哪些文字 token”。

## 二、论文要解决的问题

已用源 prompt $P$ 和随机种子 $s$ 生成图像 $I$。现在把文字改成目标 prompt $P^*$，希望得到 $I^*$：

1. 新图符合改后的文字；
2. 没要求改变的布局、物体位置、背景和外观尽量保持；
3. 核心交互只需改文字，不要求用户再画 Mask；
4. 不为每张图重新训练或微调模型。

最直接的做法是固定随机种子，只把 prompt 改掉再生成一次。但这通常失败：**相同噪声只固定了随机起点，没有固定文字与空间的交互方式。** Prompt 改变会改变文本的 $K,V$，进一步改变 Attention Map，早期去噪轨迹很快分叉，最终构图也会变化。

## 三、重新理解文生图中的 Cross-Attention

在扩散第 $t$ 步，记 U-Net 当前图像特征为 $\phi(z_t)$，文本编码为 $\psi(P)$。论文写作：

$$
Q=\ell_Q(\phi(z_t)),
\qquad
K=\ell_K(\psi(P)),
\qquad
V=\ell_V(\psi(P)).
$$

单个 Attention Head 下，设：

- 图像有 $N=H\times W$ 个空间位置；
- prompt 被切成 $L$ 个 token；
- Query、Key 的维度为 $d$。

则：

$$
Q\in\mathbb R^{N\times d},
\qquad
K\in\mathbb R^{L\times d},
\qquad
V\in\mathbb R^{L\times d_v}.
$$

Cross-Attention Map 为：

$$
\boxed{
M=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt d}\right)
}
\qquad
M\in\mathbb R^{N\times L}.
$$

这里对每个图像位置对应的一行，在 $L$ 个文本 token 上做 Softmax，所以：

$$
\sum_{j=1}^{L}M_{ij}=1.
$$

$M_{ij}$ 的含义是：**图像位置 $i$ 在更新自己时，读取第 $j$ 个文本 token 的程度。** 最后：

$$
\boxed{
\widehat\phi(z_t)=MV
}
\qquad
\widehat\phi(z_t)\in\mathbb R^{N\times d_v}.
$$

因此，每个图像位置都用该行的权重，对所有 token 的 Value 做加权求和。

例如某层特征为 $16\times16$，文本序列长度为 $77$，则每个 Head 的：

$$
M\in\mathbb R^{256\times77}.
$$

取出单词 `bear` 所在 token 对应的一列，再 reshape 为 $16\times16$，就得到“模型在各空间位置多大程度上关注 bear”的热力图。论文作图时会跨层、时间步和 Head 平均；实际编辑时各 Head 的 Map 仍分别保存和处理。

## 四、整篇论文最关键的思想：Map 管“在哪里”，Value 带“什么内容”

可以先用一个便于理解、但不是严格数学分解的说法：

- $M$：把图像位置连接到文字 token，主要体现**在哪里、以多大程度读取某个词**；
- $V$：被读取后真正送入图像特征的文本信息，主要携带**这个词提供什么语义内容**。

源 prompt：`a cat riding a bicycle`

目标 prompt：`a cat riding a car`

目标分支仍使用目标 prompt 的文本 Value，所以 `car` 的语义可以进入图像；但若把源图中 `bicycle` 的空间 Attention Map 对齐并注入给目标词 `car`，模型便倾向于让 `car` 出现在原自行车所在的位置，并继承原来的大体构图。

$$
\boxed{
\text{使用源图的 }M
+
\text{使用目标 prompt 的 }V
\Longrightarrow
\text{尽量保留空间关系，同时改变语义。}
}
$$

注意：这不是完美的“布局—语义”严格解耦。$M$ 本身由 $Q,K$ 决定，U-Net 后续还有卷积、Self-Attention 和非线性变换；因此注入太多源 Map 会过度限制新物体的几何形状。

## 五、两条同步去噪轨迹

Prompt-to-Prompt 同时维护：

- **源分支**：源 prompt $P$，得到源图和源 Attention Map $M_t$；
- **目标分支**：目标 prompt $P^*$，本来会得到目标 Map $M_t^*$，但在前向计算中将其替换或修改为 $\widehat M_t$。

两条分支从同一份初始噪声开始：

$$
z_T\sim\mathcal N(0,I),
\qquad
z_T^*=z_T.
$$

随后对 $t=T,T-1,\ldots,1$ 重复：

1. 源分支用 $(z_t,P)$ 去噪，并取得 $M_t$；
2. 目标分支根据 $(z_t^*,P^*)$ 形成自己的 $M_t^*$；
3. 编辑函数生成 $\widehat M_t=\operatorname{Edit}(M_t,M_t^*,t)$；
4. 目标分支使用目标 prompt 的 $V$，但把 Attention Map 改为 $\widehat M_t$，得到 $z_{t-1}^*$；
5. 若开启 LocalBlend，再限制本步 latent 的实际改动区域。

简写为：

$$
z_{t-1}^*
=
\operatorname{DM}(z_t^*,P^*,t)
\{M\leftarrow\widehat M_t\}.
$$

预训练模型参数在整个过程中保持冻结；没有新的训练集、Loss、反向传播或参数更新。官方实现把源与目标放进同一 batch 并在每个 Attention 层截获权重，所以这是**采样时控制**。

## 六、操作一：Word Swap（词语替换）

例子：

```text
source: a big bicycle
target: a big car
```

目标是改变主体类别，但保留大体位置和构图。论文的编辑函数为：

$$
\operatorname{Edit}(M_t,M_t^*,t)
=
\begin{cases}
M_t^*, & t<\tau,\\
M_t, & \text{otherwise}.
\end{cases}
$$

论文的反向循环是 $t=T\rightarrow1$，所以不要把方向读反：

- 高噪声、反向过程早期的 $t\ge\tau$：注入源 Map $M_t$，先固定大体构图；
- 低噪声、反向过程后期的 $t<\tau$：恢复目标 Map $M_t^*$，让新物体调整细节和几何。

官方代码中的 `cross_replace_steps=\lambda` 更直观：它表示在前 $\lambda$ 比例的去噪步骤中替换 Cross-Attention。

控制规律：

- 注入过少：目标语义自由，但原图结构容易丢失；
- 注入较多：原图保持更好；
- 全程注入：可能过度约束，例如生成的 car 仍呈 bicycle 的形状；
- 最佳比例取决于改动幅度，不存在对所有 prompt 都固定最优的数值。

若新旧词被 tokenizer 切成不同数量的 token，不能直接按可见单词位置复制；需要用 token 对齐矩阵对 Map 做复制、分配或平均。官方 `AttentionReplace` 实现还要求两个 prompt 按空格切分后的词数相同，因此它不是能理解任意改写语义的通用对齐器。

## 七、操作二：Prompt Refinement（提示词细化）

例子：

```text
source: a castle
target: children drawing of a castle
```

或者：

```text
source: a car on the side of the street
target: an old car on the side of the street
```

定义对齐函数 $A(j)$：目标 prompt 的第 $j$ 个 token 若能在源 prompt 中找到对应 token，就返回其源索引；若是新增 token，则返回 `None`。论文写作：

$$
\bigl(\operatorname{Edit}(M_t,M_t^*,t)\bigr)_{i,j}
=
\begin{cases}
(M_t^*)_{i,j}, & A(j)=\operatorname{None},\\
(M_t)_{i,A(j)}, & \text{otherwise}.
\end{cases}
$$

含义：

- 共同词沿用源 Map，保护已有对象和构图；
- 新增词保留目标 Map，让 `old`、`at sunset` 或某种画风真正产生作用。

因此它既能做局部属性修改，也能做全局风格、天气、光照等变化。

## 八、操作三：Attention Re-weighting（注意力重加权）

有些连续程度很难只靠改文字表达，例如“兔子要多蓬松一点”或“山上少一些雪”。对目标 token $j^*$ 乘系数 $c$：

$$
\bigl(\operatorname{Edit}(M_t,M_t^*,t)\bigr)_{i,j}
=
\begin{cases}
c(M_t)_{i,j}, & j=j^*,\\
(M_t)_{i,j}, & \text{otherwise}.
\end{cases}
$$

论文方法部分写作 $c\in[-2,2]$；官方 Notebook 也出现过范围外的示例值，因此这不是代码的硬约束。直观上：

- $c=1$：保持不变；
- $c>1$：增强该词的作用；
- $0<c<1$：减弱；
- $c=0$：尽量抑制；
- 负值可能产生反向或复杂效果，但不保证线性、稳定。

这里乘的是 **Softmax 之后的 Attention 权重**。修改后各项不一定仍然和为 1，因为它的目的不是重新构造概率分布，而是干预送入 $MV$ 的语义强度。

## 九、可选的 LocalBlend：从 Attention 自动得到局部区域

核心 P2P 不要求用户画 Mask，但为了进一步保护背景，可以从所编辑词的 Cross-Attention Map 自动估计区域。

对源词 $w$ 与目标词 $w^*$：

1. 聚合截至当前时刻的多个 Cross-Attention Map；
2. 对层和 Head 等维度取平均；
3. 论文用阈值 $k=0.3$ 得到二值区域；
4. 将新旧词区域取并集，覆盖原轮廓与新轮廓；
5. 用区域 $\alpha$ 融合源、目标 latent：

$$
\boxed{
z_{t-1}^*
=(1-\alpha)\odot z_{t-1}
+\alpha\odot z_{t-1}^*.
}
$$

- $\alpha=0$：取源分支，保护区域外内容；
- $\alpha=1$：取目标分支，允许区域内编辑。

它与 BLD 的区别：

| BLD | P2P LocalBlend |
|---|---|
| 用户提供空间 Mask | 由指定词的 Attention Map 自动估计 Mask |
| 背景分支来自原图编码后、处于相同噪声等级的 latent | 背景分支来自同步生成的源 prompt 轨迹 |
| 主要解决有原始输入图像的局部 inpainting | 主要增强两次相关生成之间的局部保持 |

由于 Attention Map 往往分辨率较低，这个自动区域只是粗定位，不能等同于精确分割 Mask。

## 十、为什么不只控制 Self-Attention

二者都能影响布局，但信息来源不同：

| Cross-Attention | Self-Attention |
|---|---|
| 图像位置与文本 token 交互 | 图像位置彼此交互 |
| 可指出某个位置对应 `bear`、`car` 等词 | 能反映相似区域、形状和空间结构，但没有词标签 |
| 适合按词替换、细化和重加权 | 难以指定“只保留 butterfly” |

论文发现，额外注入少量早期 Self-Attention 有时能加强源图背景与布局保持；但注入过多会压制目标 prompt。论文采用最多约前 $20\%$ 的步骤作为辅助，而把 Cross-Attention 作为核心控制对象。官方 Stable Diffusion 实现只在空间位置数不超过 $16^2$ 时替换 Self-Attention，以侧重粗布局并避免高分辨率层被锁得过死；Notebook 中的经验比例也可能高于 $20\%$，这些不是理论常数。

## 十一、生成图像与真实图像不是同一难度

### 1. 编辑刚生成的图像

源 prompt、随机种子和完整去噪轨迹都已知，可以直接同步重跑源、目标分支。这是 P2P 最自然、最强的场景。

### 2. 编辑真实图像

真实图像没有已知的 $z_T$ 和生成轨迹，必须先做 **inversion**：寻找一条能由噪声近似重建该图的扩散轨迹。论文使用确定性 DDIM inversion 做初步实验，但会出现：

- VAE 和 inversion 重建误差；
- prompt 无法完整描述真实图像；
- CFG 越强越易编辑，但可能越难精确重建；
- 降低 CFG 有助重建，却限制大幅编辑。

因此不能把“P2P 本身无需优化”误说成“任意真实图都无需 inversion 就能精确编辑”。官方项目列出的 Null-text Inversion 是后续改进，不是本论文核心算法。

## 十二、实验说明了什么

### 1. 注入步数消融

从 $0\%$ 到 $100\%$ 增加 Cross-Attention 注入比例时，结果通常越来越像源图；但全程注入可能让目标物体保留源物体几何，导致文本一致性下降。它证明：**早期 Attention Map 的确强烈影响构图，同时编辑存在“源图保持—目标语义自由度”折中。**

### 2. 不同模型

论文主要在 Imagen 上展示，也在 LDM 和 Stable Diffusion 上验证词语替换、提示词细化和重加权，说明思想不是专属于一个 backbone；前提是模型含有可访问的文本 Cross-Attention。不同 Text Encoder 的语义解耦程度不同，重加权的稳定性也会不同。

### 3. 用户研究

32 名参与者按 1–5 分评价背景/结构保持、文本一致性和真实感：

| 方法 | 背景/结构 ↑ | 文本一致性 ↑ | 真实感 ↑ |
|---|---:|---:|---:|
| VQGAN+CLIP | $1.84\pm1.11$ | $2.46\pm1.16$ | $1.32\pm0.70$ |
| Text2Live | $4.15\pm1.09$ | $2.89\pm1.22$ | $2.36\pm1.12$ |
| Baseline | $3.38\pm1.12$ | $4.26\pm1.03$ | $4.11\pm0.93$ |
| Prompt-to-Prompt | $4.64\pm0.64$ | $4.55\pm0.71$ | $4.42\pm0.82$ |

Baseline 使用同一随机种子，并在前 $20\%$ 去噪步骤后直接把源 prompt 换成目标 prompt。P2P 在三项主观评分中均最高，支持其既能保持结构又能服从编辑文本。

### 4. 自动指标不能只看一个

附录结果：

| 方法 | CLIP ↑ | MS-SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| VQGAN+CLIP | $0.282\pm0.04$ | $0.27\pm0.046$ | $0.64\pm0.05$ |
| Text2Live | $0.247\pm0.04$ | $0.82\pm0.065$ | $0.25\pm0.05$ |
| Baseline | $0.253\pm0.03$ | $0.69\pm0.13$ | $0.35\pm0.12$ |
| Prompt-to-Prompt | $0.253\pm0.04$ | $0.81\pm0.11$ | $0.22\pm0.10$ |

P2P 的 CLIP 分数并非最高；直接优化 CLIP 的 VQGAN+CLIP 更高。P2P 的 LPIPS 最好、MS-SSIM 接近最好，说明其源图相似性较强。论文也明确承认：没有 ground truth 时，单一自动指标难以同时衡量编辑正确性、源图保持和真实感，所以还需用户研究与可视化结果。

## 十三、证据边界与局限

1. **低分辨率 Attention Map**：难以提供像分割图一样精确的边界。
2. **不能直接移动物体**：它擅长保留或替换原位置上的语义，不提供显式坐标迁移。
3. **大几何变化困难**：保持越强，新类别越可能继承旧类别形状；放松控制又会破坏原构图。
4. **真实图依赖 inversion**：重建误差与可编辑性存在冲突，还需要合适的源 prompt。
5. **token 对齐并不总是简单**：一个可见单词可能拆成多个 subword token。
6. **重加权不是严格线性旋钮**：模型后续存在非线性，系数翻倍不代表视觉效果恰好翻倍。
7. **评价规模有限**：用户研究只有 32 人，论文未给出显著性检验、置信区间或参与者详细信息；大量证据仍是定性图例。
8. **方法间监督不完全一致**：GLIDE、Blended Diffusion 需要人工 Mask，因此未进入定量用户研究；各方法的交互成本和逐图优化成本也不同。
9. **安全风险**：更直观的真实图像编辑也可能被用于制造误导内容。

## 十四、四个参数的直观意义

| 参数 | 控制什么 | 增大后的典型效果 |
|---|---|---|
| `cross_replace_steps` | 前多少比例步骤注入源 Cross-Attention | 更保留源构图，但可能妨碍新几何 |
| `self_replace_steps` | 前多少比例步骤注入源 Self-Attention | 更保留整体布局/背景，但过大压制目标 prompt |
| `local_blend` | 是否用词的 Attention 自动限制编辑区域 | 区域外更稳定，但受低分辨率 Map 限制 |
| `equalizer` | 指定 token 的 Attention 倍数 | 增强或减弱某个词的视觉作用 |

这些是推理超参数，不通过反向传播学习。官方实现通过 `AttentionControl` 在每个 Attention 层的 forward 中读取、替换或缩放权重。

## 十五、必须掌握与了解即可

### 必须掌握

- [ ] 能说明相同 seed 为什么仍不能保持构图。
- [ ] 能写出 $M=\operatorname{Softmax}(QK^T/\sqrt d)$ 与 $\widehat\phi=MV$，并说明形状。
- [ ] 能解释 $M$ 与 $V$ 在 P2P 中分别起什么作用。
- [ ] 能完整描述源/目标双分支的同步采样流程。
- [ ] 能区分 Word Swap、Prompt Refinement、Attention Re-weighting。
- [ ] 能解释为什么主要注入早期步骤，以及注入太多为什么失败。
- [ ] 能解释 LocalBlend 与 BLD 的 Mask 来源和背景轨迹差异。
- [ ] 能区分 Cross-Attention 与 Self-Attention 的控制能力。
- [ ] 能说明它是采样时控制，不训练新模型。
- [ ] 能说明真实图编辑为何需要 inversion。

### 了解即可

- 官方实现中每个张量的 batch/head reshape 细节；
- token 对齐矩阵的动态规划代码；
- Imagen 三阶段 U-Net 的全部层配置；
- 六种评测 prompt 模板的词表；
- 每张附录图的具体超参数。

## 十六、汇报问答与标准答案

### 1. Prompt-to-Prompt 的核心贡献是什么？

它发现文本到图像的 Cross-Attention Map 把每个词与空间布局联系起来，并在扩散采样时注入或修改这些 Map，使用户只改 prompt 就能在尽量保持原构图的前提下进行局部替换、提示词细化和语义强度控制，不需要为每张生成图重新训练模型。

### 2. 为什么相同随机种子还不够？

随机种子只固定初始噪声。修改 prompt 会改变文本 $K,V$，从而改变每一步的 Attention Map 和噪声预测；两条去噪轨迹会从早期开始分叉，因此最终构图仍可能完全不同。

### 3. 源 Map 与目标 Value 为什么能够编辑？

源 Map 提供原图中“哪些位置读取哪些词”的路由，目标 Value 提供改后词的语义信息。目标分支使用源 Map 但保留目标 prompt 的 Value，因此能在旧位置注入新语义。不过这只是近似解耦，注入太强会限制新物体几何。

### 4. 为什么只在早期注入更合理？

高噪声早期步骤先决定粗粒度布局和几何，低噪声后期逐渐细化外观。早期注入源 Map 可锚定构图，后期放回目标 Map 则给新语义和几何留下自由度。

### 5. 三种 Cross-Attention 操作有什么区别？

Word Swap 对齐并替换词的 Map；Prompt Refinement 让共同词继承源 Map、新增词使用目标 Map；Attention Re-weighting 对某个词的 Map 乘系数，以连续增强或减弱其影响。

### 6. LocalBlend 是否说明 P2P 其实也必须手画 Mask？

不是。LocalBlend 是可选步骤，Mask 由指定词的 Cross-Attention Map 自动估计，不是用户画出的像素 Mask。核心 Attention 注入本身无需空间 Mask。

### 7. P2P 与 BLD 的根本区别是什么？

BLD 用用户 Mask 在每一步混合原图背景轨迹和文本编辑轨迹，显式约束“哪里改”；P2P 修改模型内部的词—空间 Attention 对应关系，约束“哪个位置读取哪个词”。P2P 的 LocalBlend 也可做 latent 融合，但 Mask 与背景分支来源都不同。

### 8. 为什么 Self-Attention 不能替代 Cross-Attention？

Self-Attention 只知道图像位置间关系，没有与具体文字 token 的对应，难以指定只改变 `car` 或保留 `butterfly`；Cross-Attention 同时含空间索引与文本 token 索引，因此提供词级语义控制。

### 9. 该方法有没有训练和反向传播？

没有新的训练过程。原扩散模型权重冻结，P2P 只在采样 forward 中截获并改写 Attention 权重。源、目标分支可放在同一 batch 中并行计算。

### 10. 它能否直接精确编辑任意真实照片？

不能这样保证。真实图先要 inversion 找到近似噪声轨迹；原论文的 DDIM inversion 会产生重建误差，复杂图像还需要合适的源 prompt。后来的 Null-text Inversion 才进一步改善这一环节。

### 11. 实验证明了什么，没证明什么？

消融和用户研究支持 Cross-Attention 注入能改善构图保持，并在所测样例上兼顾文本一致性与真实感；不同 backbone 的结果说明思想有一定可迁移性。但实验没有证明所有 prompt、所有模型和真实图都稳定有效，也没有解决精确边界、物体移动和大几何改变。

## 十七、三条核心公式

$$
M=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt d}\right),
\qquad
\widehat\phi(z_t)=MV.
$$

$$
\widehat M_t=\operatorname{Edit}(M_t,M_t^*,t),
\qquad
z_{t-1}^*=\operatorname{DM}(z_t^*,P^*,t)\{M\leftarrow\widehat M_t\}.
$$

$$
z_{t-1}^*=(1-\alpha)\odot z_{t-1}+\alpha\odot z_{t-1}^*.
$$

## 最容易说错的地方

1. P2P 不是训练新模型，而是采样时 Attention 控制。
2. 相同 seed 只固定噪声起点，不固定布局。
3. 注入的是 Attention Map $M$；目标分支仍保留目标 prompt 的 Value $V$。
4. 论文按 $t=T\rightarrow1$ 采样，源 Map 注入发生在高噪声的反向早期，不要把公式方向读反。
5. Attention 可近似理解为 Map 管位置、Value 带内容，但不是严格完全解耦。
6. Word Swap、Refinement、Re-weighting 不是三个重新训练的模型，而是三个 Map 编辑规则。
7. Re-weighting 后的权重不必重新和为 1。
8. LocalBlend 的 Mask 来自 Attention，不是必须的人工 Mask。
9. Cross-Attention 图作图时可平均；算法中不同 Head 不会先平均成一张图再统一计算。
10. 生成图编辑与真实图编辑不同；后者需要 inversion。
11. P2P 能保持布局不代表它能显式移动物体。
12. 用户研究领先不等于在所有自动指标上最高，也不等于没有评价局限。

## 我的理解与修改

- 

## 相关笔记

- [05-Attention Is All You Need](#/notes/attention-is-all-you-need)
- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)
- [08-Experiment-Stable Diffusion 文生图、DDIM 与图像编辑](#/notes/stable-diffusion-lab)
- [11-SDEdit - Guided Image Synthesis and Editing with Stochastic Differential Equations](#/notes/sdedit)
- [12-Blended Latent Diffusion](#/notes/blended-latent-diffusion)
- [14-FlowEdit - Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](#/notes/flowedit)
 
## 后续连接

- [15-Stable Flow - Vital Layers for Training-Free Image Editing](#/notes/stable-flow)：把 Attention 控制推进到 flow-based DiT；它不是替换 Cross-Attention map，而是在关键层替换源图视觉 token 的 $K,V$。
- [16-Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet）](#/notes/controlnet)：与推理时 Attention 操作不同，它训练一个外挂分支，把结构图变成 U-Net 的多尺度控制残差。
