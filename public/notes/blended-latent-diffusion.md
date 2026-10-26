# 12 Blended Latent Diffusion

## 来源与定位

- 论文：*Blended Latent Diffusion*
- 作者：Omri Avrahami、Ohad Fried、Dani Lischinski
- 发表：ACM TOG / SIGGRAPH 2023
- [官方项目页](https://omriavrahami.com/blended-latent-diffusion-page/)
- [arXiv](https://arxiv.org/abs/2206.02779)
- [官方代码](https://github.com/omriav/blended-latent-diffusion)

$$
\boxed{
\text{SDEdit 决定“从中途开始改”，BLD 决定“每一步只让 Mask 内的修改留下”。}
}
$$

BLD 不是新的扩散训练目标或网络结构，而是一种利用预训练文本条件 LDM 进行**零样本局部图像编辑**的推理方法。它的直接前身是 2022 年的 *Blended Diffusion*；本论文把逐步融合搬到 latent space，并处理 VAE 重建误差和细 Mask 两个新问题。

## 一、它要解决什么问题

输入：

- 原图 $x$；
- 目标文本 $d$；
- 二值 Mask $m$；
- 扩散起始噪声等级 $k$。

输出 $\hat x$ 希望满足：

$$
\hat x\odot m
\quad\text{符合文本 }d,
$$

$$
\hat x\odot(1-m)
\approx
x\odot(1-m).
$$

$\odot$ 表示逐元素乘法。本论文规定：

$$
\boxed{
m=1:\text{允许编辑；}
\qquad
m=0:\text{保护原图。}
}
$$

论文把二者称为 foreground 与 background，但这里只表示“编辑分支”和“保留分支”，不一定是语义上的人物前景与风景背景。其他软件可能采用相反的 Mask 约定，实际使用时要查文档。

## 二、为什么普通 img2img 还不够

SDEdit / img2img 的流程是：

$$
z_{\mathrm{init}}
\xrightarrow{\text{加噪到 }k}
z_k
\xrightarrow{\text{文本条件下全图去噪}}
z_0.
$$

起始噪声强度控制“改多少”，但 U-Net 的卷积、Attention 和文本条件都会作用到整张 latent，所以 Mask 外的背景也会漂移。

如果只在最终像素上粘回原背景：

$$
x_{\mathrm{paste}}
=m\odot x_{\mathrm{edit}}+(1-m)\odot x,
$$

虽然 Mask 外像素可以完全不变，但新物体生成时没有持续适应真实背景，边缘可能出现颜色、光照、阴影、纹理或几何接缝。

BLD 的办法是：**每做一步去噪，就立刻在 latent 中恢复一次 Mask 外的原图轨迹，再让下一步网络继续协调边界。**

## 三、完整数据流

以常见的 $512\times512$ 图像和 8 倍 VAE 压缩为例：

$$
x\in\mathbb R^{B\times3\times512\times512}
\xrightarrow{E}
z_{\mathrm{init}}
\in\mathbb R^{B\times4\times64\times64}.
$$

Mask 同样缩小：

$$
m\in\mathbb R^{B\times1\times512\times512}
\longrightarrow
m_{\mathrm{latent}}
\in\mathbb R^{B\times1\times64\times64},
$$

计算时沿 latent 的 4 个通道广播。

整体流程：

```text
原图 x
  ↓ VAE Encoder
干净原图 latent z_init
  ↓ 加噪到起始时刻 k
初始状态 z_k
  ↓
每一个反向时间步：
  ① 文本条件去噪，得到整张图的编辑候选 z_fg
  ② 把 z_init 加噪到相同时间，得到原图候选 z_bg
  ③ Mask 内取 z_fg，Mask 外取 z_bg
  ④ 把融合结果交给下一步去噪
  ↓
最终 latent z_0
  ↓ VAE Decoder
编辑结果
```

## 四、背景分支：原图在当前噪声等级的版本

先编码原图：

$$
z_{\mathrm{init}}=E(x).
$$

根据 DDPM 的闭式加噪公式，可以直接把它加到任意时刻 $t$：

$$
\boxed{
z_t^{\mathrm{bg}}
=
\sqrt{\bar\alpha_t}\,z_{\mathrm{init}}
+
\sqrt{1-\bar\alpha_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
}
$$

等价地：

$$
q(z_t\mid z_{\mathrm{init}})
=
\mathcal N\left(
\sqrt{\bar\alpha_t}z_{\mathrm{init}},
(1-\bar\alpha_t)I
\right).
$$

其中：

- $z_t^{\mathrm{bg}}$：原图背景在时刻 $t$ 应有的带噪 latent；
- $\bar\alpha_t$：截至时刻 $t$ 的累计信号保留率；
- $\epsilon_t$：标准高斯噪声；
- $\mathcal N(\mu,\Sigma)$：均值为 $\mu$、协方差为 $\Sigma$ 的高斯分布。

关键不是死记某次噪声，而是：

$$
\boxed{
z_t^{\mathrm{bg}}
\text{必须和当前编辑 latent 处于相同噪声等级。}
}
$$

不能把干净 $z_{\mathrm{init}}$ 直接塞进高噪声的 $z_t$，否则 Mask 内外的统计尺度完全不同，边界会成为模型训练时没见过的异常输入。

## 五、编辑分支：文本条件去噪一步

当前状态为 $z_t$。LDM 接收 $z_t$、时间 $t$ 和文本 $d$。若使用 CFG：

$$
\epsilon_{\mathrm u}
=\epsilon_\theta(z_t,t,\varnothing),
$$

$$
\epsilon_{\mathrm c}
=\epsilon_\theta(z_t,t,d),
$$

$$
\boxed{
\hat\epsilon
=
\epsilon_{\mathrm u}
+s(\epsilon_{\mathrm c}-\epsilon_{\mathrm u}).
}
$$

- $\epsilon_{\mathrm u}$：无条件/空文本预测；
- $\epsilon_{\mathrm c}$：目标文本条件预测；
- $s$：CFG 强度；
- $\hat\epsilon$：最终使用的噪声预测。

Scheduler 根据 $\hat\epsilon$ 做一步反向更新：

$$
z_t\longrightarrow z_{t-1}^{\mathrm{fg}}.
$$

若用确定性 DDIM，可以写成：

$$
\hat z_0
=
\frac{
z_t-\sqrt{1-\bar\alpha_t}\hat\epsilon
}{
\sqrt{\bar\alpha_t}
},
$$

$$
z_{t-1}^{\mathrm{fg}}
=
\sqrt{\bar\alpha_{t-1}}\hat z_0
+
\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon.
$$

不必把 DDIM 展开式当作 BLD 的新贡献。BLD 只需要一个能够给出编辑候选的反向去噪步骤；核心创新发生在下一步的融合。

## 六、整篇论文最重要的公式：逐步融合

同一目标时间得到：

- $z_{t-1}^{\mathrm{fg}}$：文本引导产生的编辑候选；
- $z_{t-1}^{\mathrm{bg}}$：原图 latent 加到相同噪声等级后的背景候选。

融合：

$$
\boxed{
z_{t-1}
=
m_{\mathrm{latent}}\odot z_{t-1}^{\mathrm{fg}}
+
(1-m_{\mathrm{latent}})\odot z_{t-1}^{\mathrm{bg}}.
}
$$

一个一维小例子：

$$
m=[1,1,0,0],
$$

$$
z^{\mathrm{fg}}=[8,7,6,5],
\qquad
z^{\mathrm{bg}}=[1,2,3,4].
$$

则：

$$
z=[8,7,3,4].
$$

前两个位置允许编辑，后两个位置恢复原图分支。

### Mask 约束的直接推导

二值 Mask 满足：

$$
m(1-m)=0,
\qquad
(1-m)^2=1-m.
$$

因此：

$$
\begin{aligned}
(1-m)\odot z_{t-1}
&=(1-m)\odot
\left[
m\odot z_{t-1}^{\mathrm{fg}}
+(1-m)\odot z_{t-1}^{\mathrm{bg}}
\right]\\
&=(1-m)\odot z_{t-1}^{\mathrm{bg}}.
\end{aligned}
$$

同理：

$$
m\odot z_{t-1}
=m\odot z_{t-1}^{\mathrm{fg}}.
$$

这说明在 latent 网格上，二值 Mask 会直接选择一个分支。若 $0<m<1$，则是软 Mask 的加权过渡。

### 严格实现中的时间下标

论文 Algorithm 1 为了简洁省略了部分 $t/t-1$ 标号，逐字阅读会出现时间下标歧义。论文实验所用的官方旧版 DDIM 代码采用更清楚的顺序：

$$
\tilde z_t
=
m_t\odot z_t
+
(1-m_t)\odot z_t^{\mathrm{bg}},
$$

$$
z_{t-1}
=
\operatorname{DenoiseStep}(\tilde z_t,d,t).
$$

也就是先把**同属时刻 $t$** 的当前编辑状态和原图背景状态融合，再去噪到 $t-1$。也可把一次去噪后的两个 $t-1$ 候选写成前面的融合式；无论采用哪种记号，真正不可违反的是：**被融合的两项必须处在同一噪声等级，并且约束要反复施加。**

## 七、为什么必须每一步都融合

### 1. 去噪网络每一步都会改动整张 latent

卷积具有跨位置感受野，Attention 可以交换全局信息，Prompt 也会通过条件机制影响所有位置。只在起点约束一次，下一步之后背景仍会漂移。

逐步融合等价于反复执行：

```text
模型自由提出整张图的更新
→ 只接受 Mask 内更新
→ Mask 外恢复到原图的同噪声轨迹
→ 再让模型协调下一步
```

### 2. 后续去噪可以逐渐协调边界

融合后，下一步网络同时看到新对象、原背景与二者边界，可以继续调整颜色、光照、阴影和纹理。一次融合未必立刻无缝，但反复去噪为协调提供机会。

### 3. 最终背景为何回到原图 latent

当 $t\to0$：

$$
\bar\alpha_0\approx1,
\qquad
1-\bar\alpha_0\approx0,
$$

所以：

$$
z_0^{\mathrm{bg}}
\approx z_{\mathrm{init}}.
$$

背景分支经历的是“原图的强噪声版本 $\to$ 中噪声版本 $\to$ 轻噪声版本 $\to$ 原图 latent”，而不是在最后突然贴回干净背景。

## 八、为什么 Mask 外像素仍不保证完全相同

BLD 强约束的是 latent，不是最终 RGB 像素。最终通常只有近似保持：

$$
D(E(x))\neq x.
$$

原因包括：

1. VAE 是有损压缩；
2. Mask 从像素空间缩小到 latent 会损失边界精度；
3. Decoder 的卷积感受野跨越 Mask 边界；
4. 软 Mask 会执行加权融合；
5. 后述 Progressive Mask Shrinking 早期会暂时扩大可编辑区。

因此准确表述是：算法在每个 latent 时刻把非编辑区域拉回原图的带噪轨迹，从而显著保护背景；它并不保证 Mask 外 RGB 逐像素相等或边界必然无缝。

## 九、VAE 背景重建补偿

人脸、文字和高频规则纹理对 VAE 重建误差尤其敏感。论文比较了四种思路。

### 1. 最终像素粘贴

Mask 外完全保留，但边界可能有接缝。

### 2. Poisson Blending

可减轻接缝，但可能造成编辑区颜色改变。

### 3. 优化最终 latent

$$
z^*
=
\arg\min_z
\left[
\|D(z)\odot m-\hat x\odot m\|
+
\lambda
\|D(z)\odot(1-m)-x\odot(1-m)\|
\right].
$$

- 第一项：Mask 内保持已经生成的编辑结果；
- 第二项：Mask 外接近原图；
- $\lambda$：背景保护权重。

固定 Decoder 时，latent 未必能表达原图的全部高频细节，论文观察到结果可能过度平滑。

### 4. 按单张图片优化 Decoder 权重

固定最终 latent $z_0$，针对这张图优化 Decoder 参数 $\theta$：

$$
\boxed{
\theta^*
=
\arg\min_\theta
\left[
\|D_\theta(z_0)\odot m-\hat x\odot m\|
+
\lambda
\|D_\theta(z_0)\odot(1-m)-x\odot(1-m)\|
\right].
}
$$

最后输出：

$$
x^*=D_{\theta^*}(z_0).
$$

论文设 $\lambda=100$，并对每张图使用 Adam、学习率 $10^{-4}$、75 个优化步骤。它是**可选的测试时单图优化**，不是重新训练扩散 U-Net。因此“基础编辑不训练新模型”成立，但“整个方法绝对没有任何优化”不准确。

## 十、细 Mask：Progressive Mask Shrinking

VAE 将空间尺寸缩小 8 倍。原图中很细的 Mask 到 latent 后可能只剩不到一个网格、发生断裂或消失。

扩散又是 coarse-to-fine：

- 早期高噪声阶段形成大致形状和布局；
- 后期低噪声阶段补充纹理与细节。

如果早期 Mask 太细，背景融合会不断抹掉刚形成的粗结构。论文采用：

$$
\boxed{
\text{早期扩大 Mask}
\longrightarrow
\text{随去噪逐渐缩小}
\longrightarrow
\text{后期恢复原始 Mask}.
}
$$

令随时间变化的 Mask 为 $m_t$：

$$
z_{t-1}
=
m_{t-1}\odot z_{t-1}^{\mathrm{fg}}
+
(1-m_{t-1})\odot z_{t-1}^{\mathrm{bg}}.
$$

论文把采样过程等分为四段，依次使用 $7\times7$、$5\times5$、$3\times3$ 膨胀核和原始 Mask。它表示可编辑区域逐渐收紧，不是把生成物体机械缩小。对极细小目标仍可能失败。

## 十一、随机采样与 CLIP 排序

扩散采样具有随机性，同一原图、Mask 和 Prompt 可生成多个候选。论文一次产生一批结果，再用 Mask 内区域与 Prompt 的 CLIP 余弦距离排序，选择更匹配文字的结果。

必须区分：

- 早期 *Blended Diffusion* 每个去噪步都需要 CLIP Loss 梯度；
- 本文 BLD 的文本条件由 LDM 自身提供，CLIP 只在生成完成后排序候选，不参与每一步去噪。

CLIP 排名有帮助，但第一名不保证一定优于第二名；若只评价 Mask 内，甚至可能选出局部看似合理、整图关系却不自然的结果。

## 十二、与几个相近概念的区别

### SDEdit / img2img

- SDEdit：选择从哪个噪声时刻开始，主要控制**改多少**；
- BLD：每一步按 Mask 选择分支，主要控制**在哪里改**。

二者是可组合的两个维度：

- strength / $k$ 大：编辑自由度更大，但更容易偏离原结构；
- Mask 大：允许更多空间发生变化；
- CFG 大：Mask 内更强地遵循文本，但过大可能产生伪影。

特殊情况：

- $m\equiv1$：近似退化为全图 img2img；
- $m\equiv0$：编辑分支全部丢弃，近似只剩 VAE 重建；
- $k=0$：几乎没有扩散编辑，只剩重建误差。

### 专门训练的 Inpainting 模型

专用 Inpainting 网络在训练时就接收 noisy latent、Mask 和 masked image 等输入，明确学过区域补全。BLD 可使用普通预训练 text-to-image LDM，Mask 由采样算法在模型外部施加，不必重新训练扩散网络。

### 2022 年 Blended Diffusion

早期方法在像素空间扩散，每步用外部 CLIP Loss 梯度推动文本匹配，因此较慢且可能出现像素截断或 CLIP 对抗性伪影。BLD 在更小的 latent space 运行，用已经具备文本条件能力的 LDM 直接生成；代价是引入 VAE 重建误差和细 Mask 下采样问题。

## 十三、实验到底证明了什么

论文用 50 张真实图像，随机生成矩形 Mask，并随机选择 ImageNet 类别名称作为 Prompt。主要指标：

- **Precision**：把 Mask 后的生成区交给 EfficientNet，目标类别是否进入 top-5；
- **Diversity**：只对分类正确的结果，在 Mask 区域计算两两 LPIPS；
- **人类评价**：两张结果二选一，分别比较视觉质量与文字匹配；
- **推理时间**：在 NVIDIA A10 上比较单图与完整候选批次。

核心数值：

| 方法 | 批次 Precision | CLIP 排名后最佳 Precision | Diversity |
|---|---:|---:|---:|
| Blended Diffusion | 10.40% | 36% | 0.106 |
| Local CLIP | 10.49% | 38% | 0.419 |
| GLIDE-filtered | 1.87% | 4% | 0.114 |
| BLD | **28.66%** | **54%** | 0.115 |

在论文定义的 50 个 ImageNet 类别局部编辑设置中，BLD 的分类 Precision 更高，并显著加快了推理。论文还展示了添加/替换对象、替换背景、文字、涂鸦、顺序编辑等应用。

但实验不能被夸大为“任何 Prompt、任何图片都全面最好”：

1. 样本只有 50 个，Prompt 限于 ImageNet 类别名；
2. 分类器 Precision 只是文字一致性的代理，不等于整体编辑质量；
3. CLIP 排序和候选批次会影响最终最好结果；
4. 人类评价时未展示原图与 Mask，因此不能直接评价背景保持；
5. GLIDE-filtered 在人类视觉质量比较中反而常胜，因为它往往几乎没有执行目标编辑；
6. 论文的速度结果依赖当时的 A10、模型、分辨率、步数和批量设置。

## 十四、局限与失败模式

- VAE 重建使 Mask 外 RGB 发生变化；
- 很细的 Mask 即使使用渐进缩小仍可能失败；
- 结果对 Prompt、Mask 大小和输入图像的小变化敏感；
- CLIP 只看 Mask 区域排序时，可能忽视整图是否合理；
- 文本偏置可能把“big mountain”等描述生成成海报或文字；
- Mask 只限制空间范围，不保证生成物在语义、透视、光照上必然合理；
- 候选生成、排序与可选 Decoder 优化仍带来时间成本；
- 继承预训练 LDM 数据偏差与潜在滥用风险。

## 十五、哪些公式必须会，哪些理解即可

必须会写并解释：

1. 原图 latent 的任意时刻加噪：

   $$
   z_t^{\mathrm{bg}}
   =\sqrt{\bar\alpha_t}z_{\mathrm{init}}
   +\sqrt{1-\bar\alpha_t}\epsilon_t.
   $$

2. 核心融合：

   $$
   z_{t-1}
   =m z_{t-1}^{\mathrm{fg}}
   +(1-m)z_{t-1}^{\mathrm{bg}}.
   $$

3. 对 $m=0$ 和 $m=1$ 分情况说明来源。

4. 用 $m(1-m)=0$ 推出 Mask 外等于背景分支。

5. 用 $\bar\alpha_0\approx1$ 解释最终背景 latent 回到原图 latent。

6. 说明 CFG 先生成编辑候选，Mask 再决定哪些位置保留。

理解含义即可：

- DDPM / DDIM 的完整概率推导；
- latent 与 Decoder 权重优化中的 $\arg\min$；
- Progressive Mask 的具体形态学实现；
- CLIP、LPIPS、EfficientNet 的内部网络细节。

## 十六、汇报问答与标准答案

### 1. 论文的核心贡献是什么？

把 Blended Diffusion 的逐步空间融合迁移到文本条件 LDM 的 latent space，使普通预训练文生图模型无需重新训练扩散网络即可进行 Mask 限定的局部编辑，并进一步处理 VAE 背景重建误差与细 Mask 问题。

### 2. 为什么每一步都要混回原图 latent？

去噪网络每一步都会全局更新 latent。只约束一次，后续步骤仍会改变背景；逐步融合是在每次更新后重新施加空间约束。

### 3. 为什么背景必须加到相同噪声等级？

当前编辑状态处于时刻 $t$ 的噪声分布。若混入干净 latent，Mask 内外统计尺度不一致，边界成为网络未见过的异常输入；相同噪声等级才能形成协调的下一步输入。

### 4. Mask 中 1 和 0 表示什么？

本论文中 $m=1$ 取文本编辑分支，$m=0$ 取原图背景分支。其他实现可能采用相反约定。

### 5. 为什么 Mask 外只能说近似不变？

约束发生在有损 VAE latent 中，Mask 还经过下采样，Decoder 感受野会跨边界，因此最终 RGB 不保证逐像素相同。

### 6. BLD 与 SDEdit/img2img 有何区别？

SDEdit/img2img 用起始噪声控制整图改多少；BLD 每一步用 Mask 控制哪里允许改。前者提供编辑自由度，后者提供空间定位。

### 7. BLD 与专用 Inpainting 网络有何区别？

专用网络训练时显式接收 Mask 与 masked image；BLD 可以直接使用普通预训练 text-to-image LDM，在采样循环外部执行 Mask 融合。

### 8. 为什么不能最后才粘回背景？

最终粘贴虽然能保存背景像素，但编辑区域没有在生成过程中持续适应背景，边缘容易出现光照、颜色、纹理和几何接缝。逐步融合让后续去噪不断协调两部分。

### 9. 为什么薄 Mask 要先扩大再缩小？

细 Mask 在低分辨率 latent 中可能消失，而扩散早期需要较大区域形成粗结构。早期扩大编辑区让对象成形，后期缩回原 Mask 再限制边界。

### 10. CFG 与 Mask 是否重复？

不重复。CFG 决定编辑候选多强地遵循文本，Mask 决定候选的哪些空间位置能留下。

### 11. 方法是否完全不需要训练或优化？

基础 BLD 不重新训练扩散模型，是 zero-shot 推理方法；但论文为高保真背景恢复提供了可选的逐图 Decoder 参数优化。

### 12. 一句话描述完整算法。

把原图编码并加噪，在文本条件下逐步去噪；每一步都把 Mask 内的编辑候选与 Mask 外、处于同一噪声等级的原图 latent 融合，最后统一经 VAE 解码。

## 十七、截至本讲的阶段知识地图

| 层面 | 已学内容 | 回答的问题 |
|---|---|---|
| 信息交互 | Attention、Transformer | token 怎样交流，网络怎样堆叠 |
| 生成动力学 | DDPM、Rectified Flow | 从噪声到数据沿什么规则移动 |
| 表示空间 | LDM | 在像素还是压缩 latent 中生成 |
| 去噪网络 | U-Net、DiT、MM-DiT | 用什么模型预测噪声/速度 |
| 推理编辑 | SDEdit、BLD | 如何利用预训练模型改已有图像 |

统一表示：

$$
f_\theta(u_t,t,c),
$$

其中 $u_t$ 可以是像素 $x_t$ 或 latent $z_t$，$c$ 可以是类别、文字等条件，$f_\theta$ 可以是 U-Net、DiT 或 MM-DiT，而 DDPM / Rectified Flow 决定它预测什么以及如何更新状态。

因此：

- LDM 与 DiT 不冲突：前者决定“在哪里算”，后者决定“用什么网络算”；
- DiT 与 Rectified Flow 不冲突：前者是 backbone，后者是生成动力学；
- MM-DiT 不等于 Rectified Flow：Stable Diffusion 3 把二者组合使用；
- SDEdit 与 BLD 通常不改变训练，只改变推理起点或中间轨迹。

## 相关笔记

- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)
- [08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)
- [08-Experiment-Stable Diffusion 文生图、DDIM 与图像编辑](#/notes/stable-diffusion-lab)
- [10-Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT）](#/notes/mmdit)
- [11-SDEdit - Guided Image Synthesis and Editing with Stochastic Differential Equations](#/notes/sdedit)
- [13-Prompt-to-Prompt Image Editing with Cross-Attention Control](#/notes/prompt-to-prompt)
- [14-FlowEdit - Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](#/notes/flowedit)
- [16-Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet）](#/notes/controlnet)：从 Mask 的区域约束扩展到由训练分支学习边缘、姿态、深度等稠密空间条件。
