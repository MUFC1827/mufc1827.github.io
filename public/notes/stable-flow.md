# 15 Stable Flow：用关键层实现无需训练的图像编辑

## 来源与一句话定位

- 论文：*Stable Flow: Vital Layers for Training-Free Image Editing*
- 作者：Omri Avrahami 等
- 正式发表：CVPR 2025
- [官方项目页](https://omriavrahami.com/stable-flow/)
- [CVPR 论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Avrahami_Stable_Flow_Vital_Layers_for_Training-Free_Image_Editing_CVPR_2025_paper.html)
- [arXiv](https://arxiv.org/abs/2411.14430)
- [官方代码](https://github.com/snap-research/stable-flow)

$$
\boxed{
\text{Stable Flow 先自动找出 DiT 中真正影响图像形成的关键层，}
\text{再只在这些层把原图分支的视觉 Attention }K,V\text{ 注入编辑分支。}
}
$$

它是一个**推理时编辑方法**：不训练新模型、不更新 FLUX/SD3 权重；但它需要先离线做逐层消融来确定关键层。对于真实图像，它还需要先做 inversion，并用 latent nudging 改善重建。

> [!warning] 名称辨析
> Stable Flow 是论文提出的“稳定编辑”方法，不是 Stable Diffusion 的另一个版本。这里的 stable 指尽量保持不该改变的内容。

## 一、它接在前面知识链的哪里

```text
Rectified Flow：学习从噪声到图像的速度场
→ DiT / MM-DiT：用 Transformer 作为速度预测网络，并联合处理图像、文字 token
→ Prompt-to-Prompt：在 Attention 内部控制编辑，但主要建立在 U-Net 扩散模型上
→ FlowEdit：不反演、不改模型内部特征，用“目标速度－源速度”直接编辑真实图
→ Stable Flow：回到模型内部，自动找 DiT 的关键层，只在这些层注入原图视觉 K、V
```

Stable Flow 解决的核心矛盾是：

- 编辑后的图要听从新文字，即具有**可编辑性**；
- 没被要求改变的人物身份、姿态、布局、背景要尽量不变，即具有**保真性**。

如果约束太弱，整张图会漂移；约束太强，模型只会复制原图，忽略编辑文字。

## 二、论文利用了流模型的一个“缺点”

作者观察到：FLUX、SD3 这类基于 flow matching 的 DiT 文生图模型，在**相同初始随机种子**下只改变 prompt 时，往往比 SDXL 等模型保留更多共同结构，即生成多样性较低。

这原本可能是缺点，但对编辑很有用：

$$
\text{同一个 seed} + \text{相近的两个 prompt}
\Longrightarrow
\text{两条生成轨迹天然较对齐}.
$$

例如把“站立的狗”改成“坐着的狗”，相同 seed 往往还能保留道路、视角和狗的大体身份。不过，只靠相同 seed 仍会发生不必要的变化，因此还需要从原图分支注入信息。

## 三、为什么不能照搬 U-Net 的“中间层注入”经验

U-Net 有明显的下采样—瓶颈—上采样结构，各层分辨率和作用比较容易解释：有的偏布局，有的偏细节。因此以往方法可以依据层的位置选择注入位置。

FLUX 的主体是连续的 Transformer block：

$$
h^{\ell+1}=h^\ell+F_\ell(h^\ell).
$$

- $h^\ell$：第 $\ell$ 层输入；
- $F_\ell$：该 Transformer block 的变换；
- 加号：残差连接。

它没有 U-Net 明显的粗—细层级，所以不能武断地说“只控制中间层”。论文必须先回答：**DiT 中到底哪些层真正决定图像？**

## 四、第一项贡献：自动寻找 Vital Layers

### 1. 逐层旁路实验

作者先准备 $k=64$ 个不同 prompt 和对应 seed，用完整模型生成参考集合 $G_{ref}$。

随后对每一层 $\ell$ 分别做一次实验：利用残差连接绕过这一层，即把

$$
h^{\ell+1}=h^\ell+F_\ell(h^\ell)
$$

临时改为

$$
h^{\ell+1}=h^\ell.
$$

模型权重没有删除，也没有重新训练，只是在本次前向计算中不使用该层的变换。用相同 prompt 和 seed 再生成集合 $G_\ell$。

直觉是：

- 绕过某层后图几乎没变，这一层对最终图像的影响较小；
- 绕过某层后出现噪声、身份改变、结构改变或纹理大变，这一层就是关键层候选。

### 2. 用 DINOv2 衡量变化

作者不用逐像素差异，因为两张语义和结构相近的图也可能有像素级小偏移；而是使用 DINOv2 特征的感知相似度 $d(\cdot,\cdot)$。

论文定义：

$$
\boxed{
\operatorname{vitality}(\ell)
=1-\frac{1}{k}\sum_{(s,p)}
d\big(M_{\mathrm{full}}(s,p),M_{-\ell}(s,p)\big)
}
\tag{1}
$$

更不容易误解的等价写法是把 64 个一一配对的 prompt/seed 记作 $(s_i,p_i)$：

$$
\operatorname{vitality}(\ell)
=1-\frac{1}{k}\sum_{i=1}^{k}
d\big(M_{\mathrm{full}}(s_i,p_i),M_{-\ell}(s_i,p_i)\big).
$$

原论文公式把求和下标排成 $s\in S,p\in P$，但补充材料明确实际生成的是 $k$ 对对应样本，而不是 $k^2$ 个笛卡尔积组合；因此这里按实际流程解释。

- $M_{\mathrm{full}}(s,p)$：完整模型在 seed $s$、prompt $p$ 下的结果；
- $M_{-\ell}(s,p)$：绕过第 $\ell$ 层后的结果；
- $d$：感知相似度，越大表示越相似；
- $1-d$：变化程度，因此越大表示该层越重要；
- 求平均：避免被某一个 prompt 的偶然现象误导。

再用阈值选集合：

$$
\boxed{
V=\{\ell\in\mathbb L\mid
\operatorname{vitality}(\ell)\ge \tau_{vit}\}.
}
\tag{2}
$$

$\mathbb L$ 是全部层，$\tau_{vit}$ 是关键性阈值，$V$ 是 vital layer 集合。

### 3. 实际找到哪些层

FLUX.1-dev 初步得到：

$$
[0,1,2,17,18,25,28,53,54,56].
$$

作者又经验性发现第 2 层可以去掉，所以官方编辑实现使用：

$$
\boxed{V_{FLUX}=[0,1,17,18,25,28,53,54,56].}
$$

这些层分散在网络前、中、后部，验证了“DiT 关键层不能只凭位置猜”的观点。对 SD3，论文得到：

$$
V_{SD3}=[0,7,8,9].
$$

不同模型要分别检测，不能直接复制 FLUX 的层号。

### 4. 不同关键层大致影响什么

论文的旁路图展示：

- 去掉 FLUX 第 0 层：结果可能接近纯噪声；
- 去掉第 18 层：全局结构和主体身份明显变化；
- 去掉第 56 层：纹理、局部细节明显变化；
- 去掉第 5、52 等非关键层：结果只发生较小变化。

这不是说每层只有一个固定功能，而是说明不同关键层对图像形成的影响尺度不同。

## 五、第二项贡献：只在关键层注入视觉 Attention

### 1. 两条并行生成分支

设：

- 源图分支 $x$：使用源 prompt，例如“一名男子”；
- 编辑分支 $\hat x$：使用目标 prompt，例如“一名拿着牛油果的男子”；
- 两个分支从相同的初始噪声/seed 出发并行运行。

在普通层，两条分支各算各的。在关键层，将编辑分支中**图像 token 对应的 Key 和 Value**替换成源图分支的图像 Key 和 Value。

### 2. 先回忆 Attention

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V.
$$

- $Q$：当前 token 正在寻找什么；
- $K$：每个候选信息的“索引/标签”；
- $QK^T$：Query 与各 Key 的匹配程度；
- softmax：把匹配分数变成权重；
- $V$：真正被加权取回的内容。

### 3. Stable Flow 到底替换什么

在 vital layer 中，编辑分支保留自己的 Query：

$$
Q^{edit}\ \text{保留}.
$$

文字 token 仍来自目标 prompt；但图像 token 的 $K,V$ 换成源分支的：

$$
K^{edit}_{img}\leftarrow K^{src}_{img},
\qquad
V^{edit}_{img}\leftarrow V^{src}_{img}.
$$

把文字与图像 token 一起写，可理解为：

$$
\boxed{
K^{edit}=[K^{edit}_{text};K^{src}_{img}],\qquad
V^{edit}=[V^{edit}_{text};V^{src}_{img}].
}
$$

随后编辑分支计算：

$$
\hat H=
\operatorname{softmax}\left(
\frac{Q^{edit}(K^{edit})^T}{\sqrt d}
\right)V^{edit}.
$$

> [!important] 必须说准确
> Stable Flow 不是把整层输出直接复制过去，也不是把两个结果取平均；它只在选中的关键层里，替换编辑分支**图像 token 所对应的 Attention K、V**。编辑 Query 和目标文字信息仍保留。

### 4. 为什么保留 Q，却替换 K、V

- 保留编辑分支的 $Q$：编辑结果仍可根据当前目标状态提出自己的信息需求；
- 使用原图的图像 $K$：原图中的人物、背景、位置仍然作为可被匹配的索引；
- 使用原图的图像 $V$：真正取回的视觉内容来自原图轨迹，有助于保留身份和布局；
- 保留目标文字 token：需要改变的区域仍能从新文字获得语义。

可以把它类比成：**编辑分支拿着新任务提问，但视觉资料库暂时换成原图资料库；文字资料库仍写着新要求。**

## 六、为什么必须“只注入关键层”

关键层的多模态 Attention 呈现一种有用的分工：

- 不需要修改的位置，更关注原图视觉 token，从而复制和保留；
- 需要修改的位置，更关注目标文字 token，例如关注“avocado”，从而产生新内容。

若在所有层注入，原图约束过强，模型会直接复制源图、忽略新 prompt；若只在非关键层注入，约束又不够，身份和背景容易变化。

因此：

$$
\boxed{
\text{vital-only injection}
=\text{原图保真约束与新文字编辑能力之间的平衡点。}
}
$$

它不需要人工 Mask；“哪里保留、哪里改变”主要由多模态 Attention 自己判断。但这也意味着它不提供像 Mask 那样严格的像素级区域保证。

## 七、生成图像的完整编辑流程

对模型自己生成、已知 seed 的源图：

1. 准备源 prompt、目标 prompt 和同一个 seed；
2. 从同一份高斯噪声复制出源分支和编辑分支；
3. 两条分支按相同时间步并行求解 flow ODE；
4. 普通层中，两条分支分别计算；
5. vital layer 中，把源分支图像 token 的 $K,V$ 注入编辑分支；
6. 编辑分支保留目标 prompt 的文字 token 和自己的 $Q$；
7. 重复全部采样步，解码得到源图与编辑图。

官方实现没有另设“只在某一段时间注入”的时间门控：所选 vital layers 会在每一个采样步执行上述替换。

这一情形不需要 inversion，因为生成源图所用的初始噪声本来就已知。

## 八、真实图像为什么多出一个 inversion 问题

真实图片只有最终图像，没有与它对应的初始噪声。要在两条对齐的 flow 轨迹上编辑，先要寻找一个能重建该图片的噪声 latent，这就是 inversion。

### 1. 连续流的基本式

论文设 $p_0$ 为高斯噪声分布，$p_1$ 为图像数据分布，速度场 $u_t$ 产生流：

$$
\boxed{
\frac{d}{dt}\phi_t(x)=u_t(\phi_t(x)),
\qquad \phi_0(x)=x.
}
$$

含义是：当前位置沿模型预测的速度移动。ODE 求解器把这条连续曲线离散成许多小步。

### 2. Euler 正向步与逆向步

论文把 FLUX 调度器的一步写成：

$$
z_{t-1}=z_t+(\sigma_{t+1}-\sigma_t)u_t(z_t).
\tag{3}
$$

- $z_t$：当前 latent；
- $u_t(z_t)$：模型在当前位置预测的速度；
- $\sigma_{t+1}-\sigma_t$：本次沿速度走多远。

反方向近似为：

$$
\boxed{
z_t=z_{t-1}+(\sigma_t-\sigma_{t+1})u_t(z_{t-1}).
}
\tag{4}
$$

它依赖局部近似：

$$
u_t(z_t)\approx u_t(z_{t-1}),
$$

即步长足够小时，相邻位置的速度差不大。必须理解的是“Euler = 当前点速度 × 小时间间隔；inversion = 反向走”，不必死背实现中的下标正负号。

### 3. 为什么朴素 inverse Euler 在 FLUX 上不够好

真实图像经 VAE 编码得到的 latent 位于数据附近。作者发现，即使增加很多步，朴素逆 Euler 仍可能导致重建时眼睛、肢体或背景变形。论文推测，实际相邻 latent 的速度变化较大，前述局部近似不够准确。

## 九、第三项贡献：Latent Nudging

作者提出一个很简单、但主要依靠实验发现的技巧：反演前把干净图像 latent 乘以一个略大于 1 的标量。

$$
\boxed{
z_0^{nudged}=\lambda z_0,
\qquad \lambda=1.15.
}
$$

它把 latent 轻微推离训练数据流形；在像素上几乎看不出变化，却使 inverse Euler 得到的轨迹更容易被后续采样重建，并使编辑更集中于目标区域。

必须注意：

- $\lambda=1.15$ 是经验超参数，不是从定理严格推导出来的；
- $\lambda=1$ 即不 nudging，论文即使使用 1000 个反演/去噪步也重建较差；
- 作者测试中，$1.15$ 是能完整重建的最小值；
- 过大如 $3.0$ 会过饱和；小于 $1$ 会严重破坏重建。

论文实际还缓存 inversion 过程中每一步的源 latent。仅缓存可以重建原图，但编辑时仍可能改坏身份；**latent nudging + latent caching** 才兼顾重建和编辑，并把实际流程降到 50 步。

## 十、真实图像的完整流程

```text
真实 RGB 图像
→ VAE 编码为图像 latent z0
→ z0 × 1.15（latent nudging）
→ 按源 prompt 做 inverse Euler，并缓存每个时间步的源 latent
→ 得到噪声端 latent
→ 复制为源重建分支与目标编辑分支
→ 50 步生成；每步使用对应的缓存源轨迹
→ 只在 vital layers 注入源图视觉 K、V
→ VAE 解码得到编辑图
```

所以 Stable Flow 的真实图编辑**不是 inversion-free**。

## 十一、与 FlowEdit 的关键区别

| 维度 | FlowEdit | Stable Flow |
|---|---|---|
| 主要对象 | 真实图编辑 | 生成图和真实图编辑 |
| 是否需要真实图 inversion | 不需要 | 需要 |
| 改哪里 | 外部采样 ODE 的速度 | 模型内部 Attention |
| 核心量 | 目标速度减源速度 | 源分支图像 K、V |
| 是否双分支 | 每步构造源/目标查询 | 源生成轨迹与编辑轨迹并行 |
| 保真来源 | 共享噪声下速度差抵消公共成分 | 关键层注入源图视觉信息 |
| 额外技巧 | Monte Carlo 平均、时间范围 | 关键层检测、latent nudging/caching |

两者都不训练新权重，但“training-free”不等于流程相同。

## 十二、训练、离线分析和推理分别发生什么

### 原始模型训练

FLUX/SD3 已经由模型作者预训练好；Stable Flow 不参与这一步，也没有新的训练损失。

### Stable Flow 的一次性离线分析

对每一层用 64 个 prompt/seed 逐层旁路、重新生成，再算 DINOv2 相似度，得到关键层集合。它有计算成本，但没有梯度下降和参数更新。

### 每张图的推理

- 生成图：两条并行采样轨迹 + 关键层 K、V 注入；
- 真实图：额外加 VAE 编码、nudging、inversion/caching，再双分支编辑。

## 十三、实验怎样证明方法有效

### 1. 数据与任务

作者从 COCO validation 中筛出含明显人或动物的图：主要非刚性主体占图像面积至少 5%、至多 33%。围绕主体设计：

- 非刚性动作编辑；
- 添加物体；
- 替换物体；
- 场景编辑。

最终共 3,200 个编辑样本。

### 2. 对比方法

- SDEdit；
- Prompt-to-Prompt + Null-text Inversion；
- InstructPix2Pix；
- MagicBrush；
- MasaCTRL。

主实验骨干模型为 FLUX.1-dev；补充材料还在 SD3 上验证同一机制。

### 3. 三个自动指标

- $CLIP_{txt}$：编辑图与目标文字是否相符；
- $CLIP_{img}$：编辑图与输入图是否相似；
- $CLIP_{dir}$：从源文字到目标文字的变化方向，是否与源图到编辑图的变化方向一致。

三个指标必须一起看。$CLIP_{img}$ 很高也可能只是完全没编辑；因此还要同时看文字遵循和方向一致性。

### 4. 主定量结果

| 方法 | $CLIP_{txt}\uparrow$ | $CLIP_{img}\uparrow$ | $CLIP_{dir}\uparrow$ |
|---|---:|---:|---:|
| SDEdit | 0.24 | 0.71 | 0.07 |
| P2P+NTI | 0.21 | 0.76 | 0.08 |
| Instruct-P2P | 0.22 | 0.87 | 0.07 |
| MagicBrush | 0.24 | 0.88 | 0.11 |
| MasaCTRL | 0.20 | 0.76 | 0.03 |
| Stable Flow | 0.23 | **0.92** | **0.14** |

严谨结论：Stable Flow 的文字相似度不是最高，但输入保持和图文变化方向最好，说明它在“该改的改、不该改的留”这一综合目标上更平衡。

### 5. 最关键的消融实验

| 方案 | $CLIP_{txt}\uparrow$ | $CLIP_{img}\uparrow$ | $CLIP_{dir}\uparrow$ |
|---|---:|---:|---:|
| Stable Flow | 0.23 | 0.92 | 0.14 |
| 所有层都注入 | 0.17 | 0.98 | 0.00 |
| 只在非关键层注入 | 0.25 | 0.72 | 0.09 |
| 所有层做 attention extension | 0.18 | 0.98 | 0.01 |
| 不做 latent nudging | 0.22 | 0.62 | 0.05 |

解释：

- 所有层注入：$CLIP_{img}=0.98$ 看似好，其实几乎只复制原图；$CLIP_{dir}=0$ 暴露出没有正确编辑；
- 非关键层注入：比较听文字，但原图保持明显变差；
- 不 nudging：真实图反演/重建质量下降，后续编辑跟着漂移；
- 完整方法取得最合理平衡。

### 6. 人类评价

AMT 采用二选一评价：目标文字遵循、输入保持、真实感、总体质量。每个样本由 5 人评价。Stable Flow 对所有基线、所有维度的胜率均超过随机水平 50%，论文报告二项检验 $p<0.05$。

> [!note] 原文数字核验
> 补充材料同时写了“每个 baseline 320 个评分”和“总计 1,920 个回答”，但表中只有 5 个 baseline，二者按字面相乘并不一致。因此汇报时只引用已明确的二选一设置、每样本 5 次评分、胜率与显著性结论，不自行替作者猜测总数。

这支持方法有效，但不能据此声称它在所有图片和所有编辑类型上都必然最好。

### 7. 稳健性补充

- DINOv2 与 DINOv1 选出的关键层集合相同、排序略不同；
- CLIP、LPIPS 会替换少数层，但实际编辑差别很小；
- 从关键层集合删掉约 20% 时变化较小，再删更多会出现身份和背景变化；
- 在 SD3 上也能找出关键层并用相同注入机制编辑，说明思想不只适用于 FLUX，但层号不能直接迁移。

## 十四、能做什么，不能做什么

### 较擅长

- 非刚性动作：站立变坐下、抬手等；
- 添加、替换、移除物体；
- 改时间、天气、场景属性；
- 连续多轮编辑；
- 保持一致风格生成一组变体；
- 图片中文字的颜色、大小写或内容修改。

### 论文明确承认的三项局限

- **大幅风格转换较差**：强视觉注入会阻碍从照片彻底变成油画、动画等；
- **物体拖拽较差**：文本模型本身空间指令遵循有限；
- **完整背景替换较差**：源视觉注入可能造成旧背景泄漏；

### 从机制可以进一步看出的工程边界

- 不使用显式 Mask，无法保证严格像素区域不变；
- 关键层集合依赖具体模型，需要为新架构重新检测；
- 逐层旁路分析和双分支采样增加离线与推理成本；
- latent nudging 是经验技巧，理论解释仍不充分。

官方公开 runner 目前直接支持的是 FLUX.1-dev，并硬编码了最终关键层；论文补充材料展示了 SD3 结果，但仓库没有提供完整的 64 样本逐层扫描脚本或 SD3 runner。因此，“方法可迁移”与“官方代码已一键覆盖所有 DiT”不是同一件事。

## 十五、论文真正的贡献与边界

### 真正贡献

1. 把流模型相同 seed 下“多样性偏低”转化为编辑优势；
2. 提出可自动复用的逐层旁路方法，找出 DiT 的 vital layers；
3. 证明只在 vital layers 注入源图视觉 K、V，可兼顾编辑和保持；
4. 提出 latent nudging，改善 FLUX 真实图 inverse Euler 的重建与编辑；
5. 通过 FLUX 主实验、SD3 补充实验、自动指标、人类评价和消融验证各组件。

### 不应夸大的地方

- 它没有提出新的生成模型训练方法；
- training-free 不表示零计算、零超参数或不需要 inversion；
- 高图像相似度不能单独证明编辑好；
- 关键层重要是消融与编辑实验给出的经验结论，不等于已完整解释每层的语义机制；
- $\lambda=1.15$ 是实验选择，不是普适常数。

## 十六、必须掌握与了解即可

### 必须能讲清楚

- 为什么同 seed 的低多样性反而利于编辑；
- U-Net 的层级经验为什么不能直接搬到 DiT；
- 怎样利用 residual bypass + DINOv2 定义 vitality；
- Attention 注入准确替换的是源图视觉 $K,V$，而非 $Q$ 或整层输出；
- 为什么所有层注入会复制原图，非关键层注入又约束不足；
- 生成图和真实图的流程差异；
- inverse Euler 的直觉、局部近似和 latent nudging 的作用；
- Stable Flow 与 FlowEdit 的本质区别；
- 三个指标和关键消融各证明什么；
- 方法的三项主要局限。

### 了解即可

- DINOv1/CLIP/LPIPS 得到的每一组精确层号；
- 用户研究的每一个胜率和 $p$ 值；
- FLUX 官方代码的每一个工程参数；
- ODE 求解误差的严格数值分析证明。

## 十七、汇报常见问题与标准答案

### Q1：Stable Flow 为什么叫 training-free？

因为它冻结预训练 FLUX/SD3 的全部权重，不反向传播、不优化新参数。逐层旁路寻找关键层是离线分析，不是训练。

### Q2：Vital layer 如何判断？

分别绕过每一层，用相同 prompt/seed 重新生成，并用 DINOv2 比较与完整模型结果的感知相似度。绕过后变化越大，$1-\text{similarity}$ 越大，该层 vitality 越高。

### Q3：为什么关键层分布在网络各处？

DiT 是连续 Transformer block，没有 U-Net 固定的下采样—上采样分辨率层级；图像结构、身份和细节的重要计算会分布在不同位置，必须通过实验检测。

### Q4：注入时到底复制了什么？

在关键层，把编辑分支中图像 token 的 Attention Key、Value 替换为源分支对应的 Key、Value；编辑分支 Query 和目标文字 token 仍保留。

### Q5：为什么不是所有层都注入？

所有层都注入会让原图约束过强，模型几乎复制原图而忽略目标 prompt；只注入非关键层又不足以稳定身份和背景。消融实验直接支持 vital-only 的平衡。

### Q6：真实图为什么需要 inversion，生成图为什么不需要？

生成图的初始 seed/noise 已知；真实图只有终点，需要先找到能重建它的噪声 latent，才能构造对齐的源/目标生成轨迹。

### Q7：Latent nudging 做了什么？

在真实图反演前令 $z_0^{nudged}=1.15z_0$，轻微移出训练分布。作者经验上发现这能改善 inverse Euler 的重建并减少非目标区域漂移；它是实验技巧，不是严格理论结论。

### Q8：它和 Prompt-to-Prompt 有什么联系？

两者都在 Attention 内部控制编辑。Prompt-to-Prompt 重点控制文本—图像 Cross-Attention map；Stable Flow 面向 DiT/MM-DiT，在自动发现的关键层替换源图视觉 token 的 $K,V$。

### Q9：它和 FlowEdit 最大区别是什么？

FlowEdit 不需要 inversion，也不改内部 Attention，而是在外部利用目标速度减源速度构造编辑 ODE；Stable Flow 对真实图需要 inversion，并在模型内部注入关键层视觉 $K,V$。

### Q10：实验最有力的证据是什么？

完整方法在主表取得最高 $CLIP_{img}=0.92$ 和 $CLIP_{dir}=0.14$；所有层注入虽然图像相似度升到 0.98，但方向分数降为 0，说明只是复制；去掉 nudging 后图像相似度降到 0.62。消融把各设计与其作用对应起来。

## 十八、30 秒口述版

> Stable Flow 是一个面向 FLUX、SD3 等 flow-based DiT 的无需训练图像编辑方法。作者先利用残差连接逐层绕过 Transformer block，并用 DINOv2 比较生成结果，从而自动找出对图像形成最重要且分散在网络各处的 vital layers。编辑时，源图与目标图使用相同初始噪声并行生成，只在这些关键层把源分支图像 token 的 Attention Key 和 Value 注入目标分支，同时保留目标 Query 和目标文字信息，因此既能保留身份背景，又能执行新 prompt。真实图还要先 inversion；作者用 $z_0\leftarrow1.15z_0$ 的 latent nudging 加缓存来改善 FLUX 的 inverse Euler 重建。实验和消融表明，所有层注入会只复制原图，非关键层注入又保持不足，关键层注入取得更好的平衡。

## 十九、与前课连接

- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)：速度场、ODE 与 Euler 是真实图 inversion 的基础。
- [09-Scalable Diffusion Models with Transformers（DiT）](#/notes/dit)：解释为什么生成网络由 Transformer block 构成。
- [10-Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT）](#/notes/mmdit)：解释图像和文字 token 如何在多模态 Attention 中共同计算。
- [13-Prompt-to-Prompt Image Editing with Cross-Attention Control](#/notes/prompt-to-prompt)：同属推理时 Attention 控制，但控制对象不同。
- [14-FlowEdit - Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](#/notes/flowedit)：同为 flow 模型的 training-free 编辑，但一个改内部特征，一个改外部速度，并且 inversion 要求相反。
- [16-Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet）](#/notes/controlnet)：对比训练免费 DiT 编辑与需要训练的 U-Net 空间条件控制。
