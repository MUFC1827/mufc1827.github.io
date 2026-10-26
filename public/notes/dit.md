# 09 Scalable Diffusion Models with Transformers（DiT）

## 来源

- 论文全名：**Scalable Diffusion Models with Transformers**
- 作者：William Peebles、Saining Xie
- 会议：ICCV 2023
- CVF 正式页面：<https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html>
- CVF 正式论文：<https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf>
- 官方代码：<https://github.com/facebookresearch/DiT>

> 正确简称是 **DiT（Diffusion Transformer）**，不是 DiM。

## 本节主线

DiT 没有发明新的扩散过程，而是在 LDM 框架中，把每个时间步负责预测噪声的卷积 U-Net 换成处理 latent patch token 的 Transformer，并研究扩大 Transformer 的计算量能否稳定改善生成质量。

$$
\boxed{
\text{DDPM 决定学什么，LDM 决定在哪里学，DiT 决定用什么网络来学。}}
$$

$$
\underbrace{\text{VAE}+\boxed{\text{U-Net}}+\text{Scheduler}}_{\text{常见 LDM}}
\quad\longrightarrow\quad
\underbrace{\text{VAE}+\boxed{\text{Transformer}}+\text{Scheduler}}_{\text{DiT}}
$$

因此，VAE、前向加噪公式、噪声预测 Loss、反向采样和 CFG 都没有因为 DiT 而消失。

## 一、DiT 保留了什么，替换了什么

### 从 DDPM 保留

$$
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I)
$$

$$
L_{\mathrm{simple}}
=\mathbb E\left\|\epsilon-\epsilon_\theta(z_t,t,y)\right\|_2^2.
$$

- 训练时随机抽一个时间步 $t$ 和一份真实噪声 $\epsilon$；
- 网络学习预测噪声；
- 推理从高斯噪声 $z_T$ 开始，按时间步逐步去噪；
- Scheduler 仍根据预测结果计算 $z_{t-1}$。

### 从 LDM 保留

$$
x\xrightarrow{\mathcal E}z_0,
\qquad
z_0\xrightarrow{\mathcal D}\widehat x.
$$

- 冻结的 VAE Encoder 把 RGB 图像压缩到 latent space；
- 扩散过程在较小的 latent feature map 中运行；
- 最后只用 VAE Decoder 解码一次。

### 唯一核心替换

$$
\epsilon_\theta^{\mathrm{U\text{-}Net}}(z_t,t,y)
\quad\longrightarrow\quad
\epsilon_\theta^{\mathrm{DiT}}(z_t,t,y).
$$

DiT 替换的是**去噪网络 backbone**，不是 VAE、Scheduler 或扩散数学。论文仍使用卷积式 VAE，因此完整系统是“卷积 VAE + Transformer 去噪器”的混合系统。

## 二、与原始 Transformer 的联系和区别

DiT 复用了已经学过的：

- Multi-Head Self-Attention；
- $Q,K,V$ 的可训练线性投影；
- FFN / MLP；
- Residual Connection；
- LayerNorm；
- Positional Embedding。

注意力公式不变：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}}\right)V.
$$

变化在于 token 的含义：语言 Transformer 的 token 是词或子词；DiT 的 token 是带噪 latent 的一小块 patch。

| 项目 | 翻译 Transformer | DiT |
|---|---|---|
| 输入 token | 词或子词 | noisy latent patch |
| 主结构 | Encoder + 自回归 Decoder | Encoder 风格的 block 堆叠 |
| 位置 | 一维序列位置 | 二维 latent 网格位置 |
| Causal Mask | Decoder 需要 | 不需要 |
| 输出 | 下一个 token 的词表分数 | 所有位置的噪声和方差参数 |
| 单次前向 | 训练时并行目标位置 | 并行处理所有图像 patch |
| 完整生成 | 按 token 顺序生成 | 按扩散时间步顺序去噪 |

DiT 没有逐 patch 生成图像。一个扩散时间步里，全部 patch 已经存在并可相互观察；串行发生在：

$$
z_T\rightarrow z_{T-1}\rightarrow\cdots\rightarrow z_0.
$$

## 三、怎样把 latent 变成 Transformer token

### 1. VAE 先压缩图像

论文使用下采样因子 $f=8$ 的预训练 VAE：

$$
[B,3,256,256]
\xrightarrow{\text{VAE Encoder}}
[B,4,32,32].
$$

VAE 得到干净 latent $z_0$，加噪后得到形状不变的 $z_t$。

### 2. Patchify

设 latent 的空间大小为 $I\times I$，通道数为 $C$，patch 边长为 $p$。不重叠 patch 的数量是：

$$
\boxed{T=\left(\frac Ip\right)^2}.
$$

每个 patch 包含 $p^2C$ 个数，拉平成向量后用同一个可训练线性层投影到 Transformer 隐藏维度 $d$：

$$
r_i\in\mathbb R^{p^2C},
\qquad
e_i=r_iW_{\mathrm{patch}}+b_{\mathrm{patch}},
\qquad
W_{\mathrm{patch}}\in\mathbb R^{p^2C\times d}.
$$

$W_{\mathrm{patch}}$ 不是人工设置的；它和 $W^Q,W^K,W^V$ 一样，由 Loss 反向传播学习。所有位置共享同一组 $W_{\mathrm{patch}},b_{\mathrm{patch}}$。

对于 $I=32,C=4$：

| patch size $p$ | patch 网格 | token 数 $T$ | 每块原始维数 $p^2C$ |
|---:|---:|---:|---:|
| 2 | $16\times16$ | 256 | 16 |
| 4 | $8\times8$ | 64 | 64 |
| 8 | $4\times4$ | 16 | 256 |

Patchify 只是把二维张量重新组织成 token 并投影维度；它不是 VAE 压缩，也不是文字 Tokenizer。

### 3. 二维位置编码

Patch Embedding 后加入固定的二维 sin-cos 位置编码：

$$
X_0=\operatorname{PatchEmbed}(z_t)+P_{\mathrm{pos}},
\qquad X_0\in\mathbb R^{B\times T\times d}.
$$

Self-Attention 单独看只知道有哪些向量，不知道它们原来在左上、右下还是相邻位置。二维位置编码把原网格坐标写入 token；它与时间步编码不是一回事。

## 四、DiT-XL/2 的完整张量流程

以 256×256 图像、batch size 为 $B$ 为例：

```text
真实图像 x                         [B, 3, 256, 256]
  ↓ 冻结 VAE Encoder
干净 latent z₀                    [B, 4, 32, 32]
  ↓ 随机 t、ε，加噪
带噪 latent zₜ                    [B, 4, 32, 32]
  ↓ p=2 的 Patchify
256 个 patch，每块 2×2×4=16 个数  [B, 256, 16]
  ↓ 共享 Linear：16 → 1152
图像 token                        [B, 256, 1152]
  ↓ 加二维位置编码
  ↓ 28 个 adaLN-Zero DiT Block
处理后的 token                    [B, 256, 1152]
  ↓ Final adaLN + Linear：1152 → 32
每个 token 的输出 patch           [B, 256, 32]
  ↓ Unpatchify
空间输出                           [B, 8, 32, 32]
  ↓ 沿通道拆成两半
噪声预测 ε̂                       [B, 4, 32, 32]
方差相关参数                       [B, 4, 32, 32]
```

`DiT-XL/2` 的含义：

- `XL`：28 层、隐藏维度 1152、16 个注意头；
- `/2`：latent patch size $p=2$，不是两层或两步去噪。

由于 VAE 已经下采样 8 倍，一个 $2\times2$ latent patch 粗略对应原图 $16\times16$ 的区域。

## 五、一个 DiT Block 内部在做什么

### 1. Self-Attention：不同图像区域交换信息

对于 DiT-XL/2：

$$
X\in\mathbb R^{B\times256\times1152}.
$$

16 个头意味着每头维度：

$$
d_h=1152/16=72.
$$

每个头的形状可写成：

$$
Q,K,V\in\mathbb R^{B\times256\times72},
\qquad
A=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{72}}\right)
\in\mathbb R^{B\times256\times256}.
$$

一个 patch 的 Query 会与所有 patch 的 Key 比较，再按权重汇总 Value。直观上，它在判断：“为了估计我这个位置的噪声，应参考图中哪些区域？”Q、K、V 都来自同一组图像 token，所以这是 Self-Attention；没有 Causal Mask。

多头输出仍按已经学过的方式汇总：

$$
\operatorname{MultiHead}
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_{16})W^O.
$$

### 2. MLP / FFN：加工每个 token 内部特征

Self-Attention 让 token 之间交换信息，MLP 对每个 token 自己的 1152 维特征做非线性加工。DiT-XL 的 MLP ratio 为 4：

$$
1152\rightarrow4608\rightarrow1152.
$$

同一层 MLP 在所有 patch 位置共享参数。

### 3. Residual 与 LayerNorm

Residual 保留原信息并让深网络更容易传递梯度；LayerNorm 稳定每个 token 内部特征的尺度。DiT 使用 pre-norm 风格，并进一步让 LayerNorm 受条件控制。

## 六、三种信息不要混淆

| 信息 | 回答的问题 | 注入位置 |
|---|---|---|
| 二维位置编码 $P_{\mathrm{pos}}$ | 这个 patch 在图中哪里？ | 输入 token 相加 |
| 时间步嵌入 $e_t(t)$ | 当前噪声有多重？ | 每个 DiT Block 的条件调制 |
| 类别嵌入 $e_y(y)$ | 要生成哪个 ImageNet 类别？ | 每个 DiT Block 的条件调制 |

原始论文的条件是 ImageNet 类别编号，不是自然语言 prompt，也没有使用 CLIP Text Encoder。时间步与类别分别编码成 $d$ 维向量，再相加：

$$
c=e_t(t)+e_y(y),
\qquad c\in\mathbb R^{B\times d}.
$$

同一张图的全体 patch 使用同一个全局条件 $c$，但各 patch 本身的内容和位置不同，所以不会变成相同 token。

## 七、adaLN 与 adaLN-Zero

### 1. 从普通 LayerNorm 开始

对一个 token $x\in\mathbb R^d$，LayerNorm 先在它的 $d$ 个特征维度上归一化：

$$
\operatorname{LN}(x)
=\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\varepsilon}}.
$$

作用是让不同样本、不同层的数值尺度更稳定，但归一化不是最终预测结果，只是让后面的 Attention 和 MLP 更容易训练。

### 2. Adaptive LayerNorm

adaLN 的全称是 Adaptive Layer Normalization。它让归一化后的缩放和偏移由条件 $c$ 决定：

$$
\operatorname{adaLN}(x,c)
=(1+\gamma(c))\odot\operatorname{LN}(x)+\beta(c).
$$

- $\gamma(c)$：根据“时间步 + 类别”决定各特征维度放大或缩小多少；
- $\beta(c)$：决定各特征维度平移多少；
- $\odot$：逐元素相乘。

直观上，普通 LayerNorm 只是“把特征整理整齐”；adaLN 还告诉每层：“现在噪声这么重、目标是这个类别，你要采用相应的加工方式。”

### 3. adaLN-Zero 的完整 block

条件 $c$ 经过一个小 MLP，一次产生 6 组 $d$ 维参数：

```text
shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp
```

公式为：

$$
u=(1+\gamma_{\mathrm{MSA}}(c))\odot\operatorname{LN}(x)
+\beta_{\mathrm{MSA}}(c),
$$

$$
x'=x+\alpha_{\mathrm{MSA}}(c)\odot\operatorname{MSA}(u),
$$

$$
v=(1+\gamma_{\mathrm{MLP}}(c))\odot\operatorname{LN}(x')
+\beta_{\mathrm{MLP}}(c),
$$

$$
x_{\mathrm{out}}
=x'+\alpha_{\mathrm{MLP}}(c)\odot\operatorname{MLP}(v).
$$

其中 $\alpha$ 是 residual gate，控制 Attention 或 MLP 的修改量有多少加回主干。

### 4. “Zero” 究竟是什么意思

生成上述 6 组参数的最后一个线性层被零初始化，所以训练刚开始时：

$$
\gamma\approx0,
\qquad
\beta\approx0,
\qquad
\alpha\approx0.
$$

因为公式里是 $1+\gamma$，LayerNorm 输出并没有被清零；真正暂时关闭残差分支的是 gate $\alpha=0$：

$$
x_{\mathrm{out}}\approx x.
$$

所以每个深层 block 初始接近“什么也不改”的恒等映射，训练较稳定；训练开始后这些参数会通过反向传播变成非零。`Zero` 不表示所有网络权重都为零，也不表示它永远不工作。

## 八、论文比较的四种条件注入方式

1. **In-context conditioning**：把 $t$ 和类别 $y$ 当作两个额外 token 拼进图像 token 序列。
2. **Cross-Attention**：图像 patch 是 Query，长度为 2 的 $t,y$ 条件序列提供 Key 和 Value。
3. **adaLN**：让 $t+y$ 产生 LayerNorm 的 shift 和 scale。
4. **adaLN-Zero**：在 adaLN 上加入零初始化 residual gate。

DiT-XL/2 训练 400K steps、无 CFG 的消融结果：

| 条件方式 | GFLOPs | FID↓ |
|---|---:|---:|
| In-context | 119.37 | 35.24 |
| Cross-Attention | 137.62 | 26.14 |
| adaLN | 118.56 | 25.21 |
| adaLN-Zero | 118.64 | **19.47** |

因此论文后续采用 adaLN-Zero。这个结论只针对“时间步 + 单个 ImageNet 类别”这一实验，不能推出长文本条件下 adaLN 永远优于 Cross-Attention。

## 九、输出层与 Unpatchify

论文在末端使用 Final adaLN 和共享 Linear，把每个 token 变回一个输出 patch：

$$
[B,T,d]
\xrightarrow{\mathrm{Linear}}
[B,T,p^2(2C)]
\xrightarrow{\mathrm{Unpatchify}}
[B,2C,I,I].
$$

当 $I=32,C=4,p=2$：

$$
[B,256,1152]
\rightarrow[B,256,32]
\rightarrow[B,8,32,32].
$$

为什么输出 $2C=8$ 个通道？因为论文同时预测：

- 前 4 通道：噪声 $\widehat\epsilon$；
- 后 4 通道：反向高斯分布的对角协方差相关参数。

后 4 通道不是可以直接当作 $\sigma^2$ 的方差数值；ADM 的 learned-range 方案会用它在两个 log-variance 边界之间插值。概念主线仍是“DiT 预测噪声”，但完整模型还学习反向过程的不确定性。

论文将这个末端称为 Transformer decoder，但它不是《Attention Is All You Need》中带 Causal Mask 的 Decoder stack。还要区分：

- **DiT output head / linear decoder**：token $\rightarrow$ latent 中的噪声与方差参数；
- **VAE Decoder**：最终干净 latent $\rightarrow$ RGB 图像。

## 十、训练流程

训练数据是 ImageNet 图像 $x$ 和类别 $y$：

1. 冻结 VAE，得到 $z_0=\mathcal E(x)$；
2. 随机抽时间步 $t$；
3. 采样真实噪声 $\epsilon\sim\mathcal N(0,I)$；
4. 直接构造：

   $$
   z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon;
   $$

5. 把 $z_t$ patchify，加入位置编码；
6. 将 $t,y$ 编成条件，送入每个 adaLN-Zero block；
7. DiT 输出 $\widehat\epsilon$ 和方差参数；
8. 噪声部分使用 MSE，方差部分使用变分项：

   $$
   L=L_{\mathrm{MSE}}+\lambda L_{\mathrm{VB}};
   $$

   其中 $\lambda$ 表示实现中对变分项的权重；核心噪声预测仍由 MSE 监督。

9. 反向传播更新 DiT 的 Patch Embedding、Attention、MLP、条件嵌入、adaLN 和输出层参数；VAE 不更新；
10. 维护 EMA 权重用于评测和采样。

一次训练样本只随机学习一个 $t$，不是从 $1000$ 到 $1$ 完整去噪一遍。

论文主要设置：AdamW、恒定学习率 $10^{-4}$、weight decay 0、global batch size 256、水平翻转、1000 个训练扩散时间步、EMA decay 0.9999。

为了支持 CFG，训练时会以一定概率把类别替换成可学习的 null class。这样同一模型既会条件预测，也会“空条件”预测。

## 十一、推理流程

1. 从 $z_T\sim\mathcal N(0,I)$ 开始；
2. 指定 ImageNet 类别 $y$；
3. DiT 对全部 latent patch 并行预测噪声和方差参数；
4. Scheduler 根据这些预测计算 $z_{t-1}$；
5. 沿多个时间步重复 3–4；
6. 得到 $z_0$ 后，VAE Decoder 只运行一次，输出 RGB 图像。

若使用 CFG：

$$
\widehat\epsilon_{\mathrm{cfg}}
=\widehat\epsilon_{\varnothing}
+s\left(\widehat\epsilon_y-\widehat\epsilon_{\varnothing}\right).
$$

DiT 仍然是“空间 token 并行、扩散时间步串行”。换用 Transformer 并没有让多步采样自动消失。

## 十二、模型规模、Patch Size 与计算量

| 型号 | Blocks $N$ | Hidden $d$ | Heads | GFLOPs（$I=32,p=4$） |
|---|---:|---:|---:|---:|
| DiT-S | 12 | 384 | 6 | 1.4 |
| DiT-B | 12 | 768 | 12 | 5.6 |
| DiT-L | 24 | 1024 | 16 | 19.7 |
| DiT-XL | 28 | 1152 | 16 | 29.1 |

扩大一次 DiT 前向计算主要有两条路：

1. **扩大模型**：增加深度 $N$、隐藏维度 $d$ 和头数；
2. **增加 token**：减小 patch size，例如 `/8 → /4 → /2`。

$p$ 减半时，token 数变为 4 倍；Self-Attention 中两两比较的部分随 $T^2$ 增长，但整个模型还含线性投影和 MLP，因此不能简单宣称总计算严格变为 16 倍。论文的表述是总 GFLOPs 至少约增至 4 倍。

论文强调 GFLOPs 而不只看参数量，因为缩小 patch size 会大幅增加 token 数和实际前向计算，却几乎不改变参数量。

GFLOPs 表示一次前向的理论浮点运算量，不等于参数量、显存、墙上时间，也不等于生成一张图的总计算；完整采样还要多次前向。

## 十三、论文实验与正确结论

### 1. 扩大模型或增加 token 都改善 FID

作者训练 $4$ 种模型规模 $\times 3$ 种 patch size，共 12 个模型。400K steps、无 CFG 时：

| 模型 | `/8` FID↓ | `/4` FID↓ | `/2` FID↓ |
|---|---:|---:|---:|
| S | 153.60 | 100.41 | 68.40 |
| B | 122.74 | 68.38 | 43.47 |
| L | 118.87 | 45.64 | 23.33 |
| XL | 106.41 | 43.01 | **19.47** |

在论文测试的范围中：

- 固定 patch size，扩大模型通常降低 FID；
- 固定模型规模，减小 patch size、增加 token 通常降低 FID；
- GFLOPs 与 FID 的相关系数约为 $-0.93$。

这是特定数据集、训练配方和 12 个配置中的经验相关规律，不是“算力无限增大，质量必然无限提高”的数学定理。

### 2. 大模型可能更有效地使用训练计算

小模型即使训练更久，最后也可能比不上用更少训练步数的大模型。这说明选择合适的模型容量本身很重要，不能只靠延长训练补偿。

### 3. 多采样几步不能完全代替更强 backbone

论文中的一个对比：

- DiT-L/2，1000 个采样步：80.7 TFLOPs/图，FID-10K 为 25.9；
- DiT-XL/2，128 个采样步：15.2 TFLOPs/图，FID-10K 为 23.7。

在这一实验设置中，更大的 XL/2 用约五分之一的采样计算仍然更好。因此，增加采样步数不能完全弥补单步去噪网络能力不足；但不能外推为任何采样器、任何任务中多采样都无用。

### 4. 最终 ImageNet 结果

- 256×256：DiT-XL/2 使用 CFG scale 1.5 时，FID 为 **2.27**；无 CFG 时 FID 为 9.62。
- 512×512：DiT-XL/2 使用 CFG scale 1.5 时，FID 为 **3.04**，优于当时此前扩散模型 ADM 的 3.85；但没有优于表中 StyleGAN-XL 的 2.41。
- CFG 提高样本的条件一致性和 precision，但 recall 会下降，体现质量/条件强化与多样性的折中。

只能说这是论文发表时、ImageNet 与该评测设置下的结果，不能说 DiT 到今天仍是所有任务上的 SOTA。

## 十四、论文贡献与局限

### 主要贡献

1. 证明标准 ViT 风格 Transformer 可以作为 latent diffusion 的去噪 backbone，卷积 U-Net 不是唯一选择。
2. 给出 Patchify、条件注入和输出恢复组成的简单 DiT 设计空间。
3. 在四种方案中发现 adaLN-Zero 在该任务上效果最好且额外计算较低。
4. 系统研究模型深宽、patch size、GFLOPs 与生成质量的关系。
5. DiT-XL/2 在论文当时取得很强的 ImageNet 类别条件生成结果。

### 局限

- 原论文主要研究 ImageNet 类别条件生成，不是开放词汇 text-to-image；
- 仍依赖冻结的卷积 VAE，不是端到端纯 Transformer；
- Self-Attention 随 token 数增加较昂贵，高分辨率或更小 patch 会显著加大计算；
- 扩散采样仍需要多次网络前向；
- U-Net 对照并非全部在完全相同训练配方和预算下重新训练；
- 结论是所测范围内的经验 scaling 趋势，不能无限外推；
- FID 衡量一批生成图与真实图的分布距离，不能完整代替人类对单张图的评价。

## 十五、最容易混淆的点

1. DiT 是去噪网络架构，不是新的加噪或采样公式。
2. DiT 没有取代 LDM；原论文中的 DiT 本身就在 latent space 做扩散。
3. 被替换的是 U-Net，不是 VAE 或 Scheduler。
4. Patch 在 latent 上切，不是在原始 RGB 图上切。
5. `/2` 是 latent patch size，不是层数、时间步或 VAE 压缩率。
6. Patchify 不是 VAE 压缩，只是张量到 token 序列的重排与线性投影。
7. DiT 更像 Transformer Encoder/ViT，不是自回归 Decoder，没有 Causal Mask。
8. 所有 patch 在一次前向中并行预测；扩散时间步仍按顺序运行。
9. 位置编码、时间步编码、类别编码分别回答“哪里、噪声多重、生成什么”。
10. 原始论文是类别条件，不是 Stable Diffusion 式文本条件。
11. adaLN-Zero 的 Zero 指条件调制层和 residual gate 的零初始化，不是全网络权重为零。
12. 最终 8 通道由 4 通道噪声与 4 通道方差相关参数组成，后者不是直接的标准差。
13. 论文末端的 linear decoder 不是翻译 Transformer 的 Decoder stack。
14. Patch size 变小，参数量近似不变，但 token 和 GFLOPs 增加。
15. FID 越低通常越好；论文中的 scaling 是相关性而不是定理。

## 十六、公式与形状速记

$$
\boxed{
\begin{aligned}
z_0&=\mathcal E(x),\\
z_t&=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,\\
T&=\left(\frac Ip\right)^2,\\
X_0&=\operatorname{PatchEmbed}(z_t)+P_{\mathrm{pos}},\\
c&=e_t(t)+e_y(y),\\
X_N&=\operatorname{DiTBlocks}(X_0,c),\\
[\widehat\epsilon,\widehat v]
&=\operatorname{Unpatchify}(\operatorname{Linear}(X_N)),\\
L_{\mathrm{simple}}
&=\|\epsilon-\epsilon_\theta(z_t,t,y)\|_2^2.
\end{aligned}}
$$

必须记住的 256×256、DiT-XL/2 形状：

$$
[B,3,256,256]
\rightarrow[B,4,32,32]
\rightarrow[B,256,16]
\rightarrow[B,256,1152]
\rightarrow[B,256,32]
\rightarrow[B,8,32,32].
$$

## 十七、汇报问答

### 1. DiT 相比 LDM 改了什么？

只把 LDM 中负责预测噪声的 U-Net backbone 换成 Transformer。VAE latent space、前向加噪、diffusion loss、CFG、Scheduler 和最终 VAE 解码仍保留。

### 2. 为什么 latent 能交给 Transformer？

latent 是二维空间特征图。DiT 把它切成不重叠 patch，每个 patch 经过共享线性投影成为一个 token，再加二维位置编码，因此可以用 Transformer 处理。

### 3. DiT-XL/2 的 `/2` 是什么？

表示 latent patch size $p=2$。XL 表示深度、宽度和头数配置；`/2` 不表示两层、两步采样或缩小两倍。

### 4. Patch size 变小会怎样？

token 数增加，空间粒度更细，前向 GFLOPs 增加。在论文测试范围内通常得到更低 FID；参数量却近似不变，所以不能只用参数量衡量复杂度。

### 5. DiT 为什么仍需要位置编码？

Self-Attention 本身不知道 token 来自 latent 网格的哪里。二维位置编码让它区分不同空间坐标和相邻关系。

### 6. 为什么不需要 Causal Mask？

DiT 不是逐 patch 自回归生成。当前时间步的所有 latent patch 都已知，模型要同时为所有位置预测噪声，因此各 patch 可以互相观察。

### 7. 时间步和类别如何进入最佳 DiT Block？

它们分别编码后相加，再生成 LayerNorm 的 shift、scale 以及两个 residual gate，通过 adaLN-Zero 调节每一层 Attention 和 MLP。

### 8. adaLN-Zero 为什么有用？

条件可以控制每层的特征处理；零初始化 gate 让深层 block 在训练初始接近恒等映射，随后逐渐学习有效修改量。论文消融中它优于另外三种测试方案。

### 9. 为什么最终输出 $2C$ 个通道？

一半预测噪声，另一半预测反向高斯分布的对角协方差相关参数。二者都被 unpatchify 回原 latent 空间布局。

### 10. 为什么论文主要比较 GFLOPs，而不只看参数量？

参数量没有反映输入 token 数。减小 patch size 可以令参数量几乎不变，却显著增加每次前向计算；GFLOPs 更能表达这一差异。

### 11. 论文怎样说明 DiT 可以 scaling？

作者训练 4 种模型尺寸和 3 种 patch size 共 12 个模型。在所测范围中，增加深宽或增加 token 都持续降低 FID，GFLOPs 与 FID 明显负相关。

### 12. 给小模型增加采样步数能代替大模型吗？

该论文实验中不能：L/2 使用 1000 步和更多总采样计算，仍不如只用 128 步的 XL/2。这说明采样计算不能完全补偿 backbone 能力不足，但不是对所有方法的普遍定理。

### 13. 原始 DiT 是不是文字版 Stable Diffusion？

不是。它是 ImageNet 类别条件生成，输入 class label；论文只提出未来可把 DiT 作为文本生成图像系统的 backbone。

### 14. DiT 的并行与串行分别在哪里？

同一个扩散时间步内，全部 patch token 并行经过 Transformer 并同时预测噪声；不同扩散时间步之间仍要依次更新 latent。

### 15. 一句话说明 DiT 的论文价值。

DiT 证明 latent diffusion 的噪声预测器可以由可扩展的 Transformer 取代 U-Net，并在作者测试范围内呈现“增加模型或 token 的前向计算量，生成质量稳定改善”的经验规律。

## 相关笔记

- [05-Attention Is All You Need](#/notes/attention-is-all-you-need)
- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)
- [08-Experiment-Stable Diffusion 文生图、DDIM 与图像编辑](#/notes/stable-diffusion-lab)
- [10-Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT）](#/notes/mmdit)
 
## 后续连接

- [15-Stable Flow - Vital Layers for Training-Free Image Editing](#/notes/stable-flow)：不再假设 DiT 层按位置具有固定粗细功能，而是通过逐层旁路实验自动识别影响图像形成的关键 Transformer 层。
