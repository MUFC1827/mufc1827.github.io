# 10 Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT）

## 来源

- 论文全名：**Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**
- 作者：Patrick Esser、Sumith Kulal、Andreas Blattmann 等
- 会议：ICML 2024
- PMLR 正式页面：<https://proceedings.mlr.press/v235/esser24a.html>
- PMLR 正式 PDF：<https://raw.githubusercontent.com/mlresearch/v235/main/assets/esser24a/esser24a.pdf>
- 老师清单中的 arXiv 页面：<https://arxiv.org/abs/2403.03206>
- Stability AI 技术介绍：<https://stability.ai/news-updates/stable-diffusion-3-research-paper>
- 官方推理参考实现：<https://github.com/Stability-AI/sd3-ref>

这篇论文通常也被称为 **Stable Diffusion 3 / MM-DiT 论文**。

## 本节主线

这篇论文把前面学过的四条路线第一次汇合：

$$
\boxed{
\text{LDM 潜空间}
+
\text{改进的 Rectified Flow}
+
\text{DiT}
+
\text{文字—图像双向 MM-DiT}
}
$$

一句话概括：

> DiT 让 Transformer 学会处理 latent patch；MM-DiT 再让文字 token 和图像 patch 使用各自的参数，并在联合 Attention 中双向交换信息。

论文不只提出 MM-DiT，还同时改进了：

1. Rectified Flow 的训练时间步抽样；
2. VAE latent 表示；
3. 文字编码与训练 caption；
4. 高分辨率训练稳定性；
5. 模型 scaling 与偏好对齐。

最终质量是整套系统共同作用的结果，不能全部归因于 MM-DiT 一个模块。

## 一、与前面知识的关系

| 已学内容 | 本文保留什么 | 本文新增什么 |
|---|---|---|
| DDPM | 随机抽时间步、直接构造带噪输入、MSE、CFG | 主要目标从噪声预测改为速度预测 |
| Rectified Flow | 直线插值、速度场、ODE、Euler 采样 | 使用 Logit-Normal 更频繁抽取中间时间步 |
| LDM | 冻结 VAE、latent space、冻结文字编码器 | VAE latent 通道由常见 4 增至 16 |
| DiT | Patchify、位置编码、Transformer、条件调制、scaling | 从类别条件变成文字—图像双流 |
| LDM Cross-Attention | 图像特征读取文字条件 | 文字与图像 token 在联合 Attention 中双向更新 |

可以把系统写成：

$$
\boxed{
\text{SD3 技术主线}
=
\text{16 通道 VAE}
+
\text{3 个文字编码器}
+
\text{MM-DiT 速度预测器}
+
\text{Rectified Flow ODE Solver}.
}
$$

## 二、最重要的记号方向陷阱

第 07 讲采用：

$$
X_s=(1-s)X_{\mathrm{noise}}+sX_{\mathrm{data}},
$$

所以 $s=0$ 是噪声、$s=1$ 是数据。

本文采用：

$$
\boxed{
z_t=(1-t)x_0+t\epsilon,
}
$$

其中：

- $x_0$：真实图像的干净 latent；
- $\epsilon\sim\mathcal N(0,I)$：高斯噪声；
- $t=0$：数据；
- $t=1$：噪声。

二者是同一条直线，只要令：

$$
s=1-t,
$$

就有：

$$
(1-s)\epsilon+sx_0
=t\epsilon+(1-t)x_0
=z_t.
$$

因此区别只是时间箭头反过来了，不是两种不同 Rectified Flow。

## 三、Rectified Flow 的核心数学

### 1. 一般路径

论文先用统一形式表示数据和噪声之间的路径：

$$
z_t=a_tx_0+b_t\epsilon.
$$

$a_t$ 决定保留多少数据，$b_t$ 决定加入多少噪声。不同扩散或 flow formulation 的一个主要区别，就是怎样选 $a_t,b_t$。

Rectified Flow 取：

$$
a_t=1-t,
\qquad
b_t=t,
$$

所以：

$$
z_t=(1-t)x_0+t\epsilon.
$$

### 2. 速度标签怎样得到

对 $t$ 求导：

$$
\frac{dz_t}{dt}
=
\frac{d}{dt}\bigl((1-t)x_0+t\epsilon\bigr)
=
-x_0+\epsilon.
$$

因此条件直线的速度是：

$$
\boxed{
u_t=\epsilon-x_0.
}
$$

它与 $t$ 无关，因为这条监督线是恒速直线。

### 3. 训练 Loss

神经网络接收当前中间状态、时间和文字条件，预测速度：

$$
v_\theta(z_t,t,c).
$$

训练目标为：

$$
\boxed{
L_{\mathrm{RF}}
=
\mathbb E
\left[
\left\|
v_\theta(z_t,t,c)-(\epsilon-x_0)
\right\|_2^2
\right].
}
$$

含义：

- 输入：带噪 latent $z_t$、时间 $t$、文字条件 $c$；
- 标签：该训练配对的直线方向 $\epsilon-x_0$；
- 参数：MM-DiT 中所有可训练权重；
- 优化：让预测速度接近真实训练速度。

与 DDPM 对比：

| 方法 | 网络主要预测 |
|---|---|
| DDPM / 原始 LDM | 加入的噪声 $\epsilon$ |
| 本文 Rectified Flow | 速度 $\epsilon-x_0$ |

### 4. 为什么训练不需要求解 ODE

训练时已知 $x_0,\epsilon,t$，可以直接构造：

$$
z_t=(1-t)x_0+t\epsilon.
$$

因此每个训练样本只需随机抽一个 $t$，运行一次网络并计算 MSE，不需要先从 $0$ 数值积分到 $t$。这就是 **simulation-free training**。

### 5. 生成为什么能从噪声走向数据

网络学习的是随 $t$ 增大时“数据 $\rightarrow$ 噪声”的速度，但生成从 $t=1$ 向 $t=0$ 反向积分：

$$
z_{t-\Delta t}
\approx
z_t-\Delta t\,v_\theta(z_t,t,c).
$$

一维例子：

$$
x_0=2,
\qquad
\epsilon=-1.
$$

则：

$$
z_t=(1-t)2+t(-1)=2-3t,
\qquad
\frac{dz_t}{dt}=-3.
$$

从 $t=1$ 时的噪声 $z_1=-1$ 出发，取 $\Delta t=0.1$：

$$
z_{0.9}
\approx
-1-0.1(-3)
=-0.7.
$$

它确实朝数据 $2$ 的方向移动。正负号没有矛盾，原因是生成时 $dt<0$。

### 6. 训练直线不等于生成轨迹严格笔直

训练中的每个 $(x_0,\epsilon)$ 配对都是直线，但同一个中间位置可能由很多配对经过。MSE 学到的是条件平均速度，因此实际 ODE 轨迹仍可能弯曲。

本文没有把 Reflow、重新配对或一步蒸馏作为核心流程，所以：

$$
\boxed{
\text{使用 Rectified Flow}
\not\Rightarrow
\text{天然一步生成}.
}
$$

推理仍需要多次 MM-DiT 前向和 ODE 数值更新。

## 四、为什么要改变训练时间步的抽样

### 1. Uniform 的问题

经典 Rectified Flow 常令：

$$
t\sim\operatorname{Uniform}(0,1),
$$

即所有噪声程度被同样频繁地抽到。

作者认为中间区域通常更难：

- $t\approx0$：输入接近清晰数据；
- $t\approx1$：输入接近纯噪声；
- $t\approx0.5$：数据与噪声充分混合，感知结构处于最模糊的阶段。

因此训练预算不一定应平均分配，而应更常练习中间的困难区域。

### 2. Logit-Normal Sampling

先采样实数：

$$
u\sim\mathcal N(m,s^2),
$$

再用 Sigmoid 映射到 $0\sim1$：

$$
\boxed{
t=\operatorname{sigmoid}(u)
=
\frac{1}{1+e^{-u}}.
}
$$

例如：

$$
u=0\Rightarrow t=0.5,
\qquad
u=2\Rightarrow t\approx0.88,
\qquad
u=-2\Rightarrow t\approx0.12.
$$

当 $m=0,s=1$ 时，正态分布更常产生接近 0 的 $u$，所以映射后更常得到中间时间步。

论文使用的密度为：

$$
\pi_{\mathrm{ln}}(t;m,s)
=
\frac{1}
{s\sqrt{2\pi}\,t(1-t)}
\exp
\left[
-\frac{
\left(
\operatorname{logit}(t)-m
\right)^2
}{2s^2}
\right],
$$

其中：

$$
\operatorname{logit}(t)
=
\log\frac{t}{1-t}.
$$

密度中的 $1/[t(1-t)]$ 来自变量变换：

$$
\frac{d}{dt}\operatorname{logit}(t)
=
\frac{1}{t(1-t)}.
$$

必须记住的是采样过程 $u\rightarrow t$；完整概率密度会解释来源即可。

### 3. 它改变了什么、没有改变什么

Logit-Normal 改变的是：

> 训练时哪些时间步被抽到得更频繁。

它没有改变：

- 数据与噪声之间的直线路径；
- 每个样本的速度标签 $\epsilon-x_0$；
- 推理时 ODE Solver 的更新规则。

所以训练 timestep sampler 与推理 Scheduler / ODE Solver 不是一回事。

论文测试 61 种 formulation。综合不同数据集、EMA 设置、步数与 CFG 后，$\mathrm{rf/lognorm}(0,1)$ 的平均排名为 1.54，优于 uniform RF 的 5.67 和 LDM-Linear 的 2.88；尤其在少步采样中，Rectified Flow 下降较慢。

## 五、整个文生图系统包含哪些模块

### 图像路线

$$
X
\xrightarrow{\text{冻结 VAE Encoder}}
x_0
\xrightarrow{\text{RF 加噪}}
z_t
\xrightarrow{\text{Patchify}}
\text{image tokens}.
$$

### 文字路线

$$
\text{prompt}
\xrightarrow{\text{CLIP-L、CLIP-G、T5}}
\left(c_{\mathrm{vec}},c_{\mathrm{ctxt}}\right).
$$

### MM-DiT 路线

$$
\left(
\text{image tokens},
\text{text tokens},
t
\right)
\xrightarrow{\text{MM-DiT}}
\widehat v.
$$

### 采样与解码

$$
z_1
\xrightarrow{\text{多步 ODE 反向积分}}
z_0
\xrightarrow{\text{VAE Decoder}}
\widehat X.
$$

## 六、图像与文字表示

### 1. 16 通道 VAE

论文仍使用下采样倍数 $f=8$ 的 Autoencoder，但最终选择 16 个 latent 通道。

对于 256×256 图像：

$$
[B,3,256,256]
\rightarrow
[B,16,32,32].
$$

对于 1024×1024 图像：

$$
[B,3,1024,1024]
\rightarrow
[B,16,128,128].
$$

VAE 重建质量是后续 latent 生成质量的上限：VAE 已永久丢失的细节，MM-DiT 无法凭空恢复。增加 latent 通道可以保存更多信息，但也让生成模型要预测更复杂的 latent。

| VAE latent 通道 | FID↓ | Perceptual Similarity↓ | SSIM↑ | PSNR↑ |
|---:|---:|---:|---:|---:|
| 4 | 2.41 | 0.85 | 0.75 | 25.12 |
| 8 | 1.56 | 0.68 | 0.79 | 26.40 |
| 16 | **1.06** | **0.45** | **0.86** | **28.62** |

这些是 Autoencoder 重建指标，不是最终 8B 文生图模型的 FID。

### 2. 图像 Patch Token

使用 $2\times2$ latent patch。一个 patch 含：

$$
2\times2\times16=64
$$

个数。

256 分辨率：

$$
[B,16,32,32]
\rightarrow
[B,256,64]
\rightarrow
[B,256,D].
$$

1024 分辨率：

$$
[B,16,128,128]
\rightarrow
[B,4096,64]
\rightarrow
[B,4096,D].
$$

$D$ 是 Transformer hidden size，与 16 个 VAE latent channels 不是同一个维度。

### 3. 三个冻结的文字编码器

论文使用：

- CLIP-L/14；
- OpenCLIP bigG/14；
- T5-v1.1-XXL。

它们产生两类文字条件。

#### 全局条件 $c_{\mathrm{vec}}$

两个 CLIP 的 pooled output 分别为 768 和 1280 维，拼接得到：

$$
c_{\mathrm{vec}}\in\mathbb R^{B\times2048}.
$$

它是整段 prompt 的全局摘要，与时间步嵌入共同控制各 MM-DiT Block 的调制。

#### 逐 token 条件 $c_{\mathrm{ctxt}}$

CLIP 序列和 T5 序列在论文设置中组合为：

$$
c_{\mathrm{ctxt}}
\in
\mathbb R^{B\times154\times4096}.
$$

它保留每个文字 token 的细节，经 Linear 投影到公共 hidden size 后，真正进入联合 Attention。

所以：

- $c_{\mathrm{vec}}$：全局告诉网络整段 prompt 大意；
- $c_{\mathrm{ctxt}}$：保留各个词和字符级语义，参与位置相关的细致交流。

文字编码器被冻结，不表示进入 MM-DiT 后的文字 token 永远固定。冻结的是 CLIP/T5 参数；MM-DiT 内部的文字流仍会逐层更新。

## 七、MM-DiT 的核心：两套参数，一场联合 Attention

MM-DiT 的 MM 是 **Multimodal**，不是 Multi-Head。

### 1. 为什么不让文字和图像完全共享参数

文字 token 与图像 patch 的含义差异很大。如果让它们共用完全相同的 LayerNorm、QKV、输出投影和 MLP，会强迫两种模态采用同一套加工方式。

MM-DiT 因此保留两套模态专用参数：

- 文字流参数；
- 图像流参数。

但两条流并非互相隔离，它们会在 Attention 中相遇。

### 2. 分别计算 Q、K、V

图像流：

$$
Q_I=X_IW_I^Q,
\qquad
K_I=X_IW_I^K,
\qquad
V_I=X_IW_I^V.
$$

文字流：

$$
Q_T=X_TW_T^Q,
\qquad
K_T=X_TW_T^K,
\qquad
V_T=X_TW_T^V.
$$

$W_I$ 与 $W_T$ 是不同的可训练矩阵，但投影后每个头的维度相同，所以可以联合计算。

### 3. 沿 token 维拼接

$$
Q=[Q_T;Q_I],
\qquad
K=[K_T;K_I],
\qquad
V=[V_T;V_I].
$$

然后只做一次联合 Attention：

$$
\boxed{
O
=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_h}}
\right)V.
}
$$

若文字 token 数为 $N_T$，图像 token 数为 $N_I$，则每头注意力矩阵为：

$$
(N_T+N_I)\times(N_T+N_I).
$$

它包含四个区域：

$$
\begin{bmatrix}
\text{文字看文字} & \text{文字看图像}\\
\text{图像看文字} & \text{图像看图像}
\end{bmatrix}.
$$

这就是“文字—图像双向信息流”的具体数学意义。

### 4. 联合后再拆分

联合 Attention 输出再按 token 位置拆成：

$$
O=[O_T;O_I].
$$

- $O_T$ 回到文字流，使用文字专用输出投影、Residual 和 MLP；
- $O_I$ 回到图像流，使用图像专用输出投影、Residual 和 MLP。

最准确的类比是：

> 文字专家和图像专家使用各自的笔记和工具；进入同一场会议交换信息；会后再回到各自工作流继续加工。

### 5. 与 LDM Cross-Attention 的区别

传统 LDM Cross-Attention：

$$
Q=Q_{\mathrm{image}},
\qquad
K,V=K_{\mathrm{text}},V_{\mathrm{text}}.
$$

主要是图像读取固定文字条件。

MM-DiT：

- 图像可以读取文字；
- 文字也可以根据当前图像状态更新；
- 两种模态还各自保留 self-attention；
- 信息在每个 block 中双向融合。

### 6. 与 DiT 的关系

如果文字和图像共用一套参数，那么拼接 token 后的普通 DiT 可以看成 MM-DiT 的特殊情况。

MM-DiT 的关键新增不是“把 token 拼起来”本身，而是：

$$
\boxed{
\text{模态专用参数}
+
\text{联合双向 Attention}.
}
$$

## 八、完整张量例子

以 1024×1024 图像为例：

$$
[B,3,1024,1024]
\xrightarrow{\text{VAE}}
[B,16,128,128].
$$

Patchify：

$$
[B,16,128,128]
\rightarrow
[B,4096,64]
\rightarrow
[B,4096,D].
$$

文字：

$$
c_{\mathrm{ctxt}}
\in
\mathbb R^{B\times154\times4096}
\rightarrow
[B,154,D].
$$

联合 Attention 的 token 总数：

$$
4096+154=4250.
$$

每个头的分数矩阵约为：

$$
[B,4250,4250].
$$

这解释了为什么高分辨率时 Attention 计算和训练稳定性成为重要问题。

## 九、输出为什么不再是 DiT 的 $2C$

第 09 讲的原始 DiT 使用 DDPM 式反向高斯建模，主模型输出：

- $C$ 个噪声通道；
- $C$ 个方差参数通道；
- 共 $2C$。

本文使用 Rectified Flow ODE，网络直接预测速度场：

$$
v_\theta(z_t,t,c)
\in
\mathbb R^{B\times C_{\mathrm{lat}}\times h\times w}.
$$

所以输出与 latent 同形状。

1024 分辨率、16 通道 VAE：

$$
[B,4096,D]
\xrightarrow{\text{Final Linear}}
[B,4096,64]
\xrightarrow{\text{Unpatchify}}
[B,16,128,128].
$$

这里没有额外的 16 个 learned variance 通道，因为确定性 ODE 不需要 DDPM 反向高斯中的协方差预测。

最终只把图像流还原为空间速度张量；文字流不会被 VAE 解码成图像。

## 十、完整训练流程

1. 输入图像 $X$ 和 caption；
2. 冻结 VAE，得到干净 latent $x_0$；
3. 冻结 CLIP-L、CLIP-G、T5，得到 $c_{\mathrm{vec}},c_{\mathrm{ctxt}}$；
4. 采样 $\epsilon\sim\mathcal N(0,I)$；
5. 通过 Logit-Normal 抽时间步：

   $$
   u\sim\mathcal N(0,1),
   \qquad
   t=\operatorname{sigmoid}(u);
   $$

6. 构造：

   $$
   z_t=(1-t)x_0+t\epsilon;
   $$

7. MM-DiT 接收图像 token、文字 token、时间和全局条件；
8. 输出速度预测：

   $$
   \widehat v=v_\theta(z_t,t,c);
   $$

9. 计算：

   $$
   L=\|\widehat v-(\epsilon-x_0)\|_2^2;
   $$

10. 反向传播更新 MM-DiT；VAE 和三个文字编码器保持冻结；
11. 更新 EMA 权重。

为了训练 CFG，三个文字编码器的输出会分别随机置零。论文实验中每个编码器的 drop rate 约为 46.3%，因此三者同时为空的概率约为：

$$
0.463^3\approx0.10.
$$

也就是约 10% 样本训练空条件分支。

## 十一、完整推理流程

1. 编码 prompt，得到文字条件；
2. 从高斯 latent 开始：

   $$
   z_1\sim\mathcal N(0,I);
   $$

3. MM-DiT 分别计算条件与空条件速度；
4. 使用 CFG 组合预测方向；
5. Euler 或其他 ODE Solver 从 $t=1$ 向 $t=0$ 更新：

   $$
   z_{t-\Delta t}
   \approx
   z_t-\Delta t\,v_\theta(z_t,t,c);
   $$

6. 重复若干次得到 $z_0$；
7. VAE Decoder 最后解码一次得到 RGB 图像。

MM-DiT 在一个时间步内并行处理全部文字和图像 token；不同 ODE 时间步仍串行。

## 十二、除 MM-DiT 外的系统改进

### 1. 更好的 caption

真实数据集 caption 经常只写主体，忽略背景、位置、数量和画面文字。作者使用 CogVLM 生成更详细的合成 caption，并采用：

$$
50\%\text{ 原始 caption}
+
50\%\text{ 合成 caption}.
$$

GenEval overall success rate 从 43.27% 提高到 49.78%。但某些子项没有同时改善，因此不能说合成 caption 对所有能力都单调提升。

### 2. T5 的作用

T5-XXL 参数量大、显存占用高。论文发现移除 T5 时：

- 普通 prompt 的视觉质量下降较小；
- 复杂长 prompt、细节组合和图中文字能力下降更明显。

因此可以在推理显存与完整文字能力之间取舍。

### 3. 预计算冻结编码器输出

VAE 和文字编码器被冻结，因此其输出可对训练集预计算：

- 优点：节省训练 GPU 显存和重复前向计算；
- 缺点：占用大量磁盘，且无法每个 epoch 随机改变图像增强结果。

### 4. QK-Normalization

高分辨率、大模型和混合精度训练时，Attention logits 可能不断变大，Attention entropy 坍缩并最终出现 NaN。

论文在点积前对 $Q,K$ 加 RMSNorm：

$$
\widetilde Q=\operatorname{RMSNorm}(Q),
\qquad
\widetilde K=\operatorname{RMSNorm}(K),
$$

再计算：

$$
\operatorname{softmax}
\left(
\frac{\widetilde Q\widetilde K^{\mathsf T}}{\sqrt{d_h}}
\right).
$$

它是训练稳定技巧，不是 adaLN 条件调制，也不是保证任何设置都绝不发散的普遍配方。

### 5. 可变宽高比与位置编码

训练高分辨率图像时采用 aspect-ratio buckets，使同一个 batch 中图像形状一致、面积近似固定。二维位置网格按目标宽高扩展、插值并中心裁剪，避免简单插值无法正确表达不同长宽。

### 6. 高分辨率 Timestep Shift

分辨率越高，像素或 latent 位置越多；在同一 $t$ 下，模型仍可能从大量位置的统计信息中恢复更多信号，因此需要把时间表移向更强噪声。

论文用常数图像的均值估计误差近似：

$$
\sigma(t,n)
=
\frac{t}{(1-t)\sqrt n},
$$

其中 $n$ 是位置数。

要求分辨率 $n$ 和 $m$ 具有相同不确定性：

$$
\sigma(t_n,n)=\sigma(t_m,m),
$$

解得：

$$
\boxed{
t_m
=
\frac{
\sqrt{m/n}\,t_n
}{
1+\left(\sqrt{m/n}-1\right)t_n
}.
}
$$

当 $m>n$ 时，$t_m>t_n$，即把时间点推向噪声端。论文在 1024×1024 训练和采样中采用经验 shift $\alpha=3.0$。

这个推导建立在简化的常数图像假设上，论文也明确承认它不完全现实；最终 shift 仍通过人类偏好实验选择。

### 7. DPO 对齐

作者还用图像偏好版本的 DPO 和 LoRA 微调 2B、8B 模型，提高 prompt following 和人类偏好。它属于后训练对齐，不是 MM-DiT 或 Rectified Flow 的必要组成。

## 十三、Scaling 怎样定义

论文用 Transformer depth $D_{\mathrm{depth}}$ 参数化模型：

$$
D_{\mathrm{model}}=64D_{\mathrm{depth}},
$$

$$
\text{Attention heads}=D_{\mathrm{depth}},
$$

$$
D_{\mathrm{MLP}}=4D_{\mathrm{model}}.
$$

所以每个头维度保持约 64。

研究范围大致从：

- depth 15、约 450M 参数；
- 到 depth 38、约 8B 参数。

最大实验达到约 $5\times10^{22}$ training FLOPs。

## 十四、实验分别说明什么

### 1. RF formulation 实验

在统一模型、数据、优化器和采样设置下，改进 timestep sampling 的 Rectified Flow 综合排名最好；uniform RF 在少步时有优势，但步数增加后相对表现下降，Logit-Normal RF 更稳健。

能说明：

> Rectified Flow 的训练时间步分配很重要，偏重中间噪声尺度可改善高分辨率文生图训练。

不能说明：

> Logit-Normal 在所有数据、模型和 solver 上永远最优。

### 2. Backbone 实验

论文比较 DiT、CrossDiT、UViT 和 MM-DiT。在 CC12M 设置中：

- Vanilla DiT 落后于 UViT；
- CrossDiT 优于 UViT；
- 两套模态参数的 MM-DiT 表现最好；
- 再把 CLIP 与 T5 分成第三套参数只有较小增益，却增加参数和显存。

因此后续选择文字流、图像流两套参数。

能说明：

> 在该文本—图像任务中，模态专用参数加联合 Attention 比测试的共享参数或单向 Cross-Attention 架构更有效。

不能说明：

> MM-DiT 在所有模态、预算和任务上必然优于其他 Transformer。

### 3. Scaling 实验

随着模型增大和训练步数增加，validation loss 平滑下降。论文测试范围中，validation loss 与：

- GenEval：$r=-0.920$；
- 人类偏好：$r=-0.982$；
- T2I-CompBench：$r=-0.898$；
- 视频人类偏好：$r=-0.908$

呈强负相关。

含义是 validation loss 越低，外部生成质量指标在这组实验中通常越高。它是经验相关，不是因果定律，也不能无限外推。

### 4. 最终结果

GenEval overall：

| 模型 | Overall↑ |
|---|---:|
| SDXL | 0.55 |
| DALL-E 3 | 0.67 |
| depth 38，512，base | 0.68 |
| depth 38，512，DPO | 0.71 |
| depth 38，1024，DPO | **0.74** |

论文当时的 8B 模型在人类偏好实验的 visual aesthetics、prompt following 和 typography 上也取得较强结果。

注意最终 0.74 同时包含：

- 最大模型；
- 1024 分辨率训练；
- DPO；
- 改进 VAE；
- 改进 caption 和数据；
- 三个文字编码器；
- MM-DiT；
- 改进 Rectified Flow。

不能把它当成“只换 MM-DiT 就得到 0.74”。

### 5. 模型大小与采样步数

相对 50 步采样，5 步时 CLIP score 的相对下降：

- depth 15：4.30%；
- depth 38：2.71%。

大模型还表现出较短的数值路径。论文据此认为更大模型能更好拟合较直的 flow，并在少步采样时更稳健。

这不表示大模型已经能无损一步生成，也不表示采样步数永远不重要。

## 十五、论文贡献与局限

### 主要贡献

1. 统一比较大量扩散与 flow formulation；
2. 提出适合 RF 的 Logit-Normal timestep sampling；
3. 提出文字与图像双向联合的 MM-DiT；
4. 将 16 通道 VAE、三个文字编码器和更好 caption 组合进高分辨率系统；
5. 给出 QK-Norm、位置网格、timestep shift 等高分辨率训练方法；
6. 进行最大到 8B 参数的 scaling study；
7. 展示 validation loss 与多种生成指标和人类偏好的强相关。

### 局限

- 最终系统同时改变多个组件，难以把全部增益归因于单一设计；
- 主体实验集中于文字—图像生成，不能自动外推到任意多模态任务；
- 联合 Attention 随图像 token 数增加很昂贵；
- 16 通道 VAE 提高上限，也增加生成模型的预测难度和计算；
- RF 的条件监督线是直线，但实际生成 ODE 不保证严格直线；
- 没有通过本论文解决一步高质量生成；
- scaling 规律只在所测模型、数据和计算范围内成立；
- 最终结果还依赖大规模数据、caption、过滤、DPO 和高计算预算；
- benchmark 与人类偏好不能完整代表所有真实使用场景；
- 原始大规模训练过程和数据并非仅凭公开论文即可完整复现。

## 十六、学习深度

### 必须会推导和解释

1. $z_t=(1-t)x_0+t\epsilon$；
2. $dz_t/dt=\epsilon-x_0$；
3. Rectified Flow 速度 MSE；
4. 为什么生成从 $t=1$ 反向走向 $t=0$；
5. Logit-Normal 的 $u\rightarrow t$；
6. 图像与文字如何分别产生 $Q,K,V$；
7. 拼接后的联合 Attention 和四个矩阵区域；
8. 训练和推理的完整模块流程；
9. MM-DiT 为什么输出 $C$ 通道速度而不是 DiT 的 $2C$；
10. 高分辨率 timestep shift 的直观意义。

### 理解结论即可

- 论文 Eq. (3)–(12) 中 FM 与 CFM 等价的完整积分证明；
- Continuity Equation 的严格推导；
- Mode Sampling 与 CosMap 的反函数密度；
- 61 种 formulation 的统一 weighted-loss 全部变形；
- 高分辨率 shift 的完整统计假设细节；
- DPO 的完整优化公式；
- 所有数据过滤与去重算法实现。

## 十七、最容易混淆的点

1. 本文 $t=0$ 是数据、$t=1$ 是噪声，与第 07 讲记号方向相反。
2. MM-DiT 的 MM 是 Multimodal，不是 Multi-Head。
3. 两套权重不表示两条流互不交流；它们在联合 Attention 中相遇。
4. $c_{\mathrm{vec}}$ 是全局文字摘要，$c_{\mathrm{ctxt}}$ 是逐 token 文字序列。
5. 三个文字编码器不等于 MM-DiT 有三套主干参数；最终主要采用文字、图像两套。
6. CLIP/T5 冻结，不表示 MM-DiT 内的文字 token 不更新。
7. Logit-Normal 抽的是训练时间步，不是 latent 噪声本身，也不是推理步长表。
8. 直线训练插值不等于生成 ODE 完全笔直。
9. 本文没有以 Reflow 或一步 Distillation 为核心。
10. 16 个 latent channels、Transformer hidden size、模型 depth 是三种不同维度。
11. QK-RMSNorm 是 Attention 稳定技巧，adaLN 是条件调制。
12. MM-DiT 最终输出 latent 速度，不直接输出 RGB。
13. 本文 RF 输出是 $C$，不是第 09 讲 DDPM DiT 的 $2C$。
14. 最终质量来自 formulation、架构、表示、数据和后训练共同作用。
15. “未见饱和”只指论文测试区间，不是无限 scaling 定律。

## 十八、公式与形状速记

$$
\boxed{
\begin{aligned}
x_0&=\mathcal E(X),\\
z_t&=(1-t)x_0+t\epsilon,\\
\frac{dz_t}{dt}&=\epsilon-x_0,\\
L_{\mathrm{RF}}
&=\mathbb E\|
v_\theta(z_t,t,c)-(\epsilon-x_0)
\|_2^2,\\
u&\sim\mathcal N(0,1),\\
t&=\operatorname{sigmoid}(u),\\
Q&=[Q_T;Q_I],\quad
K=[K_T;K_I],\quad
V=[V_T;V_I],\\
O&=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_h}}
\right)V,\\
z_{t-\Delta t}
&\approx z_t-\Delta t\,v_\theta(z_t,t,c).
\end{aligned}
}
$$

1024 分辨率形状：

$$
\boxed{
\begin{aligned}
[B,3,1024,1024]
&\rightarrow[B,16,128,128]\\
&\rightarrow[B,4096,64]\\
&\rightarrow[B,4096,D],\\
\text{text}
&\rightarrow[B,154,D],\\
\text{joint sequence}
&\rightarrow[B,4250,D],\\
\text{image output}
&\rightarrow[B,16,128,128].
\end{aligned}
}
$$

## 十九、汇报问答

### 1. 这篇论文与前四篇有什么关系？

它保留 LDM 的 latent space，使用 Rectified Flow 的直线路径和速度预测，以 DiT 为视觉 backbone，再把类别条件升级为文字—图像双流的 MM-DiT。

### 2. 本文 Rectified Flow 的训练目标是什么？

随机构造 $z_t=(1-t)x_0+t\epsilon$，让 MM-DiT 根据 $z_t,t,c$ 预测条件直线速度 $\epsilon-x_0$。

### 3. 为什么预测 $\epsilon-x_0$ 却能从噪声生成数据？

该速度对应 $t$ 增大时数据走向噪声；生成时从 $t=1$ 向 $t=0$ 反向积分，因此 Euler 更新会减去该速度，实际方向是噪声走向数据。

### 4. Logit-Normal Sampling 做了什么？

先采样高斯变量 $u$，再令 $t=\operatorname{sigmoid}(u)$，让训练更常抽到信号和噪声充分混合的中间时间步。它只改变训练重点，不改变直线路径或推理 solver。

### 5. MM-DiT 比 DiT 新在哪里？

DiT 主要处理图像 patch 和全局类别条件；MM-DiT 为文字、图像使用不同参数，再把两者的 $Q,K,V$ 沿 token 维拼接并进行联合 Attention。

### 6. “双向”具体指什么？

联合注意力矩阵同时包含文字看图像和图像看文字两个交叉区域，因此文字表示会根据当前图像 token 更新，图像 token 也会读取文字。

### 7. 为什么文字和图像不用同一套权重？

两种模态的输入结构和语义不同，专用 LayerNorm、QKV、投影和 MLP 允许它们在各自表示空间加工；投影后维度相同，又可在 Attention 中交流。

### 8. $c_{\mathrm{vec}}$ 与 $c_{\mathrm{ctxt}}$ 有何区别？

$c_{\mathrm{vec}}$ 是整段 prompt 的全局摘要，与时间步一起调制 block；$c_{\mathrm{ctxt}}$ 保留逐 token 信息，直接参与联合 Attention。

### 9. 为什么使用 16 通道 VAE？

更多 latent channels 能保留更完整的图像细节，提高 VAE 重建上限；代价是 flow 模型需要预测更复杂的 latent，因此更依赖大模型容量。

### 10. 为什么本文输出不是 $2C$？

原始 DiT 用 DDPM 反向高斯，需要噪声和方差参数；本文用确定性 Rectified Flow ODE，只需预测与 latent 同形状的速度场，所以输出是 $C$ 通道。

### 11. QK-Normalization 有什么用？

它在 Attention 点积前规范 Q、K 的尺度，避免大模型高分辨率训练时 logits 爆炸、Attention entropy 坍缩和 NaN。

### 12. 高分辨率为什么要 shift timestep？

高分辨率有更多观测位置，同一噪声比例下仍容易残留可恢复信号，因此作者把时间表推向更强噪声，使不同分辨率的有效不确定性更接近。

### 13. 这篇论文实现一步生成了吗？

没有。它没有依赖 Reflow 或一步蒸馏，实际速度场也不保证严格笔直；生成仍需多次 MM-DiT 前向和 ODE 数值积分。

### 14. 最终结果能全部归因于 MM-DiT 吗？

不能。最终模型同时使用改进 RF、16 通道 VAE、三个文字编码器、合成 caption、数据处理、高分辨率训练、8B scaling 和 DPO。

### 15. 论文的 scaling 结论是什么？

在所测范围内，模型增大和训练增加会平滑降低 validation loss，而较低 loss 与 GenEval、T2I-CompBench 和人类偏好显著相关；这是经验规律，不是无限外推的定理。

## 相关笔记

- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)
- [08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)
- [09-Scalable Diffusion Models with Transformers（DiT）](#/notes/dit)
- [11-SDEdit - Guided Image Synthesis and Editing with Stochastic Differential Equations](#/notes/sdedit)
- [14-FlowEdit - Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](#/notes/flowedit)
 
## 后续连接

- [15-Stable Flow - Vital Layers for Training-Free Image Editing](#/notes/stable-flow)：利用图像、文字 token 的联合 Attention，只在自动发现的关键层替换编辑分支的视觉 $K,V$。
