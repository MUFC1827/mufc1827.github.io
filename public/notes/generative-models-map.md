# 17 生成式模型知识体系总复习与论文串联（详细版）

> [!abstract] 这份笔记解决什么问题
> 这不是把十二篇论文各自再摘要一遍，而是把它们放进**同一个生成系统**中：每篇论文解决哪一层问题、继承了什么、改变了什么、没有改变什么、留下了什么局限，以及为什么下一篇会自然出现。
>
> 建议先通读“全局地图”和“基础链”，再读“编辑与控制链”，最后看总表与口述主线。第一次阅读目标是建立大局；汇报前再用末尾的公式与问答清单查漏补缺。

---

## 一、先建立总地图：十二篇论文不是一条替代链

最容易犯的错误，是把论文理解成：

> DDPM 被 Rectified Flow 替代，LDM 又替代 DDPM，DiT 又替代 LDM，MM-DiT 再替代 DiT。

这不对，因为它们多数不在同一个层级。一个完整生成系统可以拆成以下六个**可独立选择、也可相互组合的轴**：

| 系统轴 | 核心问题 | 代表内容 |
|---|---|---|
| 1. 表示空间 | 模型在哪里生成？ | 像素空间；LDM 的 latent 空间 |
| 2. 概率路径 / 动力学 | 数据和噪声怎样连接？ | DDPM；Rectified Flow |
| 3. 预测目标 | 网络每次究竟预测什么？ | 噪声、Score、速度、方差等 |
| 4. Backbone | 用什么神经网络做预测？ | U-Net；DiT；MM-DiT |
| 5. 条件与信息交互 | 文本、类别、Mask、边缘图怎样影响生成？ | Self/Cross/Joint Attention；adaLN；ControlNet |
| 6. Sampler / Solver | 推理时怎样从噪声数值地走到图像？ | 反向扩散采样器；DDIM；ODE Solver |

因此可以把一个生成系统写成：

$$
\boxed{
\text{生成系统}
=
(\text{表示空间},\text{路径},\text{预测目标},
\text{Backbone},\text{条件机制},\text{Sampler/Solver})
}
$$

几个真实组合能立刻说明这件事：

| 系统 | 表示空间 | 路径与目标 | Backbone | 条件机制 | 推理 |
|---|---|---|---|---|---|
| 原始 DDPM | 像素 | Gaussian Diffusion；主要预测噪声 | U-Net | 原论文以无条件为主 | 随机反向链 |
| LDM / Stable Diffusion v1.x | latent | DDPM 类扩散；预测噪声 | U-Net | 文本 Cross-Attention | 多步扩散采样器 |
| 原始 DiT | latent | DDPM 类扩散；预测噪声和方差 | Transformer | 类别经 adaLN-Zero | 多步扩散采样 |
| Stable Diffusion 3 | latent | Rectified Flow；预测速度 | MM-DiT | 文图 Joint Attention | ODE Solver |

必须记住：

- **DDPM 与 Rectified Flow**主要是“路径和学习目标”的两种方案；
- **像素空间与 latent 空间**是“在哪里生成”的选择；
- **U-Net、DiT、MM-DiT**是负责预测的网络结构；
- Rectified Flow 可以配 U-Net，也可以配 DiT；
- LDM 可以使用 DDPM，也可以使用 Rectified Flow；
- 原始 DiT 使用的是 DDPM 类扩散，并不等于 Rectified Flow；
- Stable Diffusion 3 才把 latent、Rectified Flow、MM-DiT 等组合成一个系统。

### 1. 一条统一的训练—推理主线

不论是扩散还是流模型，都可以先用一个抽象式理解：

$$
u_t
\xrightarrow[\text{时间 }t,\,\text{条件 }c]{f_\theta}
\widehat a_t,
$$

其中：

- $u_t$：当前带噪的图像或 latent；
- $t$：当前噪声程度；
- $c$：文本、类别、Mask、边缘、深度等条件；
- $f_\theta$：U-Net、DiT 或 MM-DiT；
- $\widehat a_t$：网络预测的噪声、Score、速度或其他参数。

训练时，有真实样本和人为构造出的监督目标，因此可以计算 Loss：

$$
\mathcal L(\theta)
=
\mathbb E\bigl[\|a_t-f_\theta(u_t,t,c)\|^2\bigr],
$$

然后反向传播更新参数 $\theta$。

推理时，没有真实答案、没有 Loss，也不更新参数；只重复调用已经学好的网络，用采样器或 ODE 求解器更新状态。为避免不同论文时间方向相反造成符号混乱，可以统一写成：

$$
\boxed{
u_{t+\delta}
=
\Phi\bigl(u_t,f_\theta(u_t,t,c),t,\delta\bigr)
}
$$

其中 $\delta$ 可以为正或负，$\Phi$ 表示相应的扩散采样公式或 ODE 数值更新公式。

### 2. 两条论文主线

十二篇论文可以分为两条相接的研究链：

#### 基础生成链：怎样建立一个强大的生成器？

$$
\text{Transformer}
\rightarrow
\text{DDPM / RF}
\rightarrow
\text{LDM}
\rightarrow
\text{DiT}
\rightarrow
\text{MM-DiT / SD3}
$$

它们依次回答：

1. 信息怎样在 token 之间交换？
2. 噪声怎样变成数据？
3. 为什么要在昂贵的像素空间生成？
4. 预测网络为什么一定要是 U-Net？
5. 复杂文字和图像怎样在大模型中深度融合？

#### 编辑与控制链：有了生成器，怎样精确地修改已有图像？

$$
\text{SDEdit}
\rightarrow
\text{BLD}
\rightarrow
\text{Prompt-to-Prompt}
\rightarrow
\text{FlowEdit}
\rightarrow
\text{Stable Flow}
$$

同时还有一条正交的“训练式空间控制”路线：

$$
\text{ControlNet}
$$

它们共同围绕同一个矛盾：

$$
\boxed{
\text{编辑得足够明显}
\quad\text{vs.}\quad
\text{尽量保持源图不该变化的部分}
}
$$

### 3. 时间线与依赖关系

| 序号 | 论文 | 年份 | 主要位置 |
|---:|---|---:|---|
| 01 | Attention Is All You Need | 2017 | Attention 与 Transformer 基础 |
| 02 | Denoising Diffusion Probabilistic Models | 2020 | 扩散生成底座 |
| 03 | Flow Straight and Fast | 2022 | Rectified Flow 运输路径 |
| 04 | High-Resolution Image Synthesis with Latent Diffusion Models | 2021/2022 | latent 空间与通用条件接口 |
| 05 | Scalable Diffusion Models with Transformers | 2022/2023 | DiT Backbone |
| 06 | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis | 2024 | SD3 与 MM-DiT 系统 |
| 07 | SDEdit | 2021/2022 | 加噪强度控制的通用编辑 |
| 08 | Blended Latent Diffusion | 2022 | Mask 局部编辑 |
| 09 | Prompt-to-Prompt Image Editing | 2022 | Cross-Attention 布局控制 |
| 10 | FlowEdit | 2024 | 基于速度差的免反演编辑 |
| 11 | Stable Flow | 2024/2025 | 关键层视觉特征注入 |
| 12 | Adding Conditional Control to Text-to-Image Diffusion Models | 2023 | 训练式空间条件控制 |

它们不是严格按发表时间排序，而是按本课程最容易理解的**概念依赖顺序**排列。

---

## 二、统一符号：先解决最容易混淆的时间方向

### 1. 三套常见记号

| 论文体系 | 起点 | 终点 | 时间增大时 |
|---|---|---|---|
| DDPM | $x_0$ 是数据 | $x_T$ 是噪声 | 数据逐渐加噪 |
| 原始 Rectified Flow 讲法 | $X_0$ 常取噪声 | $X_1$ 常取数据 | 噪声流向数据 |
| SD3 / FlowEdit 常用讲法 | $x_0$ 是数据 | $\epsilon$ 在 $t=1$ | 数据流向噪声；生成时反向积分 |

符号不同不表示原理矛盾。判断一条公式时，先问：

1. $t=0$ 在这篇文章中代表数据还是噪声？
2. 训练定义的正方向是什么？
3. 生成时积分是正向还是反向？

### 2. 三种预测量不要混为一谈

- **噪声预测**：估计加入样本的 $\epsilon$；
- **Score 预测**：估计 $\nabla_x\log p_t(x)$，即密度增大最快的方向；
- **速度预测**：估计连续路径上 $dx_t/dt$。

它们可以在特定参数化下互相换算，但含义不同。论文把某个量作为训练标签，通常是为了让监督更简单、尺度更稳定或更适合对应的采样过程。

---

# 第一部分：基础生成链

## 三、论文 01：Attention Is All You Need

**一句话定位：**它没有提出扩散模型，而是提供了后来 DiT、MM-DiT、Cross-Attention 编辑等工作的共同信息交互语言。

详细课内笔记：[05-Attention Is All You Need](#/notes/attention-is-all-you-need)

### 1. 它要解决什么问题

传统 RNN 按顺序更新：

$$
h_t=f(h_{t-1},x_t).
$$

必须先计算 $h_{t-1}$ 才能计算 $h_t$，句子内部不易并行；相距很远的两个词还要经过很多中间状态才能交流。CNN 可以并行，但单层主要观察局部，远距离位置通常要堆叠多层。

论文提出的问题是：

> 能否让任意两个位置一层内直接交换信息，并同时处理整个序列？

### 2. Scaled Dot-Product Attention 到底在算什么

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V
}
$$

- $Q$（Query）：当前位置想查询什么；
- $K$（Key）：每个候选位置提供的检索标签；
- $QK^{\mathsf T}$：每个 Query 与所有 Key 的匹配分数；
- Softmax：把一行分数变成非负且和为 1 的权重；
- $V$（Value）：真正被取出并加权汇总的信息。

若：

$$
Q\in\mathbb R^{n_q\times d_k},\qquad
K\in\mathbb R^{n_k\times d_k},\qquad
V\in\mathbb R^{n_k\times d_v},
$$

那么：

$$
QK^{\mathsf T}\in\mathbb R^{n_q\times n_k},
$$

输出为：

$$
\operatorname{Attention}(Q,K,V)
\in\mathbb R^{n_q\times d_v}.
$$

Softmax 后每个权重虽小于 1，但一行权重总和为 1；乘 $V$ 后得到的是多个 Value 的加权和，不是另一个 Softmax。

为什么除以 $\sqrt{d_k}$？若 $q,k$ 每维近似独立、均值 0、方差 1，则点积：

$$
q\cdot k=\sum_{j=1}^{d_k}q_jk_j
$$

的方差大约随 $d_k$ 增长到 $d_k$，标准差约为 $\sqrt{d_k}$。缩放后能防止 Softmax 输入过大、分布过早接近 one-hot、梯度太小。

### 3. $W^Q,W^K,W^V$ 从哪里来

它们不是人工指定的语义规则，而是和神经网络其他权重一样：

1. 随机初始化；
2. 前向计算得到预测和 Loss；
3. 反向传播计算 Loss 对各矩阵的梯度；
4. 优化器反复更新；
5. 最终学习出适合任务的投影空间。

输入 $X$ 通过：

$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

生成不同角色的表示。它们值不同，并不代表输入被复制成三个完全无关的数据集，而是同一输入经过三种可学习的线性观察方式。

### 4. Multi-Head 如何汇总

第 $i$ 个头：

$$
\operatorname{head}_i
=
\operatorname{Attention}
(QW_i^Q,KW_i^K,VW_i^V).
$$

汇总为：

$$
\boxed{
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}
(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
}
$$

多头之间参数和计算相对独立，可以关注不同关系；最后**不是平均，也不是再做 Softmax**，而是拼接后经 $W^O$ 混合回模型维度。原始 Base 模型：

$$
d_{\text{model}}=512,\qquad h=8,\qquad d_k=d_v=64.
$$

### 5. 完整 Transformer，而不只是 Attention

原始 Transformer 仍包含：Embedding、位置编码、FFN、Residual、LayerNorm、输出 Linear 与 Softmax。

Encoder 每层：

1. Multi-Head Self-Attention；
2. Residual + LayerNorm；
3. 位置级 FFN；
4. Residual + LayerNorm。

Decoder 每层：

1. Masked Self-Attention；
2. Residual + LayerNorm；
3. Encoder–Decoder Attention；
4. Residual + LayerNorm；
5. FFN；
6. Residual + LayerNorm。

原论文使用 Post-LN：

$$
\operatorname{LayerNorm}igl(x+\operatorname{Sublayer}(x)\bigr).
$$

FFN 对每个 token 分开计算，但所有 token 共用参数：

$$
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2,
$$

维度为：

$$
512\rightarrow2048\rightarrow512.
$$

Attention 负责 token 之间交换信息；FFN 负责进一步加工单个 token 内部的特征。

### 6. 三种 Attention 的 Q、K、V

| 模块 | $Q$ 来自 | $K,V$ 来自 | 作用 |
|---|---|---|---|
| Encoder Self-Attention | 源序列 | 源序列 | 源句内部建模 |
| Decoder Masked Self-Attention | 已有目标序列 | 已有目标序列 | 只能看已生成内容 |
| Encoder–Decoder Attention | Decoder | Encoder 输出 | 生成目标词时读取源句 |

Causal Mask 将未来位置的分数设成 $-\infty$，Softmax 后权重为 0，避免训练时偷看未来答案。

### 7. 位置编码为什么不会“被冲掉”

Attention 对排列本身不敏感，因此输入使用：

$$
e_i=\operatorname{TokenEmbedding}(x_i)+\operatorname{PositionEncoding}(i).
$$

	位置不需要在后续每个数字中单独保留一个“位置栏位”。模型训练时反复看到位置模式与任务 Loss 的关系，会学会从混合向量中利用位置特征。理论上不同“词向量 + 位置向量”可能偶然相同，但在高维连续空间与训练分布中精确碰撞几乎不构成实际问题；而且多层变换会结合上下文进一步区分。

### 8. 训练为什么并行，推理为什么仍顺序

训练翻译时，Decoder 输入为右移后的完整正确目标句；Causal Mask 保证每个位置只读过去，因此所有位置可在一次矩阵运算中并行计算 Loss。交叉熵反向传播会更新 QKV、FFN、Embedding 等全部可训练参数。

推理时未来 token 尚不存在，只能：

$$
y_1\rightarrow y_2\rightarrow\cdots\rightarrow y_T.
$$

所以 Transformer 消除了网络层内部的 RNN 式递归依赖，却没有消除自回归输出本身的时间顺序。

### 9. 贡献、证据与局限

主要贡献：

- 建立完全不依赖序列 RNN/CNN 的 Encoder–Decoder；
- 系统化 Scaled Dot-Product Attention 与 Multi-Head Attention；
- 任意两个位置一层即可交互，长距离路径缩短；
- 大幅提高训练并行性；
- 在机器翻译上取得当时很强的结果，并通过句法分析展示迁移能力。

关键实验：本地版本摘要报告 WMT14 英德翻译 28.4 BLEU、英法 41.8 BLEU；模型尺寸与 Head 等消融说明性能不是仅靠参数增加；句法分析表明架构不只适用于翻译。

局限：

- 长度为 $n$ 时完整 Self-Attention 的计算和注意力矩阵存储通常为 $O(n^2)$；
- Attention 本身无顺序，需要位置编码；
- 自回归解码仍逐 token；
- Attention Map 只能提供解释线索，不能把每个 Head 简单认定为唯一、固定的语法规则。

### 10. 它如何通向后文

- LDM：图像 Query 通过 Cross-Attention 读取文字 Key、Value；
- DiT：noisy latent patch 作为 token 做 Self-Attention；
- MM-DiT：文字和图像 token 共同进行 Joint Attention；
- Prompt-to-Prompt：直接干预 Cross-Attention Map；
- Stable Flow：在关键 Transformer 层干预视觉 Key、Value。

---

## 四、论文 02：Denoising Diffusion Probabilistic Models（DDPM）

**一句话定位：**规定“逐步加噪”这道正向题，再训练网络学习“逐步去噪”的逆过程，建立现代扩散生成的数学底座。

详细课内笔记：[06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)

### 1. 它要解决什么问题

生成模型需要学习复杂数据分布 $p_{\text{data}}(x)$，但直接从简单噪声一步映射到高维图像很难。DDPM 把难题拆成很多极小的去噪步骤：每一步只需恢复一点点信息。

整体过程：

$$
x_0\text{（真实数据）}
\xrightarrow{q}
x_T\approx\mathcal N(0,I)
\xrightarrow{p_\theta}
\widehat x_0.
$$

- $q$：人为固定的前向加噪过程，不学习；
- $p_\theta$：神经网络参数化的反向生成过程，需要学习。

### 2. $\mathcal N(\mu,\Sigma)$ 是什么

$$
x\sim\mathcal N(\mu,\Sigma)
$$

表示 $x$ 服从均值为 $\mu$、协方差为 $\Sigma$ 的高斯分布。图像是高维向量；$I$ 是单位矩阵。若协方差是 $\sigma^2I$，就表示每一维加入方差为 $\sigma^2$、彼此独立的高斯噪声。

### 3. 前向加噪与任意时刻的闭式公式

一步前向过程：

$$
q(x_t\mid x_{t-1})
=
\mathcal N\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_tI
\right).
$$

等价采样写法：

$$
x_t
=
\sqrt{1-\beta_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

定义：

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

连续代入，并利用独立高斯噪声的线性组合仍为高斯，可得到：

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
}
$$

这条公式极重要：训练时可以随机抽一个 $t$，直接从 $x_0$ 得到 $x_t$，不必真的循环加噪 $t$ 次。

### 4. 为什么需要学习反向分布

论文将一步反向过程建模为：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

训练时若已知 $x_0$，真实一步后验：

$$
q(x_{t-1}\mid x_t,x_0)
$$

可由高斯公式求出。但生成新图时只有 $x_t$，并不知道真实 $x_0$，所以必须让神经网络从大量训练样本中学会估计反向过程。

### 5. Loss 从哪里来：ELBO 到噪声 MSE

目标本来是最大化数据似然 $\log p_\theta(x_0)$。直接计算困难，于是引入已知前向过程 $q$，利用 Jensen 不等式得到可优化的变分下界；以“要最小化的负下界”记号可拆成：

$$
L
=
L_T+sum_{t>1}L_{t-1}+L_0.
$$

直观上：

- $L_T$：最终加噪分布是否接近标准高斯；
- $L_{t-1}$：学习到的反向一步是否接近真实后验；
- $L_0$：最后一步是否能恢复数据分布。

主要学习项是两个高斯分布间的 KL 散度：

$$
L_{t-1}
=
D_{\mathrm{KL}}
\bigl(
q(x_{t-1}\mid x_t,x_0)
\,\|\,
p_\theta(x_{t-1}\mid x_t)
\bigr).
$$

固定或特定参数化方差时，高斯 KL 的主要可训练部分变成均值平方误差；将真实后验均值用前向公式中的噪声 $\epsilon$ 改写，又得到加权的噪声预测误差。论文进一步去掉随 $t$ 变化的权重，形成：

$$
\boxed{
L_{\mathrm{simple}}
=
\mathbb E_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-epsilon_\theta(x_t,t)
\right\|^2
\right]
}
$$

必须会讲清的推导主线是：

$$
\boxed{
\text{最大似然}
\rightarrow
\text{ELBO}
\rightarrow
\text{逐步高斯 KL}
\rightarrow
\text{后验均值误差}
\rightarrow
\text{噪声预测 MSE}
}
$$

完整高斯配方代数可以看懂即可，不必汇报时逐行背诵。

### 6. 为什么 Loss 的数值和梯度不能混淆

Loss 数值回答“当前预测错了多少”，梯度回答“参数往哪个方向改，Loss 才会下降”。更新规则类似：

$$
\theta
\leftarrow
\theta-\eta\frac{\partial L}{\partial\theta}.
$$

反向传播直接使用的是各局部导数，但这些导数由当前预测、真实目标及 Loss 形式共同决定。Loss 并非只是训练结束后的检验分数；它定义了优化目标，并通过链式法则产生梯度。

### 7. 与 Score Matching 的联系

由高斯条件分布可得：

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

Score 表示当前位置朝向“概率密度更高区域”的方向。于是预测噪声与预测条件 Score 只差一个已知缩放系数。这解释了 DDPM 与去噪 Score Matching、Langevin Dynamics 之间的联系。

### 8. 训练与推理的完整区别

训练一次：

1. 采样真实图像 $x_0$；
2. 随机抽一个 $t$；
3. 采样已知噪声 $\epsilon$；
4. 用闭式公式构造 $x_t$；
5. U-Net 预测 $\widehat\epsilon$；
6. 计算 $\|\epsilon-\widehat\epsilon\|^2$；
7. 反向传播更新网络。

随机抽一个时间步，是对“所有数据和所有时间步的期望 Loss”的 Monte Carlo 估计。

推理：

$$
x_T\sim\mathcal N(0,I),
$$

然后反复执行类似：

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_tz.
$$

| 项目 | 训练 | 推理 |
|---|---|---|
| 起点 | 真实数据加噪 | 随机噪声 |
| 时间步 | 每个样本随机抽一个 | 从 $T$ 逐步走到 0 |
| 真实噪声 | 已知，可作标签 | 未知，由网络估计 |
| Loss / 反向传播 | 有 | 无 |
| 网络调用 | 一个训练样本通常一次 | 原始 DDPM 约 1000 次 |

U-Net 是预测器，不等于扩散过程本身；以后可以用 DiT 替换它而保留相同数学目标。

### 9. 贡献、证据与局限

主要贡献：

- 把固定高斯前向过程、可学习反向过程和变分目标连成完整系统；
- 用 $\epsilon$-parameterization 得到简单稳定的训练目标；
- 连接 Diffusion、Denoising Score Matching 与 Langevin Dynamics；
- 证明扩散模型能产生高质量图像；
- 将采样解释为逐步恢复信息的 progressive decompression。

关键实验：CIFAR-10 上报告 IS 9.46、FID 3.17；原始过程常取 $T=1000$。

局限：

- 原始采样需要大量网络前向，速度慢；
- 样本质量强，但论文中的对数似然不是最佳；
- 前向日程、反向方差与采样器仍需设计；
- 论文主要建立无条件生成基础，复杂文本控制尚未解决；
- 训练能一步构造任意 $x_t$，不代表推理能一步从 $x_T$ 跳回 $x_0$。

### 10. 它如何通向后文

- LDM 把同样的加噪和去噪搬进 latent；
- Stable Diffusion 用文本条件 U-Net 做噪声预测；
- DiT 保留 DDPM 类目标，只替换 U-Net；
- SDEdit 利用“处在某个噪声层级后再反向生成”实现编辑；
- Rectified Flow 则重新思考数据与噪声之间的路径及预测目标。

---

## 五、论文 03：Flow Straight and Fast（Rectified Flow）

**一句话定位：**用监督回归学习数据分布之间的连续速度场，并通过 Rectification 让轨迹更直、更适合少步数值求解。

详细课内笔记：[07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)

### 1. 它要解决什么问题

连续时间生成模型可用 ODE/SDE 把噪声运输到数据，但推理需要数值积分。轨迹弯曲时，大步长近似误差大，必须多次调用网络。论文希望：

- 用简单监督 MSE 学习运输；
- 统一生成、Domain Transfer 等任务；
- 让实际轨迹更直，从而少步甚至一步生成。

### 2. 人工插值线与速度监督

从两个分布取样：

$$
X_0\sim\pi_0,\qquad X_1\sim\pi_1.
$$

在端点之间画直线：

$$
\boxed{X_t=(1-t)X_0+tX_1}
$$

它的导数是常量：

$$
\boxed{\dot X_t=X_1-X_0}.
$$

训练网络：

$$
\boxed{
\mathcal L_{\mathrm{RF}}
=
\mathbb E
\left[
\left\|
(X_1-X_0)-v_\theta(X_t,t)
\right\|^2
\right]
}
$$

训练只需采样端点、时间和插值位置，不必在每个训练样本内数值求解 ODE，因此叫 simulation-free training。

### 3. MSE 最优解为什么是条件期望

令标签 $Y=X_1-X_0$、输入 $U=(X_t,t)$。平方误差可分解：

$$
\mathbb E\|Y-v(U)\|^2
=
\mathbb E\|Y-\mathbb E[Y\mid U]\|^2
+
\mathbb E\|\mathbb E[Y\mid U]-v(U)\|^2.
$$

第一项与 $v$ 无关；第二项在 $v(U)=\mathbb E[Y\mid U]$ 时最小。因此：

$$
\boxed{
v^X(x,t)
=
\mathbb E[X_1-X_0\mid X_t=x]
}
$$

直观上：若多条训练直线在同一时刻、同一位置相交，而方向不同，只看当前位置无法判断样本来自哪一条线，网络的最优预测就是这些可能方向的条件平均。

### 4. 最关键区分：训练直线不等于生成轨迹

- $X_t$：训练时人为构造的插值点，需要同时知道起点和终点；
- $Z_t$：推理时从一个新起点沿学习到的速度场求解出的 ODE 轨迹。

生成 ODE：

$$
dZ_t=v_\theta(Z_t,t)\,dt.
$$

由于方向平均，第一次学出的 $Z_t$ 仍可能弯曲。因此：

> “训练样本用直线连接”不等于“1-RF 天然可以一步生成”。

在精确速度场等理论条件下，论文证明的是边缘分布保持：

$$
\operatorname{Law}(Z_t)=\operatorname{Law}(X_t).
$$

即每个时刻所有样本的总体分布一致；它不保证每个粒子遵循原本的端点配对，也不保证整条联合轨迹相同。

### 5. Rectification、Reflow 与 Distillation

论文还证明对合适的凸代价 $c$：

$$
\mathbb E[c(Z_1-Z_0)]
\le
\mathbb E[c(X_1-X_0)].
$$

这表示 Rectification 不会增加相应的端点运输代价，但不等于必然求得任意指定代价下的全局最优运输。

轨迹直度可写为：

$$
S(Z)
=
\int_0^1
\mathbb E
\left[
\left\|
(Z_1-Z_0)-\dot Z_t
\right\|^2
\right]dt.
$$

若 $S(Z)=0$，全程就是恒速直线，一步 Euler 可以准确抵达终点。

**Reflow**：

1. 训练第一个 RF；
2. 用它产生模型自己的噪声—数据端点配对；
3. 将这些新配对重新连直线；
4. 再训练一个 RF。

新 coupling 的方向冲突通常更少，实际 ODE 更直。

**Distillation**：在已有流足够直后，训练新网络直接模仿原流的端点映射，将多步求解压缩成一步或少步。

| | Reflow | Distillation |
|---|---|---|
| 核心 | 改变 coupling 并重学速度场 | 模仿已有流的完整映射 |
| 目的 | 拉直实际轨迹 | 压缩网络调用步数 |
| 典型顺序 | 先做 | 后做 |

### 6. 训练与推理

训练：采样 $X_0,X_1,t$，构造 $X_t$，以 $X_1-X_0$ 为标签做一次速度 MSE。

推理用数值积分，例如 Euler：

$$
Z_{t+\Delta t}
\approx
Z_t+\Delta t\,v_\theta(Z_t,t).
$$

步数越少，每一步的局部近似误差越大；轨迹越直，对粗步长越友好。

### 7. 与 DDPM 的关系

| | DDPM | Rectified Flow |
|---|---|---|
| 中间状态 | 预设高斯加噪日程 | 端点插值路径 |
| 常见预测 | 噪声 / Score | 速度 |
| 生成视角 | 反向随机链或相关 ODE | 连续 ODE 运输 |
| 训练单样本 | 随机一个时间步 | 随机一个时间步 |
| 加速重点 | 改 Sampler、蒸馏 | 拉直、粗步积分、蒸馏 |

二者都可以在像素或 latent 中运行，也都可以用 U-Net 或 Transformer 作为预测器。

### 8. 贡献、证据与局限

主要贡献：

- 用简单监督 MSE 学习分布之间的 ODE 运输；
- 训练时不需数值模拟完整轨迹；
- 给出边缘分布保持、凸代价不增加等理论；
- 提出 Reflow，改善轨迹直度和粗步求解；
- 将生成、无配对图像翻译与 Domain Adaptation 放入统一框架；
- 展示 Reflow + Distillation 后的一步生成潜力。

关键实验需谨慎复述：本地 PDF Table 1 中，完整求解的 1-RF 为 FID 2.58、Recall 0.57；蒸馏 2-RF 的一步结果为 FID 4.85、Recall 0.50；蒸馏 3-RF Recall 为 0.51。不要把不同列的最佳数拼成同一个配置。

局限：1-RF 不天然一步；Reflow 要额外生成数据并重新训练；误差可能迭代累积；理论依赖速度场准确、ODE 解存在唯一等条件；凸代价改善不等于自动求得所有任务的最优运输。

### 9. 它如何通向后文

Stable Diffusion 3 的核心组合是：

$$
\text{latent space}+\text{Rectified Flow}+\text{MM-DiT}.
$$

FlowEdit 则进一步利用“源条件速度与目标条件速度之差”，构造无需显式源图反演的编辑 ODE。

---

## 六、论文 04：High-Resolution Image Synthesis with Latent Diffusion Models（LDM）

**一句话定位：**先用 Autoencoder 做感知压缩，再在保持二维结构的 latent 中扩散，把算力集中在语义建模而非像素细节上。

详细课内笔记：[08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)

### 1. 它要解决什么问题

像素空间扩散每个时间步都在高分辨率 RGB 上运行大 U-Net，很多算力被用于表达人眼不敏感的微小像素变化。论文把图像学习拆成：

1. **感知压缩**：去除不重要的高频冗余；
2. **语义建模**：学习物体、布局、关系与数据分布。

核心问题是：能否先压缩图像，再把扩散算力用在更有意义的 latent 表示上？

### 2. 两阶段系统

$$
\boxed{
x
\xrightarrow{\mathcal E}
z_0
\xrightarrow{\text{Diffusion}}
z_T
\xrightarrow{\text{Reverse}}
\widehat z_0
\xrightarrow{\mathcal D}
\widehat x
}
$$

#### 第一阶段：训练 Autoencoder

$$
z=\mathcal E(x),\qquad \widetilde x=\mathcal D(z).
$$

若空间下采样因子为：

$$
f=\frac Hh=\frac Ww,
$$

则 $512\times512$ 图像在 $f=8$ 时变成 $64\times64$ latent，空间位置数降为原来的 $1/64$。

Autoencoder 不只用像素 MSE，还组合：

- 重建 Loss；
- Perceptual Loss：比较深层视觉特征，保留感知结构；
- Patch Adversarial Loss：鼓励局部纹理真实；
- KL 或 VQ 形式的轻量 latent regularization：让 latent 空间更规则。

只用像素误差常倾向产生模糊平均结果。压缩也存在折中：

- $f$ 太小：latent 仍大，节省有限；
- $f$ 太大：细节永久丢失；
- 论文发现 $f=4\sim16$ 是较合理范围，常见 $f=4,8$ 是实验选择，不是数学定理。

#### 第二阶段：冻结 Autoencoder，训练 latent 扩散

$$
z_0=\mathcal E(x),
$$

$$
z_t
=
\sqrt{\bar\alpha_t}z_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
$$

$$
\boxed{
L_{\mathrm{LDM}}
=
\mathbb E
\left[
\left\|
\epsilon-epsilon_\theta(z_t,t)
\right\|^2
\right]
}
$$

LDM 没有重新发明 DDPM Loss；它主要把 DDPM 中的 $x_t$ 换成 latent $z_t$。

### 3. 文本如何进入图像生成：Cross-Attention

条件编码器得到文本 token：

$$
\tau_\theta(y)\in\mathbb R^{M\times d_\tau}.
$$

U-Net 某层图像特征展平为：

$$
\phi_i(z_t)\in\mathbb R^{N\times d_i}.
$$

论文的语义方向是：

$$
Q=W_Q\phi_i(z_t),\qquad
K=W_K\tau_\theta(y),\qquad
V=W_V\tau_\theta(y).
$$

即：

- Query 来自图像位置；
- Key、Value 来自文本 token；
- 每个图像位置查询“此刻应该读取哪些词的信息”。

条件 Loss：

$$
L_{\mathrm{cond}}
=
\mathbb E
\left[
\left\|
\epsilon-epsilon_\theta(z_t,t,\tau_\theta(y))
\right\|^2
\right].
$$

对于 Mask、低分辨率图、语义图等与图像网格对齐的条件，也可以沿通道维拼接；Cross-Attention 更适合文本等 token 条件。

### 4. 完整训练与推理

阶段一训练 Encoder/Decoder；阶段二通常冻结它们：

1. 图像编码为 $z_0$；
2. 随机抽 $t,\epsilon$；
3. 构造 $z_t$；
4. 编码条件；
5. 条件 U-Net 预测噪声；
6. MSE 反向传播更新生成网络。

文生图推理：

1. Prompt 编成文本 token；
2. 从随机 $z_T$ 开始；
3. U-Net 与 Sampler 多步更新 latent；
4. VAE Decoder 最后只解码一次。

纯文生图没有输入图片，所以通常不需要在推理开头使用 VAE Encoder；图生图和编辑才需将源图编码。

还要分清：LDM 原论文的文本实验使用 BERT tokenizer 和论文自己的 Transformer 条件编码器；Stable Diffusion v1.x 才典型使用冻结 CLIP ViT-L/14。不能把“LDM”直接等同于“必须使用 CLIP 的 Stable Diffusion”。

### 5. 论文实验究竟证明什么

论文不是只展示文生图，而是通过多类任务证明 latent 扩散是通用框架：

- 无条件生成：说明 latent 并未破坏生成能力；
- 文本与类别条件生成：说明条件接口有效；
- 布局条件生成：说明可接受空间控制；
- 超分辨率与 Inpainting：说明同一 latent 框架能迁移到图像到图像任务；
- 不同压缩率实验：说明计算效率与重建上限之间存在最佳区域。

### 6. 贡献与局限

主要贡献：

- 将感知压缩与语义生成拆成两阶段；
- 在保留二维布局的 latent 中扩散，显著降低计算成本；
- 一个 Autoencoder 可复用于多个生成任务；
- 提供通用 Cross-Attention 条件接口；
- 统一支持文本、类别、布局、超分、修复等任务。

局限：

- 质量上限受 Autoencoder 重建能力限制；
- 压缩丢失的细节无法被扩散准确恢复；
- 两阶段系统更复杂，可能产生 VAE 重建伪影；
- 扩散采样仍需多次网络前向，并未解决速度问题；
- 训练数据与 Caption 质量决定条件生成上限。

### 7. 它如何通向后文

- DiT 保留 VAE 和 latent，只替换 U-Net Backbone；
- SD3 仍在 latent 中生成，但改用 RF 速度目标与 MM-DiT；
- SDEdit、BLD、P2P、ControlNet 等都利用 LDM/Stable Diffusion 的 latent、U-Net 或 Cross-Attention 机制。

---

## 七、论文 05：Scalable Diffusion Models with Transformers（DiT）

**一句话定位：**在 LDM 里保留 latent 与扩散数学，只把 U-Net 去噪器换成 Transformer，并系统验证生成质量随计算规模提升。

详细课内笔记：[09-Scalable Diffusion Models with Transformers（DiT）](#/notes/dit)

### 1. 它要解决什么问题

扩散模型长期以卷积 U-Net 为默认 Backbone。作者问：

1. 卷积和 U 形多尺度结构是扩散生成必需的吗？
2. Transformer 用作扩散预测器时，是否也会展现清晰 Scaling 规律？

DiT 改变的只有：

$$
\boxed{
\epsilon_\theta^{\mathrm{U\text{-}Net}}
\longrightarrow
\epsilon_\theta^{\mathrm{Transformer}}
}
$$

它保留 VAE、latent、DDPM 类加噪、Sampler 与 CFG。因此最重要的层级关系是：

$$
\boxed{
\text{DDPM 决定学什么，LDM 决定在哪里学，DiT 决定用什么网络学。}
}
$$

### 2. 怎样把二维 latent 交给 Transformer

设 noisy latent：

$$
z_t\in\mathbb R^{I\times I\times C}.
$$

用边长为 $p$ 的 patch 切块，token 数为：

$$
\boxed{T=\left(\frac Ip\right)^2}.
$$

每块有 $p^2C$ 个数，拉平并经过共享 Linear 投影到 hidden dimension $d$：

$$
X_0=\operatorname{PatchEmbed}(z_t)+P_{\mathrm{pos}}.
$$

三类信息不可混淆：

| 信息 | 回答的问题 |
|---|---|
| 二维位置编码 | 这个 latent patch 在图像哪里？ |
| 时间步嵌入 | 当前样本噪声多重？ |
| 类别嵌入 | 要生成哪个 ImageNet 类别？ |

DiT 类似 Transformer Encoder / ViT，不使用 Causal Mask。它不是从左到右逐 patch 生成；在一个扩散时间步内，所有 patch 同时存在、共同 Self-Attention、并行预测。

### 3. adaLN-Zero 为什么关键

普通 LayerNorm：

$$
\operatorname{LN}(x)
=
\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}.
$$

条件 $c=e_t(t)+e_y(y)$ 通过 MLP 生成调制参数：

$$
\operatorname{adaLN}(x,c)
=
(1+\gamma(c))\odot\operatorname{LN}(x)+\beta(c).
$$

adaLN-Zero 进一步为两个残差分支生成 Gate：

$$
x'
=
x+alpha_{\mathrm{MSA}}(c)\odot
\operatorname{MSA}(\operatorname{adaLN}(x,c)),
$$

$$
x_{\mathrm{out}}
=
x'+\alpha_{\mathrm{MLP}}(c)\odot
\operatorname{MLP}(\operatorname{adaLN}(x',c)).
$$

产生这些参数的最后线性层零初始化，使训练初期：

$$
\alpha_{\mathrm{MSA}}\approx0,
\qquad
\alpha_{\mathrm{MLP}}\approx0.
$$

Block 初始近似恒等映射，深网络更稳定。“Zero”不是所有模型参数为零，也不是 LayerNorm 输出为零，而是残差分支最初被 Gate 暂时关闭。

### 4. 输出为什么是 $2C$

DiT 末端：

$$
[B,T,d]
\rightarrow
[B,T,p^2(2C)]
\rightarrow
[B,2C,I,I].
$$

其中一半通道预测噪声，另一半预测反向高斯方差相关参数。这与 VAE Decoder 完全不同：

- DiT Output Head：token 变成 latent 中的扩散预测量；
- VAE Decoder：最终干净 latent 变成 RGB 图像。

### 5. 完整训练与推理

训练：

1. 冻结 VAE，把图像编码为 $z_0$；
2. 随机抽 $t,\epsilon$ 并构造 $z_t$；
3. Patchify，加入二维位置编码；
4. 时间与类别条件控制 adaLN-Zero Block；
5. DiT 输出噪声和方差参数；
6. 噪声部分用 MSE，方差部分配合变分项；
7. 反向传播更新 DiT。

推理：

$$
z_T\rightarrow z_{T-1}\rightarrow\cdots\rightarrow z_0
\rightarrow\text{VAE Decoder}.
$$

单个时间步内所有 patch 并行，但不同扩散时间步仍然串行。

### 6. Scaling 实验如何理解

论文组合四种模型规模 S/B/L/XL 和三种 latent patch size /8、/4、/2，共 12 个模型。

- 固定 patch size，模型越大通常 FID 越低；
- 固定模型，patch 越小，token 越多，质量通常越好；
- GFLOPs 与 FID 相关系数约为 $-0.93$。

为什么 GFLOPs 比参数量更有解释力？减小 patch 会大幅增加 token 数和 Attention 计算，却几乎不改变模型参数量，所以参数量无法单独表达实际计算规模。

论文还比较 In-context、Cross-Attention、adaLN、adaLN-Zero 四种条件方式；在其**类别条件**设定下，adaLN-Zero 效果最好且额外计算低。但不能据此断言在长文本任务中 adaLN 永远优于 Cross-Attention。

最终 ImageNet 256 的 DiT-XL/2 + CFG 报告 FID 2.27，ImageNet 512 报告 FID 3.04。

### 7. 贡献与局限

主要贡献：

- 证明 U-Net 不是扩散模型唯一有效 Backbone；
- 提出简洁的 latent Diffusion Transformer；
- 提出并验证 adaLN-Zero；
- 系统研究模型深宽、Patch Size、GFLOPs 与生成质量的关系；
- 为后续大型 Transformer 生成模型提供 Scaling 基线。

局限：

- 原始 DiT 是 ImageNet 类别条件，不是开放文本生成；
- 仍依赖卷积 VAE，不是完全“纯 Transformer”；
- Self-Attention 对 token 数为二次复杂度；
- 小 patch 和高分辨率会显著增加计算；
- 多步扩散采样仍未消失；
- FID 不能完整代表语义正确性、人类偏好和单图质量。

### 8. 它如何通向后文

DiT 把 2017 年 Transformer 的 token 思维迁移到图像 latent：

$$
\text{词 token}\longrightarrow\text{latent patch token}.
$$

但原始 DiT 只有图像 patch 与简单类别条件。MM-DiT 的下一步是让长文本 token 与图像 token 在每一层真正双向交互。

---

## 八、论文 06：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT / SD3）

**一句话定位：**它不是只提出一个新 Block，而是把 latent、改进 Rectified Flow、双流 Transformer、多文本编码器和高分辨率训练技术组合成 Stable Diffusion 3。

详细课内笔记：[10-Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT）](#/notes/mmdit)

### 1. 它要解决什么问题

1. Rectified Flow 怎样用于大规模高分辨率文生图？
2. 普通 DiT 或单向 Cross-Attention 是否足以理解复杂文字？
3. 系统扩大时，验证 Loss 和最终生成能力是否呈可预测的 Scaling 趋势？

完整系统是：

$$
\boxed{
\text{Stable Diffusion 3}
=
\text{16 通道 VAE}
+
\text{改进 RF}
+
\text{MM-DiT}
+
\text{多文字编码器}
+
\text{高分辨率训练与后训练}
}
$$

### 2. 论文采用的 RF 方向

$$
\boxed{z_t=(1-t)x_0+t\epsilon}
$$

- $t=0$：干净数据 latent；
- $t=1$：高斯噪声。

对 $t$ 求导：

$$
\boxed{\frac{dz_t}{dt}=\epsilon-x_0}.
$$

训练目标：

$$
\boxed{
L_{\mathrm{RF}}
=
\mathbb E
\left[
\left\|
v_\theta(z_t,t,c)-(\epsilon-x_0)
\right\|^2
\right]
}
$$

网络学的是随 $t$ 增大时“数据到噪声”的速度；生成则从 $t=1$ 反向积分到 $t=0$：

$$
z_{t-\Delta t}
\approx
z_t-\Delta t\,v_\theta(z_t,t,c).
$$

符号反过来不代表从图像生成噪声；关键是生成积分方向也反过来。

### 3. Logit-Normal Timestep Sampling

普通 RF 常用 $t\sim U(0,1)$。论文认为中间噪声区更困难、更有感知意义，于是使用：

$$
u\sim\mathcal N(m,s^2),
\qquad
\boxed{t=\operatorname{sigmoid}(u)}.
$$

在 $m=0,s=1$ 时，训练更频繁采到中间时间步。它只改变：

> 有限训练预算在不同噪声强度上怎样分配。

它不改变直线路径、速度标签，也不直接规定推理 Solver 的时间步。论文比较 61 种 formulation，`rf/lognorm(0,1)` 在其设定中综合最好；不能外推为所有任务永远最优。

### 4. 图像与文字输入

图像经下采样 $f=8$、16 通道 VAE：

$$
[B,3,1024,1024]
\rightarrow
[B,16,128,128].
$$

使用 $2\times2$ latent patch：

$$
[B,16,128,128]
\rightarrow
[B,4096,64]
\rightarrow
[B,4096,D].
$$

文字编码器包括 CLIP-L/14、OpenCLIP bigG/14、T5-v1.1-XXL，形成：

- $c_{\mathrm{vec}}$：Prompt 全局摘要，用于 Block 调制；
- $c_{\mathrm{ctxt}}$：逐 token 表示，进入 Joint Attention。

文字编码器冻结只表示其自身参数不更新；文本 token 进入 MM-DiT 后仍会逐层更新。

### 5. MM-DiT 到底“多模态”在哪里

MM 表示 Multimodal，不是 Multi-Head。文字流与图像流先使用各自参数：

$$
Q_I=X_IW_I^Q,\quad
K_I=X_IW_I^K,\quad
V_I=X_IW_I^V,
$$

$$
Q_T=X_TW_T^Q,\quad
K_T=X_TW_T^K,\quad
V_T=X_TW_T^V.
$$

沿 token 维拼接：

$$
Q=[Q_T;Q_I],\qquad
K=[K_T;K_I],\qquad
V=[V_T;V_I].
$$

共同计算：

$$
\boxed{
O
=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_h}}
\right)V
}
$$

注意力矩阵包含四块：

$$
\begin{bmatrix}
\text{文字看文字} & \text{文字看图像}\\
\text{图像看文字} & \text{图像看图像}
\end{bmatrix}.
$$

输出再拆成文字与图像两部分，各自经过输出投影、Residual、FFN。其本质是：

$$
\boxed{
\text{模态专用参数}
+
\text{共享联合 Attention}
}
$$

它与 LDM Cross-Attention 的区别：

- LDM：图像 Query 读取相对固定的文本 Key、Value，主要是文本影响图像；
- MM-DiT：两类 token 都有 Query，都能读取对方，并在每层同时更新。

### 6. 为什么只输出 $C$ 而不是 DiT 的 $2C$

原始 DiT 配合反向高斯扩散，输出噪声和方差相关参数，共 $2C$。MM-DiT 配合确定性 RF ODE，只需预测与 latent 同形状的速度：

$$
v_\theta(z_t,t,c)
\in
\mathbb R^{B\times C\times h\times w}.
$$

因此输出 $C$ 通道，不再预测另一半反向高斯方差。

### 7. 完整训练与推理

训练：

1. VAE 编码图像为 $x_0$；
2. 三个文本编码器产生条件；
3. 采样 $\epsilon$；
4. Logit-Normal 抽 $t$；
5. 构造 $z_t=(1-t)x_0+t\epsilon$；
6. MM-DiT 预测速度；
7. 与 $\epsilon-x_0$ 做 MSE；
8. 更新 MM-DiT，冻结 VAE 和文字编码器；
9. 条件 Dropout 让同一模型学会空条件分支，以供 CFG。

推理：

1. 编码 Prompt；
2. 从 $z_1\sim\mathcal N(0,I)$ 开始；
3. 分别预测有条件与无条件速度；
4. 用 CFG 组合；
5. ODE Solver 从 $t=1$ 积分到 $0$；
6. VAE Decoder 解码。

单个时间步内 token 并行，但 ODE 时间步仍按顺序执行。

### 8. 其他系统改进不能漏掉

- 更详细的合成 Caption 与原始 Caption 混合；
- 16 通道 VAE，提高重建能力上限；
- QK-RMSNorm，抑制大模型和高分辨率下 Attention Logits 爆炸；
- Aspect-Ratio Buckets；
- 高分辨率 Timestep Shift；
- 冻结编码器表示预计算，节省训练计算；
- 图像偏好 DPO 与 LoRA 后训练；
- 最大约 8B 参数的 Scaling Study；
- 初步视频 Scaling 实验。

Timestep Shift 的直觉：分辨率越高，观察维度越多，同样噪声比例下仍可能恢复出信号，所以有效训练和采样日程需向更强噪声区域移动。

### 9. 贡献、证据与局限

主要贡献：

- 大规模比较 Diffusion 与 Flow Formulation；
- 提出 RF 的 Logit-Normal 时间采样；
- 提出文字—图像双向交互的 MM-DiT；
- 验证模态专用参数 + Joint Attention 优于论文测试的共享参数或单向 Cross-Attention 方案；
- 把系统扩展到约 8B 参数，并发现 Validation Loss 与多个生成指标、人类偏好相关；
- 显著改善 Prompt Following、Typography 和空间组合能力。

最终 GenEval 0.74 来自最大模型、1024 分辨率、DPO、改进 RF、MM-DiT、16 通道 VAE、多文本编码器、更好 Caption 等完整系统，不能将全部增益归因于 MM-DiT 单一模块。

局限：多项改进同时出现，单模块净贡献难完全分离；Joint Attention 随图像 token 增多而昂贵；16 通道 latent 同时增加网络负担；RF 监督路径为直线但生成 ODE 不保证完全直；仍需多步前向；训练依赖巨大数据与算力，论文信息不足以完整复现工业级系统。

### 10. 基础链在这里汇合

$$
\boxed{
\text{Transformer 提供信息路由}
}
$$

$$
\boxed{
\text{DDPM / RF 提供数据—噪声路径与训练目标}
}
$$

$$
\boxed{
\text{LDM 提供 latent 空间}
+
\text{DiT 思想提供 Transformer 预测器}
+
\text{MM-DiT 提供多模态联合交互}
}
$$

从这里开始，研究问题从“怎样训练一个强生成器”转向：

> 已经有强生成先验后，怎样在保持源图的同时，只改变用户要求的内容？

---

# 第二部分：编辑与控制链

## 九、先理解所有编辑方法的共同矛盾

输入一张源图 $x^{\mathrm{src}}$ 和目标文字 $c^{\mathrm{tar}}$ 时，我们同时希望：

1. **目标一致性**：修改后的图符合新指令；
2. **源图保持性**：身份、姿态、构图、背景等无关区域尽量不变；
3. **真实性**：结果仍在自然图像流形附近；
4. **可操作性**：不为每张图重新训练巨大模型，推理成本可接受。

这些目标彼此牵制：

- 修改太弱：源图保持好，但指令没执行；
- 修改太强：目标出现了，却像重新生成一张无关图片；
- 注入源特征太多：输出被锁死，改不动；
- 注入太少：源身份、布局或背景漂移。

后面五种编辑方法的区别，本质上是它们选择在生成过程的哪个位置插手：

| 方法 | 主要干预对象 | 最直观的旋钮 |
|---|---|---|
| SDEdit | 初始噪声层级 | 加噪强度 $t_0$ |
| BLD | 每个时间步的前景/背景 latent | Mask 与 blending |
| Prompt-to-Prompt | Cross-Attention Map | 哪些词、哪些步替换/保留 |
| FlowEdit | 源条件与目标条件的速度差 | 编辑 ODE、时间区间、平均次数 |
| Stable Flow | Transformer 关键层视觉 $K,V$ | Vital Layer 集合与注入强度 |

ControlNet 则不属于“仅在现有推理路径上做免训练干预”的同一路线。它训练额外分支，让模型长期学会服从边缘、深度、姿态等空间条件。

---

## 十、论文 07：SDEdit — Guided Image Synthesis and Editing with Stochastic Differential Equations

**一句话定位：**先把输入图加到某个中间噪声层级，再用预训练生成模型反向去噪；加噪强度就是“保持输入”与“听从生成先验”之间的连续旋钮。

详细课内笔记：[11-SDEdit - Guided Image Synthesis and Editing with Stochastic Differential Equations](#/notes/sdedit)

### 1. 它要解决什么问题

用户的输入可能不是一张自然照片，而是手绘草图、粗糙涂色、带笔触的图片或不完整结构。直接把输入当最终结果不真实；从纯噪声生成又很难严格保持它的结构。

SDEdit 的问题是：

> 不专门为每类输入训练新模型，能否利用一个预训练 Score / Diffusion 模型，把“不够真实但有结构”的输入拉回自然图像分布？

### 2. 最核心操作：从中间时刻出发

设输入为 $x^{(g)}$，$g$ 表示用户给出的 guide。先加到噪声层级 $t_0$：

$$
\boxed{
x_{t_0}
=
\sqrt{\bar\alpha_{t_0}}\,x^{(g)}
+
\sqrt{1-\bar\alpha_{t_0}}\,\epsilon
}
$$

然后从 $t_0$ 开始执行预训练模型的反向生成：

$$
x_{t_0}\rightarrow x_{t_0-1}\rightarrow\cdots\rightarrow x_0^{\mathrm{edit}}.
$$

它并没有先寻找一个“能够精确重建源图的纯噪声”；因此经典 SDEdit 的关键不是 inversion，而是**有意破坏一部分输入信息，再让生成先验补全**。

### 3. 为什么加噪后反而更容易编辑

自然图像分布只占高维像素空间很小区域。手绘或粗糙输入可能离这个分布很远，模型难以直接处理。适量加噪有两个作用：

1. 抹去不真实的局部细节和伪影；
2. 保留较低频的轮廓、颜色和结构；
3. 把输入推到预训练生成模型熟悉的中间噪声分布附近；
4. 反向过程再利用生成先验恢复自然纹理。

### 4. $t_0$ 是真实性—忠实度旋钮

| $t_0$ | 输入保留 | 生成自由度 | 常见结果 |
|---|---|---|---|
| 很小 | 很强 | 很低 | 几乎照抄，原输入伪影也可能留下 |
| 中等 | 适中 | 适中 | 通常最平衡 |
| 很大 | 很弱 | 很高 | 更真实，但结构/身份容易漂移 |

论文称这两个方向为：

- **Faithfulness**：对输入 guide 的忠实程度；
- **Realism**：结果属于真实数据分布的程度。

SDEdit 没有神奇地同时最大化二者，而是给出一条可调节的折中曲线。

### 5. 条件引导怎样加入

若模型还有类别、文本、分割图等条件 $y$，反向过程使用条件 Score 或条件噪声预测：

$$
s_\theta(x_t,t,y)
\approx
\nabla_{x_t}\log p_t(x_t\mid y).
$$

这样输入 guide 主要提供结构起点，条件 $y$ 提供目标语义，预训练生成先验负责真实性。

### 6. 训练与推理

SDEdit 通常**不重新训练基础模型**。使用阶段：

1. 选择源输入与条件；
2. 选择 $t_0$；
3. 给输入加对应强度的噪声；
4. 调用预训练 Score / Diffusion 模型反向采样；
5. 必要时多次采样，选择合适结果。

“Training-free”只表示无需为该编辑方法训练新网络，不表示没有多步推理成本。

### 7. 实验、贡献与局限

论文在 stroke-based synthesis、图像编辑、补全、上色等任务展示通用性。人类评价中的 98.09% 是与某个具体基线做成对比较时的偏好比例，不是分类准确率，也不能脱离对照设置表述为“算法准确率 98.09%”。

主要贡献：

- 提出极简、通用的中间时刻编辑接口；
- 不要求输入本身来自真实数据分布；
- 用单一噪声强度刻画 realism–faithfulness 折中；
- 可复用不同预训练 Score 模型和多种条件。

局限：

- $t_0$ 需按图像和任务调节；
- 大改动会破坏身份与构图，小改动又可能不够明显；
- 没有显式的局部 Mask 或词—区域控制；
- 随机采样导致结果不确定；
- 高保真真实图编辑并非它最强的场景。

### 8. 它如何通向下一篇

SDEdit 只有一个全局噪声旋钮。若只想换图像中一个局部物体，而背景几乎逐像素保持，就需要在每个去噪步骤显式区分前景与背景——这正是 BLD 的出发点。

---

## 十一、论文 08：Blended Latent Diffusion（BLD）

**一句话定位：**在 latent 扩散的每一个时间步，把模型生成的 Mask 内前景与正确噪声层级的源图背景拼接，从而实现训练自由的局部文本编辑。

详细课内笔记：[12-Blended Latent Diffusion](#/notes/blended-latent-diffusion)

### 1. 它要解决什么问题

SDEdit 对整张图加噪，局部编辑时背景也可能变化。BLD 接收：

- 一张源图；
- 一个二值或软 Mask；
- 一个目标文本。

目标是：Mask 内按文字重绘，Mask 外尽量恢复源图。

### 2. 为什么不能只在最后一步粘贴

简单做法是“生成完整图片后，把原背景拷回去”。但前景的边界、光照、阴影与背景没有共同演化，容易出现接缝。

BLD 的核心不只是 blending，而是：

> 在每一个扩散时间步，都把前景与背景置于同一个噪声水平后再融合。

### 3. 每一步的两条候选分支

先把源图编码到 latent：

$$
z_0^{\mathrm{src}}=\mathcal E(x^{\mathrm{src}}).
$$

在从 $t$ 到 $t-1$ 的一步中，构造：

1. **前景候选** $z_{t-1}^{\mathrm{fg}}$：由文本条件扩散模型对当前 latent 去噪得到；
2. **背景候选** $z_{t-1}^{\mathrm{bg}}$：从源 latent 直接加到同一噪声层级得到。

背景候选类似：

$$
z_{t-1}^{\mathrm{bg}}
=
\sqrt{\bar\alpha_{t-1}}z_0^{\mathrm{src}}
+
\sqrt{1-\bar\alpha_{t-1}}\epsilon.
$$

然后融合：

$$
\boxed{
z_{t-1}
=
m\odot z_{t-1}^{\mathrm{fg}}
+
(1-m)\odot z_{t-1}^{\mathrm{bg}}
}
$$

- $m=1$：采用模型生成前景；
- $m=0$：采用源图背景；
- 软边界区域为二者加权过渡。

### 4. 为什么强调“相同噪声层级”

若一边是近乎干净的源 latent，另一边是高噪声生成 latent，二者统计分布不同，拼接后模型会看到异常状态。把源背景也加到对应 $t-1$，可使整个 latent 在同一噪声尺度上，再交给下一步去噪。

这是 BLD 最应记住的不变量：

$$
\boxed{\text{每一步融合前，前景和背景必须处于相同噪声时间。}}
$$

### 5. Progressive Mask Shrinking 与后处理

论文逐步缩小或调整编辑区域，使早期阶段允许新物体重构较大结构，后期阶段更多保护边缘与背景，减少接缝。

还可进行 decoder optimization：由于 VAE 编码再解码本身会造成重建误差，作者可微调解码相关过程，使 Mask 外像素更接近原图。多候选结果还可用 CLIP 语义分数排序。

### 6. 完整推理流程

1. VAE 编码源图；
2. 准备与 latent 尺寸对齐的 Mask；
3. 从较高噪声 latent 开始；
4. 每步用目标文本产生前景候选；
5. 用源 latent 产生同时间步背景候选；
6. 按 Mask 融合；
7. 重复到 $z_0$；
8. VAE 解码；
9. 可选 decoder optimization 与 CLIP 排序。

方法不要求重新训练扩散模型，但每一步仍要执行生成网络，并维护源背景分支。

### 7. 贡献与局限

主要贡献：

- 把 Mask 局部融合搬入 latent 扩散的每一个时间步；
- 显式保护背景，同时让前景受文本控制；
- 不需针对任务重新训练基础模型；
- 可借助 progressive mask 与优化减轻边界问题。

局限：

- 需要用户给 Mask；
- 编辑质量高度依赖 Mask 大小和边界；
- 过小 Mask 限制新物体形状，过大 Mask 破坏背景；
- VAE 重建会让 Mask 外也产生轻微变化；
- 物体带来的真实阴影、反射、遮挡可能需要改变 Mask 外区域，硬保护背景反而不自然；
- CLIP 排序增加成本，也不保证符合所有人类偏好。

### 8. 它如何通向下一篇

BLD 用用户 Mask 指定“哪里能变”。Prompt-to-Prompt 则问：生成模型内部是否已经通过文字—图像 Attention 自己形成了空间布局？若是，是否可以通过控制 Attention Map，在不画 Mask 的情况下保持构图？

---

## 十二、论文 09：Prompt-to-Prompt Image Editing with Cross-Attention Control

**一句话定位：**同步运行源 Prompt 与目标 Prompt 的生成分支，并在一定时间内向编辑分支注入或修改源分支的 Cross-Attention Map，使文字改变而空间布局尽量保持。

详细课内笔记：[13-Prompt-to-Prompt Image Editing with Cross-Attention Control](#/notes/prompt-to-prompt)

### 1. 它要解决什么问题

若用同一个随机种子，只把 Prompt 从 “a red car” 改成 “a blue car”，图像布局仍可能大幅变化。原因是生成过程中 Cross-Attention 不只控制“有没有这个词”，还参与决定词与空间位置的对应关系。

论文观察到：

> Cross-Attention Map 承载了大量空间布局信息；改 Prompt 时若保留合适的 Map，就可能保留构图。

### 2. Attention Map 为什么能代表空间关系

LDM 中图像位置作为 Query，文本 token 作为 Key、Value：

$$
A_t
=
\operatorname{softmax}
\left(
\frac{Q_tK_t^{\mathsf T}}{\sqrt{d_k}}
\right).
$$

$A_t[i,j]$ 表示图像位置 $i$ 在当前层、当前时间步对第 $j$ 个文本 token 的注意程度。把固定词 $j$ 对所有图像位置的权重画出来，就得到一张空间热力图。

P2P 直接控制的是 **Cross-Attention Map $A$**。目标分支仍自然使用目标 Prompt 产生的 Value；不应笼统说成“算法显式把源 Value 换进目标分支”。

### 3. 双分支同步生成

准备：

- 源 Prompt $P$；
- 编辑 Prompt $P^*$；
- 同一个初始噪声；
- 同样的采样时间表。

源分支产生源图及 Attention Map，目标分支生成编辑图。在指定的早期时间步和层，把源 Map 或经过映射的源 Map 注入编辑分支：

$$
A_t^{\mathrm{edit}}
\leftarrow
\operatorname{Edit}
\bigl(A_t^{\mathrm{src}},A_t^{\mathrm{tar}}\bigr).
$$

早期扩散步骤主要决定全局形状与布局，后期更多补纹理。因此常见策略是早期保留/注入较多源 Attention，后期放开让目标语义细化。

### 4. 三种核心操作

#### Word Swap：词替换

源和目标 Prompt 结构相近，如：

> a **dog** sitting on the grass  
> a **cat** sitting on the grass

用 token 对齐矩阵把源 Attention 的对应列映射到目标 token，使位置关系保留，而新 Value 和后续特征使物体语义改变。

#### Prompt Refinement：增加或删除词

例如：

> a castle  
> a castle **in the snow**

共有词继承源 Attention，新词使用目标分支自身 Attention，使原主体布局保持，同时允许新增描述进入。

#### Attention Reweighting：调节词强度

对某个词的 Attention 列乘系数：

$$
A'[:,j]=\lambda_jA[:,j].
$$

$\lambda_j>1$ 强化该词，$0<\lambda_j<1$ 弱化。它更像连续编辑，不必改 Prompt 文本本身。

### 5. LocalBlend

若只希望某些对象区域变化，可把选定词在多个层、Head 的 Attention Map 聚合并阈值化，得到空间 Mask，再融合源、目标 latent。它与 BLD 都会做空间混合，但 Mask 来源不同：

- BLD：用户显式提供 Mask；
- LocalBlend：从词的 Attention Map 自动估计区域。

### 6. 对生成图与真实图的差别

P2P 最自然的场景是：源图本来就是模型由源 Prompt 和某个噪声生成的，因而源路径已知。若输入任意真实照片，必须先找到可在模型中重建它的 latent/noise 轨迹，通常需要 DDIM inversion、Null-Text Inversion 等额外方法。

因此“P2P 本身不训练模型”不表示任意真实图都能直接无成本编辑。

### 7. 贡献与局限

主要贡献：

- 揭示 Cross-Attention Map 对图像空间布局的重要作用；
- 不训练新模型，通过推理时 Attention 控制完成编辑；
- 统一词替换、Prompt 增删与权重调节；
- 通过注入时间窗控制源图保持与目标变化的折中；
- LocalBlend 提供词级局部控制。

局限：

- 依赖模型的 Attention 语义与空间对齐质量；
- 复杂 Prompt 的 token 对齐可能不稳定；
- 大几何变化与姿态改变不适合强制保留旧 Map；
- 真实图通常需要 inversion，重建误差会影响编辑；
- 双分支、缓存和 Attention 操作增加内存与计算；
- Attention Map 不是完美分割图。

### 8. 它如何通向后文

P2P 展示“控制内部表示”比只改输入 Prompt 更精确，但真实图反演成为明显成本。FlowEdit 下一步直接问：对于 Rectified Flow 模型，能否不先把源图积分到噪声端，直接建立从源图到目标分布的编辑 ODE？

---

## 十三、论文 10：FlowEdit — Inversion-Free Text-Based Editing Using Pre-Trained Flow Models

**一句话定位：**将传统“先反演、再生成”路径代数重写为源条件速度与目标条件速度的差，得到无需显式反演和额外训练的编辑 ODE。

详细课内笔记：[14-FlowEdit - Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](#/notes/flowedit)

### 1. 它要解决什么问题

基于流模型编辑真实图的传统思路：

1. 用源条件把真实图从数据端积分到噪声端，即 inversion；
2. 再用目标条件从噪声端积分回目标图。

问题是反演需要完整额外轨迹、计算较贵，数值误差会累积；源路径与目标路径在高噪声区还可能分离，影响内容保持。

### 2. 从 inversion 路径推导速度差

记源图为 $Z_0^{\mathrm{src}}$。传统源反演状态与目标生成状态的关系，可写出一个差量路径：

$$
Z_t^{\mathrm{inv}}
=
Z_0^{\mathrm{src}}
+
Z_t^{\mathrm{tar}}
-
Z_t^{\mathrm{src}}.
$$

对时间求导：

$$
\boxed{
dZ_t^{\mathrm{inv}}
=
\left[
V^{\mathrm{tar}}(Z_t^{\mathrm{tar}},t)
-
V^{\mathrm{src}}(Z_t^{\mathrm{src}},t)
\right]dt
}
$$

最关键直觉：

> 不必真的保存“源图走到噪声”的整个轨迹；每一步只要知道目标条件想怎样推、源条件原本会怎样推，二者之差就是应施加的编辑方向。

### 3. 如何在不反演时构造两个查询点

每一步采样新的噪声 $N_t$，把源图与噪声插值：

$$
\boxed{
\widehat Z_t^{\mathrm{src}}
=(1-t)Z_0^{\mathrm{src}}+tN_t
}
$$

再用当前编辑状态 $Z_t^{\mathrm{FE}}$ 构造对应目标查询点：

$$
\boxed{
\widehat Z_t^{\mathrm{tar}}
=
Z_t^{\mathrm{FE}}
+
\widehat Z_t^{\mathrm{src}}
-
Z_0^{\mathrm{src}}
}
$$

随后分别计算：

$$
V^{\mathrm{src}}
=
v_\theta(\widehat Z_t^{\mathrm{src}},t,c_{\mathrm{src}}),
$$

$$
V^{\mathrm{tar}}
=
v_\theta(\widehat Z_t^{\mathrm{tar}},t,c_{\mathrm{tar}}).
$$

编辑状态按速度差更新。抽象写法：

$$
\boxed{
\frac{dZ_t^{\mathrm{FE}}}{dt}
=
\mathbb E_{N_t}
\left[
V^{\mathrm{tar}}-V^{\mathrm{src}}
\right]
}
$$

具体积分方向与实现的时间记号有关，但核心永远是“目标速度减源速度”。

### 4. 为什么速度差有利于保持内容

若某个结构在源条件和目标条件下都合理，两边速度会有相似分量，做差后被抵消；只有目标条件要求改变的分量会较突出。因此它天然带有“共同内容相消、差异内容保留”的倾向。

这不是严格保证：若两个速度场在无关区域也不一致，背景仍可能漂移；若目标改动很大，也会产生大范围变化。

### 5. $n_{\max}$、$n_{\min}$、$n_{\mathrm{avg}}$

- $n_{\max}$：从多高噪声时刻开始编辑；越高通常自由度越大；
- $n_{\min}$：在哪个低噪声时刻停止更新，避免后期过度改动；
- $n_{\mathrm{avg}}$：每个时间步采多少噪声样本平均速度差；越大越稳定，但网络调用越多。

至少要分别评估源、目标两个条件，因此一个时间步通常至少两次模型前向；若做多噪声平均，成本继续乘以 $n_{\mathrm{avg}}$。

### 6. 完整推理流程

1. VAE 编码源图；
2. 得到源 Prompt 与目标 Prompt 表示；
3. 初始化编辑状态为源 latent；
4. 在选定时间区间内，为每一步采样噪声；
5. 构造源查询点与目标查询点；
6. 同一预训练 flow model 分别预测源/目标速度；
7. 以速度差更新编辑状态；
8. 可对多份噪声求平均；
9. VAE 解码结果。

没有模型训练，也没有显式完整 inversion；但仍需多步、双条件甚至多次平均的推理。

### 7. 贡献与局限

主要贡献：

- 从 inversion-based editing 推导出速度差编辑 ODE；
- 对预训练 flow 模型实现 training-free、inversion-free 编辑；
- 不依赖源噪声轨迹缓存；
- 通过共享随机插值与速度差增强源内容保持；
- 能用于多种文本语义编辑。

局限：

- inversion-free 不等于零成本，每步至少双分支；
- 依赖预训练模型的源/目标速度估计质量；
- 超参数区间与平均次数会显著影响结果；
- 很大的结构变化仍可能破坏源图；
- 速度差路径并不自动保证是某种全局最优传输；
- 与扩散 U-Net 方法相比，它面向的是预训练 Flow 模型生态。

### 8. 它如何通向下一篇

FlowEdit 干预的是模型输出的速度场。Stable Flow 则观察到，MM-DiT 内部并非所有层都同等重要：少数层对视觉结构特别关键。若只在这些层注入源图特征，可能用更局部的网络内部干预实现稳定编辑。

---

## 十四、论文 11：Stable Flow — Vital Layers for Training-Free Image Editing

**一句话定位：**先用系统性消融定位 flow Transformer 中决定视觉结构的少数 Vital Layers，再只向编辑分支这些层注入源分支视觉 Key、Value。

详细课内笔记：[15-Stable Flow - Vital Layers for Training-Free Image Editing](#/notes/stable-flow)

### 1. 它要解决什么问题

现代 Stable Diffusion 3、FLUX 一类模型使用深层多模态 Flow Transformer。已有特征注入编辑常：

- 向许多层甚至所有层注入源特征；
- 缺乏“为什么选这些层”的解释；
- 注入太多会把编辑分支锁成源图；
- 注入太少又无法保持结构。

论文的问题是：

> 哪些层真正控制图像结构？能否先测量，再只干预少数关键层？

### 2. Vital Layer 如何被发现

作者对每层做 bypass / ablation：让模型在一次实验中绕过某层，再用固定 Prompt 和噪声生成图像，与完整模型输出比较。

借助 DINOv2 视觉特征相似度衡量输出变化。可写成：

$$
\boxed{
\operatorname{vitality}(\ell)
=
1-
\frac1k
\sum_{i=1}^{k}
d\bigl(M_{\mathrm{full}}^{(i)},M_{-\ell}^{(i)}\bigr)
}
$$

其中 $d$ 是感知相似度；绕过第 $\ell$ 层后相似度越低，$1-d$ 越大，该层越 vital。

作者发现重要性高度集中，而不是每层平均承担同样作用。层的重要性取决于具体模型架构，不能把某个模型的层号直接照搬到另一模型。

论文报告的示例集合：

- FLUX：$[0,1,17,18,25,28,53,54,56]$；
- SD3：$[0,7,8,9]$。

这些是对应实现中的经验层号，不是普适定律。

### 3. 为什么注入视觉 $K,V$，保留编辑 $Q$

源分支与目标编辑分支同步运行。在 Vital Layer 中：

- 编辑分支保留自己的视觉 Query $Q_I^{\mathrm{edit}}$；
- 将源分支的视觉 Key、Value $K_I^{\mathrm{src}},V_I^{\mathrm{src}}$ 提供给编辑分支；
- 目标文本的 Key、Value 仍来自目标 Prompt。

直观上：

- 编辑 Query 表示目标分支当前“想找什么”；
- 源视觉 Key 提供原图结构的匹配地址；
- 源视觉 Value 提供原图视觉内容；
- 目标文本条件仍推动语义改变。

因此它既不像全盘复制源 latent，也不完全放任目标分支重构。

### 4. 为什么只选 Vital Layers

三种消融对比最能说明贡献：

- **所有层注入**：源图保持过强，目标编辑被抑制；
- **只注入非关键层**：保持作用太弱，布局和身份漂移；
- **只注入 Vital Layers**：较好平衡编辑强度与源图保持。

这说明关键不是“注入越多越好”，而是识别少数对结构因果敏感的计算位置。

### 5. 生成图与真实图流程

对于模型自己生成的源图，可以直接保存同一生成轨迹上的源视觉特征并同步注入。

对真实照片，仍通常需要先获得模型可重建的 latent/noise 轨迹，即 inversion；论文还使用 latent nudging，例如：

$$
z_0^{\mathrm{nudged}}=1.15z_0,
$$

并缓存源分支特征，以改善重建与编辑稳定性。

所以 Stable Flow 的“training-free”表示不训练新参数，不等于真实图无需反演，也不等于无需额外缓存和双分支计算。

### 6. 完整推理流程

1. 确定模型的 Vital Layer 集合；
2. 准备源与目标 Prompt；
3. 对生成图复用已知轨迹；对真实图做 inversion / nudging；
4. 源、编辑分支同步求解 flow；
5. 缓存源分支 Vital Layer 的视觉 $K,V$；
6. 只在编辑分支对应层注入；
7. 其余层正常使用目标分支特征；
8. ODE 完成后 VAE 解码。

### 7. 贡献与局限

主要贡献：

- 提出可测量的 layer vitality 分析；
- 发现大规模 Flow Transformer 的视觉控制集中在少数层；
- 只在关键层注入源视觉 $K,V$，减少不必要干预；
- 将训练自由编辑扩展到 SD3、FLUX 一类模型；
- 通过所有层、非关键层、关键层对比解释方法为何有效。

局限：

- Vital Layers 是模型特定、版本特定的，换架构需重新扫描；
- 层消融指标是经验代理，不是完备因果证明；
- 真实图仍可能依赖 inversion 和 latent nudging；
- 双分支与特征缓存增加计算和显存；
- 大幅几何编辑依然可能与源结构注入冲突；
- DINO 距离关注视觉相似性，但不能覆盖所有语义与人类偏好。

### 8. 它如何与前文形成闭环

Stable Flow 把最早的 QKV 概念变成可操作编辑工具：

$$
\text{Transformer 的 }Q,K,V
\rightarrow
\text{MM-DiT 文图联合建模}
\rightarrow
\text{关键层视觉 }K,V\text{ 注入}.
$$

它也展示新的研究范式：不一定重新训练完整模型，可以先解释模型内部功能分工，再寻找最小干预点。

---

## 十五、论文 12：Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet）

**一句话定位：**冻结强大的预训练文生图 U-Net，复制其编码与中间模块作为可训练控制分支，并用零初始化卷积把边缘、深度、姿态等空间条件安全地注入原模型。

详细课内笔记：[16-Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet）](#/notes/controlnet)

### 1. 它要解决什么问题

纯文本擅长说明“画什么”，却很难精确规定：

- 人的骨骼姿态；
- 物体边缘与轮廓；
- 深度关系；
- 分割布局；
- 法线、线稿、涂鸦等空间结构。

直接微调整个 Stable Diffusion 成本高，而且小数据集可能破坏原有能力。只把控制图简单拼到输入，也不一定能充分利用预训练知识。

### 2. 锁定分支与可训练分支

对预训练网络块 $\mathcal F(x;\Theta)$：

- 一份保持冻结，参数 $\Theta$ 不更新；
- 一份复制为可训练块，参数 $\Theta_c$ 从预训练权重初始化；
- 条件 $c$ 经一个 zero convolution 注入复制分支；
- 复制分支输出再经 zero convolution 作为残差加回冻结分支。

抽象公式：

$$
\boxed{
y_c
=
\mathcal F(x;\Theta)
+
\mathcal Z
\left(
\mathcal F
\left(x+\mathcal Z(c;\Theta_{z1});\Theta_c\right);
\Theta_{z2}
\right)
}
$$

其中 $\mathcal Z$ 是权重和偏置都零初始化的 $1\times1$ 卷积。

### 3. 为什么 Zero Convolution 不是“永远学不动”

训练刚开始：

$$
\mathcal Z(\cdot)=0,
$$

所以 ControlNet 对原模型没有扰动，完整系统一开始等价于原有 Stable Diffusion。

虽然某些上游复制分支参数在第一步可能暂时收不到有效梯度，但输出端 zero conv 自己的权重梯度通常不为零；一旦它从零更新，梯度就会继续传回复制分支。于是训练表现为：

1. 从完全不破坏原模型开始；
2. 控制残差逐渐增强；
3. 可训练分支慢慢学会把空间条件翻译成预训练 U-Net 可用的特征。

### 4. 为什么只复制 Encoder 与 Middle Block

Stable Diffusion U-Net 含 12 个 Encoder Block、1 个 Middle Block，以及 Decoder。ControlNet 复制 12 个编码块和 1 个中间块，产生 13 组多尺度残差，注入冻结 U-Net 对应的 skip / decoder 路径；若把最开始条件输入的 zero conv 也计算在内，常说共有 14 个 zero convolution connections。

Encoder 多尺度特征已覆盖从局部边缘到全局结构的信息；复用冻结 Decoder 可保留原模型的生成能力并降低可训练成本。

### 5. 训练数据怎样构造

训练样本通常包含：

- 目标图像 $x_0$；
- 文本 Prompt $c_t$；
- 从目标图像提取或配对得到的控制图 $c_f$，如 Canny、Depth、Pose。

仍然使用原扩散噪声目标：

$$
\boxed{
\mathcal L
=
\mathbb E
\left[
\left\|
\epsilon-epsilon_\theta(z_t,t,c_t,c_f)
\right\|^2
\right]
}
$$

变化的是网络多接收一个控制条件，并训练复制分支；DDPM/LDM 的数学目标不需要重写。

论文训练中会以约 50% 概率将文本 Prompt 置空，使模型不得不从空间控制图中学习语义与结构，而不是总依赖文字捷径。

### 6. 推理与多条件组合

推理时：

1. 输入 Prompt；
2. 输入一张与目标布局相符的控制图；
3. ControlNet 产生多尺度残差；
4. 冻结 U-Net 同时接收文本与这些残差；
5. Sampler 多步去噪；
6. VAE 解码。

若使用多种控制，可把多个 ControlNet 的残差相加后注入：

$$
r_{\mathrm{total}}^{(\ell)}
=
\sum_j\lambda_j r_j^{(\ell)}.
$$

$\lambda_j$ 控制每种条件强度。但多个条件若彼此冲突，并不会因相加自动变得一致。

### 7. 贡献、证据与局限

主要贡献：

- 提出冻结主干 + 可训练副本的稳健条件控制结构；
- Zero Convolution 保证训练初期不破坏预训练模型；
- 支持 Canny、HED、Depth、Normal、Pose、Segmentation、Scribble 等多种条件；
- 在较小到较大数据规模下都展示可训练性；
- 支持多个空间条件组合；
- 把大型预训练模型的语义能力与任务特定的精确空间控制连接起来。

局限：

- ControlNet **需要训练**，不是 training-free 编辑；
- 每类条件通常需要相应模型或训练数据；
- 复制大量 U-Net 模块增加训练与推理成本；
- 控制图质量决定上限，错误姿态或深度会误导生成；
- 文本与控制条件冲突时需要权衡；
- 它擅长“按结构生成”，不天然保证任意真实源图的身份和纹理像素级保持。

### 8. 它在体系中的位置

ControlNet 与五篇编辑方法是正交补充：

- P2P、FlowEdit、Stable Flow：重点是利用现有模型对一张图做训练自由的语义编辑；
- BLD：通过 Mask 保护局部；
- ControlNet：通过额外训练让模型长期服从一类结构条件。

它继承前文三个关键思想：

$$
\text{DDPM/LDM 的噪声 Loss}
+
\text{U-Net 多尺度结构}
+
\text{Residual Add}
\rightarrow
\text{可训练空间控制器}.
$$

---

# 第三部分：把所有论文装回同一套系统

## 十六、三个代表性系统的完整“装配图”

### 1. Stable Diffusion v1.x

$$
\boxed{
\text{VAE latent}
+
\text{DDPM 类噪声预测}
+
\text{U-Net}
+
\text{CLIP 文本条件}
+
\text{Cross-Attention}
+
\text{Sampler}
}
$$

一次去噪调用中的信息流：

1. 当前 noisy latent $z_t$ 进入 U-Net；
2. 时间嵌入告诉网络噪声强度；
3. CLIP 将 Prompt 变成文本 token；
4. U-Net 图像特征作为 Query，通过 Cross-Attention 读取文本 Key、Value；
5. U-Net 输出噪声估计；
6. Sampler 根据噪声估计得到下一时刻 latent；
7. 重复多步后，VAE Decoder 把 $z_0$ 变成 RGB 图像。

对应论文来源：

- DDPM：噪声路径与噪声 MSE；
- LDM：VAE latent 与 Cross-Attention 条件接口；
- Transformer：QKV、Multi-Head、FFN、Normalization 等基础模块。

### 2. 原始 DiT 系统

$$
\boxed{
\text{VAE latent}
+
\text{DDPM 类扩散}
+
\text{DiT}
+
\text{类别 adaLN-Zero}
+
\text{Sampler}
}
$$

它相对 LDM 的关键变化仅是：

$$
\text{U-Net 图像特征图}
\longrightarrow
\text{latent patch tokens + Transformer}.
$$

因此不能说 DiT “取消了扩散”或“取消了 latent”；它仍在每个扩散时间步预测噪声，并仍依赖 VAE 解码。

### 3. Stable Diffusion 3

$$
\boxed{
\text{16ch VAE latent}
+
\text{Rectified Flow}
+
\text{MM-DiT}
+
\text{CLIP/T5 多文本条件}
+
\text{Joint Attention}
+
\text{CFG + ODE Solver}
}
$$

一次速度预测中的信息流：

1. noisy latent 被 Patchify 成图像 token；
2. 多文本编码器产生全局向量和文本 token；
3. 时间与全局文字向量调制 Block；
4. 文本流、图像流分别生成自己的 QKV；
5. 两类 QKV 拼接后做 Joint Attention；
6. 文字和图像表示都被更新；
7. MM-DiT 输出 latent 速度；
8. ODE Solver 用速度更新状态；
9. 从 $t=1$ 走到 $t=0$ 后由 VAE 解码。

### 4. 一个模型名字往往代表“配方”，不是单一公式

“Stable Diffusion 3 效果更好”不能只归因于 MM-DiT。完整结果同时受到：

- VAE 通道数与重建上限；
- 数据和 Caption 质量；
- Rectified Flow formulation；
- 时间步采样；
- Backbone 规模；
- 文本编码器能力；
- QK-Norm 等训练稳定技术；
- 分辨率课程、DPO 与 CFG

影响。科研阅读时必须分开“论文的核心创新”和“系统最终成绩所依赖的完整 recipe”。

---

## 十七、相邻论文为什么会出现：真正的递进关系

### 1. Transformer → 后续生成模型

Transformer 并不是 DDPM 的前一代生成器。它提供的是一种通用的信息路由结构：

$$
QK^{\mathsf T}\rightarrow\operatorname{softmax}\rightarrow V.
$$

后续论文拿走了这个部件：

- LDM 用它让图像位置查询文本；
- DiT 用它让所有 latent patch 全局交互；
- MM-DiT 用它让文本与图像双向交互；
- P2P 直接改 Attention Map；
- Stable Flow 直接换关键层视觉 $K,V$。

所以 Transformer 是贯穿两条研究链的“共同语法”。

### 2. DDPM → LDM：数学不变，计算空间改变

DDPM 证明逐步噪声预测可以生成高质量图像，却在像素空间计算昂贵。LDM 没有否定 DDPM，而是做模块化重构：

$$
x_t\in\mathbb R^{H\times W\times3}
\quad\longrightarrow\quad
z_t\in\mathbb R^{h\times w\times c}.
$$

生成数学大致保持，运行空间被压缩。这是“效率层”的创新，而不是另一种反向概率过程。

### 3. DDPM ↔ Rectified Flow：同一任务的两种动力学表达

二者都学噪声与数据之间的连续层级，却给出不同监督：

$$
\text{DDPM：预测加入的 }\epsilon,
$$

$$
\text{RF：预测插值路径的 }\frac{dx_t}{dt}.
$$

DDPM 的经典表述含离散随机反向链；RF 强调确定性 ODE 运输和路径直度。二者不是和 LDM、DiT 处于同一选择层。

### 4. LDM → DiT：空间不变，Backbone 改变

LDM 用卷积 U-Net 预测噪声。DiT 问的是 U-Net 是否必须：

$$
\text{latent feature maps + U-Net}
\quad\longrightarrow\quad
\text{latent patch tokens + Transformer}.
$$

这使扩散生成更容易继承 Transformer 的规模化经验，但代价是 Attention 对 token 数的 $O(n^2)$ 成本。

### 5. DiT → MM-DiT：简单类别条件变成深层多模态交互

原始 DiT 通过 adaLN 注入一个类别向量，无法充分表示长 Prompt 中多个词和复杂关系。MM-DiT 保留图文各自参数，却让它们在每层共享 Joint Attention：

$$
\text{单一图像 token 流 + 类别向量}
\quad\longrightarrow\quad
\text{文字流与图像流双向更新}.
$$

### 6. DDPM/LDM → SDEdit：把生成轨迹变成编辑接口

扩散模型已经学会从不同噪声层级回到自然图像。SDEdit 发现：不必总从纯噪声开始；从用户输入的中间噪声版本开始，就能在保留结构的同时利用生成先验。

### 7. SDEdit → BLD：全局旋钮变成空间局部控制

SDEdit 的 $t_0$ 对整张图统一作用。BLD 引入 Mask 并在每一步混合，让“哪里改变、哪里恢复源图”明确分开。

### 8. BLD → P2P：外部 Mask 变成内部语义布局控制

BLD 依赖用户画 Mask。P2P 发现 Cross-Attention Map 已隐含“哪个词对应哪里”，因此直接控制内部 Map，可以在 Prompt 变化时保持布局，并用 LocalBlend 自动得到词级区域。

### 9. P2P → FlowEdit：从 Attention 干预转向动力学干预

P2P 对真实图常需 inversion。FlowEdit 在预训练 Flow 模型中重新推导编辑路径，用目标速度减源速度，避免显式走完源图到噪声的反演轨迹。

### 10. FlowEdit → Stable Flow：从输出速度转向关键内部层

FlowEdit 直接修改最终速度场。Stable Flow 深入网络内部，先找少数决定视觉结构的 Vital Layers，再只注入这些层的源视觉 $K,V$，体现“理解内部机制后做最小干预”的路线。

### 11. ControlNet：与推理时编辑平行的训练式路线

ControlNet 不是以上方法的最后替代者。它回答的是另一个问题：

> 如果我们愿意为某类条件训练额外参数，怎样让预训练模型长期、稳定地服从精确空间控制？

因此它用冻结主干保留生成先验，用复制分支学习条件，用 Zero Convolution 安全接入。

---

## 十八、十二篇论文总表：问题—方法—贡献—局限

| # | 论文 | 它解决的核心问题 | 核心方法 | 最主要贡献 | 最关键局限 |
|---:|---|---|---|---|---|
| 01 | Transformer | 序列位置怎样直接、并行交换信息？ | Scaled Dot-Product + Multi-Head + FFN | 建立纯 Attention Encoder–Decoder | $O(n^2)$；自回归推理仍串行 |
| 02 | DDPM | 怎样稳定学习噪声到数据的生成？ | 固定高斯前向链 + 学习反向链 + 噪声 MSE | 建立高质量扩散生成底座 | 原始采样约千步，慢 |
| 03 | Rectified Flow | 怎样用简单监督学习更直的连续运输？ | 直线插值 + 速度回归 + Reflow | simulation-free ODE 学习与路径拉直 | 1-RF 不天然一步，Reflow 成本高 |
| 04 | LDM | 像素扩散为何如此昂贵？ | Autoencoder 压缩 + latent diffusion + Cross-Attention | 高效 latent 生成与通用条件接口 | 质量受 VAE 重建上限约束 |
| 05 | DiT | U-Net 是否是扩散唯一 Backbone？ | Patchify latent + Transformer + adaLN-Zero | 证明 Transformer 扩散 Scaling | Attention 成本高；原论文仅类别条件 |
| 06 | MM-DiT / SD3 | RF 怎样规模化并融合复杂文图条件？ | 16ch VAE + Logit-Normal RF + 双流 Joint Attention | 建立大型多模态 Flow Transformer 系统 | 改进耦合多、算力大，单模块贡献难拆 |
| 07 | SDEdit | 如何复用生成先验处理草图/输入图？ | 中间时刻加噪后反向生成 | 通用 training-free 编辑入口 | 全局折中粗糙；结构易随强噪声漂移 |
| 08 | BLD | 怎样只编辑 Mask 内并保护背景？ | 每步同噪声层级 latent blending | 局部文本编辑与背景保持 | 依赖 Mask；VAE 与边界伪影 |
| 09 | P2P | 改 Prompt 时怎样保留原布局？ | 双分支 + Cross-Attention Map 注入/重加权 | 将内部语义布局变成编辑接口 | 真实图常需 inversion；大形变困难 |
| 10 | FlowEdit | Flow 编辑能否免去显式 inversion？ | 源/目标查询 + 速度差 ODE | inversion-free、training-free Flow 编辑 | 每步双分支且依赖超参数，不是零成本 |
| 11 | Stable Flow | 哪些层控制视觉结构，怎样最小干预？ | 层消融定位 Vital Layers + 源视觉 KV 注入 | 关键层分析与稀疏内部注入 | 层集合模型特定；真实图仍可能反演 |
| 12 | ControlNet | 怎样让文生图严格服从边缘/深度/姿态？ | 冻结 U-Net + 可训练副本 + Zero Conv | 稳定复用大模型的训练式空间控制 | 需为条件训练，算力与存储增加 |

---

## 十九、六种控制/编辑路线的横向比较

### 1. 方法属性总表

| 方法 | 是否训练新参数 | 真实图是否通常需反演 | 是否需用户 Mask | 主要干预层级 | 每步额外成本 | 最擅长 |
|---|---:|---:|---:|---|---|---|
| SDEdit | 否 | 否 | 否 | 初始噪声状态 | 无额外分支，但仍多步采样 | 草图到真实图、全局重绘 |
| BLD | 否 | 否，直接编码源图 | 是 | latent 空间逐步融合 | 源背景构造与融合 | 局部替换、背景保护 |
| P2P | 否 | 对真实图通常是 | 否，可选 LocalBlend | Cross-Attention Map | 双分支/Map 缓存 | Prompt 小改、布局保持 |
| FlowEdit | 否 | 否 | 否 | Flow 速度场 | 至少源/目标两次前向 | 真实图语义编辑、免显式反演 |
| Stable Flow | 否 | 对真实图通常是 | 否 | Vital Layers 的视觉 $K,V$ | 双分支与 KV 缓存 | Flow Transformer 身份/结构保持 |
| ControlNet | **是** | 否 | 视条件而定 | 多尺度可训练残差 | 额外 ControlNet 分支 | 边缘、深度、姿态等精确控制 |

### 2. 显式空间控制与隐式空间控制

**显式空间条件**直接告诉模型每个位置应是什么结构：

- BLD 的 Mask；
- ControlNet 的 Edge、Depth、Pose、Segmentation。

**隐式内部控制**利用模型已学到的表示：

- P2P 的 Cross-Attention Map；
- Stable Flow 的关键层视觉 $K,V$；
- FlowEdit 的速度差。

显式控制更直观，但需要额外输入；隐式控制更方便，却依赖模型内部表示是否可靠。

### 3. 生成图与真实图为什么不同

模型自己生成的图通常知道：

- 初始噪声；
- Prompt；
- 每一步 latent / feature 轨迹。

真实照片只给最终像素，不知道它在生成模型中的起始噪声和路径。因此 P2P、Stable Flow 等若要同步复现源分支，常需 inversion。SDEdit、BLD、FlowEdit 各自用别的方式避开或放松了精确反演要求：

- SDEdit：主动加噪，不要求精确重建；
- BLD：源图经 VAE 编码后作为每步背景锚点；
- FlowEdit：用共享随机插值与速度差代替完整 inversion 路径。

### 4. “Training-free”到底是什么意思

它只表示：

$$
\boxed{\text{编辑时不再通过反向传播训练新的模型参数}}
$$

它不表示：

- 不需要预训练基础模型；
- 不需要多步采样；
- 不需要双分支；
- 不需要 inversion；
- 不需要缓存中间特征；
- 推理免费或即时。

汇报时应分别报告训练成本、反演成本、网络前向次数、缓存/显存，而不要只用一个“training-free”概括效率。

### 5. 一个实用的选择逻辑

- 输入是粗草图，希望生成自然图：优先理解 **SDEdit**；
- 有明确局部区域，只改 Mask 内：**BLD**；
- 只小改 Prompt，必须保留构图：**P2P**；
- 使用 RF 模型、希望真实图免显式反演：**FlowEdit**；
- 使用 SD3/FLUX，愿意双分支并强调关键层结构保持：**Stable Flow**；
- 有边缘/深度/姿态数据，愿意训练稳定控制模块：**ControlNet**。

这些不是绝对排名，真实系统还可组合。例如：Attention 控制可叠加 Mask，ControlNet 也可与其他采样引导配合。

---

## 二十、从宏观视角看：这套文献真正展示了怎样的研究演进

### 1. 从“端到端黑盒”走向“模块化生成系统”

现代图像生成不是一个公式包办全部，而是把问题拆成：表示、路径、目标、Backbone、条件、Solver。模块化带来两个研究优势：

1. 可以单独替换一个轴并做公平消融；
2. 可以把已验证组件重新组合成新系统。

例如 SD3 不是从零出现，而是组合：

$$
\text{LDM latent}
+
\text{RF dynamics}
+
\text{DiT scaling}
+
\text{multimodal attention}.
$$

### 2. 从“能生成”走向“高效生成”

- DDPM：质量强，但像素与千步采样昂贵；
- LDM：压缩空间，降低每次网络调用成本；
- RF：改变路径，追求更直、更适合少步求解；
- DiT/MM-DiT：探索计算规模如何转化成质量与条件理解。

效率至少有三种完全不同含义：

| 效率维度 | 典型工作 |
|---|---|
| 每步状态尺寸更小 | LDM |
| 所需步数更少 | RF、Reflow、Distillation、Solver |
| 训练计算更易扩展 | Transformer / DiT Scaling |

不能只说“某论文更快”，必须说明快在状态空间、步数、并行性还是训练复用。

### 3. 从“单模态预测”走向“条件对齐”

- Transformer 给出 token 交互；
- LDM Cross-Attention 让图像读取文本；
- MM-DiT 让图文双向更新；
- ControlNet 再引入空间条件；
- P2P 与 Stable Flow 把这些内部交互直接变成编辑控制手柄。

### 4. 从“重新训练模型”走向“复用生成先验”

强大预训练模型本身已包含纹理、结构、物体和语言对应知识。编辑论文越来越关注：

- 能否不重训整个模型；
- 能否不为每张图做昂贵优化；
- 能否找到最小干预位置；
- 能否解释为什么该干预既保留源图又执行目标。

SDEdit 干预状态，BLD 干预局部 latent，P2P 干预 Attention Map，FlowEdit 干预速度，Stable Flow 干预关键层 KV。它们实际上在逐渐深入生成器内部。

### 5. 从“经验技巧”走向“机制发现”

- P2P：Attention Map 承载布局；
- FlowEdit：编辑可以写成两个条件速度场之差；
- Stable Flow：视觉控制集中于少数关键层；
- ControlNet：零初始化接口能安全地把新能力接入冻结网络。

这一趋势对科研很重要：好工作不仅给结果，还应该说明**干预对象为何对应所需性质**，并通过消融验证。

---

## 二十一、所有关键公式串成一条线

### 1. 信息交互

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V
}
$$

意义：用 Query–Key 相似度选择并汇总 Value。

### 2. DDPM 任意时刻加噪

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
}
$$

意义：训练时可一步得到任意噪声层级。

### 3. DDPM 简化目标

$$
\boxed{
L_{\mathrm{simple}}
=
\mathbb E\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2
}
$$

意义：把复杂反向分布学习化成监督式噪声回归。

### 4. Rectified Flow 路径与速度

$$
\boxed{X_t=(1-t)X_0+tX_1},
\qquad
\boxed{\dot X_t=X_1-X_0}.
$$

$$
\boxed{
L_{\mathrm{RF}}
=
\mathbb E\|(X_1-X_0)-v_\theta(X_t,t,c)\|^2
}
$$

意义：从插值端点直接构造速度监督。

### 5. LDM

$$
\boxed{z_0=\mathcal E(x_0)},
\qquad
\boxed{\widehat x=\mathcal D(\widehat z_0)}.
$$

意义：生成数学搬到压缩但保留空间结构的 latent。

### 6. DiT token 数

$$
\boxed{T=(I/p)^2}.
$$

意义：patch 越小，token 越多、计算越大、细粒度能力通常越强。

### 7. SD3 RF 与 Logit-Normal

$$
\boxed{z_t=(1-t)x_0+t\epsilon},
\qquad
\boxed{v^*=\epsilon-x_0},
$$

$$
\boxed{u\sim\mathcal N(0,1),\quad t=\operatorname{sigmoid}(u)}.
$$

意义：在 latent 中学数据到噪声的速度，训练预算更多投向中间时刻；生成时反向积分。

### 8. BLD

$$
\boxed{
z_{t-1}
=
m\odot z_{t-1}^{\mathrm{fg}}
+
(1-m)\odot z_{t-1}^{\mathrm{bg}}
}
$$

意义：每一步在同噪声层级融合编辑前景与源背景。

### 9. Prompt-to-Prompt

$$
\boxed{
A_t
=
\operatorname{softmax}
\left(\frac{Q_tK_t^{\mathsf T}}{\sqrt{d_k}}\right)
}
$$

意义：控制词—图像位置关系，而不是简单复制最终像素。

### 10. FlowEdit

$$
\boxed{
\frac{dZ_t^{\mathrm{FE}}}{dt}
=
\mathbb E
\left[
V^{\mathrm{tar}}-V^{\mathrm{src}}
\right]
}
$$

意义：共同变化相消，目标与源条件的差异形成编辑方向。

### 11. ControlNet

$$
\boxed{
y_c
=
\mathcal F(x;\Theta)
+
\mathcal Z\bigl(
\mathcal F(x+\mathcal Z(c);\Theta_c)
\bigr)
}
$$

意义：冻结主干提供原能力，复制分支学习新条件，零卷积使新分支从“无干扰”开始。

### 12. 哪些推导必须会，哪些了解即可

**必须能自己解释：**

- Attention 的形状、QKV 角色与 $\sqrt{d_k}$；
- DDPM 闭式加噪为何让训练随机抽一个 $t$；
- DDPM Loss 的推导路线为何落到噪声 MSE；
- RF 插值求导为何得到端点差；
- MSE 最优预测为何是条件期望；
- LDM 为什么节省空间计算、又受 VAE 上限约束；
- DiT 如何 Patchify，adaLN-Zero 为什么初始近似恒等；
- MM-DiT 怎样拼 QKV 并形成四块 Joint Attention；
- BLD 为什么必须在同一噪声层级融合；
- FlowEdit 为什么是目标速度减源速度；
- ControlNet Zero Conv 为什么不会永远学不动。

**理解思路、不必现场逐行推：**

- DDPM 完整 ELBO 的所有高斯配方细节；
- Rectified Flow 边缘分布保持的严格连续性方程证明；
- 各 ODE Solver 的高阶误差公式；
- DINO vitality 指标的所有实现细节；
- 大模型训练中的完整分布式系统工程。

---

## 二十二、关键实验各自证明了什么

不能只背指标。汇报时最好说“实验设计 → 结论边界”。

| 论文 | 关键证据 | 它真正支持的结论 |
|---|---|---|
| Transformer | 翻译成绩、组件消融、句法分析 | 无 RNN/CNN 的 Attention 架构可高效建模序列并迁移任务 |
| DDPM | CIFAR-10 IS/FID、采样与率失真分析 | 简化噪声目标能训练高质量扩散生成器，生成逐步恢复信息 |
| Rectified Flow | 2D 运输、CIFAR、Domain Transfer、Reflow/蒸馏 | 简单速度回归可学运输；更直 coupling 更利于少步求解 |
| LDM | 压缩率对比、多类生成和条件任务 | 适度 latent 压缩能在效率与质量间取得良好平衡，条件接口通用 |
| DiT | 12 个规模/patch 配置、条件机制消融 | Transformer 可作扩散 Backbone，GFLOPs 在测试范围内与 FID 强相关 |
| MM-DiT | 61 种路径配方、架构对比、Scaling、人评 | RF 时间采样和双流架构有效，Loss 可预测多项指标，但最终效果来自完整系统 |
| SDEdit | 多任务展示、真实性/忠实度评价 | 单个中间噪声接口能在多种输入上复用生成先验 |
| BLD | Mask 编辑对比、背景保持、消融 | 逐步 latent blending 比最后粘贴更自然，能显式保护背景 |
| P2P | Word Swap/Refinement/Reweight、注入时间消融 | Cross-Attention Map 对布局有关键作用，可作为训练自由编辑接口 |
| FlowEdit | 多种 Flow Backbone/编辑任务与基线比较 | 速度差可在不显式反演下实现有效语义编辑 |
| Stable Flow | 逐层 bypass、DINO 差异、注入层消融 | 层重要性高度不均，少数 Vital Layers 足以强烈影响结构保持 |
| ControlNet | 多类控制、数据规模、Zero Conv 训练曲线、组合控制 | 冻结主干与可训练副本能稳健学习多种精确空间条件 |

实验结论的三条纪律：

1. 单一指标不等于完整能力，例如 FID 不直接衡量 Prompt 服从；
2. 最终系统结果不能全部归给某个新模块；
3. 人类偏好百分比必须说明对照方法、任务和评价方式。

---

## 二十三、从“研究问题”而不是论文标题来记忆

### 研究问题 A：怎样表示与交换信息？

回答链：

$$
\text{Attention}
\rightarrow
\text{图像 Patch Self-Attention}
\rightarrow
\text{文图 Joint Attention}
\rightarrow
\text{Attention / KV 编辑}.
$$

相关论文：Transformer、LDM、DiT、MM-DiT、P2P、Stable Flow。

### 研究问题 B：怎样连接简单噪声与复杂数据？

回答链：

$$
\text{DDPM 的高斯逐步去噪}
\quad\leftrightarrow\quad
\text{RF 的连续速度运输}.
$$

相关论文：DDPM、Rectified Flow、SD3、FlowEdit。

### 研究问题 C：怎样降低高分辨率生成成本？

回答链：

$$
\text{像素扩散}
\rightarrow
\text{latent 扩散}
\rightarrow
\text{可扩展 Transformer}
\rightarrow
\text{更合适路径与 Solver}.
$$

相关论文：DDPM、LDM、DiT、MM-DiT/RF。

### 研究问题 D：怎样控制“改什么”和“别改什么”？

回答链：

$$
\text{全局噪声强度}
\rightarrow
\text{显式 Mask}
\rightarrow
\text{词—区域 Attention}
\rightarrow
\text{条件速度差}
\rightarrow
\text{关键层视觉特征}.
$$

相关论文：SDEdit、BLD、P2P、FlowEdit、Stable Flow。

另一条回答是通过训练获得可复用空间控制：ControlNet。

### 研究问题 E：怎样保护预训练能力并添加新能力？

- 不训练、只改推理：SDEdit、BLD、P2P、FlowEdit、Stable Flow；
- 冻结主干、训练旁路：ControlNet；
- 冻结表示模型、训练生成器：LDM 阶段二、DiT、SD3；
- 零初始化新路径：DiT 的 adaLN-Zero Gate、ControlNet 的 Zero Conv。

注意最后两者都用“从接近恒等/零扰动开始”的思想，但所在结构和功能不同。

---

## 二十四、可直接用于汇报的完整叙事线

### 1. 30 秒总述

> 我的学习主线是现代生成模型如何从基础生成走向高效、可扩展和可控。Transformer 提供 token 间的信息交互；DDPM 与 Rectified Flow 描述噪声和数据之间的生成动力学；LDM 把生成移到压缩 latent；DiT 和 MM-DiT 用 Transformer 扩展预测器与文图融合。建立强生成器之后，SDEdit、BLD、Prompt-to-Prompt、FlowEdit 和 Stable Flow 分别从初始状态、latent、Attention、速度场和关键层特征介入编辑；ControlNet 则通过冻结主干和训练控制分支实现精确空间条件。整套工作的共同目标，是在质量、效率、条件服从和源图保持之间取得更好的平衡。

### 2. 约 3 分钟基础生成链

> 首先，Transformer 用 Q、K、V 将信息检索写成矩阵运算，使任意 token 可以直接交互，也为之后的文本—图像 Attention 打下基础。DDPM 随后定义固定高斯前向加噪过程，并学习其反向过程；它从变分目标得到噪声预测 MSE，把复杂概率建模转化成稳定监督学习，但原始像素空间千步采样昂贵。LDM 因此先用 Autoencoder 去掉感知冗余，再在具有二维结构的 latent 中扩散，并用 Cross-Attention 让图像位置查询文本。Rectified Flow 从另一条轴重新定义生成路径：在噪声和数据间插值，学习速度场，并用 Reflow 追求更直轨迹。DiT 又在保持 latent 和扩散目标的前提下，把 U-Net 换成 Patch Transformer，验证生成质量随 GFLOPs 扩展。Stable Diffusion 3 最后把这些思想组合起来：16 通道 VAE 提供表示空间，Rectified Flow 提供速度目标，MM-DiT 让文字和图像用各自参数生成 QKV、再联合 Attention，配合多文本编码器和高分辨率训练形成完整系统。

### 3. 约 3 分钟编辑与控制链

> 有了强生成先验，问题变成怎样既完成目标修改又保留源图。SDEdit 把输入加到某个中间噪声层级再反向去噪，用噪声强度控制真实性和忠实度。BLD 加入显式 Mask，在每个去噪步把目标前景和同噪声层级的源背景混合。Prompt-to-Prompt 发现 Cross-Attention Map 承载词与空间位置的对应，因此同步运行源、目标 Prompt 并注入 Attention Map，使语义改变时布局保持。对于 Flow 模型，FlowEdit 将传统反演再生成的路径改写成目标速度减源速度，从而避免显式 inversion。Stable Flow 更深入网络内部，通过逐层消融找到少数 Vital Layers，只在这些层注入源视觉 Key、Value。ControlNet 是平行的训练式路线：冻结原 U-Net，复制编码和中间块学习边缘、深度、姿态等控制，并用 Zero Convolution 保证训练初期不破坏原模型。

### 4. 最后的研究结论

> 这些论文不是一条简单替代史，而是一套模块化研究框架。表示空间、生成路径、预测网络、条件机制、求解器和编辑干预点可以相互组合。未来研究的关键不是只提出更大的模型，而是明确某种能力由系统哪一层产生、怎样用最低成本干预它，并用消融与多维评价证明质量、效率和保持性之间的真实改进。

---

## 二十五、每篇论文的“答辩卡片”

### 01 Transformer

- **问题：**RNN 难并行、远距离路径长。
- **答案：**所有位置用 Attention 直接交换信息。
- **公式：**$\operatorname{softmax}(QK^{\mathsf T}/\sqrt{d_k})V$。
- **贡献：**建立纯 Attention Encoder–Decoder 与多头机制。
- **局限：**$O(n^2)$；自回归推理仍逐 token。
- **后文作用：**所有 DiT、Cross/Joint Attention 和 KV 编辑的基础。

### 02 DDPM

- **问题：**如何稳定学习复杂图像分布。
- **答案：**固定逐步加噪，学习反向小步去噪。
- **公式：**$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$；噪声 MSE。
- **贡献：**现代高质量扩散生成底座，连接 Score Matching。
- **局限：**原始多步采样慢。
- **后文作用：**LDM、DiT、SDEdit 等的数学基础。

### 03 Rectified Flow

- **问题：**怎样学习更直接、更易少步求解的分布运输。
- **答案：**直线插值监督速度，Reflow 改善 coupling。
- **公式：**$X_t=(1-t)X_0+tX_1$，目标 $X_1-X_0$。
- **贡献：**简单 simulation-free 速度回归及直化框架。
- **局限：**监督线直不等于第一次生成轨迹直。
- **后文作用：**SD3 动力学与 FlowEdit 基础。

### 04 LDM

- **问题：**像素空间扩散成本太高。
- **答案：**Autoencoder 压缩后在 latent 中扩散。
- **公式：**$z=\mathcal E(x)$，在 $z_t$ 上预测噪声。
- **贡献：**高效生成空间与通用 Cross-Attention 条件接口。
- **局限：**受 VAE 重建上限和伪影影响。
- **后文作用：**Stable Diffusion、DiT 和多种编辑方法的平台。

### 05 DiT

- **问题：**扩散预测器是否必须是 U-Net。
- **答案：**把 noisy latent Patchify 后交给 Transformer。
- **公式：**$T=(I/p)^2$；adaLN-Zero 调制残差分支。
- **贡献：**证明 Transformer Backbone 与 GFLOPs Scaling。
- **局限：**原论文仅类别条件，Attention 成本高。
- **后文作用：**MM-DiT 和大型生成 Transformer 起点。

### 06 MM-DiT / SD3

- **问题：**如何规模化 RF 文生图并加强复杂文图融合。
- **答案：**文本、图像各自 QKV 与 FFN，但共同 Joint Attention。
- **公式：**$z_t=(1-t)x_0+t\epsilon$，速度 $\epsilon-x_0$。
- **贡献：**改进 RF + MM-DiT + 大规模系统化训练。
- **局限：**组件多、成本高，最终收益难归因单个模块。
- **后文作用：**FlowEdit、Stable Flow 的基础模型生态。

### 07 SDEdit

- **问题：**怎样把粗糙 guide 变成真实图并保留结构。
- **答案：**加到中间噪声层级，再调用预训练反向过程。
- **旋钮：**$t_0$。
- **贡献：**通用、无需重训的 realism–faithfulness 接口。
- **局限：**全局控制粗糙，强编辑易漂移。

### 08 BLD

- **问题：**怎样只改 Mask 内、保护背景。
- **答案：**每一步融合生成前景和同噪声层级的源背景。
- **公式：**$m\odot z^{fg}+(1-m)\odot z^{bg}$。
- **贡献：**训练自由的 latent 局部混合。
- **局限：**依赖 Mask，边界与 VAE 重建可能出错。

### 09 Prompt-to-Prompt

- **问题：**改词时怎样保持构图。
- **答案：**控制源/目标生成分支的 Cross-Attention Map。
- **操作：**Word Swap、Refinement、Reweighting、LocalBlend。
- **贡献：**把词—位置对齐变成编辑接口。
- **局限：**真实图常需反演，大形变不适合强布局锁定。

### 10 FlowEdit

- **问题：**预训练 Flow 模型编辑真实图能否免显式 inversion。
- **答案：**使用目标条件速度减源条件速度更新编辑状态。
- **公式：**$dZ/dt=\mathbb E[V^{tar}-V^{src}]$。
- **贡献：**无需训练和显式反演的 Flow 编辑 ODE。
- **局限：**每步多次网络调用，超参数敏感。

### 11 Stable Flow

- **问题：**哪些层真正决定视觉结构，如何最小注入。
- **答案：**逐层 bypass 找 Vital Layers，只注入源视觉 $K,V$。
- **贡献：**关键层机制分析与稀疏特征注入。
- **局限：**层号不通用，真实图通常仍需 inversion。

### 12 ControlNet

- **问题：**怎样让预训练文生图模型精确服从空间条件。
- **答案：**冻结主干、训练复制分支，用 Zero Conv 注入残差。
- **贡献：**稳定、通用的边缘/深度/姿态等条件控制。
- **局限：**必须训练新分支，每类条件需要数据与模型。

---

## 二十六、老师可能追问的问题与直接答案

### Q1：DDPM、LDM、DiT 到底是什么关系？

DDPM 是路径与训练目标，LDM 是生成所在空间，DiT 是预测网络。它们可以同时出现在一个系统里；原始 DiT 正是 latent 中的 DDPM 类扩散加 Transformer Backbone。

### Q2：Rectified Flow 是否替代了扩散模型？

它提供另一种数据—噪声路径和速度预测目标。在某些新系统中替换了 DDPM 类目标，但它不替代 VAE、Backbone、条件机制等其他模块；也不存在所有任务上绝对优于扩散的结论。

### Q3：监督路径明明是直线，为什么还需要 Reflow？

因为很多随机端点直线会相交。同一位置对应多种速度时，MSE 网络输出条件平均，实际 ODE 轨迹会弯。Reflow 用模型自身产生的 coupling 重新配对，减少方向冲突。

### Q4：LDM 为什么快，代价是什么？

空间下采样后 latent 位置数大幅减少，所以每个去噪步便宜。代价是 Autoencoder 压缩会丢失细节，生成质量上限受重建能力限制。

### Q5：DiT 和 Transformer Decoder 一样逐 patch 生成吗？

不一样。DiT 类似 Transformer Encoder；一个扩散时间步内所有 patch 同时存在并行处理，不用 Causal Mask。顺序发生在扩散/ODE 时间步之间，而不是空间 patch 之间。

### Q6：MM-DiT 与普通 Cross-Attention 有什么本质差别？

Cross-Attention 通常是图像 Query 单向读取文本 Key、Value；MM-DiT 让文本和图像各自生成 QKV，再合并做 Joint Attention，两种 token 在每层双向更新，同时保留模态专用参数。

### Q7：为什么 SD3 的结果不能全归功于 MM-DiT？

因为最终系统还改变了 RF formulation、时间采样、VAE、文字编码器、Caption、分辨率训练、QK-Norm 和 DPO。只有针对架构的受控消融能支持 MM-DiT 的净贡献。

### Q8：SDEdit 为什么不需要 inversion？

它不要求精确重建源图，而是主动把源输入加噪到中间分布，再由生成先验恢复。它牺牲部分精确保持换取真实性和可编辑性。

### Q9：BLD 为什么不能最后才把背景贴回去？

前景与背景需要在整个生成过程中共同形成一致边界、纹理和光照。逐步在相同噪声层级融合，下一步网络还能修正接缝；最终硬贴容易分布不连续。

### Q10：P2P 控制的是 Value 还是 Attention Map？

核心方法显式操纵或注入的是 Cross-Attention Map。目标分支仍由目标 Prompt 自然产生相应 Value；不能把它简化成“直接替换 Value”。

### Q11：FlowEdit 的 inversion-free 是否意味着更便宜？

它省去显式完整反演轨迹，但每个时间步至少要评估源、目标两个速度，可能还要多噪声平均。因此它减少的是某类反演流程，不保证总网络调用一定最少。

### Q12：Stable Flow 为什么保留编辑 Query，只注入源视觉 Key、Value？

编辑 Query 保留目标分支当前需求，源 Key、Value提供源图可被检索的结构与内容。这样目标语义仍可驱动查询，同时关键视觉信息得到锚定。

### Q13：ControlNet 的 Zero Conv 初始化为零，为什么能训练？

输出端 zero conv 权重的梯度不一定为零，它先从零更新；随后非零连接让梯度继续传入复制分支。零初始化的目的是初始不扰动原模型，而不是永久封闭梯度。

### Q14：这些编辑方法共同的评价维度是什么？

至少同时看目标指令执行、源图无关内容保持、视觉真实性、局部边界、身份/布局保持，以及训练、反演、网络调用和显存成本。只看一个 CLIP 或 DINO 分数不够。

### Q15：整套文献最重要的科研方法论是什么？

先把复杂系统拆成正交模块，再明确能力来自哪个模块；提出干预时要说明机制，使用消融分离因果贡献，并在质量、效率、控制、保持等多维指标上公平比较。

---

## 二十七、复习顺序与使用方法

### 第一遍：抓住大局（约 40–60 分钟）

只读：

1. 第一节“六个系统轴”；
2. 每篇的“一句话定位、问题、核心方法、贡献、局限”；
3. 第十七节递进关系；
4. 第十八、十九节总表。

目标：能不看笔记讲出两条主线和每篇所在层级。

### 第二遍：补数学与流程（约 90–120 分钟）

重点读每篇公式、训练/推理和第二十一节公式串联。每看到公式，都用四句话解释：

1. 输入是什么；
2. 输出是什么；
3. 每个符号是什么；
4. 它解决了哪一步困难。

### 第三遍：准备老师提问（约 45–60 分钟）

使用第二十五节答辩卡片和第二十六节问答。不要只背答案，要尝试从六个系统轴定位：老师问的究竟是空间、路径、目标、Backbone、条件还是 Solver。

### 汇报时的幻灯片结构

1. 研究问题与全局六轴图；
2. Transformer 信息交互基础；
3. DDPM 与 RF 两种生成动力学；
4. LDM 的表示空间；
5. DiT 与 MM-DiT 的架构演进；
6. SDEdit 到 Stable Flow 的编辑干预层级；
7. ControlNet 的训练式控制；
8. 横向对比、局限与可能研究问题；
9. 总结。


---

## 二十八、最终必须形成的心智模型

不要把十二篇论文背成十二座孤岛，而要记成三层：

### 第一层：生成器怎样成立

$$
\boxed{
\text{Attention 负责信息交换}
}
$$

$$
\boxed{
\text{DDPM / RF 负责从噪声到数据的路径}
}
$$

$$
\boxed{
\text{LDM 负责高效表示空间}
}
$$

$$
\boxed{
\text{U-Net / DiT / MM-DiT 负责预测}
}
$$

### 第二层：条件怎样进入生成器

$$
\boxed{
\text{类别向量、Cross-Attention、Joint Attention、空间残差}
}
$$

分别对应简单类别、文本读取、文图双向融合和精确结构控制。

### 第三层：编辑怎样干预生成器

$$
\boxed{
\text{状态}
\rightarrow
\text{局部 latent}
\rightarrow
\text{Attention Map}
\rightarrow
\text{速度场}
\rightarrow
\text{关键层 }K,V
}
$$

对应：

$$
\boxed{
\text{SDEdit}
\rightarrow
\text{BLD}
\rightarrow
\text{P2P}
\rightarrow
\text{FlowEdit}
\rightarrow
\text{Stable Flow}
}
$$

ControlNet 则从训练路线添加可复用空间控制。

最后用一句话收束整套学习：

> 现代生成模型的发展，不只是把网络越做越大，而是在表示空间、生成动力学、预测架构、条件交互和编辑干预之间不断重新分工；真正有价值的研究，是找到能力产生的位置，并以更清楚、更高效、更可控的方式利用它。
