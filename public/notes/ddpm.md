# 06 Denoising Diffusion Probabilistic Models（DDPM）

## 来源

- 论文全名：**Denoising Diffusion Probabilistic Models**
- 作者：Jonathan Ho、Ajay Jain、Pieter Abbeel
- 会议：NeurIPS 2020
- 官方论文：<https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf>
- 官方代码：<https://github.com/hojonathanho/diffusion>

## 本节主线

DDPM 学习的不是“凭空画图”，而是**怎样逐步去掉高斯噪声**：

$$
\boxed{
x_0\text{（真实图像）}
\xrightarrow[\text{固定，不训练}]{\text{逐步加噪}}
x_T\approx\mathcal N(0,I)
\xrightarrow[\text{神经网络学习}]{\text{逐步去噪}}
\widehat x_0\text{（生成图像）}}
$$

- **前向过程** $q$：人为规定怎样加噪，最终把数据变成近似标准高斯噪声。
- **反向过程** $p_\theta$：神经网络学习怎样从 $x_t$ 得到更干净的 $x_{t-1}$。
- **训练技巧**：随机选择一个时间步 $t$，直接构造 $x_t$，让网络预测当时加入的噪声 $\epsilon$。
- **生成过程**：从 $x_T\sim\mathcal N(0,I)$ 开始，按 $T,T-1,\ldots,1$ 反复去噪。

## 符号与约定

| 符号 | 含义 |
|---|---|
| $x_0$ | 干净的真实数据；在图像任务中就是原图 |
| $x_t$ | 第 $t$ 步的带噪数据 |
| $T$ | 总扩散步数；原论文实验取 $T=1000$ |
| $\beta_t$ | 第 $t$ 步加入噪声的强度，通常很小 |
| $\alpha_t=1-\beta_t$ | 第 $t$ 步保留原信号的比例 |
| $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$ | 从第 $1$ 步到第 $t$ 步累计保留的信号比例 |
| $\epsilon\sim\mathcal N(0,I)$ | 标准高斯噪声 |
| $q$ | 固定的前向加噪过程，没有可学习参数 |
| $p_\theta$ | 参数为 $\theta$ 的反向去噪过程 |
| $\epsilon_\theta(x_t,t)$ | 神经网络根据 $x_t,t$ 预测出的噪声 |

> [!important] 记号提醒
> 本文中 $x_0$ 是**数据**，$x_T$ 是**噪声**。下一篇 Rectified Flow 的常用记号恰好是 $X_0$ 表示噪声、$X_1$ 表示数据，不要混淆。

### 阅读公式前必须认识的概率记号

| 写法 | 直观含义 |
|---|---|
| Eq. (4) | Equation (4)，即“公式（4）”，不是一个新变量 |
| $x\sim\mathcal N(\mu,\Sigma)$ | 随机变量 $x$ 按均值为 $\mu$、协方差为 $\Sigma$ 的高斯分布取值 |
| $\mathcal N(x;\mu,\Sigma)$ | 把同一高斯分布写成关于 $x$ 的概率密度 |
| $I$ | 单位矩阵；$\beta I$ 表示各维方差均为 $\beta$ 且彼此独立 |
| $p(a\mid b)$ | 已知 $b$ 时，$a$ 的条件分布；竖线读作“在……条件下” |
| $\mathbb E[\cdot]$ | 对许多随机样本取平均，即期望 |
| $\prod_t$ | 连乘；$\sum_t$ 是连加 |
| $\|a-b\|^2$ | 各维差值平方后相加，即均方误差的核心 |
| $D_{\mathrm{KL}}(q\|p)$ | 衡量两个概率分布差异的 KL 散度；非负，相同为零 |

高斯记号最重要的不是背密度函数，而是会把“分布写法”变成“采样写法”：

$$
x\sim\mathcal N(\mu,\beta I)
\quad\Longleftrightarrow\quad
x=\mu+\sqrt\beta\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

$\mu$ 决定随机点的中心，$\beta$ 是方差，$\sqrt\beta$ 才是噪声的标准差。

## 一、前向扩散：怎样把图像变成噪声

### 1. 一步加噪

论文式（2）规定：

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I
\right).
$$

等价的采样写法是：

$$
x_t=\sqrt{\alpha_t}\,x_{t-1}
+\sqrt{1-\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

含义是：

- $\sqrt{\alpha_t}\,x_{t-1}$：保留上一时刻的大部分信号；
- $\sqrt{1-\alpha_t}\,\epsilon$：加入一小部分新噪声；
- 连续做很多次后，原图信息逐渐消失，$x_T$ 接近 $\mathcal N(0,I)$。

### 2. 为什么可以直接从 $x_0$ 得到任意 $x_t$

定义

$$
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

论文式（4）给出闭式结果：

$$
q(x_t\mid x_0)
=\mathcal N\!\left(
x_t;\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I
\right),
$$

因此可直接采样：

$$
\boxed{
x_t=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon.}
$$

推导主线如下。连续代入

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

后，$x_0$ 前面的系数变为

$$
\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}
=\sqrt{\bar\alpha_t}.
$$

各步独立高斯噪声的线性组合仍是高斯噪声，其总方差递推为

$$
\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)
=1-\bar\alpha_t.
$$

所以许多次加噪可以合并为“一次缩放原图 + 一次加高斯噪声”。这使训练时不必真的从 $1$ 循环到 $t$。

## 二、反向过程：怎样从噪声逐步恢复数据

论文把生成模型写成：

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),
\qquad
p(x_T)=\mathcal N(0,I).
$$

每个反向步骤建模为高斯分布：

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N\!\left(
x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)
\right).
$$

直观上，若每一步 $\beta_t$ 足够小，相邻的 $x_t$ 与 $x_{t-1}$ 只差一点，反向条件分布可以用高斯分布近似。原论文主要让网络学习均值，方差采用预先设定的形式。

### 已知 $x_0$ 时的真实后验

训练时拥有原图 $x_0$，因此可以得到：

$$
q(x_{t-1}\mid x_t,x_0)
=\mathcal N(x_{t-1};\widetilde\mu_t,\widetilde\beta_t I),
$$

其中

$$
\widetilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t,
$$

$$
\widetilde\beta_t
=\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t}\beta_t.
$$

推导只需记住主线：

$$
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
$$

右边是两个关于 $x_{t-1}$ 的高斯密度。把指数中的二次项展开、合并并配方，就能读出上面的均值和方差。

> [!note] 为什么推理时不能直接用这个后验
> 它含有真实 $x_0$。生成新图时只有随机噪声 $x_T$，并不知道目标原图，所以必须让网络从 $x_t$ 中估计反向分布。

## 三、训练目标：从似然到“预测噪声”

### 1. 为什么出现 ELBO

我们希望最大化真实数据的似然 $p_\theta(x_0)$，等价于最小化 $-\log p_\theta(x_0)$。直接对所有隐藏变量积分很困难，于是插入已知的前向分布 $q$：

$$
\begin{aligned}
-\log p_\theta(x_0)
&=-\log
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]\\
&\le
\mathbb E_q
\left[
-\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
=L.
\end{aligned}
$$

不等号来自 Jensen 不等式，因为 $-\log$ 是凸函数。$L$ 是负对数似然的可计算上界；最小化它，就在推动模型提高数据似然。

论文式（5）把它拆成：

$$
\boxed{
L=
\underbrace{D_{\mathrm{KL}}(q(x_T\mid x_0)\Vert p(x_T))}_{L_T}
+
\sum_{t>1}
\underbrace{D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\Vert
p_\theta(x_{t-1}\mid x_t)
\right)}_{L_{t-1}}
+
\underbrace{-\log p_\theta(x_0\mid x_1)}_{L_0}.}
$$

- $L_T$：让最终噪声分布接近 $\mathcal N(0,I)$；前向过程固定后，它对 $\theta$ 是常数。
- $L_{t-1}$：让模型的反向一步接近真实后验，是主要训练部分。
- $L_0$：最后一步重建数据的代价。

### 2. 为什么网络预测 $\epsilon$

由

$$
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon
$$

可解出

$$
x_0=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}
{\sqrt{\bar\alpha_t}}.
$$

把它代入真实后验均值，可化为：

$$
\widetilde\mu_t
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon
\right).
$$

因此只要网络能从 $(x_t,t)$ 预测出 $\epsilon$，就能构造反向均值：

$$
\boxed{
\mu_\theta(x_t,t)
=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).}
$$

高斯分布之间的 KL 散度在方差固定时，会化成均值之间的加权平方误差。继续代入上式，得到论文式（12）的形式：

$$
L_{t-1}-C
=
\mathbb E_{x_0,\epsilon}
\left[
\frac{\beta_t^2}
{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right].
$$

论文进一步去掉随 $t$ 变化的权重，使用简单目标（式（14））：

$$
\boxed{
L_{\mathrm{simple}}(\theta)
=
\mathbb E_{t,x_0,\epsilon}
\left[
\left\|
\epsilon-
\epsilon_\theta\!\left(
\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,t
\right)
\right\|^2
\right].}
$$

训练时加入的 $\epsilon$ 是我们自己采样的，所以“正确答案”天然已知。Loss 越小，说明网络越能识别并去掉不同噪声强度下的噪声。

### 3. 与 score matching 的联系

由 $q(x_t\mid x_0)$ 的高斯形式可得：

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{x_t-\sqrt{\bar\alpha_t}x_0}
{1-\bar\alpha_t}
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

score 是“对数概率密度上升最快的方向”：

$$
s(x)=\nabla_x\log p(x).
$$

它指向更可能出现的数据区域。上式先给出的是“已知干净图 $x_0$ 时”的条件 score。训练中，同一个 $x_t$ 可能由不同 $x_0$ 和噪声产生；在均方误差下，网络的最优输出是

$$
\epsilon_\theta^*(x_t,t)=\mathbb E[\epsilon\mid x_t,t].
$$

因此把网络输出乘上已知系数

$$
s_\theta(x_t,t)
=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}},
$$

就得到对带噪数据边缘分布 $q_t(x_t)$ 的 score 的估计。也就是说，**预测噪声与学习 score 只差一个已知缩放系数**。

Langevin dynamics（朗之万动力学）的直觉写法是：

$$
x\leftarrow x+\eta\,s_\theta(x)+\sqrt{2\eta}\,z,
\qquad z\sim\mathcal N(0,I).
$$

第一项沿 score 往高概率区域移动，第二项保留随机探索。DDPM 的反向采样与这种“朝高概率方向走，同时加入适量随机性”的思想密切相关，但二者的离散更新公式并不完全相同。

## 四、完整训练流程

一次训练迭代：

1. 从数据集采样真实图像 $x_0$。
2. 均匀采样 $t\in\{1,\ldots,T\}$。
3. 采样 $\epsilon\sim\mathcal N(0,I)$。
4. 直接构造

   $$
   x_t=\sqrt{\bar\alpha_t}x_0+
   \sqrt{1-\bar\alpha_t}\epsilon.
   $$

5. 网络计算 $\widehat\epsilon=\epsilon_\theta(x_t,t)$。
6. 计算 $\|\epsilon-\widehat\epsilon\|^2$，反向传播并更新 $\theta$。

随机抽一个 $t$ 是对“所有时间步期望”的 Monte Carlo 估计。因此训练一次只需调用一次网络，不需要跑完整的 $T$ 步链。

### U-Net 在这里究竟做什么

原论文用 U-Net 实现 $\epsilon_\theta$。它不是另一种扩散过程，而是**每一步负责预测噪声的神经网络**：

$$
(x_t,t)\xrightarrow{\text{U-Net}}\widehat\epsilon,
\qquad \widehat\epsilon\text{ 与 }x_t\text{ 形状相同}.
$$

流程为：

1. **输入**：带噪图像 $x_t$，以及表示当前噪声强度的时间步 $t$。
2. **下采样路径**：逐渐缩小空间尺寸、增加通道，提取从局部纹理到整体结构的特征。
3. **中间层**：在最低分辨率处加工全局语义特征。
4. **上采样路径**：逐渐恢复原来的空间尺寸。
5. **跳跃连接**：把下采样阶段同尺寸的细节直接送给上采样阶段，避免位置和边缘信息在压缩中丢失。
6. **输出**：每个像素位置、每个通道上预测的噪声 $\widehat\epsilon$。

$t$ 会先变成 time embedding，并注入多个网络层，使同一个 U-Net 知道自己面对的是轻噪声还是重噪声。U-Net、时间嵌入和其中的 Attention 参数都由 Loss 反向传播学习；$\beta_t,\alpha_t,\bar\alpha_t$ 是预设的噪声日程，不由这份基础模型学习。

## 五、完整采样流程

先采样：

$$
x_T\sim\mathcal N(0,I).
$$

然后对 $t=T,T-1,\ldots,1$：

$$
\boxed{
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)
+\sigma_t z,}
$$

其中

$$
z\sim\mathcal N(0,I)\quad(t>1),
\qquad z=0\quad(t=1).
$$

- 括号内部分：根据预测噪声，把 $x_t$ 往更干净的方向移动。
- $\sigma_tz$：按照反向高斯分布保留合理的随机性，所以同一模型能生成多样样本。
- 最后一步不再加噪声，得到 $x_0$。

| | 训练 | 生成 / 推理 |
|---|---|---|
| 起点 | 真实数据 $x_0$ | 随机噪声 $x_T$ |
| 是否知道真实噪声 $\epsilon$ | 知道，是自己加入的 | 不知道，由网络预测 |
| 时间步 | 随机抽一个 $t$ | 从 $T$ 依次走到 $1$ |
| 是否计算 Loss、更新参数 | 是 | 否 |
| 网络调用次数 | 每个样本通常一次 | 原论文约 $T=1000$ 次 |

## 六、论文结论与实验

- 原论文在 CIFAR-10 无条件生成上报告 **Inception Score 9.46、FID 3.17**。
- 反向均值采用 $\epsilon$ 参数化并使用 $L_{\mathrm{simple}}$，样本质量明显优于论文中的直接均值预测基线。
- 原论文的噪声日程为 $T=1000$，$\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$。
- 生成过程中通常先出现整体轮廓和大尺度结构，随后补充纹理与细节，体现出逐级恢复信息的特点。
- 模型生成质量很强，但当时的对数似然并非最佳；同时逐步采样需要大量网络调用，速度是明显局限。

论文的核心贡献不是只提出“加噪再去噪”，而是把以下内容连成了可训练、有效的系统：

1. 固定高斯前向扩散与可学习高斯反向过程；
2. 可分解的变分训练目标；
3. $\epsilon$ 参数化与简单噪声预测 Loss；
4. 与 denoising score matching、Langevin dynamics 的联系；
5. 高质量图像生成以及逐级有损压缩的解释。

## 七、汇报时最应讲清楚的五个问题

1. **$q$ 与 $p_\theta$ 有什么区别？**  
   $q$ 是人为规定且固定的加噪过程；$p_\theta$ 是神经网络学习的去噪过程。

2. **为什么式（4）重要？**  
   它让我们能从 $x_0$ 一步得到任意 $x_t$，训练不必模拟前面所有加噪步骤。

3. **为什么预测噪声，而不是直接预测图像？**  
   真实后验均值可改写成含 $\epsilon$ 的形式；预测 $\epsilon$ 就足以构造反向均值，并把 KL 目标化成简单的均方误差。

4. **训练时为什么只随机抽一个 $t$？**  
   完整 Loss 是对时间步的期望，随机抽样给出无偏的随机梯度估计，计算量更小。

5. **为什么训练快于采样？**  
   一次训练样本只需处理一个随机 $t$；采样却必须从 $T$ 到 $1$ 连续去噪，后一步依赖前一步。

## 容易混淆的点

- $x_t$ 不是网络参数，而是某一噪声强度下的数据。
- $\epsilon$ 是训练时真实加入的噪声；$\epsilon_\theta$ 是网络的预测。
- $\beta_t$ 越大表示本步加入噪声越多；$\bar\alpha_t$ 越小表示累计保留的原信号越少。
- 网络训练的是反向过程，不是前向过程。
- $L_{\mathrm{simple}}$ 是从变分目标推导后简化、重新加权得到的实用 Loss，不等于未经修改的完整 ELBO。
- 训练时可以直接构造任意 $x_t$，不代表生成时也能直接从 $x_T$ 跳到高质量 $x_0$。

## 公式速记

$$
\boxed{
\begin{aligned}
\alpha_t&=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,\\
x_t&=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,\\
L_{\mathrm{simple}}
&=\mathbb E\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2,\\
x_{t-1}
&=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)+\sigma_tz.
\end{aligned}}
$$

## 我的理解与修改

- 

## 掌握检查

- [ ] 能画出“数据 $\rightarrow$ 噪声 $\rightarrow$ 数据”的完整流程。
- [ ] 能解释 $\beta_t,\alpha_t,\bar\alpha_t$ 各自表示什么。
- [ ] 能从逐步加噪说明为什么得到 $q(x_t\mid x_0)$ 的闭式形式。
- [ ] 能解释真实后验为什么依赖 $x_0$，以及推理时为什么需要神经网络。
- [ ] 能复述 ELBO 拆成 $L_T,L_{t-1},L_0$ 后各项的意义。
- [ ] 能说明 $\epsilon$ 参数化怎样把训练变成噪声预测 MSE。
- [ ] 能区分训练算法与采样算法。
- [ ] 能不看笔记写出“加噪、Loss、采样”三条核心公式。

## 相关笔记

- [05-Attention Is All You Need](#/notes/attention-is-all-you-need)
- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)
