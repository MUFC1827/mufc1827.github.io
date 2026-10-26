# 08 High-Resolution Image Synthesis with Latent Diffusion Models（LDM）

## 来源

- 论文全名：**High-Resolution Image Synthesis with Latent Diffusion Models**
- 作者：Robin Rombach、Andreas Blattmann、Dominik Lorenz、Patrick Esser、Björn Ommer
- 会议：CVPR 2022
- CVPR 正式论文：<https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf>
- 官方代码：<https://github.com/CompVis/latent-diffusion>
- Stable Diffusion 官方代码：<https://github.com/CompVis/stable-diffusion>

## 本节主线

像素空间扩散质量高，但每一步都要处理高分辨率图像，训练和采样都很昂贵。LDM 先用预训练 Autoencoder 把图像压缩成仍保留空间结构的 latent feature map，再在 latent space 中运行扩散模型，最后只解码一次：

$$
\boxed{
x\xrightarrow{\mathcal E}z_0
\xrightarrow{\text{加噪}}z_t
\xrightarrow{\text{U-Net 学习去噪}}z_0
\xrightarrow{\mathcal D}\widehat x.}
$$

一句话定位：

> DDPM 主要回答“怎样加噪和去噪”，Rectified Flow 主要回答“怎样学习噪声到数据的运输路径”，LDM 主要回答“这个生成过程在哪个表示空间里做更高效”。这些方法可以组合，而不是互相替代。

## 术语速查

| 缩写 / 术语 | 全称与含义 |
|---|---|
| DM | Diffusion Model，扩散模型 |
| LDM | Latent Diffusion Model，潜空间扩散模型 |
| latent / $z$ | 编码器得到的潜在表示；这里通常是 $h\times w\times c$ 的二维空间特征图，不是一个类别标签 |
| Autoencoder | 自动编码器，由 Encoder $\mathcal E$ 和 Decoder $\mathcal D$ 组成 |
| VAE | Variational Autoencoder，变分自动编码器；使用概率分布和 KL 正则约束 latent |
| KL | Kullback-Leibler divergence，衡量两个概率分布差异 |
| VQ | Vector Quantization，向量量化，把连续特征映射到离散码本向量 |
| U-Net | 带下采样、上采样和 skip connection 的网络；在 LDM 中负责预测 latent 中的噪声 |
| CLIP | Contrastive Language-Image Pre-training，把文字和图像学习到可对应的语义表示 |
| token | 文本切分后的基本单元；不一定等于一个完整单词 |
| conditioning | 条件控制，例如文字、类别、语义图或低分辨率图像 |
| cross-attention | Query 来自图像特征，Key/Value 来自条件特征的注意力 |
| FID | Fréchet Inception Distance，衡量生成图像分布与真实图像分布差异，通常越低越好 |
| PSNR / SSIM | 偏向逐像素重建一致性的指标，通常越高越好，但不一定等同于人眼感觉更真实 |
| LPIPS | 基于深层视觉特征的感知距离，通常越低表示两图越相似 |
| throughput | 吞吐率，单位时间能训练或生成多少样本 |

## 一、为什么不直接在像素空间扩散

高分辨率 RGB 图像 $x\in\mathbb R^{H\times W\times 3}$ 包含大量数值，其中许多是人眼几乎察觉不到的高频细节。像素扩散即使通过 Loss 降低这些细节的重要性，训练时仍必须：

- 在所有像素上运行 U-Net；
- 在所有像素上保存中间激活并计算梯度；
- 采样时重复运行网络许多步。

论文把图像建模粗略分成两个任务：

1. **Perceptual compression（感知压缩）**：去掉人眼不敏感的像素冗余，同时保留视觉结构。
2. **Semantic compression（语义建模）**：学习物体、布局、关系和整体图像分布。

LDM 显式拆开这两个任务：Autoencoder 负责第一项，Diffusion Model 负责第二项。这样扩散模型不必浪费主要算力反复处理几乎无关的像素细节。

## 二、第一阶段：Perceptual Image Compression

### 1. Encoder 与 Decoder

给定图像：

$$
x\in\mathbb R^{H\times W\times 3},
$$

编码器和解码器执行：

$$
\boxed{
z=\mathcal E(x),\qquad
\widetilde x=\mathcal D(z)=\mathcal D(\mathcal E(x)).}
$$

其中：

- $z\in\mathbb R^{h\times w\times c}$ 是较小的空间特征图；
- $\widetilde x$ 是重建图像；
- latent 仍保留“哪里有什么”的二维布局，所以适合继续用卷积 U-Net 处理。

### 2. 下采样因子 $f$

论文定义：

$$
\boxed{f=\frac Hh=\frac Ww.}
$$

例如 $f=8$ 表示高和宽分别缩小 8 倍：

$$
512\times512\longrightarrow64\times64.
$$

若通道数先不考虑，仅空间位置数已经缩小 $8^2=64$ 倍，而不是 8 倍。

- $f$ 太小：压缩不足，扩散模型仍然昂贵；
- $f$ 太大：第一阶段永久丢失太多细节，生成质量上限下降；
- 主文认为 $f=4\sim16$ 有较好的速度—质量平衡；图 7 重点展示了 $f=4,8$ 的强结果，但不能据此说只有这两个取值最优。

### 3. 为什么不能只用像素 MSE

若只最小化 $L_1/L_2$ 像素误差，模型倾向于把多种可能细节平均，重建结果容易模糊。论文的第一阶段结合：

- **perceptual loss**：比较深层视觉特征，使重建在视觉语义上接近原图；
- **patch-based adversarial objective**：判别局部图像块是否真实，改善纹理与锐度；
- **轻量 latent regularization**：避免 latent 的数值尺度和分布失控。

论文比较两种正则：

- **KL-reg**：用较轻的 KL penalty 把 latent 约束得接近标准正态，类似 VAE；
- **VQ-reg**：使用向量量化，可理解为量化层被吸收到 Decoder 内的 VQGAN 形式。

两者不能都笼统叫作 VQ-VAE。Stable Diffusion v1.x 的 `AutoencoderKL` 属于 **KL-regularized autoencoder**。

### 4. 为什么分两阶段训练

Autoencoder 先单独训练好，之后可以冻结并复用于多个扩散任务。这样不需要同时权衡：

- 图像是否重建得好；
- latent 的生成分布是否容易学习。

第二阶段只需学习固定 latent space 中的数据分布，系统更稳定，也便于复用。

## 三、第二阶段：Latent Diffusion Model

### 1. 从 DDPM 公式得到 LDM

像素空间 DM 的论文式（1）为：

$$
L_{\mathrm{DM}}
=\mathbb E_{x,\epsilon,t}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2
\right].
$$

先把真实图像编码为：

$$
z_0=\mathcal E(x),
$$

然后沿用 DDPM 的闭式加噪：

$$
\boxed{
z_t=\sqrt{\bar\alpha_t}z_0+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).}
$$

把 $x_t$ 换成 $z_t$，得到论文式（2）：

$$
\boxed{
L_{\mathrm{LDM}}
=\mathbb E_{\mathcal E(x),\epsilon,t}
\left[
\left\|
\epsilon-\epsilon_\theta(z_t,t)
\right\|_2^2
\right].}
$$

因此 LDM 的主要数学变化不是重新发明扩散 Loss，而是把扩散过程搬到更合适的表示空间。

### 2. U-Net 在做什么

$\epsilon_\theta$ 由 time-conditional U-Net 实现：

$$
(z_t,t)\xrightarrow{\text{U-Net}}\widehat\epsilon.
$$

- 输入 $z_t$：当前带噪 latent；
- 输入 $t$：当前噪声强度，经 time embedding 注入网络；
- 下采样：提取更大感受野和高层语义；
- skip connection：保存位置和局部细节；
- 上采样：恢复空间分辨率；
- 输出：与 $z_t$ 同形状的预测噪声。

Autoencoder 的 downsampling 与 U-Net 内部的 downsampling 不是一回事：前者把像素变成 latent，只做一次；后者是去噪网络内部的多尺度特征处理，每个扩散时间步都会运行。

## 四、条件生成：怎样让模型听懂文字

### 1. 条件分布

无条件模型学习 $p(z)$；条件模型希望学习：

$$
p(z\mid y),
$$

其中 $y$ 可为文字、类别、语义分割图、边界框、低分辨率图像等。

条件编码器把不同模态变成特征：

$$
\tau_\theta(y)\in\mathbb R^{M\times d_\tau}.
$$

$M$ 是条件 token / 元素数量，$d_\tau$ 是每个条件特征的维度。

### 2. Cross-Attention

在 U-Net 的第 $i$ 个中间层，把空间特征展平成：

$$
\phi_i(z_t)\in\mathbb R^{N\times d_i},
$$

其中 $N$ 是该层空间位置数。论文写作：

$$
Q=W_Q^{(i)}\phi_i(z_t),\qquad
K=W_K^{(i)}\tau_\theta(y),\qquad
V=W_V^{(i)}\tau_\theta(y),
$$

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(\frac{QK^T}{\sqrt d}\right)V.}
$$

语义必须记清：

- $Q$ 来自 U-Net 的图像 latent 特征：当前位置“想查什么”；
- $K$ 来自条件：每个 token 的“检索标签”；
- $V$ 来自条件：真正被汇总的条件内容；
- $QK^T$ 比较每个图像位置与每个条件 token 的相关性；
- softmax 在条件 token 上分配权重；
- 输出再融合回 U-Net 特征，而不是直接修改像素。

$W_Q,W_K,W_V$ 都是通过 Loss 和反向传播学出的投影矩阵。论文排版采用左乘记法，与常见代码的右乘/转置约定可能不同；本阶段以来源和功能为重点，不强行用一种矩阵朝向套所有实现。

### 3. 条件 Loss

论文式（3）为：

$$
\boxed{
L_{\mathrm{LDM}}
=\mathbb E_{\mathcal E(x),y,\epsilon,t}
\left[
\left\|
\epsilon-
\epsilon_\theta\!\left(z_t,t,\tau_\theta(y)\right)
\right\|_2^2
\right].}
$$

与式（2）相比，只是让预测噪声的 U-Net 多获得条件编码 $\tau_\theta(y)$。原论文式（3）中 $\tau_\theta$ 与 $\epsilon_\theta$ 联合优化；具体实现也可以冻结预训练条件编码器。

### 4. Concatenation 与 Cross-Attention

- 空间对齐的条件，如低分辨率图、mask、语义图，可以沿通道维与 $z_t$ **concatenate（拼接）**；
- token 型或长度可变的条件，如文本、布局 token，更适合用 **cross-attention**；
- 二者也可以组合。

## 五、从原论文 LDM 到 Stable Diffusion

老师要求掌握的 Stable Diffusion 三块是：

| 模块 | Stable Diffusion v1.x 中的典型实现 | 负责的问题 |
|---|---|---|
| Autoencoder | `AutoencoderKL`，典型空间因子 $f=8$ | 把图像压缩到 latent，并在最后恢复图像 |
| Text Encoder | 冻结的 CLIP ViT-L/14 text encoder | 把 prompt token 编成语义序列 |
| Denoiser | 带 Cross-Attention 的 conditional U-Net | 根据 $z_t,t$ 和文字条件预测噪声 |

Stable Diffusion 常用 **Classifier-Free Guidance（CFG，无分类器引导）**增强 prompt 控制。U-Net 对同一个 $z_t$ 分别作无/负条件预测 $\epsilon_{\mathrm{uncond}}$ 与正条件预测 $\epsilon_{\mathrm{cond}}$，再组合：

$$
\boxed{
\widehat\epsilon
=\epsilon_{\mathrm{uncond}}
+s\bigl(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}\bigr).}
$$

$s$ 是 guidance scale：$s=1$ 等于直接使用条件预测；适度增大通常更贴合文字，过大则可能过饱和、失真或牺牲多样性。CFG 改变的是推理时采用的噪声预测方向，不是重新训练模型。

需要区分：

- **LDM 原论文文本实验**：BERT tokenizer + 论文自己实现的 Transformer $\tau_\theta$；
- **Stable Diffusion v1.x**：在 LDM 框架上使用冻结的 CLIP ViT-L/14 文本编码器；
- 所以“LDM 必须使用 CLIP”不正确，CLIP 是 Stable Diffusion 的具体选择。

Stable Diffusion v1.x 中，典型的 $512\times512\times3$ 图像被编码为约 $64\times64\times4$ 的 latent。标量数量比例约为：

$$
\frac{64\times64\times4}{512\times512\times3}
=\frac1{48}.
$$

这只能说明表示规模大幅降低，不表示实际运行一定精确快 48 倍，因为计算量还受通道数、网络层和注意力等因素影响。

## 六、完整训练与推理流程

### 训练第一阶段 Autoencoder

1. 输入真实图像 $x$；
2. $z=\mathcal E(x)$；
3. $\widetilde x=\mathcal D(z)$；
4. 计算重建、感知、对抗与 latent 正则相关 Loss；
5. 反向传播，更新 $\mathcal E,\mathcal D$；
6. 训练完成后固定这一表示空间。

### 训练第二阶段条件 LDM

1. 取训练对 $(x,y)$；
2. $z_0=\mathcal E(x)$；
3. 随机采样 $t$ 与 $\epsilon$；
4. 构造 $z_t$；
5. 条件编码 $c=\tau_\theta(y)$；
6. U-Net 输出 $\widehat\epsilon=\epsilon_\theta(z_t,t,c)$；
7. 计算 $\|\epsilon-\widehat\epsilon\|^2$；
8. 反向传播，更新 U-Net；原论文条件实验还更新 $\tau_\theta$，Stable Diffusion v1.x 的 CLIP 则冻结。

### 文生图推理

1. prompt 经 tokenizer 切成 token；
2. Text Encoder 得到条件特征 $c$；
3. 采样 $z_T\sim\mathcal N(0,I)$；
4. 采样器让 U-Net 从 $T$ 到 $0$ 反复预测噪声并更新 latent；
5. 得到 $z_0$ 后，只运行一次 $\mathcal D$；
6. 输出图像 $\widehat x$。

纯文本生成时没有输入图像，所以通常不用 Encoder $\mathcal E$；Decoder $\mathcal D$ 只在最后使用。推理没有真实噪声标签，不计算训练 Loss、不反向传播、不更新参数。

## 七、论文实验与结论

### 4.1 Perceptual Compression Tradeoffs

比较 $f\in\{1,2,4,8,16,32\}$：

- LDM-1 就是像素空间 baseline；
- $f=1,2$ 压缩太少，训练进展慢、吞吐率低；
- $f=32$ 压缩太强，第一阶段信息损失限制最终质量；
- $f=4\sim16$ 较平衡；图 7 重点比较并展示了 $f=4,8$ 的有竞争力结果。

这证明存在“感知压缩交给 Autoencoder、语义建模交给 DM”的合理分工点。

### 4.2 Image Generation with Latent Diffusion

论文在 CelebA-HQ、FFHQ、LSUN Churches、LSUN Bedrooms 等数据集测试无条件生成，并用 FID、Precision、Recall 衡量质量与分布覆盖。

核心结论：在固定的、重建质量良好的 latent space 中训练扩散模型，可以用更低资源获得有竞争力的生成质量；相比同时学习 Autoencoder 与 prior，也避免了两个目标之间的困难权衡。

### 4.3 Conditional Latent Diffusion

- 文本生成：在 LAION-400M 上训练 1.45B 参数模型，使用 BERT tokenizer 与 Transformer 条件编码器；
- cross-attention 使模型能处理用户自定义复杂 prompt；
- classifier-free guidance（CFG）明显提升文字匹配与样本质量；
- 同一接口还支持类别、布局和语义图条件。

对于空间条件，卷积结构允许在训练分辨率之外应用模型。论文展示了在 $256^2$ 训练后生成更大尺寸图像的能力，但分辨率变化会改变特征统计，并非任何尺寸都无条件稳定。

### 4.4 Super-Resolution

超分辨率把低分辨率图像作为条件，与 U-Net 输入在通道维拼接。论文发现：

- LDM-SR 能生成较真实的纹理；
- 纯回归模型的 PSNR/SSIM 可能更高，却容易偏模糊；
- FID、感知质量与 PSNR/SSIM 衡量的目标不同，评价生成结果不能只看一个指标；
- 仅用 bicubic degradation 训练的模型对真实未知退化泛化不足，因此还训练了更通用退化模型。

### 4.5 Inpainting

Inpainting 是根据原图、mask 和可见区域补全被遮挡内容。论文比较像素 DM 与 LDM：

- latent diffusion 的训练和采样吞吐率更高；
- 主文报告像素与 latent 方案之间至少约 $2.7\times$ 加速，同时 FID 至少改善约 $1.6\times$；
- LDM 可产生多样补全，因此与只给出单一平均结果的方法相比，未遮挡区 LPIPS 不一定占优；
- 用户研究更偏好 LDM 的结果。

## 八、主要贡献与局限

### 主要贡献

1. 把感知压缩与生成学习分开，在较温和压缩的二维 latent 中训练扩散模型。
2. 显著降低高分辨率扩散的训练与采样成本，同时保留生成质量。
3. Autoencoder 可复用于不同扩散模型和下游任务。
4. 在 U-Net 中引入通用 Cross-Attention 条件接口。
5. 用统一框架覆盖无条件生成、类别生成、文生图、布局生成、超分辨率和修复。

### 局限

- 两阶段系统比单模型更复杂；
- 生成质量上限受 Autoencoder 重建能力限制；
- 过强压缩丢失的细节无法由第二阶段准确恢复；
- 扩散采样仍需多次 U-Net 前向计算；
- Autoencoder 可能带来纹理或重建伪影；
- 训练数据偏差和文字—图像数据质量会反映在生成结果中。

## 九、容易混淆的点

- latent 是空间特征图，不是一个简单语义向量。
- LDM 不是一种全新的去噪 Loss；核心是把 DDPM 搬到 latent space。
- $f=8$ 表示高、宽各缩 8 倍，而不是总数据量缩 8 倍。
- Autoencoder downsampling 与 U-Net 内部 downsampling 不是同一个过程。
- 第二阶段不会每一步都把 $z_t$ 解码成图像；正常生成只在最后解码一次。
- 原论文的条件编码器不是 CLIP；Stable Diffusion v1.x 才使用冻结 CLIP。
- `AutoencoderKL` 是 KL-reg autoencoder，不是 VQ-VAE。
- Cross-Attention 的 Q 来自图像，K/V 来自条件；Self-Attention 的 Q/K/V 才来自同一组图像特征。
- Eq. (1) 是像素 DM Loss，Eq. (2) 是 latent DM Loss，Eq. (3) 是条件 LDM Loss；论文的 Cross-Attention 公式没有编号。
- 论文总体支持 $f=4\sim16$ 的折中；图 7 突出 $f=4,8$，但不应说成数学定理规定只能取 4 或 8。

## 十、实操验证与结论

### 1. 实验环境与基准

- 本地设备：Intel Arc 130T GPU（16 GB），使用 PyTorch XPU；
- 主要版本：Python 3.10.19、PyTorch 2.11.0+xpu、Diffusers 0.39.0；
- 精度：FP16；模型：Stable Diffusion 1.5；输出：$512\times512$；
- 基准设置：DDIM、20 steps、CFG $=7.5$、seed $=1$；
- 模型首次加载后生成基准图耗时 20.54 s；缓存和预热后，相同 20 步实验约 7.68 s。

这说明 Intel XPU 可以完成本节全部推理实验，本次不需要租用 4090。首次耗时包含初始化等额外开销，不能直接与预热后的生成耗时比较。

### 2. 参数对照得到的结论

| 对照 | 实际观察 | 正确结论 |
|---|---|---|
| 10 / 20 / 40 steps | 耗时分别为 3.82 / 7.68 / 14.37 s；三者均能成图，20 与 40 的质量差距不明显，眼睛伪影仍存在 | 主要运行时间近似随步数增长；更多步不保证无限改善质量 |
| CFG 1.0 / 7.5 / 15.0 | 1.0 条件约束较弱；7.5 主体和写实风格更明确；15.0 对比度、饱和度和锐化感过强 | CFG 是软引导；过大可能放大伪影并降低自然度，单张图不能证明普遍最优值 |
| Negative Prompt off / on | 开启后轮廓和写实感改善，但眼睛异常并未完全消除，构图也发生变化 | Negative Prompt 只改变概率方向，不是局部删除或硬性禁令 |
| DDIM / DDPM | 同模型、同 prompt 和步数下，DDIM 得到全身构图，DDPM 得到脸部特写 | Scheduler 不改变 U-Net 权重，但不同数值轨迹可以产生完全不同的样本；单样本不能判定谁更好 |
| 红壶 / 蓝壶 / 水彩壶 | 颜色和风格词都生效，但构图、位置和附加杯子也随之变化 | Prompt 是全局概率条件，不是只修改指定属性的程序指令 |

### 3. 组件与张量形状

实际输出验证了 Stable Diffusion 1.5 的数据流：

| 阶段 | Tensor shape | 含义 |
|---|---|---|
| Token IDs | `[1, 77]` | 1 条补齐或截断到 77 个 token 的文本 |
| CLIP 文本特征 | `[1, 77, 768]` | 每个 token 变成 768 维条件向量 |
| 初始 latent | `[1, 4, 64, 64]` | 1 张图的四通道空间潜变量 |
| CFG 的 U-Net 输入 / 输出 | `[2, 4, 64, 64]` | batch 中并行放入 negative 与 positive 两个分支 |
| 每个 CFG 分支 | `[1, 4, 64, 64]` | 两组噪声预测拆开后再组合 |
| VAE 解码输出 | `[1, 3, 512, 512]` | 最终 RGB 图像张量 |

因此，CFG 不是把图像通道从 4 永久变成 8，而是临时把 batch 从 1 扩为 2；组合后仍回到 `[1,4,64,64]`。

### 4. 手写 DDIM 的验证

手写一次 DDIM 更新与 `scheduler.step(..., eta=0)` 的结果形状均为 `[1,4,64,64]`：

- 最大绝对误差：`0.00390625`；
- 平均绝对误差：`0.0002799034`。

这是 FP16 下很小的舍入差异。完整手写 DDIM 生成图与 Pipeline 基准图的 SHA-256 完全相同，说明本次实现正确复现了“U-Net 预测噪声，Scheduler 根据公式更新 latent”的核心循环。

### 5. VAE 重建

实际执行了：

$$
[1,3,512,512]\xrightarrow{\mathcal E}
[1,4,64,64]\xrightarrow{\mathcal D}
[1,3,512,512].
$$

- `mode()` 重建 MSE：`0.00341097`；
- `sample()` 重建 MSE：`0.00340905`。

两张重建图肉眼非常接近原图，但细纹理存在轻微变化，验证了 VAE 是有损感知压缩。本次单次采样中 `sample()` 的 MSE 略低，不能据此推出它普遍优于 `mode()`；两者的本质区别是 `mode()` 确定、稳定，而 `sample()` 引入后验随机性。

### 6. DDIM inversion 与编辑

本次 inversion 程序成功运行，但“程序跑通”不等于“忠实重建成功”：

- 模型生成猫图经过 inversion 再用原 prompt 去噪后，猫的颜色、姿态、前腿和视角明显改变；
- 真实小狗重建发生更大漂移，彩色站立侧身图变成黑白近景小狗；
- 这说明朴素 DDIM inversion 是近似过程，并受到 VAE 有损、prompt 描述不完备、有限步离散化和 CFG 条件轨迹不完全可逆等影响。

编辑结果：

- `puppy → cat, skip20`：主体较成功地变成猫，同时保留草地、朝向、位置和大致背景；
- `puppy → cat, skip0`：改动更激进，但颜色、构图、姿态和背景漂移更大；
- `grass → sand, skip20`：草地基本没有变成沙地，属于负结果，说明较低噪声起点保留能力更强，但编辑能力可能不足。

由此得到的核心规律是：从越高噪声处重新生成，编辑自由度越大、原图保真度越低；从越低噪声处开始，结构保留越好、目标修改越可能失败。该规律是本次样本支持的趋势，不应写成对所有图像都严格成立的定理。

完整代码、图片和实验手册见 [08-Experiment-Stable Diffusion 文生图、DDIM 与图像编辑](#/notes/stable-diffusion-lab)。

## 十一、公式速记

$$
\boxed{
\begin{aligned}
z_0&=\mathcal E(x),
&\widetilde x&=\mathcal D(z_0),\\
f&=H/h=W/w,\\
z_t&=\sqrt{\bar\alpha_t}z_0+
\sqrt{1-\bar\alpha_t}\epsilon,\\
L_{\mathrm{LDM}}
&=\mathbb E\left\|
\epsilon-\epsilon_\theta(z_t,t)
\right\|_2^2,\\
Q&=W_Q\phi_i(z_t),
&K&=W_K\tau_\theta(y),
&V&=W_V\tau_\theta(y),\\
\operatorname{Attention}(Q,K,V)
&=\operatorname{softmax}
\left(\frac{QK^T}{\sqrt d}\right)V,\\
L_{\mathrm{cond}}
&=\mathbb E\left\|
\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))
\right\|_2^2.
\end{aligned}}
$$

## 十二、汇报问答

### 1. 像素空间扩散为什么昂贵，latent space 为什么省计算？

像素空间扩散要在高分辨率 RGB 张量上反复运行 U-Net。LDM 先把 $512\times512\times3$ 图像压缩为约 $64\times64\times4$ 的空间特征图，再在较小张量上完成几十次去噪，最后只解码一次；主要计算因此显著降低。

### 2. Autoencoder、Text Encoder、U-Net 分别负责什么？

Autoencoder 在像素与 latent 之间转换；Text Encoder 把 prompt token 编成条件向量；U-Net 接收 $z_t,t$ 和文本条件并预测 latent 中的噪声。Scheduler 再根据噪声预测计算下一步 latent。

### 3. 为什么采用两阶段训练？

第一阶段先学习稳定的感知压缩空间，第二阶段冻结 Autoencoder、只学习该空间的数据分布。这样把“如何重建图像”和“如何生成合理 latent”两个目标分开，便于复用 Autoencoder，也避免同时优化时互相牵制。

### 4. LDM 的噪声预测 Loss 怎样由 DDPM 得到？

DDPM 在像素 $x_t$ 上预测噪声；LDM 先令 $z_0=\mathcal E(x)$，再按同样闭式公式得到 $z_t$，把 Loss 中的 $x_t$ 换成 $z_t$：

$$
L_{\mathrm{LDM}}=\mathbb E\|\epsilon-\epsilon_\theta(z_t,t)\|_2^2.
$$

扩散数学主线没有被推翻，变化在于建模空间。

### 5. Cross-Attention 中 Q、K、V 来自哪里？

Q 来自 U-Net 当前图像 latent 特征，表示每个图像位置想查询什么；K/V 来自文字或其他条件特征，分别负责匹配和提供被汇总的信息。它让不同图像位置动态关注不同文本 token。

### 6. 训练和纯文本推理时，各模块何时运行？

训练第二阶段时，真实图像先经 VAE Encoder 得到 $z_0$，U-Net 在随机时间步预测加入的噪声，并通过 Loss 反向传播。纯文本推理没有输入图像，因此不用 VAE Encoder；U-Net 与 Scheduler 从随机 $z_T$ 多步得到 $z_0$，VAE Decoder 最后只运行一次。

### 7. 为什么压缩率不是越大越好？

压缩更强会减少扩散计算，但也会永久丢掉更多视觉信息，使 Decoder 无法恢复细节；压缩太弱又节省不了多少算力。因此存在速度与重建质量的折中，论文实验显示 $f=4\sim16$ 较平衡，而不是数学上规定只能选某个值。

### 8. 原论文 LDM 与 Stable Diffusion v1.x 有什么区别？

LDM 是潜空间扩散与通用条件接口的框架；原论文文本实验使用 BERT tokenizer 和自建 Transformer 条件编码器。Stable Diffusion v1.x 是该框架的具体系统，典型地使用 `AutoencoderKL`、冻结的 CLIP ViT-L/14 Text Encoder 和带 Cross-Attention 的条件 U-Net。

### 9. U-Net 和 Scheduler 有什么区别？

U-Net 是有参数、通过训练学得的噪声预测器；Scheduler 通常没有可训练权重，它规定时间步和从 $z_t$ 到下一状态的数值公式。更换 Scheduler 不等于重新训练或更换 U-Net。

### 10. CFG 为什么需要两份预测？

一份来自 negative/空条件，另一份来自正向 prompt。两者的差向量近似表示文字条件希望推动的方向，CFG 用 $\epsilon_{\mathrm{neg}}+s(\epsilon_{\mathrm{pos}}-\epsilon_{\mathrm{neg}})$ 放大这一方向；$s$ 太大也会放大伪影。

### 11. DDIM inversion 是严格逆过程吗？

不是。它利用当前 U-Net 噪声预测近似沿确定性 DDIM 轨迹反向移动，还受到 VAE 有损压缩、有限时间步、prompt 不完整和 CFG 的影响。本次实验中程序成功执行，但原 prompt 重建明显漂移，正是这一局限的直接证据。

## 相关笔记

- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)
- [08-Experiment-Stable Diffusion 文生图、DDIM 与图像编辑](#/notes/stable-diffusion-lab)
- [09-Scalable Diffusion Models with Transformers（DiT）](#/notes/dit)
- [12-Blended Latent Diffusion](#/notes/blended-latent-diffusion)
- [13-Prompt-to-Prompt Image Editing with Cross-Attention Control](#/notes/prompt-to-prompt)
- [16-Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet）](#/notes/controlnet)：冻结 LDM 的 U-Net 主干，用多尺度可训练分支增加边缘、深度、姿态等空间控制。
