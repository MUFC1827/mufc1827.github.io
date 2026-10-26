# 16 ControlNet：给大规模扩散模型增加空间条件控制

## 来源与一句话定位

- 论文：*Adding Conditional Control to Text-to-Image Diffusion Models*
- 作者：Lvmin Zhang、Anyi Rao、Maneesh Agrawala
- 会议：ICCV 2023，Best Paper（Marr Prize）
- [CVF 官方论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)
- [官方代码](https://github.com/lllyasviel/ControlNet)

$$
\boxed{
\text{ControlNet 冻结原 Stable Diffusion，复制其 U-Net 编码器作为可训练控制分支，}
\text{再用零初始化的卷积把多尺度空间条件安全地注入原模型。}
}
$$

最短直觉：

- 文本 prompt 告诉模型“画什么”；
- 边缘、深度、姿态、分割图等告诉模型“在哪里、摆成什么形状”；
- 冻结主干保住原模型已经学会的生成能力；
- 可训练副本学习怎样读懂新条件；
- 零卷积保证训练刚开始时控制分支的影响恰好为零，之后再由梯度逐渐学出影响。

## 一、它接在整个知识链的哪里

```text
DDPM：学习预测噪声，建立扩散训练与逐步去噪
→ LDM / Stable Diffusion：在低维 latent 中扩散，用 U-Net + 文本 Cross-Attention 生成图像
→ DiT / MM-DiT：把去噪网络换成 Transformer，并发展到 flow matching
→ SDEdit / BLD / P2P / FlowEdit / Stable Flow：在推理阶段编辑已有图像
→ ControlNet：训练一个外挂控制分支，使 Stable Diffusion 长期获得边缘、深度、姿态等空间控制能力
```

ControlNet 与前几种编辑方法最重要的区别：

| 方法 | 是否训练新参数 | 主要输入控制 | 控制方式 |
|---|---:|---|---|
| Blended Latent Diffusion | 否 | Mask + 参考背景 | 每步融合 latent |
| Prompt-to-Prompt | 否 | 源/目标 prompt | 修改 Cross-Attention map |
| FlowEdit | 否 | 源/目标 prompt | 用目标速度减源速度形成编辑 ODE |
| Stable Flow | 否 | 源图 + 目标 prompt | 在 DiT 关键层注入视觉 $K,V$ |
| **ControlNet** | **是** | 边缘、深度、姿态、分割等 | 训练一个与冻结主干相连的多尺度控制分支 |

因此它不是“临时改一次图”的技巧，而是“为基础模型训练一种可重复使用的新控制能力”。

## 二、论文要解决什么问题

只给文本：

> “一个人在厨房里做饭。”

模型知道人物、厨房和做饭，但很难仅靠文字精确指定：人物站在哪里、手臂怎样弯、灶台轮廓怎样、相机视角和深度怎样。

空间条件可以补足这些信息：

- Canny / HED 边缘：约束物体轮廓；
- M-LSD 直线：约束建筑等规则线条；
- Scribble：用用户粗线稿指定构图；
- Human Pose：约束人体关节点和姿态；
- Semantic Segmentation：约束每个语义区域的位置；
- Depth：约束远近和三维几何布局；
- Normal Map：约束表面朝向；
- Cartoon Line Drawing：约束动漫线稿。

但“把条件塞进模型”还遇到两个矛盾：

1. 直接微调整个大模型，可能在较小控制数据集上过拟合并灾难性遗忘；
2. 只接一个很浅的小网络，又可能没有能力理解真实世界中复杂的边缘、深度或姿态。

ControlNet 的答案是：**原模型不动，把它强大的编码器复制一份来学条件，并从零影响开始接回原模型。**

## 三、先回忆 Stable Diffusion 主干

训练图像 $x_0$ 先经 VAE Encoder 得到 latent $z_0$。随机采样时间步 $t$ 和高斯噪声 $\epsilon$，构造带噪 latent：

$$
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

U-Net 接收：

- 当前带噪 latent $z_t$；
- 时间步 $t$ 的 embedding；
- 文本 prompt 的 embedding $c_t$；

输出噪声预测 $\epsilon_\theta(z_t,t,c_t)$。训练目标是让预测噪声接近真实加入的噪声。

Stable Diffusion 的 U-Net 有下采样 Encoder、Middle Block、上采样 Decoder，以及 Encoder 到 Decoder 的 Skip Connection。512×512 图像在 VAE 中被压到 64×64 latent，因此 U-Net 内部常见四个尺度：

$$
64\times64\rightarrow32\times32\rightarrow16\times16\rightarrow8\times8.
$$

ControlNet 不改变“预测噪声—迭代去噪—VAE 解码”这条主流程，只给 U-Net 增加额外空间条件 $c_f$。

## 四、先理解抽象的一个网络块

普通神经网络块记为：

$$
\boxed{y=\mathcal F(x;\Theta).}
\tag{1}
$$

- $x$：输入特征图，形状可写为 $h\times w\times c$；
- $\mathcal F$：这个网络块执行的变换；
- $\Theta$：网络块内部的可学习参数；
- $y$：输出特征图。

ControlNet 对它做四件事：

1. 保留原块 $\mathcal F(x;\Theta)$，锁住参数 $\Theta$；
2. 复制同一个块，参数记为 $\Theta_c$，允许训练；
3. 用一个零卷积把条件 $c$ 加到副本输入；
4. 用另一个零卷积把副本输出变成残差，加回原块输出。

完整公式：

$$
\boxed{
y_c=
\mathcal F(x;\Theta)
+
\mathcal Z\Big(
\mathcal F\big(x+\mathcal Z(c;\Theta_{z1});\Theta_c\big);
\Theta_{z2}
\Big).
}
\tag{2}
$$

逐层读公式：

1. $\mathcal Z(c;\Theta_{z1})$：先把外部条件映射到与 $x$ 可相加的形状；
2. $x+\mathcal Z(c;\Theta_{z1})$：把条件送入可训练副本；
3. $\mathcal F(\cdot;\Theta_c)$：副本利用预训练能力理解条件；
4. 外层 $\mathcal Z(\cdot;\Theta_{z2})$：把副本结果转成“应该给主干增加多少”的控制残差；
5. 与冻结原块输出相加，得到 $y_c$。

这里的 $+$ 是逐元素相加，不是 Concat，也不是 Softmax 加权平均。

## 五、Zero Convolution 到底是什么

Zero Convolution 不是“永远输出零的特殊运算”。它就是普通 $1\times1$ 卷积，只是初始时：

$$
W=0,\qquad b=0.
$$

对空间位置 $p$、输出通道 $i$：

$$
\mathcal Z(I;W,b)_{p,i}
=b_i+\sum_j I_{p,j}W_{i,j}.
$$

$1\times1$ 的直觉：它不看相邻像素，只在同一位置把各通道重新线性组合，所以保持 $H,W$ 不变，并能调整通道表达。

### 1. 为什么训练开始时不破坏原模型

由于两个零卷积最初都输出零：

$$
\mathcal Z(c;\Theta_{z1})=0,\qquad
\mathcal Z(\cdot;\Theta_{z2})=0.
$$

代回式 (2)：

$$
\boxed{y_c=\mathcal F(x;\Theta)=y.}
\tag{3}
$$

因此第 0 步的 ControlNet 与原模型行为完全一致：原模型仍能正常生成高质量图像，只是暂时不听额外空间条件。

### 2. 权重为零，为什么还能学习

用最简单的一维式子：

$$
y=wx+b.
$$

其偏导是：

$$
\frac{\partial y}{\partial w}=x,qquad
\frac{\partial y}{\partial b}=1,qquad
\frac{\partial y}{\partial x}=w.
$$

当 $w=0$ 且 $x\neq0$：

$$
\frac{\partial y}{\partial w}=x\neq0,qquad
\frac{\partial y}{\partial b}=1\neq0,qquad
\frac{\partial y}{\partial x}=0.
$$

结论：

- 第一次反向传播时，零卷积自己的 $W,b$ 能获得非零梯度；
- 更新一步后 $W$ 变成非零；
- 此后梯度也能继续传到零卷积前面的控制副本；
- 所以控制影响是从 0 开始、逐渐由数据学习出来的。

外层零卷积在训练最初像一道安全闸门：先让自己学会怎样输出，再逐步让更前面的副本参与学习。

> 零初始化不会在这里产生“所有神经元永远对称”的问题，因为被零初始化的只是连接用的卷积；副本内部已经继承了不同的预训练权重，并不是整张深网都从同一个零状态开始。

## 六、完整 ControlNet 怎样接进 Stable Diffusion

论文所用 Stable Diffusion U-Net 有：

- 12 个 Encoder Block；
- 1 个 Middle Block；
- 12 个 Decoder Block；
- 合计 25 个块。

ControlNet：

1. 冻结原 U-Net；
2. 只复制 12 个 Encoder Block 和 1 个 Middle Block；
3. 复制的 13 个块从原预训练权重开始，但允许更新；
4. 每个副本输出经过一个零卷积；
5. 13 路控制残差分别加到原 U-Net 的 12 条 Skip Connection 和 Middle Feature；
6. 原 Decoder 不复制、也不训练。

```text
带噪 latent z_t ───────────────→ 冻结的 SD Encoder ─→ Middle ─→ 冻结 Decoder ─→ 预测噪声
       │                              │                  ↑          ↑
       └→ 可训练 Encoder 副本 ← 条件图 │── 12 路零卷积 ───┘          │
                         └→ Middle 副本 ── 1 路零卷积 ──────────────┘
```

更精确地说，Decoder 在每个尺度原本接收 Skip Feature $s_i$；加入 ControlNet 后使用：

$$
s_i+r_i,
$$

其中 $r_i$ 是对应控制分支经零卷积给出的残差。Middle 处同样做一次相加。

### 13、14 为什么会同时出现

- 论文正文：12 个编码块 + 1 个中间块，因此是 **13 路控制输出**；
- 官方代码：`control_scales` 长度也是 13；
- 官方 README 的“重复 14 次”把最前面将条件送入副本的输入零卷积也计入，所以可理解为 **总计 14 个零初始化连接模块**。

汇报时最安全的表述：**13 个可训练的预训练块/13 路多尺度控制残差；若把条件入口的零卷积也算上，共有 14 个零卷积连接。**

### 为什么只复制 Encoder 和 Middle

U-Net Encoder 会逐层提取从局部边缘到全局布局的多尺度特征；Skip Connection 正好把不同尺度信息送给 Decoder。ControlNet 只需在这些接口提供“修正量”，原 Decoder 就能利用原有生成能力完成图像。这样比再复制整个 U-Net 更省计算和显存。

论文报告：与直接优化 Stable Diffusion 相比，加入 ControlNet 后单次训练迭代大约多用 23% 显存、增加 34% 时间（其 A100 40GB 设置）。这是论文特定实现的测量，不是所有硬件的固定比例。

## 七、条件图怎样变成模型能用的特征

原始条件图 $c_i$ 通常是 512×512，而 Stable Diffusion latent 是 64×64，不能直接相加。论文用小型编码器 $\mathcal E$ 做转换：

$$
\boxed{c_f=\mathcal E(c_i).}
\tag{4}
$$

论文正文把 $\mathcal E$ 概括为一个很小的卷积网络，并写出 4 个卷积层、$4\times4$ 卷积核、步长 2、ReLU 以及 16、32、64、128 等通道设置；它的目标输出是与 64×64 latent 对齐的条件特征。官方实现把这一“微型编码器”具体展开为若干 $3\times3$ 卷积和 3 次 stride-2 下采样，最后再接零初始化卷积。这里不必死背实现层数，必须掌握的是：它与 ControlNet 联合训练，把 512×512 条件图转换为 64×64、通道可匹配的特征。

不同实现的具体卷积层数与通道可有工程调整；核心要求不变：

$$
\text{像素空间条件图}
\longrightarrow
\text{与 latent/第一层特征在空间尺寸上对齐的条件特征}.
$$

## 八、训练数据长什么样

每个样本至少需要三项：

$$
(\text{目标图像 }x_0,;\text{文本 }c_t,;\text{空间条件 }c_i).
$$

例如训练 Canny ControlNet：

1. 从真实图像 $x_0$ 用 Canny 检测器提取边缘 $c_i$；
2. 图像本身或自动标注得到文字 $c_t$；
3. VAE 把 $x_0$ 编码为 $z_0$；
4. 加随机噪声得到 $z_t$；
5. ControlNet 看到 $z_t,t,c_t,c_i$，预测加入的噪声。

不同控制类型通常训练不同 ControlNet：深度模型用“图像—深度—文本”数据，姿态模型用“图像—骨架—文本”数据。

## 九、训练目标没有换，仍然是扩散噪声预测

论文直接使用 Stable Diffusion 的噪声预测损失：

$$
\boxed{
\mathcal L=
\mathbb E_{z_0,t,c_t,c_f,\epsilon\sim\mathcal N(0,I)}
\left[
\left\|
\epsilon-\epsilon_\theta(z_t,t,c_t,c_f)
\right\|_2^2
\right].
}
\tag{5}
$$

逐项解释：

- $z_0$：目标图像的干净 latent；
- $t$：随机时间步；
- $\epsilon\sim\mathcal N(0,I)$：随机采样的标准高斯噪声；
- $z_t$：给 $z_0$ 加到第 $t$ 步后的带噪 latent；
- $c_t$：文字条件；
- $c_f$：空间条件特征；
- $\epsilon_\theta$：加入 ControlNet 后整个系统给出的噪声预测；
- $\|\epsilon-\hat\epsilon\|_2^2$：预测噪声与真实噪声的平方误差。

反向传播时：

- 原 SD U-Net 参数 $\Theta$ 冻结，不更新；
- ControlNet 副本参数 $\Theta_c$ 更新；
- 条件编码器参数更新；
- 零卷积参数 $\Theta_{z1},\Theta_{z2}$ 更新。

因此 Loss 是整套输出与真实噪声之间的误差，但优化器只修正未被冻结的参数。

### 为什么 50% 的文字被换为空字符串

训练时作者随机把 50% 的文本条件换为空字符串，使 ControlNet 不能总依赖文字告诉它“这是什么”，而要学会从边缘、姿态、深度等条件中识别结构和一定的语义。这也是论文能够展示“无 prompt 生成”的重要原因。

## 十、Sudden Convergence：为什么会突然学会控制

作者观察到：训练早期图像一直保持较高质量，却不太服从控制；到某个时刻（示例约第 6133 步）突然明显贴合条件，并通常在 1 万步以内出现，称为 sudden convergence。

合理的机制直觉是：

1. 零卷积一开始关闭控制残差，原模型仍正常生成；
2. 零卷积和控制副本逐渐积累“怎样把条件变成有效修正量”的能力；
3. 当控制信号足够强且各尺度协同后，肉眼可见的条件遵循突然出现。

这是论文的经验观察，不是证明所有任务都会在固定步数突变的数学定理。

## 十一、推理时完整流程

以“骨架图控制宇航员姿态”为例：

1. 输入骨架条件图 $c_i$ 和 prompt “an astronaut”；
2. 条件编码器得到 $c_f$；
3. 采样初始随机噪声 latent $z_T$；
4. 在每个去噪时间步 $t$：
   - 冻结 SD 主干处理 $z_t,t,c_t$；
   - ControlNet 副本同时处理 $z_t,t,c_t,c_f$；
   - 得到 13 个不同尺度的控制残差；
   - 分别加到主干 Middle 和 12 条 Skip Connection；
   - 冻结 Decoder 输出噪声预测 $\hat\epsilon$；
   - 采样器据此从 $z_t$ 算到 $z_{t-1}$；
5. 最后用 VAE Decoder 把 $z_0$ 解码为图像。

因此 ControlNet 不是只在开头处理一次条件；在每一个去噪步里，条件分支都会参与 U-Net 的噪声预测。

同一条件图：

- 换 seed：结构大致保持，人物外观、纹理等可变化；
- 换 prompt：空间骨架保持，但语义、服装、风格可变化；
- 换条件图：改变布局、姿态或几何。

## 十二、Classifier-Free Guidance 与 CFG-RW

Stable Diffusion 常用 CFG：

$$
\boxed{
\epsilon_{\mathrm{prd}}
=\epsilon_{\mathrm{uc}}
+\beta_{\mathrm{cfg}}
(\epsilon_c-\epsilon_{\mathrm{uc}}).
}
$$

- $\epsilon_{\mathrm{uc}}$：无文本条件预测；
- $\epsilon_c$：有文本条件预测；
- $\beta_{\mathrm{cfg}}$：文字引导强度。

加入 ControlNet 后，空间条件究竟放进哪一支会产生问题：

- 两支都加：二者共同的控制成分会在差值中抵消，尤其无 prompt 时 CFG 失效；
- 只给 conditional 支路加：乘上较大的 CFG 后，控制可能过强。

论文提出 CFG Resolution Weighting：只给 conditional 分支加条件，并按每个连接处的空间分辨率 $h_i$ 乘权重

$$
w_i=\frac{64}{h_i},
$$

例如论文按连接顺序给出 $h_1=8,h_2=16,\ldots,h_{13}=64$；低分辨率的粗结构连接因而得到更大的倍率。作者同时降低 CFG 的总体引导强度，避免“条件只放在 conditional 支路”时被 CFG 过分放大，并把这一组合称为 CFG Resolution Weighting。它主要用于无 prompt 或文字不足的困难设置。

注意：现代库中的 `controlnet_conditioning_scale`、`guess_mode` 等工程接口与此思想有关，但具体实现可能已经变化；汇报原论文时应以式中的 CFG-RW 为准。

## 十三、多个 ControlNet 怎样组合

如果同时想用 Pose 和 Depth：

1. Pose ControlNet 输出 13 路残差 $r_i^{pose}$；
2. Depth ControlNet 输出 13 路残差 $r_i^{depth}$；
3. 对应尺度直接相加：

$$
r_i=r_i^{pose}+r_i^{depth};
$$

4. 再注入同一冻结主干。

论文的组合实验不需要额外训练，也没有要求额外插值权重。实际应用若条件冲突，现代实现常提供每个 ControlNet 的 scale 调节强度；这是合理工程扩展，不应误说成原论文必需步骤。

## 十四、不同条件到底约束了什么

| 条件 | 主要提供的信息 | 典型优势 | 典型不足 |
|---|---|---|---|
| Canny | 清晰硬边缘 | 细节轮廓保留强 | 易过度拘泥输入细线 |
| HED | 语义化软边界 | 结构自然、比 Canny 柔和 | 细节不一定精确 |
| M-LSD | 直线段 | 建筑与室内透视 | 不适合弯曲物体 |
| Scribble | 稀疏手绘线 | 交互方便、自由度高 | 条件歧义大 |
| Pose | 人体关键点 | 姿态可控 | 不指定人物外观与背景 |
| Segmentation | 类别和区域 | 场景布局明确 | 边界与纹理较粗 |
| Depth | 远近和几何 | 三维布局与视角稳定 | 不保留材质和细节 |
| Normal | 表面方向 | 几何表面控制细 | 需要可靠法线估计 |

条件图并不是目标 RGB 图，它只携带某一种结构信息；Stable Diffusion 的生成先验和 prompt 决定剩余外观。

## 十五、消融实验分别证明什么

论文比较四种 prompt 场景：无 prompt、信息不足、语义冲突、完整 prompt。这比只在“完美描述”下测试更能看出控制分支是否真正理解条件。

### 1. 把零卷积换成普通随机初始化卷积

结果明显下降，接近轻量版。原因是随机卷积从第一步就向冻结主干注入不受控噪声，并破坏可训练副本继承的预训练特征。

证明：**零初始化的安全连接不是装饰，它保护了预训练生成能力和训练稳定性。**

### 2. 不复制预训练块，只训练浅卷积（ControlNet-lite）

完整 prompt 时尚可，但在无 prompt 或信息不足时不能充分解释复杂条件。

证明：**条件理解需要足够深、且继承预训练知识的强分支；只有浅层适配器容量不足。**

### 3. 直接微调整个原 Stable Diffusion

补充材料指出它难以自然组合多个控制，在小数据上更易过拟合/遗忘，也难把控制迁移到兼容的社区模型。

证明：**冻结主干 + 外挂控制分支在保真、组合和迁移方面有结构性优势。**

### 4. 连接 Encoder 或 Decoder、是否使用零卷积

扩展消融显示，单纯把轻量卷积接到 Encoder/Decoder 都不如完整方案稳定；只加零卷积也不能弥补浅分支表达能力不足。

证明：**“预训练深副本”和“零卷积安全连接”缺一不可。**

## 十六、定量实验应该怎样读

### 1. 人类评价

12 名参与者对 20 个未见手绘草图、5 种方法分别按图像质量和草图忠实度排序。平均排名 1–5，越高越好：

| 方法 | 图像质量 ↑ | 条件忠实度 ↑ |
|---|---:|---:|
| PITI | $1.10\pm0.05$ | $1.02\pm0.01$ |
| Sketch-Guided，$\beta=1.6$ | $3.21\pm0.62$ | $2.31\pm0.57$ |
| Sketch-Guided，$\beta=3.2$ | $2.52\pm0.44$ | $3.28\pm0.72$ |
| ControlNet-lite | $3.93\pm0.59$ | $4.09\pm0.46$ |
| **ControlNet** | **$4.22\pm0.43$** | **$4.28\pm0.45$** |

结论：完整 ControlNet 在该小规模用户研究中同时兼顾质量和控制忠实度。样本和参与者都不大，因此不能把它夸成对所有任务的绝对证明。

### 2. 语义分割条件重建 IoU

用 OneFormer 再分割生成图，与输入分割条件比较：ControlNet 的 IoU 为 $0.35\pm0.14$，高于 ControlNet-lite 的 $0.32\pm0.12$、LDM 的 $0.31\pm0.09$ 等，但仍低于把真实图重新分割所得的参考上限 $0.58\pm0.10$。

这证明控制更忠实，但不是像素级完美复原。

### 3. 为什么不能只看 FID

| 方法 | FID ↓ | CLIP ↑ | 美学分 ↑ |
|---|---:|---:|---:|
| 无空间控制的 Stable Diffusion | 6.09 | 0.26 | 6.32 |
| ControlNet-lite | 17.92 | 0.26 | 6.30 |
| **ControlNet** | **15.27** | **0.26** | **6.31** |

无控制的 SD 得到最低 FID，但它可以完全忽略给定分割布局，所以这不能说明它“控制得最好”。评价可控生成必须同时看：

$$
\text{图像真实性/质量} + \text{文本一致性} + \text{条件忠实度}.
$$

这是本论文很值得在汇报中指出的评价思想。

### 4. 与工业深度模型的比较

论文称 SD v2 Depth-to-Image 使用超过 1200 万图像和大量集群训练；ControlNet 的对比模型只用 20 万样本、单张 RTX 3090 Ti、5 天训练。12 人辨别测试平均精度约 $0.52\pm0.17$，接近随机猜测。

可以说：**在该实验设置和小规模感知测试中，两者视觉结果难以区分。**不能外推为所有数据、指标和场景上性能完全相等。

## 十七、数据规模、迁移与可组合性

作者用 1k、10k、50k、500k、3M 样本训练 Canny ControlNet：

- 1k 也没有直接崩溃，能开始跟随结构，但泛化、风格和语义更弱；
- 数据增多后条件理解和开放场景鲁棒性明显改善；
- 论文据此称架构可用于小于 50k 和大于 1M 的不同规模数据。

不要把“1k 能训练”理解为“1k 与 3M 同样好”。架构稳定和最终泛化质量是两回事。

由于 Stable Diffusion 社区中许多模型保留相同 U-Net 拓扑并由同一基础权重微调，一个 ControlNet 可以接到部分兼容社区模型上。这里的“可迁移”依赖架构和特征兼容，不能接到任意新模型。

## 十八、论文的局限

1. 原始方案通常每种条件训练一个模型，需要配对数据和训练成本；
2. 条件检测器本身的误差与偏差会传给生成结果；
3. 输入边缘或线稿有歧义时，ControlNet 可能误判语义，强 prompt 也未必完全纠正；
4. 控制过强会限制创造性，多个条件冲突时无法保证都满足；
5. 条件忠实不是像素级严格约束；
6. 可训练副本较大，并不是 LoRA 式参数极少的方法；
7. 原论文围绕 SD1.5/2.1 的 U-Net 展开，不能把结构细节直接套到任意 DiT；
8. 很多结果是定性展示，用户研究规模有限；
9. 社区检查点迁移建立在结构兼容的经验上，不是对任意模型的理论保证。

如果只想较粗地跟随条件，补充材料给出一种简单办法：前 50% 去噪步使用 ControlNet，后 50% 换回普通 Stable Diffusion，让后期细节生成获得更多自由。

## 十九、真正贡献与不应夸大的地方

### 真正贡献

1. 提出冻结主干、复制预训练编码器的强控制分支；
2. 用零卷积保证新增分支初始不改变原模型，并可平滑学出残差；
3. 在 U-Net 多尺度 Skip/Middle 接口注入结构条件；
4. 统一支持边缘、线稿、姿态、分割、深度、法线等控制；
5. 展示单条件、多条件、有/无 prompt 的控制；
6. 用消融证明零卷积与深预训练副本各自必要；
7. 展示不同数据规模、社区模型迁移和训练资源下的可用性。

### 不应夸大

- 它没有改变 DDPM/LDM 的基本噪声预测目标；
- “冻结主干”不等于整个系统无需训练；
- “零卷积”不等于参数永远为零；
- “从预训练权重开始”不等于所有新增层都不是从头训练：条件编码器和零卷积仍是新增参数；
- 一张条件图只约束其中一类结构信息，不会精确规定所有像素；
- 一个模型能迁移到兼容 SD 检查点，不代表任意架构都能直接使用。

## 二十、必须掌握与了解即可

### 必须能讲清楚

- 文本控制为什么不能替代空间条件；
- ControlNet 的冻结主干、可训练副本和零卷积分别负责什么；
- 式 (2) 每一层的计算顺序；
- 零卷积初始输出为零却仍能学习的偏导原因；
- 为什么第 0 步 ControlNet 与原 Stable Diffusion 完全相同；
- 12 Encoder + 1 Middle 怎样形成 13 路多尺度残差；
- 条件图怎样从 512×512 转为 64×64 特征；
- 训练数据三元组、加噪过程和式 (5)；
- 哪些参数训练，哪些参数冻结；
- 推理时为何每个去噪步都运行 ControlNet；
- 多 ControlNet 怎样相加；
- 两项核心消融分别证明什么；
- 为什么 FID 必须与条件忠实度一起看；
- ControlNet 与 P2P、Stable Flow 等 training-free 编辑的本质区别。

### 了解即可

- 每一种条件模型的精确训练 GPU-hours；
- CFG-RW 每个具体层的编号；
- 所有数据集与检测器名称；
- 用户研究每个标准差；
- 官方代码每一层的具体通道数。

## 二十一、汇报常见问题与标准答案

### Q1：ControlNet 一句话解决了什么问题？

它让预训练文生图扩散模型不仅听文字，还能精确跟随边缘、深度、姿态、分割等空间条件，同时尽量保留原模型的生成质量。

### Q2：为什么不直接微调整个 Stable Diffusion？

控制数据通常远小于基础模型预训练数据，整体微调容易过拟合和遗忘，且不便组合多种控制。ControlNet 冻结主干，把新能力放进外挂分支，风险更小，也更易复用。

### Q3：为什么复制的是预训练 Encoder，而不是随机小网络？

复杂边缘、深度和姿态需要语义理解。复制分支继承了 Stable Diffusion 在大规模数据上学到的强特征；消融中的浅层 ControlNet-lite 在无 prompt 和文字不足时明显更弱。

### Q4：零卷积的作用是什么？

让新增控制残差在训练开始时严格为零，保证系统初始行为等同原模型；随后权重与偏置通过非零梯度逐渐学成普通卷积，实现安全、渐进的条件注入。

### Q5：权重全是零，梯度不会全为零吗？

不会。对 $y=wx+b$，$\partial y/\partial w=x$、$\partial y/\partial b=1$，所以输入非零且上游梯度非零时，$w,b$ 第一步就会更新。只是 $\partial y/\partial x=w=0$，前一层在最初一刻暂时收不到梯度；更新一步后便可继续传播。

### Q6：ControlNet 到底复制多少块？

复制 SD U-Net 的 12 个 Encoder Block 和 1 个 Middle Block，共 13 个可训练预训练块，输出 13 路残差；Decoder 不复制。若把条件入口的零卷积也计入，官方 README 会说有 14 个零卷积连接。

### Q7：控制残差加在哪里？

1 路加到 U-Net Middle Feature，另外 12 路加到对应尺度的 Encoder–Decoder Skip Connection，再交给冻结 Decoder 生成噪声预测。

### Q8：训练 Loss 与 DDPM/LDM 有什么不同？

本质没有不同，仍是预测噪声与真实噪声的均方误差。不同点是网络额外接收空间条件，反向传播只更新 ControlNet 分支、条件编码器和零卷积，原 SD 主干冻结。

### Q9：为什么训练时把一半 prompt 置空？

迫使控制分支从空间条件本身学习结构和语义，而不是完全依赖文字，因此提升无 prompt 或 prompt 信息不足时的能力。

### Q10：推理时条件只输入一次吗？

不是。每个去噪时间步，ControlNet 都根据当前 $z_t$、时间、文本和条件产生多尺度残差，再帮助 U-Net 预测本步噪声。

### Q11：多个 ControlNet 怎样一起用？

各 ControlNet 在对应尺度输出残差，逐尺度直接相加后注入同一个冻结主干。它们不需要联合重新训练；实际工程中可再用 scale 处理强度或冲突。

### Q12：为什么无控制 SD 的 FID 反而最好？

FID 只衡量生成分布是否像真实图，不检查是否遵守给定分割图。无控制 SD 可以忽略任务而获得好 FID，所以必须同时评价条件重建 IoU 或其他条件忠实度。

### Q13：ControlNet 与 Prompt-to-Prompt 的区别？

P2P 不训练新参数，通过替换或重加权 Cross-Attention map 临时编辑；ControlNet 需要训练一个控制分支，把结构图变成多尺度残差，之后可反复用于生成。

### Q14：ControlNet 与 Stable Flow 的区别？

Stable Flow 面向 flow-based DiT，在推理时寻找并注入关键层视觉 $K,V$，无需训练；ControlNet 面向原论文中的 SD U-Net，训练一个外挂空间条件编码分支，并通过 Skip/Middle 残差控制每步去噪。

### Q15：论文实验最关键的结论是什么？

完整方案在图像质量和条件忠实度上优于比较方法；去掉零卷积会破坏安全微调，去掉预训练深副本会削弱复杂条件理解，说明两个设计缺一不可。

## 二十二、30 秒口述版

> ControlNet 是给预训练 Stable Diffusion 增加空间控制的可训练外挂。它冻结原 U-Net，复制 12 个编码块和 1 个中间块作为可训练分支，把边缘、深度、姿态或分割条件编码后送入副本。副本产生的 13 路多尺度特征经零初始化的 $1\times1$ 卷积，加到原 U-Net 的 12 条 Skip Connection 和 Middle Feature。因为零卷积初始输出为零，训练第 0 步与原 SD 完全相同，不会破坏生成能力；其权重梯度仍可非零，所以之后会逐渐学出控制残差。训练目标仍是 DDPM/LDM 的噪声预测 MSE，只更新新增分支。消融说明预训练深副本负责理解复杂条件，零卷积负责安全连接，二者共同实现有或无 prompt、单或多条件下的稳定控制。

## 二十三、一页纸复习主线

```text
问题：文字难以精确控制位置、轮廓、姿态和几何
↓
风险：全量微调会忘；浅分支又读不懂复杂条件
↓
结构：冻结 SD 主干 + 复制 Encoder/Middle 可训练副本
↓
入口：条件图 512×512 → 条件编码器 → 64×64 特征
↓
安全：零卷积初始 W=b=0，因此新增残差初始为 0
↓
学习：d(wx+b)/dw=x，零权重仍能在第一步更新
↓
注入：13 路残差 → 12 条 Skip + 1 个 Middle
↓
训练：仍预测噪声，只更新 ControlNet/条件编码器/零卷积
↓
推理：每个去噪步都运行主干和控制分支
↓
效果：支持边缘、线稿、姿态、分割、深度、法线及多条件
↓
证据：零卷积与预训练深副本的消融；质量 + 条件忠实度评价
```

## 二十四、与前课连接

- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)：式 (5) 仍是预测真实噪声的 MSE。
- [08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)：ControlNet 直接建立在 latent diffusion、VAE 和 U-Net 上。
- [12-Blended Latent Diffusion](#/notes/blended-latent-diffusion)：BLD 用 Mask 每步融合 latent；ControlNet 用训练得到的多尺度残差表达更丰富空间结构。
- [13-Prompt-to-Prompt Image Editing with Cross-Attention Control](#/notes/prompt-to-prompt)：P2P 修改 Attention，ControlNet 训练条件分支。
- [15-Stable Flow - Vital Layers for Training-Free Image Editing](#/notes/stable-flow)：Stable Flow 在 DiT 关键层推理时注入视觉 $K,V$；ControlNet 在 SD U-Net 的多尺度 Skip/Middle 上训练控制残差。
