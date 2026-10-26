# 11 SDEdit：Guided Image Synthesis and Editing with Stochastic Differential Equations

## 来源与定位

- 论文：*SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations*
- 作者：Chenlin Meng 等
- 会议：ICLR 2022
- [arXiv](https://arxiv.org/abs/2108.01073)
- [官方项目页](https://sde-image-editing.github.io/)
- [官方代码](https://github.com/ermongroup/SDEdit)

$$
\boxed{
\text{SDEdit 不训练新的编辑网络，而是把输入加噪到中途，再接上预训练扩散模型的后半段反向过程。}
}
$$

它是**推理/采样方法**，不是新的网络架构，也不是新的训练目标。

知识链：

```text
DDPM / SDE：提供加噪与去噪过程
→ LDM：把过程移到 latent space
→ DiT / MM-DiT：更换去噪网络
→ SDEdit：改变生成过程的起点，实现图生图和编辑
```

## 一、要解决的问题

用户提供一张 guide $x^{(g)}$，它可能是粗糙涂鸦、被画了几笔的照片或有明显接缝的拼贴图。输出需要同时满足：

- **Realism（真实性）**：像自然图像；
- **Faithfulness（忠实度）**：保留 guide 的颜色、轮廓、布局和编辑意图。

二者存在冲突：完全照抄 guide 会保留粗糙痕迹；完全依赖生成模型又可能忘掉输入。

早期方法的主要问题：

- Conditional GAN 往往需要为特定条件重新训练；
- GAN inversion 要为每张图寻找或优化 latent，且不自然的 guide 未必在 GAN 流形上；
- SDEdit 利用已有扩散先验，不需要逐图 inversion，也不需要为每种编辑任务重新训练。

## 二、必须会读的概率符号

$$
x\sim\mathcal N(\mu,\Sigma)
$$

表示“随机变量 $x$ 从均值为 $\mu$、协方差为 $\Sigma$ 的高斯分布中采样”。这里的 $\mathcal N$ **不是神经网络**。

若：

$$
x\sim\mathcal N(\mu,\sigma^2I),
$$

则等价的采样写法是：

$$
x=\mu+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

$I$ 是单位矩阵；$\sigma^2$ 是方差，$\sigma$ 是标准差，也是噪声前的乘数。

## 三、ODE、SDE 与前向加噪

普通微分方程 ODE：

$$
d x=f(x,t)\,dt.
$$

下一小步只由确定方向 $f$ 决定。

随机微分方程 SDE：

$$
d x=f(x,t)\,dt+g(t)\,dW_t.
$$

$dW_t$ 表示随机扰动。离散地看：

$$
x_{k+1}
\approx
x_k+f(x_k,t_k)\Delta t
+g(t_k)\sqrt{\Delta t}\,\epsilon_k.
$$

最小理解：**SDE 每一步既有确定移动，也有随机扰动。** 不要求掌握 Itô 积分的严格定义。

扩散前向过程可统一写成：

$$
\boxed{
x(t)=\alpha(t)x(0)+\sigma(t)\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
}
$$

- $\alpha(t)$：原图信号还保留多少；
- $\sigma(t)$：噪声有多强；
- $t$ 越靠近噪声端，原图信息通常越少。

### VE-SDE

原论文主文主要使用 VE-SDE。其转移分布为：

$$
x(t)\mid x(0)
\sim
\mathcal N
\left(
x(0),
[\sigma^2(t)-\sigma^2(0)]I
\right).
$$

若 $\sigma(0)$ 很小，可直观写成：

$$
x(t)\approx x(0)+\sigma(t)\epsilon.
$$

### VP-SDE / DDPM

与已经学过的 DDPM 对应：

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon.
}
$$

VE 和 VP 只是不同的噪声日程，不改变 SDEdit 的核心思想。

## 四、Score 是什么

$$
s_\theta(x,t)
\approx
\nabla_x\log p_t(x).
$$

- $p_t(x)$：时刻 $t$ 的带噪图像分布；
- $\log p_t(x)$：对数概率密度；
- $\nabla_x$：对图像中所有数值求梯度；
- Score 可理解为一个与图像同形状的箭头，指向“模型认为更可能、更像训练数据”的方向。

对条件高斯分布：

$$
q(x_t\mid x_0)
=
\mathcal N(\alpha_tx_0,\sigma_t^2I),
$$

其对数密度为：

$$
\log q
=
\text{常数}
-
\frac{\|x_t-\alpha_tx_0\|^2}{2\sigma_t^2}.
$$

对 $x_t$ 求梯度：

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{x_t-\alpha_tx_0}{\sigma_t^2}.
$$

又因为：

$$
x_t-\alpha_tx_0=\sigma_t\epsilon,
$$

所以：

$$
\boxed{
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{\epsilon}{\sigma_t}.
}
$$

因此 Score prediction 与 DDPM 的 noise prediction 本质上携带同一信息，只差已知的比例和符号：

$$
s_\theta(x_t,t)
\approx
-\frac{\epsilon_\theta(x_t,t)}{\sigma_t}.
$$

不同论文可能使用不同符号约定，必须先看其定义。

## 五、反向 SDE

一般前向 SDE：

$$
d x=f(x,t)dt+g(t)dW_t.
$$

其反向时间过程可写为：

$$
d x
=
[f(x,t)-g(t)^2s_\theta(x,t)]dt
+g(t)d\bar W_t,
$$

并从大 $t$ 向小 $t$ 积分。由于时间在倒着走，不要脱离积分方向单独死抠负号。必须理解的是：

> 反向过程使用 Score，把带噪状态逐步拉向高概率的自然图像区域，同时保留采样随机性。

反向 SDE 的严格时间反演需要随机微积分，本阶段只需会解释，不要求推导。

## 六、SDEdit 核心算法

给定 guide $x^{(g)}$，选择中间时刻 $t_0$。

### 第一步：给 guide 加噪

一般形式：

$$
\boxed{
x^{(g)}(t_0)
=
\alpha(t_0)x^{(g)}
+
\sigma(t_0)\epsilon.
}
$$

原论文 VE-SDE 形式：

$$
x^{(g)}(t_0)
\sim
\mathcal N
\left(
x^{(g)},
[\sigma^2(t_0)-\sigma^2(0)]I
\right).
$$

DDPM 形式：

$$
x^{(g)}_{t_0}
=
\sqrt{\bar\alpha_{t_0}}x^{(g)}
+
\sqrt{1-\bar\alpha_{t_0}}\epsilon.
$$

### 第二步：从 $t_0$ 开始反向生成

$$
x^{(g)}_{t_0}
\rightarrow
x_{t_0-1}
\rightarrow
\cdots
\rightarrow
x_0=x_{\text{out}}.
$$

完整流程：

```text
guide
→ 选择 t₀
→ 直接构造对应噪声等级的 noisy guide
→ 用预训练模型从 t₀ 去噪到 0
→ 得到较真实且仍保留输入结构的结果
```

扩散模型训练时已经学习了所有噪声时刻的去噪，因此并不强制只能从纯噪声端点开始。

## 七、为什么加噪再去噪不会原样返回

同一个带噪状态可能由许多不同干净图像产生：

$$
x_{t_0}
\not\Rightarrow
\text{唯一的 }x_0.
$$

反向模型不知道“刚才那张 guide 和那份噪声的唯一答案”，而是依据学到的自然图像先验，从：

$$
p_\theta(x_0\mid x_{t_0})
$$

中选择或采样一个合理结果。因此：

- 输出不一定等于 guide；
- 同一 guide 换随机噪声可以产生多个结果；
- 不自然的笔触和接缝可被模型重新解释成自然纹理。

## 八、为什么“先破坏”反而有用

粗糙 guide 往往位于自然图像分布之外。适量噪声会：

1. 先破坏局部尖锐笔触、拼贴边界和不自然细节；
2. 暂时保留较粗的颜色、轮廓和布局；
3. 给生成模型留出重新解释的空间；
4. 让反向过程用自然图像先验补出合理纹理。

直观比喻：噪声像一层雾。薄雾只允许小修，浓雾允许大改；适中的雾遮住粗糙细节，但仍看得见整体构图。

注意：这里加的是高斯噪声，不是普通模糊滤镜。

## 九、最关键参数 $t_0$

| $t_0$ / 噪声强度 | 保留 guide | 模型自由度 | 常见结果 |
|---|---:|---:|---|
| 很小 | 多 | 小 | 忠实，但粗糙痕迹或目标编辑可能保留/失败 |
| 中等 | 适中 | 适中 | 真实性与忠实度之间折中 |
| 很大 | 少 | 大 | 更自由、更多样，但身份和构图容易漂移 |
| 接近噪声端 | 几乎无 | 最大 | 接近普通随机生成 |

$$
t_0=0
\Rightarrow
\text{近似保留 guide},
$$

$$
0<t_0<1
\Rightarrow
\text{真实性—忠实度折中},
$$

$$
t_0\approx1
\Rightarrow
\text{近似普通生成}.
$$

论文在特定实验中观察到 $t_0\in[0.3,0.6]$ 较合适，但这不是通用答案。最优值还取决于数据域、模型、噪声日程、采样步数、Prompt、CFG、mask 和 VAE。

现代 Diffusers 的 `strength` 与起始噪声等级相关，但其具体映射取决于 Scheduler，不能普遍写成 `strength = t_0`，也不能把 `0.5` 理解成“50% 像素被替换”。

## 十、三种主要应用

1. **Stroke painting → image**：粗糙色块或草图生成自然图像；
2. **Stroke-based editing**：在原图上涂改，再把涂改自然化；
3. **Image compositing**：把一块内容粘贴到另一张图，让接缝和整体外观协调。

随机噪声不同，同一 guide 可以得到多个合理变体。

## 十一、Mask 版本

令 $M=1$ 为允许修改区域，$M=0$ 为必须保留区域。每个反向时间步都进行：

$$
x_t
=
M\odot x_t^{\text{generated}}
+
(1-M)\odot x_t^{\text{original-noised}}.
$$

其中 $x_t^{\text{original-noised}}$ 是原图在当前噪声等级下的版本。随着 $t\to0$，其噪声消失，所以 mask 外逐步回到原图。

关键点：**不是最后一步才把原图粗暴粘回去，而是在每个去噪步骤维持已知区域的一致性。** 这为下一篇 Inpainting 方法铺路。

## 十二、Stable Diffusion Img2Img

原始 SDEdit 在像素空间、没有文字 Prompt。现代 LDM/Stable Diffusion 的对应流程是：

$$
x^{(g)}
\xrightarrow{\mathcal E}
z^{(g)}_0
\xrightarrow{\text{加噪到 }t_0}
z^{(g)}_{t_0}
\xrightarrow[\text{Prompt + CFG}]{\text{去噪}}
z_{\text{out}}
\xrightarrow{\mathcal D}
x_{\text{out}}.
$$

$$
z^{(g)}_{t_0}
=
\sqrt{\bar\alpha_{t_0}}z^{(g)}_0
+
\sqrt{1-\bar\alpha_{t_0}}\epsilon.
$$

相同点：都从输入构造中间噪声状态，再接上反向生成。

不同点：现代版本在 latent space 中操作，并加入文本条件、CFG、VAE 和各种 Scheduler。VAE 本身也是有损的，所以即使噪声很低，也可能出现轻微变化。

## 十三、与已完成实验的联系

此前 Stable Diffusion 实验中的现象正是 SDEdit trade-off：

- `puppy → cat`、较低噪声起点：姿态和背景保留更多，编辑空间较小；
- 更高噪声起点：主体更容易变化，但颜色、姿态和构图更容易漂移；
- `grass → sand`、较低噪声起点失败：输入信息太强，文本条件不足以改写背景。

因此所谓 `strength` 不是简单的“画质参数”，而是：

$$
\boxed{
\text{输入约束强度}
\longleftrightarrow
\text{模型改写自由度}
}
$$

之间的旋钮。

## 十四、SDEdit 与 Inversion

| 方法 | 起点怎样得到 | 是否故意丢信息 | 主要目标 |
|---|---|---:|---|
| SDEdit | 给输入直接随机加噪 | 是 | 在忠实度和编辑自由度之间折中 |
| Stable Diffusion img2img | VAE 编码后随机加噪 | 是 | 文本控制的编辑与变体 |
| DDIM inversion | 沿近似确定性轨迹把输入推向高噪声 latent | 尽量不故意丢失 | 先重建原图，再改条件编辑 |
| VAE encoding | 把 RGB 压缩到干净 latent | 只有压缩损失 | 表示转换，不是扩散 inversion |

SDEdit 不寻找输入对应的唯一噪声，也不以高保真重建为首要目标；它主动加噪来制造编辑自由度。

## 十五、与 Rectified Flow / MM-DiT

SDEdit 是采样初始化策略，MM-DiT 是预测网络架构，二者可以组合。

若 Rectified Flow 约定数据端 $t=0$、噪声端 $t=1$：

$$
z^{(g)}_{t_0}
=
(1-t_0)z^{(g)}_0+t_0\epsilon.
$$

MM-DiT 预测速度：

$$
v_\theta(z_t,t,c),
$$

再从 $t_0$ 向 $0$ 积分：

$$
z_{t-\Delta t}
\approx
z_t-\Delta t\,v_\theta(z_t,t,c).
$$

核心 trade-off 不变：小 $t_0$ 更忠实，大 $t_0$ 更自由。严格地说，这是 **SDEdit-style flow img2img**：思想相同，但原论文运行反向 SDE，Rectified Flow 运行速度 ODE。

## 十六、训练与推理

### 预训练阶段

模型用大量真实图像和各噪声时刻训练，学习 Score 或等价的噪声/速度预测。这个阶段有反向传播和参数更新。

### SDEdit 编辑阶段

只执行：

```text
读取预训练模型 → 加噪 guide → 反向采样 → 输出
```

没有为当前编辑重新训练、没有反向传播，也没有修改参数。因此应说：

> SDEdit 对编辑任务是 training-free，但不是 model-free；它依赖昂贵的预训练生成模型。

## 十七、论文实验与证据

### 指标

- $L_2\downarrow$：输出与 guide 的像素距离，主要反映忠实度；
- KID $\downarrow$：生成集合与真实集合的分布距离，主要反映总体真实性；
- Masked LPIPS $\downarrow$：衡量 mask 外不应变化区域的感知差异；
- MTurk preference：人类在两张结果中选择哪张更真实或总体更满意。

一个指标不能包办全部目标：$L_2$ 小不等于图像一定自然，KID 小也不保证单张结果忠实。

### 主要结果

1. 人工 bedroom stroke 实验中，SDEdit 的 $L_2=32.55$，低于所比较的 GAN inversion 基线。与 StyleGAN2-ADA 成对比较时，98.09% 的判断选择 SDEdit 更真实，91.72% 选择 SDEdit 总体更满意。
2. 模拟 stroke 实验中，SDEdit 的 Bedroom KID 为 0.0030、Church KID 为 0.0156，并同时获得较低 $L_2$，支持其在这些设置中兼顾真实性和忠实度。
3. 人脸 compositing 中，SDEdit 的 $L_2=21.70$、masked LPIPS $=0.03$。但与 e4e 比较时，只有 43.67% 认为 SDEdit 更真实，即 e4e 在该项 realism 上更高；SDEdit 的总体满意度仍以 66% 胜出，说明它更好地平衡了真实性与忠实度。
4. Stroke-based editing 主要是定性图片，没有与生成、拼贴同等强度的独立定量验证。

数字口径：98.09% 和 91.72% 是**特定基线下的成对选择比例**，不是准确率，也不是“性能提高了这么多”。

用户研究报告的是判断次数，而不是独立参与者人数；论文没有给置信区间、显著性检验或完整参与者统计，结论不能无限外推。

## 十八、Proposition 1

论文在 Score 范数有界等假设下，给出 guide 与输出距离的高概率上界：

$$
\left\|
x^{(g)}-operatorname{SDEdit}(x^{(g)};t_0,\theta)
\right\|_2^2
\le
\sigma^2(t_0)
\left[
C\sigma^2(t_0)+d
+2\sqrt{-d\log\delta}
-2\log\delta
\right].
$$

- $C$：Score 范数上界；
- $d$：图像向量维度，不是 Attention 中的 $d_k$；
- $\delta$：允许失败的小概率；
- $\sigma(t_0)$ 增大时，右侧上界增大，表示输出**被允许**离 guide 更远。

不能推出：实际距离一定达到上界、真实感必然单调提高、或存在通用最优 $t_0$。

### 技术审读提醒

附录 Equation (8) 按文稿印刷，直接把 $\|A+B\|^2$ 上界成 $\|A\|^2+\|B\|^2$；一般情况下至少需要类似：

$$
\|A+B\|^2
\le
2\|A\|^2+2\|B\|^2,
$$

除非另有正交或交叉项非正条件。论文没有充分说明这些条件，随机轨迹相关性也需要额外论证。因此该证明至少存在常数或论证步骤上的缺口。汇报时可说：

> 作者尝试用距离上界解释 trade-off；更可靠的支持来自 Figure 3、KID 和人类评价实验。

不要求掌握附录的卡方尾界推导。

## 十九、论文真正支持与不支持的结论

论文支持：

- 中间加噪再反向去噪可以把粗糙 guide 转成更自然的图像；
- $t_0$ 经验性地控制真实性—忠实度权衡；
- 无需为三类任务重新训练条件模型或逐图做 GAN inversion；
- 同一 guide 可通过随机性得到多个结果；
- mask 采样可持续保护未编辑区域。

论文没有证明：

- 对任意领域都无需合适的预训练模型；
- 原始方法支持文字 Prompt 或 Stable Diffusion；
- $t_0=0.5$ 对所有任务最优；
- 在每一项 realism 比较中都获胜；
- 输出严格保持身份、语义或所有 mask 外像素；
- 优于后来出现的现代扩散编辑方法。

## 二十、局限

1. $t_0$ 需要人为调节，没有可靠的逐图自动最优方案；
2. 噪声太大会丢失身份、结构和细节，太小又可能编辑失败；
3. 结果受预训练模型数据域和先验限制；
4. 原始多步 SDE 采样较慢；
5. 原始方法不理解高级文本指令；
6. 人类评价与定量指标都有局限，不能代表所有用户和任务。

## 二十一、学习深度

### 必须会

- 用一句话说清 SDEdit；
- 读懂 $\mathcal N(\mu,\sigma^2I)$ 与对应采样式；
- 写出 guide 的 DDPM 加噪公式；
- 推导高斯 Score $-\epsilon/\sigma_t$；
- 解释为何可以从中间时刻开始；
- 解释 $t_0$ 的真实性—忠实度权衡；
- 区分 SDEdit、img2img、VAE encoding 和 DDIM inversion；
- 说明原始方法如何迁移到 LDM 与 Rectified Flow；
- 准确解释主要实验和 98.09% 的口径。

### 理解结论即可

- 反向 SDE 的严格时间反演；
- Wiener process 与 Itô calculus；
- Euler-Maruyama 收敛证明；
- Proposition 1 的完整概率推导；
- VE/VP 连续解与 mask 算法逐行证明。

## 二十二、汇报问答与标准答案

### 1. SDEdit 的核心创新是什么？

不从纯噪声开始，而是把用户 guide 加到合适的中间噪声等级，再调用预训练扩散模型的后半段反向过程，从而无需任务专用训练或逐图 inversion。

### 2. 为什么要先破坏输入？

适量噪声削弱不自然的笔触、接缝和细节，却仍保留粗结构；反向模型再用自然图像先验重建合理纹理。

### 3. $t_0$ 控制什么？

控制输入信息与模型先验的相对强弱：小 $t_0$ 更忠实，大 $t_0$ 更自由、更易改变，但也更容易漂移。

### 4. 为什么可以从中间开始去噪？

扩散模型训练时见过所有噪声时刻，并把当前状态和时间作为输入，因此可以从任意已有噪声状态继续反向采样。

### 5. 为什么不会还原成原图？

带噪状态对应许多可能的干净图，模型按学到的条件分布采样，而不是知道并撤销唯一的原始噪声。

### 6. Score 是什么？

是 $\nabla_x\log p_t(x)$，表示当前带噪图像朝哪个方向移动会进入更高概率区域；对高斯加噪，它与噪声预测只差比例和负号。

### 7. Training-free 是否表示完全没有训练？

不是。编辑时不重新训练，但仍依赖已经训练好的扩散/Score 模型。

### 8. 原始 SDEdit 与 Stable Diffusion img2img 有何区别？

原论文主要在像素空间、没有文本条件；Stable Diffusion 先用 VAE 编码，在 latent 中加噪，并用 Prompt、CFG 和 Scheduler 去噪。

### 9. SDEdit 与 DDIM inversion 有何区别？

SDEdit 主动随机加噪以丢失部分信息、获得编辑自由度；inversion 尽量找到能重建输入的生成轨迹。

### 10. Mask 如何保护未编辑区域？

每个反向步骤都把 mask 外替换成原图在当前噪声等级的版本，而不是最后才粘回去。

### 11. 98.09% 表示什么？

在人工 bedroom stroke 实验中，SDEdit 与 StyleGAN2-ADA 成对比较时，98.09% 的人类判断选择 SDEdit 更真实；不是准确率或相对提升。

### 12. SDEdit 是什么类型的方法？

它是利用预训练生成模型的推理/采样方法，不是 backbone、训练损失或新数据集。

### 13. 论文最重要的局限是什么？

$t_0$ 需要手工权衡，忠实度和编辑自由度不能同时最大化，而且效果受预训练模型数据域限制。

## 公式速记

$$
x\sim\mathcal N(\mu,\sigma^2I)
\iff
x=\mu+\sigma\epsilon.
$$

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{x_t-\alpha_tx_0}{\sigma_t^2}
=
-\frac{\epsilon}{\sigma_t}.
$$

$$
x^{(g)}_{t_0}
\xrightarrow{\text{reverse process}}
x_{\text{out}}.
$$

$$
z^{(g)}_{t_0}
=
(1-t_0)z^{(g)}_0+t_0\epsilon
\qquad
\text{(Rectified Flow convention)}.
$$

## 相关笔记

- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)
- [08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)
- [08-Experiment-Stable Diffusion 文生图、DDIM 与图像编辑](#/notes/stable-diffusion-lab)
- [10-Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT）](#/notes/mmdit)
- [12-Blended Latent Diffusion](#/notes/blended-latent-diffusion)
- [14-FlowEdit - Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](#/notes/flowedit)
