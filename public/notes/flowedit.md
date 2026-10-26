# 14 FlowEdit：无需反演的文本图像编辑

## 来源与定位

- 论文：*FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models*
- 作者：Vladimir Kulikov、Matan Kleiner、Inbar Huberman-Spiegelglas、Tomer Michaeli
- 正式发表：ICCV 2025，Best Student Paper
- [官方项目页](https://matankleiner.github.io/flowedit/)
- [arXiv](https://arxiv.org/abs/2412.08629)
- [官方代码](https://github.com/fallenshock/FlowEdit)

> [!note] 书目信息修正
> 老师清单将它写作“CVPR 2025 best paper”；正式出版信息是 **ICCV 2025 Best Student Paper**。论文内容与我们的学习顺序不受影响。

$$
\boxed{
\text{FlowEdit 不先把原图反演成噪声，而是利用“目标速度场}-\text{源速度场”直接推动原图变成目标图。}
}
$$

它是一个**推理时编辑算法**：不训练新模型、不更新参数、不做逐图优化，也不修改模型内部 Attention；它反复调用冻结的文本条件流模型来估计编辑方向。

## 一、它接在前面知识链的哪里

```text
Rectified Flow：学习“噪声分布 ↔ 图像分布”的速度场
→ MM-DiT / SD3、FLUX：用文本条件控制流模型的速度预测
→ SDEdit：给原图加一次噪，再按目标文本去噪；简单但容易改坏结构
→ BLD：用人工 Mask 限制哪里能改
→ Prompt-to-Prompt：修改 Cross-Attention，生成图编辑很好；真实图通常仍需 inversion
→ FlowEdit：不做 inversion，也不碰内部特征，直接构造“源图分布 → 目标图分布”的编辑 ODE
```

最宏观的理解是：

- 生成模型原本会回答：“处在位置 $z$、时刻 $t$、文本条件为 $c$ 时，应该沿什么速度移动？”
- FlowEdit 围绕同一个当前编辑位移，构造一对共享噪声的源/目标查询点；
- 两个回答中共有的“保留内容、处理噪声”部分会部分抵消，剩下的差值主要体现“文本发生变化后应该怎样改”。

## 二、先解决一个记号陷阱

我们学习原始 Rectified Flow 时使用：

$$
t=0:\text{噪声},\qquad t=1:\text{数据}.
$$

FlowEdit 论文为了与扩散模型习惯一致，反过来写：

$$
\boxed{
t=0:\text{数据/图像},\qquad t=1:\text{高斯噪声}.
}
$$

因此本文的标准流插值是：

$$
Z_t=(1-t)X_0+tX_1,
$$

其中 $X_0$ 是图像，$X_1\sim\mathcal N(0,I)$ 是噪声。生成或编辑时从较大的 $t$ 向 $0$ 积分。两种写法令 $t'=1-t$ 就能互换，理论没有矛盾。

## 三、基线：传统 ODE inversion 怎样编辑真实图

给定：

- 原图 $X^{\mathrm{src}}$；
- 源描述 $c_{\mathrm{src}}$；
- 目标描述 $c_{\mathrm{tar}}$。

冻结的文本条件流模型给出速度：

$$
V^{\mathrm{src}}(z,t)=V(z,t,c_{\mathrm{src}}),
\qquad
V^{\mathrm{tar}}(z,t)=V(z,t,c_{\mathrm{tar}}).
$$

传统方法分两段走。

### 1. 先反演：原图走到噪声

$$
dZ_t^{\mathrm{src}}
=V^{\mathrm{src}}(Z_t^{\mathrm{src}},t)\,dt,
\qquad
Z_0^{\mathrm{src}}=X^{\mathrm{src}}.
$$

从 $t=0$ 积分到 $t=1$，得到与原图对应的噪声 $Z_1^{\mathrm{src}}$。

### 2. 再生成：同一噪声走到目标图

$$
dZ_t^{\mathrm{tar}}
=V^{\mathrm{tar}}(Z_t^{\mathrm{tar}},t)\,dt,
\qquad
Z_1^{\mathrm{tar}}=Z_1^{\mathrm{src}}.
$$

从 $t=1$ 反向积分到 $t=0$，得到编辑图 $Z_0^{\mathrm{tar}}$。

完整路线是：

$$
\text{原图}
\xrightarrow[\text{源 prompt}]{\text{inversion}}
\text{噪声}
\xrightarrow[\text{目标 prompt}]{\text{sampling}}
\text{编辑图}.
$$

问题在于：经由公共高斯噪声会绕远路；数值反演也可能有重建误差。即使反演完全精确，经噪声建立的源—目标样本配对也未必是“改动最小”的配对。

## 四、论文第一步：把“两段绕行”改写成一条直接路径

论文先不发明新算法，而是重新观察传统 inversion。

定义一条新的直接轨迹：

$$
\boxed{
Z_t^{\mathrm{inv}}
=Z_0^{\mathrm{src}}+Z_t^{\mathrm{tar}}-Z_t^{\mathrm{src}}.
}
\tag{5}
$$

它看似突然，其实只是在记录“目标轨迹相对源轨迹移动了多少”：

$$
Z_t^{\mathrm{inv}}-Z_0^{\mathrm{src}}
=Z_t^{\mathrm{tar}}-Z_t^{\mathrm{src}}.
$$

### 1. 检查两个端点

在 $t=1$，两条轨迹使用同一噪声：

$$
Z_1^{\mathrm{tar}}=Z_1^{\mathrm{src}}.
$$

所以：

$$
Z_1^{\mathrm{inv}}=Z_0^{\mathrm{src}}.
$$

在 $t=0$：

$$
Z_0^{\mathrm{inv}}
=Z_0^{\mathrm{src}}+Z_0^{\mathrm{tar}}-Z_0^{\mathrm{src}}
=Z_0^{\mathrm{tar}}.
$$

因此当 $t:1\rightarrow0$ 时，这条新轨迹真的从原图走到编辑图。

### 2. 对式（5）求微分

$Z_0^{\mathrm{src}}$ 是固定原图，微分为 $0$：

$$
dZ_t^{\mathrm{inv}}
=dZ_t^{\mathrm{tar}}-dZ_t^{\mathrm{src}}.
$$

代入两条 ODE：

$$
\boxed{
dZ_t^{\mathrm{inv}}
=\left[
V^{\mathrm{tar}}(Z_t^{\mathrm{tar}},t)
-V^{\mathrm{src}}(Z_t^{\mathrm{src}},t)
\right]dt.
}
\tag{6}
$$

定义速度差：

$$
V_t^\Delta(Z_t^{\mathrm{src}},Z_t^{\mathrm{tar}})
=V^{\mathrm{tar}}(Z_t^{\mathrm{tar}},t)
-V^{\mathrm{src}}(Z_t^{\mathrm{src}},t).
$$

再由式（5）整理出：

$$
Z_t^{\mathrm{tar}}
=Z_t^{\mathrm{inv}}+Z_t^{\mathrm{src}}-Z_0^{\mathrm{src}},
$$

于是：

$$
\boxed{
dZ_t^{\mathrm{inv}}
=V_t^\Delta
\left(
Z_t^{\mathrm{src}},
Z_t^{\mathrm{inv}}+Z_t^{\mathrm{src}}-Z_0^{\mathrm{src}}
\right)dt,
\quad Z_1^{\mathrm{inv}}=Z_0^{\mathrm{src}}.
}
\tag{7}
$$

这一步的重要性是：**传统 inversion 隐含着一条从原图直接到编辑图的 ODE；它的速度正是目标速度减源速度。**

## 五、为什么直接路径看起来没有噪声

式（5）包含两条带噪轨迹之差：

$$
Z_t^{\mathrm{tar}}-Z_t^{\mathrm{src}}.
$$

二者来自同一个噪声端点，含有高度相关的公共噪声。相减后公共部分大量抵消，保留下来的主要是“目标条件相对源条件造成的改变”。因此：

- $Z_t^{\mathrm{src}}$、$Z_t^{\mathrm{tar}}$ 本身可能很吵；
- $Z_t^{\mathrm{inv}}$ 却能始终保持像一张可辨认的图；
- 高噪声阶段先改变低频、整体结构；
- 随 $t$ 降低，再逐渐改变高频纹理和细节。

这就是论文所说的 **noise-free、coarse-to-fine path**。这里的 noise-free 是“主编辑轨迹不呈现标准噪声图”，不是算法完全不使用随机噪声。

## 六、真正的难点：式（7）仍然需要 inversion

式（7）虽然描述了直接路径，但每一步仍要知道真实的 $Z_t^{\mathrm{src}}$。要得到整条 $Z_t^{\mathrm{src}}$，还是得先对原图做 source inversion。

FlowEdit 的关键问题因此是：

> 能不能不计算原图的那一条确定反演轨迹，而只构造“在每个时刻分布正确”的源侧带噪样本来询问模型？

答案是可以。

## 七、FlowEdit 的核心构造

### 1. 不反演，直接制造源侧带噪点

在每个时刻 $t$ 独立取：

$$
N_t\sim\mathcal N(0,I),
$$

并构造：

$$
\boxed{
\widehat Z_t^{\mathrm{src}}
=(1-t)Z_0^{\mathrm{src}}+tN_t.
}
\tag{8}
$$

它就是把原图与高斯噪声按当前噪声等级做线性混合：

- $t=0$：$\widehat Z_0^{\mathrm{src}}=Z_0^{\mathrm{src}}$；
- $t\approx1$：主要是高斯噪声；
- 每个时刻的边缘分布与流模型训练时见过的源侧中间分布一致。

重点：FlowEdit 不需要这些点在时间上组成一条真实 inversion 轨迹；它只需要当前 $t$ 的样本分布正确。

### 2. 用平行四边形构造目标侧查询点

设当前主编辑状态为 $Z_t^{\mathrm{FE}}$，定义：

$$
\boxed{
\widehat Z_t^{\mathrm{tar}}
=Z_t^{\mathrm{FE}}+\widehat Z_t^{\mathrm{src}}-Z_0^{\mathrm{src}}.
}
$$

等价地：

$$
\widehat Z_t^{\mathrm{tar}}-\widehat Z_t^{\mathrm{src}}
=Z_t^{\mathrm{FE}}-Z_0^{\mathrm{src}}.
$$

右边是主轨迹当前已经完成的“编辑位移”。所以该公式把同样的位移复制到带噪源点上，得到与它共享噪声、但位于目标侧的查询点。四个点构成平行四边形。

### 3. 分别询问源、目标速度，再做差

$$
\widehat V_t^\Delta
=V^{\mathrm{tar}}(\widehat Z_t^{\mathrm{tar}},t)
-V^{\mathrm{src}}(\widehat Z_t^{\mathrm{src}},t).
$$

不要误写成在同一个输入点上算两个速度。FlowEdit 用的是：

- 源 prompt 查询源侧点 $\widehat Z_t^{\mathrm{src}}$；
- 目标 prompt 查询目标侧点 $\widehat Z_t^{\mathrm{tar}}$；
- 两点共享同一份噪声，且差值等于当前编辑位移。

### 4. 对随机噪声取平均，得到主 ODE

不同 $N_t$ 会给出略有不同的方向，所以论文取条件期望：

$$
\boxed{
dZ_t^{\mathrm{FE}}
=\mathbb E\!\left[
V_t^\Delta\!\left(
\widehat Z_t^{\mathrm{src}},
Z_t^{\mathrm{FE}}+\widehat Z_t^{\mathrm{src}}-Z_0^{\mathrm{src}}
\right)
\middle| Z_0^{\mathrm{src}}
\right]dt,
\quad
Z_1^{\mathrm{FE}}=Z_0^{\mathrm{src}}.
}
\tag{9}
$$

这就是全文最核心公式。

## 八、把核心算法翻译成人话

从高时间步向低时间步重复：

1. 保留一张主图 $Z_t^{\mathrm{FE}}$，开始时就是原图；
2. 当前时刻重新采一份噪声 $N_t$；
3. 用式（8）做出“原图在当前噪声等级下的样子” $\widehat Z_t^{\mathrm{src}}$；
4. 把主图已经发生的编辑位移搬到该带噪点上，得到 $\widehat Z_t^{\mathrm{tar}}$；
5. 用源 prompt 问模型源点该怎么走；
6. 用目标 prompt 问模型目标点该怎么走；
7. 两个速度相减，得到当前编辑变化；
8. 用 Euler 法更新主图；
9. 直到 $t=0$，得到编辑结果。

离散更新为：

$$
\boxed{
Z_{t_{i-1}}^{\mathrm{FE}}
=Z_{t_i}^{\mathrm{FE}}
+(t_{i-1}-t_i)\,\overline V_{t_i}^\Delta.
}
$$

因为 $t_{i-1}<t_i$，所以 $(t_{i-1}-t_i)<0$。不要把它误当成普通“加一个正步长”；符号已经由从 $1$ 向 $0$ 积分决定。

### 一个一维手算直觉

下面只是解释“共同噪声为何会抵消”的理想化例子。假设源值为 $2$、目标值为 $3$、共同噪声为 $1$，理想直线流的两条速度为：

$$
V^{\mathrm{src}}=1-2=-1,
\qquad
V^{\mathrm{tar}}=1-3=-2.
$$

因此：

$$
V^\Delta=(-2)-(-1)=-1.
$$

共同噪声 $1$ 在相减时消失。若反向积分的一小步为 $\Delta t=-0.2$，则：

$$
Z_{\mathrm{new}}^{\mathrm{FE}}
=2+(-0.2)(-1)=2.2,
$$

主状态从 $2$ 向目标 $3$ 移动了一点。真实模型不知道具体目标图“等于 3”，而是由目标文本条件速度场提供分布层面的方向。

## 九、四类变量必须分清

| 变量 | 是否带噪 | 是否是主编辑结果 | 来源 |
|---|---|---|---|
| $Z_0^{\mathrm{src}}$ | 否 | 固定原图 | 用户输入 |
| $Z_t^{\mathrm{FE}}$ | 主路径保持图像状 | 是 | ODE 逐步更新 |
| $\widehat Z_t^{\mathrm{src}}$ | 是 | 否 | 原图与随机噪声线性混合 |
| $\widehat Z_t^{\mathrm{tar}}$ | 是 | 否 | 源侧带噪点加当前编辑位移 |

帽子 $\widehat{\phantom Z}$ 的意思可以记成“临时带噪查询点”。真正要输出的是无帽子的 $Z_0^{\mathrm{FE}}$。

## 十、为什么要取期望，实际怎样近似

式（9）的期望无法精确积分，实际用 Monte Carlo 平均：

$$
\overline V_t^\Delta
=\frac1{n_{\mathrm{avg}}}
\sum_{j=1}^{n_{\mathrm{avg}}}
\left[
V^{\mathrm{tar}}(\widehat Z_{t,j}^{\mathrm{tar}},t)
-V^{\mathrm{src}}(\widehat Z_{t,j}^{\mathrm{src}},t)
\right].
$$

- $n_{\mathrm{avg}}$ 越大，随机方向越稳定，但模型调用次数线性增加；
- 论文主实验取 $n_{\mathrm{avg}}=1$；
- 因为源、目标查询点共享同一份噪声，共同噪声影响在速度差中会部分抵消；
- 时间步足够多时，还能利用跨时间步的平均效果；
- 官方实现会在不同时间步重新采样噪声；“同一噪声”只指同一时间步内配对的源、目标查询点；
- 附录显示：$T=50,n_{\max}=33$ 时增大 $n_{\mathrm{avg}}$ 收益很小；仅用 $T=10,n_{\max}=7$ 时，增大平均次数才明显有益。

若 $n_{\mathrm{avg}}$ 很小，算法仍是随机的：换 seed 会得到不同编辑，有时也会产生额外肢体或结构错误。

## 十一、两个强度旋钮：$n_{\max}$ 与 $n_{\min}$

### 1. $n_{\max}$：决定从多早开始编辑

设总离散步数为 $T$，算法初始化：

$$
Z_{t_{n_{\max}}}^{\mathrm{FE}}=X^{\mathrm{src}}.
$$

- $n_{\max}=T$：走完整直接路径，通常编辑最强；
- $n_{\max}<T$：跳过最前面的高噪声步骤，路径更短，通常编辑更弱、原图保持更多。

注意主变量依然从干净原图开始。这里的 $t$ 是编辑路径的索引，不表示 $Z_t^{\mathrm{FE}}$ 必须是一张标准带噪图。

### 2. $n_{\min}$：最后多少步改用普通目标采样

FlowEdit 很擅长保护细节，有时保护过头，画风、高频纹理改不动。为此附录引入 $n_{\min}$：

- 从 $n_{\max}$ 到 $n_{\min}$ 使用 FlowEdit；
- 当 $i<n_{\min}$ 时，停止做源—目标速度差；
- 改用目标 prompt 的普通生成轨迹走到 $t=0$。

因此：

- $n_{\min}=0$：全程 FlowEdit，结构保持最强；
- $n_{\min}$ 较大：更早释放给目标模型，风格与细节变化更强，但更偏离原图。

它对风格编辑有效，但论文承认对大范围背景变化和姿势变化仍常常不够。

## 十二、FlowEdit 为什么可能比 inversion 改得更少

### 1. “到目标分布”不等于“配对合理”

假设源分布有两类样本，目标分布也有两类样本。理想编辑应把每个源样本送到语义相符且距离较近的目标模式。

传统 inversion 先把所有源样本送进共同高斯分布，再从噪声送到目标。共享噪声所建立的配对可能把一个源样本送到较远的目标模式，导致大幅改变。

FlowEdit 直接利用局部速度差，倾向于把源样本送到更近的目标区域，所以运输代价更低、结构保存更好。

### 2. 但它不是已证明的最优运输

作者明确说明：FlowEdit 是启发式方法：

- 不保证精确得到目标分布；
- 不保证就是 optimal transport map；
- “更短、更少改动”是玩具实验与图像实验支持的经验现象，不是普遍数学定理。

## 十三、与前面三种编辑方法的本质区别

| 方法 | 从哪里开始 | 主要控制手段 | 是否需要 Mask | 真实图是否需 inversion |
|---|---|---|---|---|
| SDEdit | 原图加到某一噪声等级 | 只按目标条件反向去噪 | 否 | 不需精确 inversion |
| BLD | 原图背景轨迹 + 编辑轨迹 | 每步按人工 Mask 融合 latent | 是 | 否 |
| Prompt-to-Prompt | 同一噪声的源/目标生成分支 | 注入或重加权 Attention Map | 核心不需；LocalBlend 可自动生成 | 生成图不需；真实图通常需要 |
| FlowEdit | 直接以真实原图为主状态 | 目标速度减源速度，构造直接 ODE | 否 | 否 |

### FlowEdit 与 SDEdit

SDEdit 在某个固定噪声等级把原图加噪一次，然后只听目标 prompt 去噪；噪声强度决定自由度。FlowEdit 在每个时刻都临时构造源、目标带噪查询点，用两个条件速度的差异隔离“该改什么”，因此结构保护更明确。

### FlowEdit 与 Prompt-to-Prompt

Prompt-to-Prompt 打开模型内部，直接控制 Cross-Attention；FlowEdit 把模型当作只需输入图、时间和文本就能查询的黑箱，不注入 Attention、Feature 或 KV。所以 FlowEdit 更容易跨 SD3、FLUX 等不同流模型架构迁移。

“model agnostic”应准确说成：**对文本条件流模型的内部架构较不敏感**，不是任何扩散模型或任何生成模型都能直接使用。

## 十四、实验到底证明了什么

### 1. 模型与数据

- 在 Stable Diffusion 3 Medium 与 FLUX.1-dev 上测试；
- 超过 70 张 $1024\times1024$ 真实图，来自 DIV2K 与可授权在线图片；
- 经 LLaVA 生成后人工修订源 prompt，再人工设计多个目标 prompt；
- 总计超过 250 个文本—图像编辑对；
- SD3：$T=50,n_{\max}=33$，源/目标 CFG 为 $3.5/13.5$；
- FLUX：$T=28,n_{\max}=24$，源/目标 conditioning scale 为 $1.5/5.5$；
- 主实验均用 $n_{\mathrm{avg}}=1$。

### 2. 对比方法

- SD3：SDEdit、ODE inversion、iRFDS；
- FLUX：SDEdit、ODE inversion、RF-Inversion、RF-Edit。

论文没有把所有旧扩散编辑方法都搬来比较，因为它们不一定能公平、直接地适配流模型。

### 3. 指标含义

| 指标 | 趋势 | 主要衡量 |
|---|---:|---|
| CLIP-T | 越高越好 | 编辑图与目标文本一致性 |
| CLIP-I | 越高越好 | 编辑图与原图整体相似性 |
| LPIPS | 越低越好 | 感知特征距离 |
| DINO | 越高越好 | 结构/语义特征相似性 |
| DreamSim | 越低越好 | 人类感知相关的图像差异 |

编辑没有唯一满分指标：完全复制原图会有很高相似度却没有完成编辑；完全重画可符合文本却丢掉结构。因此要看“文本服从度—原图保持度”的折中，而不能只挑一个最高数字。

### 4. 主要定量结论

在 SD3 汇总表中，FlowEdit 的：

$$
\text{CLIP-T}=0.344,
\quad \text{CLIP-I}=0.872,
\quad \text{LPIPS}=0.181,
\quad \text{DINO}=0.719.
$$

它取得该表最佳 CLIP-T、LPIPS、DINO，但并非每一个指标都第一：SDEdit 0.2 的 CLIP-I 与 DreamSim 更好。FLUX 表中 FlowEdit 的 DINO 最好，其他若干指标与 RF-Edit、SDEdit 各有领先。因此严谨结论应是：**FlowEdit 给出了很强、常处于 Pareto 前沿的文本一致性与结构保持折中**，而不是“所有数字全面碾压”。

### 5. 猫到狗运输实验

作者先用 SD3 生成 1000 张猫，再分别用 inversion 和 FlowEdit 编辑为狗，并与直接生成的 1000 张狗比较：

| 方法 | 潜空间均方运输代价 ↓ | LPIPS ↓ | FID ↓ | KID ↓ |
|---|---:|---:|---:|---:|
| Editing by inversion | 2239 | 0.25 | 55.88 | 0.023 |
| FlowEdit | 1376 | 0.15 | 51.14 | 0.017 |

这支持 FlowEdit 在该受控实验中建立了更短的源—目标配对，同时没有明显牺牲目标分布质量。但这仍不是对任意真实数据的最优运输证明。

## 十五、计算量与“无需优化”的准确含义

每个时间步、每个 Monte Carlo 样本至少要查询：

1. 一次源条件速度；
2. 一次目标条件速度。

大致模型前向次数为：

$$
2\times n_{\mathrm{avg}}\times(n_{\max}-n_{\min})
$$

再加 $n_{\min}$ 次普通目标采样（若启用）。因此它不是一次前向，也不一定比普通文生图便宜。

“optimization-free”只表示：

- 没有 Loss 最小化循环；
- 没有反向传播；
- 不更新模型、文本 embedding 或 latent 参数。

它仍要进行多步 ODE 数值积分。

## 十六、局限与证据边界

1. **过度保结构**：大范围背景替换和姿势变化经常不充分。
2. **随机性**：$n_{\mathrm{avg}}$ 小时不同 seed 结果不同，也可能出现伪影。
3. **不是最优运输保证**：更低 transport cost 是实验证据，不是一般定理。
4. **并非任意模型通用**：需要能按文本条件查询速度场的预训练 flow model。
5. **计算仍不低**：每步通常需要源、目标两次模型预测。
6. **超参数依赖模型和任务**：$n_{\max},n_{\min}$、源/目标 CFG 都影响结果。
7. **评测规模有限**：约 70 多张真实图、250 多个编辑对，prompt 有人工设计成分，论文没有大规模用户研究。
8. **指标不完美**：图像相似度可能奖励“根本没改”，文本指标又可能忽视身份和细节。
9. **源 prompt 并非完全关键**：附录中换错或留空源 prompt，平均指标仅小幅变化；这说明共享图像内容对速度差有很大作用，但不能据此保证所有编辑都无需准确源描述。

## 十七、必须掌握与了解即可

### 必须掌握

- [ ] 能说明 FlowEdit 的任务是文本引导真实图编辑，不是训练生成 backbone。
- [ ] 能说明传统 inversion 为何是“图像 → 噪声 → 目标图”的绕行。
- [ ] 能解释 FlowEdit 论文使用 $t=0$ 图像、$t=1$ 噪声的记号。
- [ ] 能推导式（5）到式（7），并解释 $V^{\mathrm{tar}}-V^{\mathrm{src}}$。
- [ ] 能写出 $\widehat Z_t^{\mathrm{src}}=(1-t)Z_0^{\mathrm{src}}+tN_t$。
- [ ] 能解释平行四边形关系与 $\widehat Z_t^{\mathrm{tar}}$。
- [ ] 能说明主轨迹无帽、辅助查询点有帽；“noise-free”不等于不使用噪声。
- [ ] 能完整复述每个时间步的两个模型查询和一次更新。
- [ ] 能解释 $n_{\max}$、$n_{\min}$、$n_{\mathrm{avg}}$ 的作用。
- [ ] 能比较 FlowEdit、SDEdit、P2P 和 BLD。
- [ ] 能说明实验支持什么、没有证明什么。

### 了解即可

- DDS、PDS 更新式与 FlowEdit 在噪声预测参数化下的逐项差异；
- 附录对 step size 缩放系数的全部实验；
- 每张定性图的精确 CFG 与 seed；
- SD3、FLUX 调度器的代码接口细节；
- 玩具高斯混合例子的所有坐标与解析解。

## 十八、汇报问答与标准答案

### 1. FlowEdit 的一句话贡献是什么？

它利用预训练文本条件流模型中“目标条件速度减源条件速度”构造一条由真实原图直接通往目标分布的 ODE，从而不做反演、不逐图优化、也不修改模型内部特征，就能完成结构保持较好的文本图像编辑。

### 2. 为什么传统 inversion 即使精确也可能编辑不好？

精确只保证图像能到某个噪声并从该噪声生成目标图，不保证这个共同噪声把源样本配到了距离最近、语义最合适的目标样本。错误的 coupling 会产生较大的无关改动。

### 3. 式（5）为什么确实连接源图和目标图？

在 $t=1$，源和目标轨迹共享同一噪声，二者相减为零，所以直接路径等于源图；在 $t=0$，源轨迹等于源图，恰好被减掉，直接路径等于目标图。

### 4. 为什么速度要相减？

因为直接路径是“目标轨迹减源轨迹再加固定源图”。对它求微分时固定源图消失，剩下 $dZ^{\mathrm{tar}}-dZ^{\mathrm{src}}$，代入两条 ODE 就得到目标速度减源速度。

### 5. FlowEdit 是否简单地在同一张图上换两个 prompt 算速度差？

不是。源速度在 $\widehat Z_t^{\mathrm{src}}$ 上计算，目标速度在 $\widehat Z_t^{\mathrm{tar}}$ 上计算；后者等于前者加当前主轨迹的编辑位移。两个查询点共享噪声并形成平行四边形。

### 6. 为什么能跳过 inversion？

FlowEdit 不需要真实源 inversion 的逐时刻对应点，只需要模型训练分布中的源侧中间样本。式（8）可直接把原图与新高斯噪声混合，得到当前时刻边缘分布正确的查询点。

### 7. 为什么主路径叫 noise-free，算法却又加了噪声？

加噪只发生在临时查询点 $\widehat Z_t^{\mathrm{src}},\widehat Z_t^{\mathrm{tar}}$。被连续更新并最终输出的 $Z_t^{\mathrm{FE}}$ 是源到目标的图像状直接路径，并不先退化成一张纯噪声图。

### 8. 期望有什么作用？

一次随机噪声只给出编辑方向的 Monte Carlo 估计。对多份噪声求平均可以消除偶然波动，更接近分布层面的平均速度差；实际因共享噪声抵消和跨时间步平均，论文用 $n_{\mathrm{avg}}=1$ 也能工作。

### 9. $n_{\max}$ 与 $n_{\min}$ 有何不同？

$n_{\max}$ 控制是否跳过高噪声端的早期编辑，越大通常编辑越强；$n_{\min}$ 控制低噪声末段有多少步改用普通目标采样，越大通常风格和细节自由度越高，但原图保持更弱。

### 10. FlowEdit 与 SDEdit 最本质的区别是什么？

SDEdit 把原图加到一个噪声等级后，只使用目标条件反向生成；FlowEdit 每个时刻都构造共享噪声的源/目标查询点，并用两者速度差来提取编辑变化。

### 11. FlowEdit 与 Prompt-to-Prompt 最本质的区别是什么？

P2P 进入网络内部修改 Cross-Attention Map，真实图通常还要 inversion；FlowEdit 不控制 Attention 或 Feature，只把流模型当速度查询器，并直接从真实图开始构造编辑 ODE。

### 12. 它是否训练或反向传播？

没有。模型参数完全冻结；算法只有随机采样、前向预测、速度相减与 Euler 更新。“无需优化”不等于“只需一步”。

### 13. 实验最稳妥的结论是什么？

在 SD3、FLUX 和论文所测数据上，FlowEdit 通常在目标文本一致性与源图结构保持之间取得很强的折中，并在受控猫到狗实验中降低运输代价；实验不证明它对所有编辑都是最优，也不证明所有指标都第一。

### 14. 最主要的失败场景是什么？

需要大范围重构的任务，例如显著改变姿势或整个背景。方法的优势——强结构保持——在此反而成为限制。

## 十九、公式速记

传统 inversion：

$$
dZ_t^{\mathrm{src}}=V^{\mathrm{src}}(Z_t^{\mathrm{src}},t)dt,
\qquad
dZ_t^{\mathrm{tar}}=V^{\mathrm{tar}}(Z_t^{\mathrm{tar}},t)dt.
$$

直接路径与速度差：

$$
Z_t^{\mathrm{inv}}
=Z_0^{\mathrm{src}}+Z_t^{\mathrm{tar}}-Z_t^{\mathrm{src}},
$$

$$
dZ_t^{\mathrm{inv}}
=\left[V^{\mathrm{tar}}(Z_t^{\mathrm{tar}},t)
-V^{\mathrm{src}}(Z_t^{\mathrm{src}},t)\right]dt.
$$

FlowEdit 两个查询点：

$$
\widehat Z_t^{\mathrm{src}}
=(1-t)Z_0^{\mathrm{src}}+tN_t,
\qquad N_t\sim\mathcal N(0,I),
$$

$$
\widehat Z_t^{\mathrm{tar}}
=Z_t^{\mathrm{FE}}+\widehat Z_t^{\mathrm{src}}-Z_0^{\mathrm{src}}.
$$

FlowEdit ODE：

$$
dZ_t^{\mathrm{FE}}
=\mathbb E\left[
V^{\mathrm{tar}}(\widehat Z_t^{\mathrm{tar}},t)
-V^{\mathrm{src}}(\widehat Z_t^{\mathrm{src}},t)
\mid Z_0^{\mathrm{src}}
\right]dt.
$$

## 最容易说错的地方

1. FlowEdit 论文的时间方向是 $t=0$ 图像、$t=1$ 噪声，与我们此前 RF 笔记相反。
2. 它不是新的 backbone，也没有训练新速度场，而是对已有流模型的推理调用方式。
3. “无 inversion”不等于“无多步采样”。
4. “optimization-free”不等于“计算免费”或“一次前向”。
5. 主路径 $Z_t^{\mathrm{FE}}$ 没有帽子；带帽变量是临时带噪查询点。
6. 目标与源速度不是在完全相同的输入点上计算。
7. noise-free 指主路径不经过可见纯噪声，不代表不用随机噪声。
8. $V^{\mathrm{tar}}-V^{\mathrm{src}}$ 是由直接路径求导得到，不是凭直觉随便相减。
9. $n_{\max}$ 越大通常路径越完整、编辑越强；$n_{\min}$ 越大通常末段目标自由度越大。
10. model-agnostic 只是在预训练条件流模型范围内不依赖特定内部层。
11. 更低运输代价不是一般最优运输证明。
12. 论文结果是强折中，不是每个指标都第一。

## 我的理解与修改

- 

## 相关笔记

- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)
- [10-Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（MM-DiT）](#/notes/mmdit)
- [11-SDEdit - Guided Image Synthesis and Editing with Stochastic Differential Equations](#/notes/sdedit)
- [12-Blended Latent Diffusion](#/notes/blended-latent-diffusion)
- [13-Prompt-to-Prompt Image Editing with Cross-Attention Control](#/notes/prompt-to-prompt)
 
## 后续连接

- [15-Stable Flow - Vital Layers for Training-Free Image Editing](#/notes/stable-flow)：同为无需训练的 flow 编辑，但 Stable Flow 对真实图需要 inversion，并在模型内部的关键 Attention 层注入源图视觉 $K,V$；FlowEdit 则在模型外使用目标速度减源速度。
