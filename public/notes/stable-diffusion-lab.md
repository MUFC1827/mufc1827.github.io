# 08-Experiment Stable Diffusion 文生图、DDIM 与图像编辑

## 文件与完成情况


主 Notebook 共 40 个单元；除末尾空白单元外，所有非空代码单元均已成功执行，保存结果中没有 error 输出。共生成 24 张 $512\times512$ 实验图片。

## 一、实验主线

本实验把 Stable Diffusion 1.5 的完整推理流程拆开：

$$
\text{prompt}
\xrightarrow{\text{Tokenizer}}
\text{Token IDs}
\xrightarrow{\text{CLIP}}
c,
$$

$$
z_T\sim\mathcal N(0,I)
\xrightarrow[\text{Scheduler}]{\text{U-Net 多步预测噪声}}
z_0
\xrightarrow{\text{VAE Decoder}}
x.
$$

随后再从图片出发：

$$
x\xrightarrow{\text{VAE Encoder}}z_0
\xrightarrow{\text{DDIM inversion}}
z_T,
$$

并用原 prompt 重建或用新 prompt 编辑。

本次使用的是预训练模型推理，没有计算训练 Loss、反向传播或更新权重。

## 二、实际环境

| 项目 | 实际值 |
|---|---|
| 操作系统 | Windows build 26200 |
| Python | 3.10.19 |
| PyTorch | 2.11.0+xpu |
| torchvision | 0.26.0+xpu |
| Diffusers | 0.39.0 |
| Transformers | 5.15.1 |
| Accelerate | 1.14.0 |
| GPU | Intel Arc 130T GPU（16 GB） |
| device / dtype | XPU / FP16 |
| 模型 | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| revision | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| U-Net 输入通道 | 4 |
| VAE scaling factor | 0.18215 |

XPU FP16 矩阵测试的输出全部为有限数，随后全部实验都在本地完成，因此本节无需租用 AutoDL 4090。

### 对老师代码做的必要适配

| 老师旧写法 / 说法 | 本实验处理 |
|---|---|
| `runwayml/stable-diffusion-v1-5` | 改为当前公开模型 ID |
| `cuda`、`torch.cuda` | 改为 `xpu`、`torch.xpu` |
| `_encode_prompt()` | 使用公开的 `encode_prompt()` |
| `decode_latents()` | 使用 `vae.decode()` 与 `image_processor.postprocess()` |
| 把 `AutoencoderKL` 称为 VQ-VAE/VQ-GAN | 更正为 KL-regularized continuous autoencoder |
| 写死 latent scaling | 使用 `pipe.vae.config.scaling_factor` |

## 三、E01 基准文生图

### 设置

- Prompt：`a cute cat in the garden, RAW photo, high quality`
- Negative Prompt：`semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, blurry, distorted, text, watermark`
- Seed：1
- DDIM，20 steps，CFG $=7.5$
- 分辨率：$512\times512$

### Prompt 的意义

- `a cute cat` 指定主体与大致观感；
- `in the garden` 指定场景；
- `RAW photo` 只是模型学到的摄影风格关联，不代表输出真正的 RAW 文件；
- `high quality` 是泛化质量提示词，不是分辨率开关；
- Negative Prompt 是反方向软引导，不是硬性禁令。

### 结果

- 首次生成耗时 20.54 s，其中包含模型初始化等额外开销；
- 预热后相同 20 步生成耗时约 7.68 s；
- 得到花园中的写实三花猫，主体与场景符合 Prompt；
- 猫眼仍有不自然之处，说明生成成功不等于没有局部伪影。

## 四、E02 单变量对照

### 1. Steps

| Steps | 耗时 | 观察 |
|---:|---:|---|
| 10 | 3.82 s | 已能生成完整猫和花园，但构图与后两组不同 |
| 20 | 7.68 s | 与基准图完全相同，主体和写实感明确 |
| 40 | 14.37 s | 细节没有相对 20 步出现决定性提升，眼睛伪影仍在 |

耗时随步数近似线性增加。改变步数会改变实际选取的时间步和采样轨迹，所以即使 seed 相同，构图也可能变化。不能总结成“步数越多必然越好”。

### 2. CFG

CFG 组合公式为：

$$
\widehat\epsilon
=\epsilon_{\mathrm{neg}}
+s\left(\epsilon_{\mathrm{pos}}-\epsilon_{\mathrm{neg}}\right).
$$

| CFG | 耗时 | 观察 |
|---:|---:|---|
| 1.0 | 7.39 s | 条件强调较弱，画面更松散、主体较不突出 |
| 7.5 | 7.74 s | 主体、场景与摄影风格较明确 |
| 15.0 | 7.81 s | 对比度、饱和度和锐化感更强，有过度处理倾向 |

CFG 增大主要增加条件方向的强度，不增加去噪步数，因此三组耗时接近。过高 CFG 可能同时放大文本匹配和模型伪影；单个样本不能证明普遍最优值。

### 3. Negative Prompt

- 关闭后，猫眼呈异常绿色，局部身体结构较不自然；
- 开启后，画面更干净、轮廓和摄影感改善；
- 眼睛问题并未完全消失，而且整体构图也发生变化。

结论：Negative Prompt 会影响整条随机采样轨迹，不是只从固定图片中删除某个局部元素。

### 4. DDIM 与 DDPM

在同模型、同 prompt、同 seed、30 steps 和 CFG $=7.5$ 下：

| Scheduler | 耗时 | 结果 |
|---|---:|---|
| DDIM | 11.36 s | 全身猫、石台和花园 |
| DDPM | 11.64 s | 猫脸近景、绿色虚化背景 |

二者使用同一个 U-Net 权重，但更新公式和随机轨迹不同，所以最终构图可以完全不同。本次单样本只能证明 Scheduler 会影响采样，不能证明某一个普遍更好。

### 5. Prompt 颜色与风格

- `red ceramic teapot + studio photograph`：生成红色写实茶壶，同时额外出现杯子；
- 只把 `red` 改为 `blue`：颜色成功改变，但茶壶移到左侧，`centered` 没有被严格满足；
- 改为 `watercolor painting`：水彩风格非常明显，同时出现额外容器。

这说明颜色和风格 token 确实能控制结果，但 Prompt 是全局概率条件。只改一个词也可能让其他内容与构图一起变化。

## 五、E03 组件与 Tensor 形状

| 阶段 | 实际 shape | 含义 |
|---|---|---|
| Token IDs | `[1,77]` | 一条固定长度文本序列 |
| CLIP 输出 | `[1,77,768]` | 每个 token 对应 768 维条件特征 |
| 正 / 负文本特征 | 各 `[1,77,768]` | CFG 的两个条件分支 |
| 初始 latent | `[1,4,64,64]` | 一张图的四通道空间潜变量 |
| CFG U-Net 输入 / 输出 | `[2,4,64,64]` | batch 中并行放入负、正两支 |
| 每个噪声分支 | `[1,4,64,64]` | 拆开后进行 CFG 组合 |
| CFG 汇总 | `[1,4,64,64]` | 下一步 Scheduler 所需噪声 |
| VAE 输出 | `[1,3,512,512]` | RGB 图像 |

不足 77 个位置的文本会由结束 token 补齐。本次实际 token 开头为：

```text
<|startoftext|>, a, cute, cat, in, the, garden,
<|endoftext|>, <|endoftext|>, ...
```

CFG 的“2”是 batch 临时扩充，不是 latent 通道从 4 变成 8。

## 六、E04 手写 DDIM

### 一步更新

U-Net 先预测噪声 $\widehat\epsilon$，再估计干净 latent：

$$
\widehat z_0=
\frac{
z_t-\sqrt{1-\bar\alpha_t}\widehat\epsilon
}{
\sqrt{\bar\alpha_t}
}.
$$

确定性 DDIM（$\eta=0$）再计算：

$$
z_{t-1}
=
\sqrt{\bar\alpha_{t-1}}\widehat z_0
+\sqrt{1-\bar\alpha_{t-1}}\widehat\epsilon.
$$

### 数值核对

- 手写结果与官方 Scheduler 输出均为 `[1,4,64,64]`；
- 最大绝对误差：`0.00390625`；
- 平均绝对误差：`0.0002799034`。

该误差来自 FP16 舍入，数值上已经很小。

### 完整循环

手写循环实际复现了：

```text
文本编码
→ 随机 latent
→ U-Net 预测正负两支噪声
→ CFG 汇总
→ Scheduler 更新
→ 重复 20 次
→ VAE 解码
```

手写完整生成图与 Pipeline 基准图的 SHA-256 完全相同。这是比“肉眼看起来相似”更强的证据，说明本次手写流程与 Pipeline 在该配置下等价。

## 七、E05 VAE 重建

数据流：

$$
[1,3,512,512]
\xrightarrow{\mathcal E}
[1,4,64,64]
\xrightarrow{\mathcal D}
[1,3,512,512].
$$

| latent 选择 | 重建 MSE |
|---|---:|
| `posterior.mode()` | 0.00341097 |
| `posterior.sample()` | 0.00340905 |

两张重建图都与原图非常接近，但毛发、高频纹理和局部边缘有轻微变化，验证了 VAE 是有损压缩。

`mode()` 取后验分布中心，结果确定、稳定；`sample()` 从后验中随机采样。本次 sample 的单次 MSE 略低，差值极小，不能据此推出 sample 通常比 mode 更好。

## 八、E06 DDIM Inversion

正常采样是：

$$
z_T\longrightarrow z_0\longrightarrow x.
$$

Inversion 试图近似反向移动：

$$
x\xrightarrow{\mathcal E}z_0
\longrightarrow z_1\longrightarrow\cdots\longrightarrow z_T.
$$

本次设置为 50 steps、guidance $=3.5$，先测试模型生成图，再测试真实小狗图。

### 实际结果

程序成功执行，但两次都没有实现忠实重建：

- 生成猫图的重建仍是猫和花园，却改变了颜色、姿态、前腿和视角；
- 真实小狗从彩色站立侧身图变成近似黑白、正面低头的小狗，结构和外观明显漂移。

因此应写成：

> 本次朴素 DDIM inversion 成功跑通并保留部分语义，但高保真重建失败。

误差来源包括：

- VAE 编码有损；
- 有限步 DDIM inversion 是局部近似，不是严格逆函数；
- Prompt 无法完整描述原图的每个视觉细节；
- CFG 条件轨迹不一定严格可逆；
- 真实图片不一定处于 SD 1.5 最容易表达的生成流形上。

## 九、E07 Prompt 编辑

### 1. `puppy → cat, skip20`

目标 Prompt：`Photograph of a cat on the grass`。

结果较成功：小狗变成猫，同时较好保留了草地、左向姿态、主体位置和树景。不过它不是局部像素替换，身体纹理和细节仍发生变化。

### 2. `puppy → cat, skip0`

同样成功生成猫，但画面接近黑白，尾巴、姿态、构图和背景变化更明显。

### 3. `grass → sand, skip20`

目标 Prompt：`Photograph of a puppy on sand`。

结果仍主要是草地，目标背景编辑基本失败；小狗和原图结构却保存得较好。这是一个有价值的负结果，说明较低噪声起点虽然保真度高，却可能没有足够自由度完成较大的场景变化。

### 噪声起点的折中

| 起点 | 编辑能力 | 原图保留 |
|---|---|---|
| 更高噪声，如 skip0 | 强 | 弱，容易改变姿态、颜色和构图 |
| 更低噪声，如 skip20 | 弱一些 | 强，但目标修改可能失败 |

这只是本次实验支持的经验趋势，不是对所有图片严格成立的定理。

## 十、可复现实验证据

- 共保存 24 张输出图；
- 基准图、E02 steps-20、E02 CFG-7.5、E02 negative-on、E04 手写生成、E05 original、E06 generated source 这 7 个文件逐字节相同；
- 固定模型、Scheduler、Prompt、Seed、Steps、CFG、dtype 和设备后，DDIM 本次表现为确定性生成；
- E04 手写输出与 Pipeline 完全一致；
- 主 Notebook 中没有保存的报错；
- 模型缓存约 5.48 GB。

尚未完成的可选扩展：

- 没有做不同输出分辨率的控制变量实验；
- E04 完整生成、E05、E06、E07 没有单独记录运行耗时；
- inversion 重建仍有明显漂移，若报告展示，必须作为局限或待改进结果。

## 十一、实验汇报问答

### 1. 为什么相同 seed 的不同 steps 结果仍会改变？

Seed 固定的是初始随机数；改变 steps 会改变 Scheduler 使用的离散时间步和数值轨迹，所以最终图像不必相同。

### 2. 为什么 CFG 两支可以放进一次 U-Net 前向？

把同一个 latent 沿 batch 维复制成两份，再分别配上负条件和正条件文本向量。U-Net 并行输出两份噪声预测，拆开后按 CFG 公式组合，因此不是训练了两个模型。

### 3. 手写 DDIM 实验验证了什么？

它验证 U-Net 负责给出 $\widehat\epsilon$，Scheduler 负责按公式把 $z_t$ 更新为下一状态。完整手写图与 Pipeline 文件哈希相同，说明隐藏在 Pipeline 中的核心流程已被正确展开。

### 4. VAE 重建相似，为什么仍说是有损？

肉眼相似不等于每个像素一致。本次 MSE 非零，高频纹理也有细小变化；Encoder 丢掉的信息不能由 Decoder 精确恢复。

### 5. 为什么不能说 inversion 已经成功重建？

因为“代码完成运行”只是流程成功；生成猫和真实小狗的重建都出现明显颜色、姿态和结构漂移。正确结论是保留了部分语义，但高保真重建失败。

### 6. 为什么 `skip20` 更保留原图？

它跳过最高噪声的一段，从更接近干净图像的 latent 开始去噪。此时原始结构约束更强，因此更保真，但可供新 Prompt 改写的自由度也更小。

### 7. Scheduler 与 U-Net 的区别是什么？

U-Net 是训练得到的噪声预测网络；Scheduler 是数值更新规则。更换 Scheduler 不改变 U-Net 参数，却会改变采样路径和最终图像。

### 8. Negative Prompt 为什么不是硬约束？

它只改变 CFG 中负条件分支的预测方向。模型仍在连续概率空间中采样，所以不能保证某个概念绝对消失，也可能连带改变全局构图。

## 十二、完成检查

- [x] 在 Intel XPU 上加载并运行 Stable Diffusion 1.5；
- [x] 生成并保存 $512\times512$ 基准图；
- [x] 完成 steps、CFG、Negative Prompt、Scheduler、颜色与风格对照；
- [x] 检查 Token、CLIP、latent、CFG、U-Net 和 VAE 的 shape；
- [x] 手算一次 DDIM 更新并进行数值核对；
- [x] 手写完整 DDIM 生成循环；
- [x] 比较 VAE `mode()` 与 `sample()`；
- [x] 对生成图和真实图执行 DDIM inversion；
- [x] 完成主体编辑及不同噪声起点对照；
- [x] 如实记录 inversion 和背景编辑的失败结果。

## 相关笔记

- [08-High-Resolution Image Synthesis with Latent Diffusion Models（LDM）](#/notes/latent-diffusion)
- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [07-Flow Straight and Fast（Rectified Flow）](#/notes/rectified-flow)
- [11-SDEdit - Guided Image Synthesis and Editing with Stochastic Differential Equations](#/notes/sdedit)
