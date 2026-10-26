# 05 Attention Is All You Need

## 来源

- 论文全名：**Attention Is All You Need**
- 作者：Ashish Vaswani 等
- 会议：NeurIPS 2017
- 老师清单中的 arXiv 页面：<https://arxiv.org/abs/1706.03762>
- 官方 NeurIPS 页面：<https://papers.neurips.cc/paper/7181-attention-is-all-you-need>

## 一句话主线

Transformer 用 **Self-Attention 作为序列位置之间交换信息的核心机制**，构造完整 Encoder–Decoder，取消 RNN 的逐位置循环和 CNN 的序列卷积，从而提高训练并行性并缩短长距离信息的传播路径。

> 标题不代表网络里真的只有 Attention。原模型还有 Embedding、Positional Encoding、FFN、Residual Connection、LayerNorm、Linear 和 Softmax。

- RNN：Recurrent Neural Network，循环神经网络；
- CNN：Convolutional Neural Network，卷积神经网络；
- Seq2Seq：Sequence-to-Sequence，序列到序列；
- NMT：Neural Machine Translation，神经机器翻译。

## 记号约定

论文的矩阵公式把**每个 token 向量放在一行**。这与前两节视频笔记把 token 放在一列不同，但单个分数始终是同一件事：$q_i$ 与 $k_j$ 的点积。

| 符号 | 含义 | 形状 |
|---|---|---|
| $n_q$ | Query 数量 | 标量 |
| $n_k$ | Key/Value 数量 | 标量 |
| $d_{\mathrm{model}}$ | token 主表示维度 | Base 中为 512 |
| $d_k$ | Query、Key 的维度 | Base 每头为 64 |
| $d_v$ | Value 的维度 | Base 每头为 64 |
| $Q$ | 所有 Query | $n_q\times d_k$ |
| $K$ | 所有 Key | $n_k\times d_k$ |
| $V$ | 所有 Value | $n_k\times d_v$ |

视频列向量记法中的 $K^{\mathsf T}Q$ 与论文行向量记法中的 $QK^{\mathsf T}$ 互为转置；本节按论文原式书写：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V.
$$

## 一、论文为什么提出 Transformer

### RNN 的限制

RNN 按顺序计算：

$$
h_t=f(h_{t-1},x_t).
$$

必须先得到 $h_{t-1}$ 才能计算 $h_t$，所以同一句话内部存在：

$$
h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_n.
$$

这会导致：

- 位置之间难以完全并行；
- 长距离信息要经过很多步；
- 长序列会限制 batch 大小和训练速度。

### CNN 的限制

CNN 可并行，但一层卷积通常只看局部窗口。句首与句尾要交流，需要堆很多层或使用空洞卷积，传播路径仍比全局 Self-Attention 长。

### 论文的核心假设

如果每个位置能在一层中直接读取所有位置，就可能同时得到：

1. 更强的并行训练能力；
2. 更短的长距离依赖路径；
3. 不依赖 RNN/CNN 的 Seq2Seq 模型。

## 二、完整 Encoder–Decoder 流程

以翻译为例：

$$
\boxed{
\text{源 token}
\rightarrow
\text{Embedding + Position}
\rightarrow
6\text{ 层 Encoder}
\rightarrow
\text{上下文表示}
}
$$

$$
\boxed{
\text{已知目标前缀}
\rightarrow
6\text{ 层 Decoder}
\rightarrow
\text{Linear}
\rightarrow
\text{Softmax}
\rightarrow
\text{下一个 token}
}
$$

### Encoder

原论文堆叠 $N=6$ 层，每层包含：

1. Multi-Head Self-Attention；
2. Position-wise FFN。

每个子层外都有 Residual、Dropout 与 LayerNorm：

$$
\boxed{
\operatorname{LayerNorm}
\left(x+\operatorname{Dropout}(\operatorname{Sublayer}(x))\right)
}
$$

原始 Transformer 是 **Post-LN**：先计算子层并做残差相加，再 LayerNorm。现代模型常改用 Pre-LN，不能反过来说成原论文结构。

### Decoder

每层有三个子层：

1. Masked Multi-Head Self-Attention；
2. Encoder–Decoder Attention（Cross-Attention）；
3. Position-wise FFN。

Masked Self-Attention 防止预测当前位置时偷看未来答案。将未来位置的分数加上 $-\infty$：

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

因为 $e^{-\infty}=0$，Softmax 后未来位置权重为 0。

## 三、Scaled Dot-Product Attention

核心公式：

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V
}
$$

形状变化：

$$
QK^{\mathsf T}:
[n_q,d_k][d_k,n_k]
\rightarrow[n_q,n_k].
$$

其中：

$$
(QK^{\mathsf T})_{ij}=q_i\cdot k_j,
$$

表示第 $i$ 个 Query 与第 $j$ 个 Key 的匹配分数。对每一行做 Softmax 后得到：

$$
A\in\mathbb R^{n_q\times n_k},
\qquad
\sum_{j=1}^{n_k}A_{ij}=1.
$$

最后：

$$
AV:[n_q,n_k][n_k,d_v]\rightarrow[n_q,d_v].
$$

所以：

- $Q,K$ 决定“关注谁、关注多少”；
- $V$ 决定“真正取走什么信息”；
- Softmax 只产生权重，乘 $V$ 后才得到 Attention 输出；
- $Q,K,V$ 的投影矩阵都是训练学到的，不由人手工选择。

## 四、为什么除以 $\sqrt{d_k}$

设：

$$
q\cdot k=\sum_{r=1}^{d_k}q_rk_r.
$$

在各分量相互独立、均值为 0、方差为 1 的近似假设下：

$$
\mathbb E[q\cdot k]=0,
\qquad
\operatorname{Var}(q\cdot k)=d_k.
$$

因此标准差为：

$$
\operatorname{Std}(q\cdot k)=\sqrt{d_k}.
$$

$d_k$ 越大，点积分数的尺度越大，Softmax 越容易接近 one-hot，进入饱和区域并产生很小的梯度。缩放后：

$$
\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{d_k}}\right)\approx1.
$$

必须会说：**除以 $\sqrt{d_k}$ 是把点积分数的标准差稳定到约 1，而不是随便加的经验常数。**

## 五、Multi-Head Attention

$$
\operatorname{head}_i
=
\operatorname{Attention}
\left(QW_i^Q,KW_i^K,VW_i^V\right),
$$

$$
\boxed{
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
}
$$

Base 模型使用：

$$
h=8,
\qquad
d_k=d_v=\frac{512}{8}=64.
$$

每头输出 64 维，8 个头拼接后回到 512 维，再由 $W^O$ 融合。

多头的意义：不同投影让模型能在不同表示子空间中学习不同关系。头的职责不是人工规定的，也不是头越多越好。

多头汇总不是 Softmax、平均或直接相加，而是：

$$
\boxed{\text{各头独立 Attention}\rightarrow\text{Concat}\rightarrow W^O}
$$

## 六、三种 Attention 的 Q、K、V 来源

| 模块 | $Q$ 来源 | $K,V$ 来源 | 可读取范围 |
|---|---|---|---|
| Encoder Self-Attention | Encoder 当前序列 | 同一序列 | 全部源 token |
| Decoder Masked Self-Attention | Decoder 当前序列 | 同一序列 | 当前及以前目标 token |
| Encoder–Decoder Attention | Decoder | Encoder 最终输出 | 全部源 token |

Cross-Attention 的本质：Decoder 用当前状态提出 Query，再从 Encoder 的 Key–Value 中读取与当前生成最相关的源句信息。

## 七、Position-wise FFN

FFN 是 Feed-Forward Network，前馈网络：

$$
\boxed{
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
}
$$

Base 模型的维度：

$$
512\rightarrow2048\rightarrow\operatorname{ReLU}\rightarrow512.
$$

- Attention：让不同 token 横向交换信息；
- FFN：对一个 token 内部的特征做非线性加工；
- 同一层所有位置共享一组 $W_1,W_2$；
- 不同 Transformer 层使用不同参数。

## 八、Embedding、输出层与权重共享

Token id 通过 Embedding Table 变成 $d_{\mathrm{model}}$ 维向量。论文把 Embedding 乘以：

$$
\sqrt{d_{\mathrm{model}}},
$$

再与位置编码相加。

Decoder 最终表示通过词表线性层：

$$
Z=HW_{\mathrm{vocab}}+b,
$$

再经 Softmax 得到每个 token 的概率。

论文共享源 Embedding、目标 Embedding 与输出 pre-softmax linear 的权重矩阵，称为 **Weight Tying**。

## 九、Sinusoidal Positional Encoding

$$
\operatorname{PE}_{(pos,2i)}
=
\sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right),
$$

$$
\operatorname{PE}_{(pos,2i+1)}
=
\cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right).
$$

- $pos$：token 的位置；
- $i$：频率组编号；
- 不同特征维使用不同波长；
- PE 与 Embedding 维度相同，二者直接相加。

相加后仍能表示位置，是因为每个位置加入了不同且有规律的向量；后续线性层和 Attention 可学习读取这些差异。

对固定偏移 $k$：

对第 $i$ 组频率，令 $\omega_i=10000^{-2i/d_{\mathrm{model}}}$，则：

$$
\sin(\omega_i(pos+k))
=
\sin(\omega_i pos)\cos(\omega_i k)
+
\cos(\omega_i pos)\sin(\omega_i k),
$$

$$
\cos(\omega_i(pos+k))
=
\cos(\omega_i pos)\cos(\omega_i k)
-
\sin(\omega_i pos)\sin(\omega_i k).
$$

因此偏移后的位置编码可由原位置的一对 sin/cos 特征做线性变换得到，这支持模型学习相对位置。论文中 learned PE 与 sinusoidal PE 效果接近；作者选择后者主要因为它可能外推到更长序列。

## 十、训练目标与推理

论文没有详细展开交叉熵推导，但完整训练目标是：

$$
P_\theta(y^*\mid x)
=
\prod_{t=1}^{m}
P_\theta(y_t^*\mid y_{<t}^*,x).
$$

最大化正确句子的概率，等价于最小化：

$$
\boxed{
L=-\sum_{t=1}^{m}\log P_\theta(y_t^*\mid y_{<t}^*,x)
}
$$

训练时：

- Decoder 输入是右移后的正确答案；
- 使用 Teacher Forcing；
- Causal Mask 防止看到未来答案；
- 所有目标位置可以并行计算 Loss；
- 反向传播更新所有可训练矩阵。

推理时没有正确答案前缀，必须根据已经生成的 token 自回归地逐步生成。因此 Transformer **训练时高度并行，但原始 Decoder 推理时仍是顺序的**。

## 十一、为什么 Self-Attention 有优势

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolution | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted Self-Attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

- $n$：序列长度；
- $d$：表示维度；
- $k$：卷积核宽度；
- $r$：局部注意范围。

当 $n<d$ 时，$O(n^2d)$ 可小于 $O(nd^2)$，这在论文当时的机器翻译句长中常成立。Self-Attention 任意两位置一层就能交流，最大路径为 $O(1)$。

注意：

- 表中卷积的 $O(\log_k n)$ 对应空洞卷积；普通连续卷积的最长路径约为 $O(n/k)$；
- Table 1 比较的是跨位置的信息混合操作，完整 Transformer 层的投影和 FFN 还包含 $O(nd^2)$ 量级计算；
- $O(1)$ 指**顺序步骤或路径长度**，不代表总计算为常数；
- 完整 Self-Attention 对长序列仍有 $O(n^2)$ 的计算与显存代价。

## 十二、训练设置

### 数据

- WMT14 English–German：约 4.5M 句对，约 37K 共享 BPE subword 词表；
- WMT14 English–French：约 36M 句对，约 32K word-piece 词表；
- 每个 batch 约含 25K source token 和 25K target token，而不是固定句子数。

### 优化器与学习率

Adam：

$$
\beta_1=0.9,\qquad\beta_2=0.98,\qquad\epsilon=10^{-9}.
$$

学习率：

$$
\boxed{
\operatorname{lrate}
=
d_{\mathrm{model}}^{-0.5}
\min\left(step^{-0.5},\ step\cdot warmup^{-1.5}\right)
}
$$

其中 $warmup=4000$：开始阶段近似线性增大，之后按 $step^{-1/2}$ 下降。

### 正则化

- Base dropout：$P_{\mathrm{drop}}=0.1$；
- Big 英德模型使用 0.3，Big 英法模型使用 0.1；
- Label Smoothing：$\epsilon_{ls}=0.1$。

Label Smoothing 会让模型不那么过度自信，论文中它使 perplexity 变差，却改善 accuracy 与 BLEU，因为这些指标衡量的侧重点不同。

### 硬件与时间

- 8 张 NVIDIA P100；
- Base：100K steps，约 12 小时；
- Big：300K steps，约 3.5 天。

## 十三、实验结论

### 机器翻译

- WMT14 English–German：Base 27.3 BLEU，Big 28.4 BLEU；
- Big 超过当时此前最佳结果（包括 ensemble）2 BLEU 以上；
- 当前 arXiv PDF 的 English–French 数字存在版本残留：摘要和 Table 2 为 41.8，Section 6.1 正文仍写 41.0。汇报时应说明版本，不混用。

BLEU 是机器翻译自动评价指标，核心比较候选译文与参考译文的 n-gram 重合，并加入短句惩罚；越高通常越好，但不能完整评价语义和流畅性。

推理使用 Beam Search：beam size 4、length penalty $\alpha=0.6$。

### 消融实验

论文的主要结论：

- 单头不如合适的多头；
- 头数太多导致每头维度太小，也可能变差；
- 缩小 $d_k$ 会降低质量；
- 增大层数、$d_{\mathrm{model}}$ 或 $d_{\mathrm{ff}}$ 通常改善结果；
- 适当 Dropout 和 Label Smoothing 有帮助；
- learned PE 与 sinusoidal PE 结果近似。

Base：约 65M 参数；Big：$N=6,d_{\mathrm{model}}=1024,d_{\mathrm{ff}}=4096,h=16,d_k=d_v=64$，约 213M 参数。

### 成分句法分析

论文又把 4 层 Transformer 用于 English Constituency Parsing：

- WSJ-only：91.3 F1；
- 半监督约 17M 句：92.7 F1。

这说明架构可推广到翻译之外的结构化序列任务；但表中仍有 93.0、93.3 的方法，不能说 Transformer 超过所有方法。

## 十四、论文边界与后续联系

- 原论文主要验证翻译与句法分析，不是现代所有 Transformer 任务的证明；
- 完整 Attention 对序列长度有 $O(n^2)$ 代价；
- Attention 图可以提供线索，但不能把每个 head 都当成稳定、唯一、可解释的规则；
- 原论文是 Encoder–Decoder，不等于只含 Decoder 的 GPT；
- DiT 保留 token、Self-Attention、FFN、Residual 和 LayerNorm 的骨架，把文本 token 换成 noisy latent patch，并加入扩散时间与类别条件。

## 必须掌握

- [ ] 能解释 Transformer 为什么替换 RNN/CNN。
- [ ] 能画出 Encoder 与 Decoder 每层结构。
- [ ] 能写出原始 Post-LN 残差公式。
- [ ] 能写出 Scaled Dot-Product Attention 公式并检查形状。
- [ ] 能推导为什么除以 $\sqrt{d_k}$。
- [ ] 能解释多头的独立投影、Concat 与 $W^O$。
- [ ] 能说出三种 Attention 的 Q、K、V 来源。
- [ ] 能解释 Causal Mask、FFN 和 Positional Encoding。
- [ ] 能区分训练并行与自回归推理。
- [ ] 能解释 Table 1 的复杂度、顺序操作与路径长度。
- [ ] 能准确复述核心翻译结果和消融结论。

## 了解即可

- Additive Attention 的完整实现；
- BPE/WordPiece 的具体建词表算法；
- Adam 的内部更新公式；
- 所有 baseline 与表格数字；
- Beam Search length penalty 的完整公式；
- 句法分析的数据处理细节。

## 三条核心公式

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V
$$

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

$$
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
$$

## 最容易说错的地方

1. 标题不表示模型里只有 Attention。
2. 原论文是 Encoder–Decoder，并使用 Post-LN。
3. $Q,K,V$ 的投影矩阵通过训练得到，不是人工指定。
4. Softmax 产生权重，输出还要乘 $V$。
5. 多头通过 Concat + $W^O$ 汇总，不是平均。
6. Encoder Self-Attention 不使用 causal mask。
7. Cross-Attention 的 $Q$ 来自 Decoder，$K,V$ 来自 Encoder。
8. 同层 FFN 在所有位置共享参数，但不同层不共享。
9. Positional Encoding 与 Embedding 相加，不是拼接。
10. 训练可以并行目标位置，原始 Decoder 推理仍逐 token 生成。
11. $O(1)$ sequential operations 不等于总计算是 $O(1)$。
12. 多头不是越多越好，正弦位置编码也不是明显优于 learned PE。

## 我的理解与修改

- 

## 相关笔记

- [04-Transformer（下）](#/notes/transformer-2)
- [06-Denoising Diffusion Probabilistic Models（DDPM）](#/notes/ddpm)
- [13-Prompt-to-Prompt Image Editing with Cross-Attention Control](#/notes/prompt-to-prompt)
