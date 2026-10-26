# 04 Transformer（下）

## 来源

- 视频：【李宏毅机器学习】Transformer（下）
- Bilibili：<https://www.bilibili.com/video/BV1r8nMz4EAj?p=4>
- 李宏毅官方视频：<https://www.youtube.com/watch?v=N6aRv06iv2g>
- 官方讲义 Seq2Seq / Transformer：<https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/seq2seq_v9.pdf>

## 本节主线

Encoder 先把输入序列变成一组上下文表示；Decoder 一边查看已经生成的内容，一边读取 Encoder 的输出，再预测下一个 token。

$$
\boxed{
\text{输入}
\rightarrow\text{Encoder}
\rightarrow\text{Decoder}
\rightarrow\text{词表概率}
\rightarrow\text{选择 token}}
$$

Decoder 内部的三层主干是：

$$
\boxed{
\text{Masked Self-Attention}
\rightarrow\text{Cross Attention}
\rightarrow\text{FFN}}
$$

三个子层都配有 Residual Connection 和 LayerNorm。Masked Self-Attention 查看已经生成的目标内容，Cross Attention 读取原输入；FFN 则进一步加工每个 token 的特征。FFN 是 Feed-Forward Network（前馈神经网络），通常是对每个 token 分别使用的两层全连接：

$$
\operatorname{FFN}(u^i)
=W_2\phi(W_1u^i+b_1)+b_2.
$$

它负责进一步加工 token 内部的特征，不负责从词表选择 token。

## 一、Autoregressive Decoder

Autoregressive（AT，自回归）表示每次只生成一个 token，下一步依赖此前已经生成的结果。给定输入 $x$：

$$
P(y_1,\ldots,y_T\mid x)
=\prod_{t=1}^{T}
P(y_t\mid y_1,\ldots,y_{t-1},x).
$$

这是概率链式法则：整句话的概率等于每一步条件概率的乘积。

以生成“机器学习”为例：

| 步骤 | Decoder 已有输入 | 本步输出 |
|---:|---|---|
| 1 | START | 机 |
| 2 | START 机 | 器 |
| 3 | START 机 器 | 学 |
| 4 | START 机 器 学 | 习 |
| 5 | START 机 器 学 习 | END |

设 Decoder 当前表示为 $u^t$，最终输出层把它变成词表中 $V$ 个 token 的分数：

$$
z^t=W_{\mathrm{vocab}}u^t+b_{\mathrm{vocab}},
\qquad z^t\in\mathbb R^V,
$$

$$
p^t=\operatorname{softmax}(z^t).
$$

$W_{\mathrm{vocab}},b_{\mathrm{vocab}}$ 是训练得到的参数。softmax 只产生词表概率分布；Greedy、Beam Search 或 Sampling 才负责选择具体 token。选出的 token 会放回 Decoder，所以前面预测错误可能影响后面。

### 为什么需要词表线性层

$u^t$ 是 Decoder 的内部特征，不是词表概率，而且通常只有 $d_{\mathrm{model}}$ 个数；softmax 不会改变向量长度。若词表有 $V$ 个 token，就必须先用

$$
W_{\mathrm{vocab}}\in
\mathbb R^{V\times d_{\mathrm{model}}}
$$

把内部特征映射成 $V$ 个词表分数。可以把它理解为“把 Decoder 的特征语言翻译成每个 token 的得分”。例如词表为“机、器、END”，若线性层输出 $[2,1,-1]$，softmax 后约为 $[0.705,0.259,0.035]$。

Block 内的 FFN 是“加工特征”；最后的词表线性层是“把特征映射成词表分数”，二者用途不同。

- START、BOS：开始生成。
- END、EOS：结束生成；它也属于词表中的特殊 token。

### Masked Self-Attention

生成当前位置时，未来 token 还不存在，因此 Decoder 只能关注当前位置及更早位置。对 Query 位置 $i$、Key 位置 $j$：

$$
\widetilde\alpha_{i,j}
=
\begin{cases}
\alpha_{i,j},&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

未来分数经过 softmax 后权重为 $0$。训练时计算机虽然拥有完整正确答案，Mask 仍能防止当前位置偷看未来；同时所有位置仍可并行计算。

## 二、AT 与 NAT

| | AT | NAT |
|---|---|---|
| 生成方式 | 按顺序逐个生成 | 多个位置同时生成 |
| 是否依赖已生成 token | 是 | 基础形式不依赖此前目标 token |
| 速度 | 推理难以完全并行 | 可并行，通常更快 |
| 错误反馈 | 可能逐步累积 | 不会把上一步错误继续喂回 |
| 常见效果 | 通常较好 | 通常较难达到 AT 的质量 |

NAT 需要先知道输出长度，常见做法是：

1. 另设模型预测输出长度；
2. 先输出足够长的序列，再忽略 END 后面的内容。

NAT 的主要困难是 Multi-modality（多种合理答案）：同一输入可能对应多种正确表达。AT 在选定前面的表达后，后面会以它为条件继续；基础 NAT 同时预测各位置，可能把不同表达方式的局部结果混在一起，造成重复、遗漏或不协调。

## 三、Cross Attention

Cross Attention 解决“Decoder 怎样读取 Encoder”的问题。设 Encoder 输出：

$$
a^1,a^2,\ldots,a^L,
$$

Decoder 经 Masked Self-Attention 后在第 $t$ 个位置得到表示 $d^t$。Cross Attention 中：

$$
q^t=W^q d^t,
\qquad
k^j=W^k a^j,
\qquad
v^j=W^v a^j.
$$

视频采用点积匹配：

$$
\alpha_{t,j}=(q^t)^{\mathsf T}k^j,
$$

$$
\alpha'_{t,j}
=\frac{\exp(\alpha_{t,j})}
{\sum_{m=1}^{L}\exp(\alpha_{t,m})},
\qquad
c^t=\sum_{j=1}^{L}\alpha'_{t,j}v^j.
$$

其中：

- $q^t$：Decoder 当前想从输入中查找什么；
- $k^j$：Encoder 第 $j$ 个位置可供匹配的特征；
- $v^j$：该输入位置真正提供的信息；
- $c^t$：Decoder 从整段输入中取回的加权信息。

| Attention | Query 来源 | Key、Value 来源 | 作用 |
|---|---|---|---|
| Encoder Self-Attention | Encoder | Encoder | 建立输入内部关系 |
| Masked Self-Attention | Decoder | Decoder | 查看已生成的目标内容 |
| Cross Attention | Decoder | Encoder | 根据当前需要读取原输入 |

Cross Attention 通常可以查看完整 Encoder 输出，不需要遮住未来输入，因为原输入在生成前已经全部给出。

## 四、Transformer 的训练

假设正确输出是：

$$
\text{机，器，学，习，EOS}.
$$

训练时把正确答案右移一位：

$$
\text{Decoder 输入：BOS，机，器，学，习},
$$

$$
\text{监督标签：机，器，学，习，EOS}.
$$

训练时使用正确的历史 token 作为 Decoder 输入，叫作 Teacher Forcing。每个位置的目标都是预测右边的正确 token。

设第 $t$ 个正确 token 为 $y_t^*$，模型给它的概率为 $p_t(y_t^*)$。one-hot 标签的交叉熵为：

$$
\mathcal L_t
=-\sum_{v=1}^{V}y_{t,v}\log p_{t,v}
=-\log p_t(y_t^*).
$$

整个序列的损失为：

$$
\mathcal L
=-\sum_t\log p_t(y_t^*).
$$

这个式子也可以由最大似然得到：

$$
\underbrace{\max\prod_t p_t(y_t^*)}_{\text{让正确句子概率最大}}
\Longleftrightarrow
\underbrace{\max\sum_t\log p_t(y_t^*)}_{\log\text{ 把乘法变加法}}
\Longleftrightarrow
\underbrace{\min-\sum_t\log p_t(y_t^*)}_{\text{改写为最小化 Loss}}.
$$

对数函数单调递增，所以取对数不改变哪个参数最好；再加负号只是把“最大化”改写成“最小化”。本阶段只需能复述这三步并知道正确 token 概率越大，Loss 越小，不要求证明最大似然理论。

Loss 反向传播后，会更新 Embedding、Self-Attention、Cross Attention、FFN、LayerNorm 和词表输出层等所有可学习参数。

| | 训练 | AT 推理 |
|---|---|---|
| Decoder 历史输入 | 正确答案 | 模型自己的输出 |
| 目标序列是否已知 | 已知 | 未知 |
| 位置计算 | Mask 后可并行 | 必须逐步生成 |
| 是否计算 Loss、更新参数 | 是 | 否 |

## 五、输出序列怎样选择

### Greedy Decoding

每一步都选择当前概率最大的 token：

$$
y_t=\arg\max_v P(v\mid y_{<t},x).
$$

它速度快，但“每一步局部最优”不保证“整句话概率最大”。

### Beam Search

所有可能序列约有 $V^T$ 条，无法穷举。Beam Search 每一步只保留分数最高的 $B$ 个候选前缀，再继续扩展：

$$
\log P(y\mid x)
=\sum_t\log P(y_t\mid y_{<t},x).
$$

$B=1$ 时近似 Greedy；$B$ 越大，搜索更充分但计算量更高。Beam Search 是推理时的近似搜索，不是训练算法，也不是另一种 softmax。

### Sampling

Sampling 是“按概率抽签”，不是等概率乱选。若

$$
P(\text{猫})=0.6,\qquad
P(\text{狗})=0.3,\qquad
P(\text{鸟})=0.1,
$$

就把转盘的 $60\%$ 分给“猫”、$30\%$ 分给“狗”、$10\%$ 分给“鸟”，转到谁就输出谁。猫最容易被抽到，但狗和鸟也有机会。抽出的 token 会放回 Decoder，继续计算下一步，直到 EOS。

Greedy 每次必选概率最大的 token；Sampling 允许同一输入多次运行得到不同结果。翻译、语音识别等答案较明确的任务常重视搜索，聊天、故事等开放式生成需要多样性时，适当随机性可能更合适。

## 六、常见改进技巧

### Copy Mechanism

允许模型直接复制输入中的 token，而不只从固定词表生成。适合人名、地名、专有词和摘要中的原文关键词。它是额外机制，不是基础 Transformer 必备模块。

### Guided Attention

在语音识别、TTS 等输入与输出近似单调对齐的任务中，正常关注位置应随输出逐渐向后移动。Guided、Monotonic 或 Location-aware Attention 利用这种规律，减少重复、漏读和注意力乱跳。

### 训练目标与评价指标

Cross Entropy 优化逐 token 的预测概率，而翻译可能使用 BLEU 评价整句话。BLEU 的基本流程是：

1. 准备模型译文和人工参考译文；
2. 比较相同的 1-gram、2-gram、3-gram、4-gram；
3. 合并各长度词组的匹配率；
4. 对过短输出施加长度惩罚。

例如“机器 学习 我 喜欢”与“我 喜欢 机器 学习”包含相同单词，但连续词组顺序不同，因此高阶 n-gram 匹配会下降。BLEU 是外部评价指标，不是 Transformer 内部组件，也通常不是基础训练 Loss；语义正确但措辞不同也可能得分不高。

BLEU 不能像普通网络运算一样直接求导，因此视频提到可把它作为奖励，用强化学习尝试直接优化；这不是基础 Transformer 的必备训练步骤。

### Exposure Bias 与 Scheduled Sampling

Teacher Forcing 造成训练与推理不一致：

- 训练：总能看到正确历史；
- 推理：只能看到自己的历史，一次错误可能继续累积。

模型训练时很少接触自己的错误前缀，这叫 Exposure Bias。Scheduled Sampling 会在训练中逐渐混入一部分模型自己的预测，使模型练习从错误历史中恢复；Transformer 为保持并行性，需要专门的改进版本。

## 七、完整流程

训练：

$$
\boxed{
\text{输入}\rightarrow\text{Encoder}
\rightarrow
\text{右移后的正确答案}\rightarrow
\text{Masked Self-Attention}
\rightarrow\text{Cross Attention}
\rightarrow\text{FFN}
\rightarrow\text{词表 softmax}
\rightarrow\text{Cross Entropy}
\rightarrow\text{反向传播}}
$$

AT 推理：

$$
\boxed{
\text{已有输出}
\rightarrow\text{Masked Self-Attention}
\rightarrow\text{Cross Attention}
\rightarrow\text{词表概率}
\rightarrow\text{Greedy / Beam / Sampling}
\rightarrow\text{追加 token}}
$$

不断循环，直到生成 EOS。

## 容易混淆的点

- Attention softmax 产生“对位置的权重”；最终 softmax 产生“对词表的概率”。
- Masked Self-Attention 查看 Decoder 历史；Cross Attention 读取 Encoder 输入。
- Block 内 FFN 加工特征；最终词表输出层把 $d_{\mathrm{model}}$ 维表示变成 $V$ 个分数。
- Teacher Forcing 属于训练；Greedy、Beam Search、Sampling 属于推理。
- AT 推理逐步生成，但训练时借助正确答案和 Mask 可以并行。
- Copy、Guided Attention、Scheduled Sampling 都是可选改进，不是基础 Transformer 的固定结构。

## 复习自测

- [ ] 能完整复述 AT 从 BOS 到 EOS 的循环。
- [ ] 能解释 Decoder 为什么需要 Mask。
- [ ] 能说出 Cross Attention 中 $Q,K,V$ 分别来自哪里。
- [ ] 能写出训练时 Decoder 输入与标签的右移关系。
- [ ] 能解释 Cross Entropy 为什么等于正确 token 的负对数概率。
- [ ] 能区分 Teacher Forcing、Beam Search 和 Sampling。
- [ ] 能解释 NAT 的速度优势与 Multi-modality 问题。
- [ ] 能解释 Exposure Bias 和 Scheduled Sampling。

## 相关笔记

- [03-Transformer（上）](#/notes/transformer-1)
- [05-Attention Is All You Need](#/notes/attention-is-all-you-need)
