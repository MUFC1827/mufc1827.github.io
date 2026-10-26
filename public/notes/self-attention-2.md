# 02 Self-Attention（下）

## 来源

- 视频：`李宏毅｜自注意力机制和Transformer详细解析，目前最合适新手看的Transformer课程！`
- 本节：`自注意力机制（Self-attention）（下）`
- Bilibili：<https://www.bilibili.com/video/BV1r8nMz4EAj?p=2>
- 李宏毅官方课件：<https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/self_v7.pdf>

## 本节主线

1. 把逐个位置的 Self-Attention 改写成可以并行计算的矩阵形式。
2. 用多个注意力头学习不同类型的相关性。
3. 用位置编码弥补纯 Self-Attention 缺少顺序信息的问题。
4. 了解它在语音、图像和图结构中的应用，以及与 CNN、RNN 的区别。

## 一、矩阵形式

视频采用**每个向量是一列**的记法。设序列长度为 $L$：

$$
I=\begin{bmatrix}a^1&a^2&\cdots&a^L\end{bmatrix}
\in\mathbb{R}^{d_a\times L}.
$$

由于每个位置共用同一组参数：

$$
Q=W^qI,\qquad K=W^kI,\qquad V=W^vI.
$$

其中：

$$
Q,K\in\mathbb{R}^{d_k\times L},\qquad
V\in\mathbb{R}^{d_v\times L}.
$$

### 1. 计算所有 Query-Key 分数

$$
A=K^{\mathsf T}Q\in\mathbb{R}^{L\times L}.
$$

$K^{\mathsf T}$ 的第 $j$ 行是 $(k^j)^{\mathsf T}$，$Q$ 的第 $i$ 列是 $q^i$，所以：

$$
A_{j,i}=(k^j)^{\mathsf T}q^i.
$$

它表示第 $i$ 个 Query 与第 $j$ 个 Key 的相关程度。固定 Query $i$ 时，要沿着 $A$ 的第 $i$ 列比较全部 Key。

### 2. 对每个 Query 分别做 softmax

$$
A'_{j,i}
=\frac{\exp(A_{j,i})}
{\sum_{m=1}^{L}\exp(A_{m,i})}.
$$

因此对固定的第 $i$ 列：

$$
\sum_{j=1}^{L}A'_{j,i}=1.
$$

视频把 Query $i$ 对 Key $j$ 的权重记作 $\alpha'_{i,j}$，所以 $A'_{j,i}=\alpha'_{i,j}$。

### 3. 汇总 Value

$$
O=VA'\in\mathbb{R}^{d_v\times L}.
$$

$O$ 的第 $i$ 列是：

$$
b^i=\sum_{j=1}^{L}\alpha'_{i,j}v^j.
$$

因此，$K^{\mathsf T}Q$ 一次得到所有位置之间的分数，$VA'$ 一次得到所有位置的输出，整套计算可以并行进行。

> [!note] 缩放形式补充
> 视频此处使用普通点积。Transformer 中常见的补充写法是在 softmax 前计算
> $$
> A=\frac{K^{\mathsf T}Q}{\sqrt{d_k}},
> $$
> 用来避免 $d_k$ 较大时点积数值过大。它不改变后续流程。

## 二、多头 Self-Attention

单头只有一种学习“相关性”的方式。多头把每个位置的 $q^i,k^i,v^i$ 再投影成多组，让不同组学习不同类型的相关性。

以第 $r$ 个头为例，视频中的写法是：

$$
q^{i,r}=W^{q,r}q^i,\qquad
k^{i,r}=W^{k,r}k^i,\qquad
v^{i,r}=W^{v,r}v^i.
$$

这个头在自己的组内计算：

$$
\alpha^{r}_{i,j}=(q^{i,r})^{\mathsf T}k^{j,r},
$$

$$
\alpha'^{r}_{i,j}
=\frac{\exp(\alpha^{r}_{i,j})}
{\sum_m\exp(\alpha^{r}_{i,m})},
\qquad
b^{i,r}=\sum_j\alpha'^{r}_{i,j}v^{j,r}.
$$

每个头都看到完整序列，并在自己的头内部做 softmax 和对 $v$ 的加权求和。各头输出不是再做一次 softmax，也不是简单平均，而是先拼接，再通过可学习矩阵 $W^O$：

$$
b^i=W^O
\begin{bmatrix}
b^{i,1}\\
b^{i,2}\\
\vdots\\
b^{i,H}
\end{bmatrix}.
$$

一次前向计算中，各头在汇总前相互独立；$W^O$ 把拼接结果混合并恢复到模型需要的维度。各头由同一个最终目标共同训练，所以整体上共同协作。

## 三、位置编码

纯 Self-Attention 的公式只使用输入内容，没有直接使用“第几个位置”。因此给第 $i$ 个位置加入位置向量：

$$
\widetilde a^i=a^i+e^i.
$$

位置会继续参与 Q、K、V 的计算。例如：

$$
q^i=W^q\widetilde a^i
=W^qa^i+W^qe^i.
$$

所以位置贡献没有消失，而是融入了新的表示。位置向量 $e^i$ 可以人工设计，也可以从数据中学习；本节没有要求推导正弦、余弦公式。

不同的“内容 + 位置”理论上可能碰巧相同；一旦完全相同，后续就无法区分，但高维表示中精确碰撞通常很少。

## 四、长序列与应用

长度为 $L$ 的序列会产生 $L\times L$ 的注意力矩阵，共有 $L^2$ 个分数。因此完整 Self-Attention 在长序列上计算和存储开销较大。

- **语音**：序列很长，可以使用 Truncated Self-Attention，只关注附近范围，以减少开销。
- **图像**：可以把像素或图像区域看成一组向量，让远距离区域直接交互。
- **图结构**：节点是向量，并利用图的边限制每个节点只关注相连节点；这可以构成一种图神经网络。
- **NLP**：Transformer、BERT 等模型广泛使用 Self-Attention；本节只作应用举例。

## 五、与 CNN、RNN 的比较

| | CNN | RNN | Self-Attention |
|---|---|---|---|
| 信息交互 | 人工规定的局部感受野 | 按顺序通过隐藏状态传递 | 根据内容直接计算任意位置之间的相关性 |
| 并行性 | 可以并行 | 后一步依赖前一步，难以并行 | 所有位置可以并行 |
| 长距离关系 | 需要多层逐步扩大感受野 | 需要经过许多隐藏状态 | 两个位置可以直接建立联系 |
| 主要代价/限制 | 灵活性较低但局部先验强 | 长期信息可能逐渐丢失 | 注意力矩阵开销随 $L^2$ 增长 |

老师所说“CNN 是简化的 Self-Attention”主要是指：CNN 预先限制只能看局部区域，而 Self-Attention 可以从数据中学习应该关注哪些位置。这是帮助理解的比较，不表示二者在数学上完全相等。

数据较少时，CNN 的局部先验可能更有帮助；数据足够多时，Self-Attention 的灵活性更可能体现优势。这是经验性规律，不是无条件成立的定理。

### 常见模型概念地图

| 名称 | 属于什么 | 核心思路 |
|---|---|---|
| CNN | 网络结构 | 用局部窗口提取特征 |
| RNN | 序列网络结构 | 按顺序传递隐藏状态 |
| LSTM | 一种改进 RNN | 用门结构控制记忆、遗忘和输出 |
| GAN | 生成模型训练框架 | 生成器与判别器相互对抗训练 |

LSTM 属于 RNN；GAN 与前三者不在同一分类层级，它可以让不同类型的生成器参与对抗训练。

## 六、容易混淆的点

- 视频始终使用“一个位置对应一列”的记法，所以核心公式是 $A=K^{\mathsf T}Q$、$O=VA'$。
- 对固定 Query $i$，softmax 沿注意力矩阵的第 $i$ 列进行。
- 每个注意力头都看到完整输入，不是把不同位置分给不同头。
- softmax 发生在每个头内部；多头之间通过“拼接 + $W^O$”汇总。
- 位置编码参与后续数值运算，而不是作为一个独立标签保存在旁边。
- CNN、RNN、LSTM 是网络结构概念；GAN 是包含生成器与判别器的训练框架。

## 我的理解与修改

- 

## 掌握检查

- [x] 能说明 $K^{\mathsf T}Q$ 为什么包含所有 Query-Key 点积。
- [x] 能说明 $VA'$ 为什么得到所有加权输出。
- [x] 能解释多头结果如何汇总。
- [x] 能解释为什么要加入位置编码。
- [x] 能区分 Self-Attention、CNN 和 RNN 的基本流程。
- [ ] 之后复习时能够脱稿写出完整矩阵计算链。

## 相关笔记

- [01-Self-Attention（上）](#/notes/self-attention-1)
