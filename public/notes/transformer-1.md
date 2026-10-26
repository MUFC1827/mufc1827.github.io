# 03 Transformer（上）

## 来源

- 视频：【李宏毅机器学习】Transformer（上）
- Bilibili：<https://www.bilibili.com/video/BV1r8nMz4EAj?p=3>
- 李宏毅官方视频：<https://www.youtube.com/watch?v=n9TlOhRjYoc>
- 官方讲义 Seq2Seq / Transformer：<https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/seq2seq_v9.pdf>
- Batch Normalization 补充讲义：<https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/normalization_v4.pdf>

## 本节主线

Seq2seq 接收一个序列并输出另一个序列，输入和输出长度可以不同。Transformer 分为 Encoder 与 Decoder；本节学习 Encoder：

$$
\boxed{
\text{输入与位置编码}
\rightarrow
\text{Self-Attention}
\rightarrow
\text{Add + LayerNorm}
\rightarrow
\text{FFN}
\rightarrow
\text{Add + LayerNorm}}
$$

Encoder、Decoder 是功能角色，不一定只能由 Transformer 实现，也可以使用 RNN 或 CNN。BERT 的主体使用堆叠的 Transformer Encoder。

## 一、Seq2seq 与 Encoder

$$
\text{输入序列}\longrightarrow\text{模型}\longrightarrow\text{输出序列}.
$$

语音识别、机器翻译、语音合成、聊天机器人等都可以写成 Seq2seq 任务。

采用视频中“一个 token 是一列”的记法：

$$
X=\begin{bmatrix}x^1&x^2&\cdots&x^L\end{bmatrix}
\in\mathbb R^{d_{\mathrm{model}}\times L}.
$$

先加入位置编码：

$$
X^{(0)}=X+P.
$$

经过多个 Encoder Block 后：

$$
H=\begin{bmatrix}h^1&h^2&\cdots&h^L\end{bmatrix}.
$$

Encoder 不直接输出文字，而是输出 $L$ 个上下文向量。每个 $h^i$ 仍对应一个输入位置，但已经融合了其他 token 的信息。

## 二、Fully Connected 与 FFN

最基本的全连接层：

$$
y=Wx+b,
\qquad
y_r=\sum_sW_{r,s}x_s+b_r.
$$

每个输出特征都能使用所有输入特征，所以叫“全连接”。$W,b$ 是训练得到的参数，不是人工填写的规则。

例如：

$$
x=\begin{bmatrix}2\\1\end{bmatrix},
\quad
W=\begin{bmatrix}1&3\\-2&1\end{bmatrix},
\quad
b=\begin{bmatrix}1\\0\end{bmatrix},
$$

$$
Wx+b
=\begin{bmatrix}6\\-3\end{bmatrix}.
$$

Transformer 通常对每个 token 分别使用同一组两层 FFN：

$$
\operatorname{FFN}(u^i)
=W_2\phi(W_1u^i+b_1)+b_2.
$$

第一层通常升维，非线性函数 $\phi$ 增加表达能力，第二层再变回 $d_{\mathrm{model}}$。因此：

- Self-Attention：让不同 token 交换信息；
- FFN：加工一个 token 内部的各个特征。

## 三、Residual Connection

设子层运算为 $F(x)$，残差连接是逐元素相加：

$$
y=x+F(x).
$$

两项形状必须相同。它保留原信息，让子层主要学习“需要增加的修正”。反向传播时：

$$
\frac{\partial y}{\partial x}
=I+\frac{\partial F(x)}{\partial x},
$$

其中 $I$ 提供直接的梯度通路，使深层网络更容易训练。Encoder Block 在 Self-Attention 后和 FFN 后各有一次 Residual。

## 四、BatchNorm 与 LayerNorm

Norm 会减去均值、除以标准差，使数值回到较稳定的尺度：

$$
\widehat z=\frac{z-\mu}{\sigma}.
$$

结果表示相对位置：$0$ 是平均水平，正负表示高于或低于平均值，绝对值表示相距多少个标准差。它不是概率，也不要求元素之和为 $1$。实际实现还会加入很小的 $\varepsilon$，并用可学习的 $\gamma,\beta$ 再缩放和平移。

### Batch Normalization

固定一个特征，跨 batch 中的不同样本计算均值和标准差。设 batch 中有 $B$ 个样本：

$$
\mu_r=\frac1B\sum_{b=1}^{B}Z_{r,b},
\qquad
\operatorname{BN}(Z_{r,b})
=\frac{Z_{r,b}-\mu_r}{\sigma_r}.
$$

Batch 是一次同时送入模型的一组样本，不是一句话中的多个 token。BatchNorm 训练时依赖其他样本，推理时通常使用训练期间保存的统计量。

### Layer Normalization

固定一个 token，在它自己的 $d$ 个特征之间计算：

$$
\mu_i=\frac1d\sum_{r=1}^{d}z^i_r,
\qquad
\operatorname{LN}(z^i)_r
=\frac{z^i_r-\mu_i}{\sigma_i}.
$$

每个 token 单独归一化，不依赖 batch 中其他句子，所以 Transformer 使用 LayerNorm。

| | BatchNorm | LayerNorm |
|---|---|---|
| 固定对象 | 一个特征 | 一个样本或 token |
| 比较范围 | 不同样本的同一特征 | 同一 token 的不同特征 |
| 是否依赖其他样本 | 是 | 否 |
| Transformer | 通常不用 | 使用 |

用矩阵

$$
Z=\begin{bmatrix}1&5\\2&6\\3&7\end{bmatrix}
$$

记忆方向：BatchNorm 横向比较每一行；LayerNorm 纵向比较每一列。可以类比为“BN 固定科目比较不同学生；LN 固定学生比较自己的不同科目”。

## 五、一个 Encoder Block

设第 $\ell$ 个 Block 的输入为 $X^{(\ell-1)}$。

1. 多头 Self-Attention：

$$
S^{(\ell)}
=\operatorname{MultiHeadSelfAttention}(X^{(\ell-1)}).
$$

2. 第一次 Residual + LayerNorm：

$$
U^{(\ell)}
=\operatorname{LayerNorm}
\left(X^{(\ell-1)}+S^{(\ell)}\right).
$$

3. 对每个 token 使用同一组 FFN：

$$
F^{(\ell)}=\operatorname{FFN}(U^{(\ell)}).
$$

4. 第二次 Residual + LayerNorm：

$$
X^{(\ell)}
=\operatorname{LayerNorm}
\left(U^{(\ell)}+F^{(\ell)}\right).
$$

视频与原始 Transformer 使用：

$$
\boxed{
\text{Self-Attention}
\rightarrow\text{Add}
\rightarrow\text{LayerNorm}
\rightarrow\text{FFN}
\rightarrow\text{Add}
\rightarrow\text{LayerNorm}}
$$

多个 Block 依次堆叠，形状通常保持 $d_{\mathrm{model}}\times L$；变化的是向量中包含的信息。

## 六、训练、Loss 与反向传播

Encoder 中的 Embedding、Attention、Residual、LayerNorm 和 FFN 都属于前向传播。训练与推理都会做前向传播，区别是训练之后还要：

$$
\widehat y=f(x;\theta),
\qquad
\mathcal L=\operatorname{Loss}(\widehat y,y),
$$

$$
\frac{\partial\mathcal L}{\partial\theta}
\xrightarrow{\text{反向传播}}
\theta\leftarrow
\theta-\eta\frac{\partial\mathcal L}{\partial\theta}.
$$

Loss 定义“预测有多不好”，梯度说明“参数向哪个方向改变能让 Loss 降低”。只有 Loss 数值不能决定更新方向，但没有 Loss 函数也无法求出梯度。

最小例子：

$$
\widehat y=wx,\qquad
\mathcal L=\frac12(\widehat y-y)^2.
$$

取 $x=2,y=10,w=3$：

$$
\widehat y=6,\qquad
\mathcal L=8,
\qquad
\frac{\partial\mathcal L}{\partial w}
=(\widehat y-y)x=-8.
$$

若 $\eta=0.1$：

$$
w\leftarrow3-0.1(-8)=3.8,
$$

新预测变为 $7.6$，向正确答案靠近。真正的 Transformer 只是同时对大量参数应用同样的链式法则。

可学习参数包括 $W^q,W^k,W^v,W^O$、FFN 参数、Embedding 和 LayerNorm 的 $\gamma,\beta$。softmax 公式、Residual 加法以及均值计算本身不是可学习参数。

## 七、Tensor 与 PyTorch

Tensor 是多维数组：

| 阶数 | 例子 |
|---:|---|
| 0 维 | 一个 Loss 标量 |
| 1 维 | 一个 token 向量 |
| 2 维 | 一句话的 token 矩阵 |
| 3 维 | 一批句子，常见形状 $[B,L,d_{\mathrm{model}}]$ |
| 4 维 | 一批图像，常见形状 $[B,C,H,W]$ |

PyTorch 适合神经网络，因为它：

1. 用 Tensor 统一表示输入、参数和中间结果；
2. 高效调用并行矩阵运算，并可使用 GPU；
3. 记录计算图，通过自动微分计算所有参数梯度。

PyTorch 不会改变反向传播的数学原理，只是高效、自动地完成大量张量运算和链式求导。

## 容易混淆的点

- 一句话中的 $L$ 个 token 不等于 batch；batch 是多条样本。
- LayerNorm 沿一个 token 的特征维计算，不是沿不同 token 计算。
- Residual 是同形状数据逐元素相加，不是拼接或平均。
- Self-Attention 负责跨 token 交流，FFN 负责 token 内部变换。
- Encoder 输出仍是 $L$ 个向量，而不是最终文字或分类结果。
- 本视频采用 Add 后再 LayerNorm；其他 Transformer 变体可能使用不同顺序。

## 我的理解与修改

- 

## 掌握检查

- [x] 能区分 BatchNorm 与 LayerNorm 的统计方向。
- [x] 能解释 Residual 的计算和作用。
- [x] 能解释 Self-Attention 与 FFN 的分工。
- [x] 能写出 Encoder Block 的顺序。
- [x] 能说明 Encoder 输出是什么。
- [x] 能区分前向传播、Loss、梯度和参数更新。
- [x] 能解释 Tensor 与 PyTorch 自动微分的基本作用。

## 相关笔记

- [02-Self-Attention（下）](#/notes/self-attention-2)
- [04-Transformer（下）](#/notes/transformer-2)
