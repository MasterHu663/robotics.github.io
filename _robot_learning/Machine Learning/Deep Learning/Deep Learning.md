---
title: "Deep Learning"
permalink: /ml/dl/
toc: true
mathjax: true
---

## 1. 线性神经网络

### 1.1 线性模型

$$
\hat {\mathbf y}=\mathbf X \mathbf w +b
$$

$\hat{\mathbf y} \in \mathbb R^n$为预测值， $\mathbf X \in \mathbb R^{n \times d}$ 为整个数据集的n个样本，其中$\mathbf X$的每一行是一个样本，每一列是一种特征。  

### 1.2 损失函数

一种模型质量的度量方式，平方误差定义为：

$$
l^{(i)}(\mathbf w, b)=\frac{1}{2}(\hat y ^{(i)}-y^{(i)})^2
$$

在训练模型时，我们希望找一组参数 $(\mathbf w ^ \ast, b^ \ast)$ ，这组参数可以最小化在所有训练样本上的总损失：

$$
\mathbf w^*, b^*=\arg \min_{\mathbf w,b} L(\mathbf w,b)
$$

其中，

$$
L(\mathbf w, b)= \frac{1}{n} \sum^n_{i=1}\frac{1}{2}(\mathbf w^{\mathrm T}\mathbf x^{(i)}+b-y^{(i)})^2
$$

### 1.3 随机梯度下降

$$
\mathbf w \leftarrow \mathbf w -\frac{\eta}{|\mathcal B|}\sum_{i \in \mathcal B}  \partial_w l^{(i)}(\mathbf w , b)=
 \mathbf w -\frac{\eta}{|\mathcal B|}\sum_{i \in \mathcal B}\mathbf x^{(i)}(\mathbf w^{\mathrm T}\mathbf x^{(i)}+b-y^{(i)}),
\\
b \leftarrow b \mathbf w -\frac{\eta}{|\mathcal B|}\sum_{i \in \mathcal B}  \partial_b l^{(i)}(\mathbf w , b)= -\frac{\eta}{|\mathcal B|}\sum_{i \in \mathcal B}(\mathbf w^{\mathrm T}\mathbf x^{(i)}+b-y^{(i)}),
$$

$\eta$（learnng rate）为学习率，$\lvert \mathcal B \lvert$（batch size）表示每个小批量样本的样本数，两者均为超参数。

### 1.4 正态分布与平方损失

高斯分布：若随机变量$x$具有均值$\mu$和方差$\sigma^2$（标准差$\sigma$），其正态分布概率密度函数如下：

$$
p(x)=\frac{1}{\sqrt {2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)
$$

均方损失误差函数（简称均方损失）可以用于线性回归的一个原因是：我们假设观测中包含噪声，其中噪声服从正态分布。噪声正态分布如下：

$$
y=\mathbf w^\mathrm T \mathbf x + b + \mathbf \epsilon
$$

其中，$\epsilon \sim \mathcal N(0,\sigma^2)$。因此，我们现在可以写出通过给定的$\mathbf x$观测到特定$y$的似然：

$$
P(y \ |\ \mathbf x) =\frac{1}{\sqrt {2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(y-\mathbf w^\mathrm T \mathbf x - b)^2)
$$

现在，根据极大似然估计法，参数$\mathbf w$和b的最优值是使整个数据集的似然最大的值：

$$
P(\mathbf y \  |\ \mathbf X)= \prod^n_{i=1}p(y^{(i)}|\mathbf x^{(i)})
$$

改为最小化负对数似然：

$$
-\log P (\mathbf y \  |\ \mathbf X)= \sum^n_{i=1}\frac{1}{2}\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}(y^{(i)}-\mathbf w^\mathrm T \mathbf x^{(i)} - b)^2
$$

因此，在高斯噪声的假设下，**最小化均方误差等价于对线性模型的极大似然估计**，以此可以解读平方损失目标函数的。

注：**概率**是已知参数，推测数据出现的可能性，**似然**是已知数据反推参数的可能性

### 1.5 从线性回归到深度网络

将线性回归模型视为仅由单个人工神经元组成的神经网络，称为单层神经网络。对于线性回归，每个输入都与每个输出相连，我们将这种变换称为全连接层或称为稠密层。

### 1.6 softmax回归

机器学习实践者用分类这个词来描述两个有微妙差别的问题： 1. 我们只对样本的“硬性”类别感兴趣，即属于哪个类别； 2. 我们希望得到“软性”类别，即得到属于每个类别的概率。 这两者的界限往往很模糊。其中的一个原因是：即使我们只关心硬类别，我们仍然使用软类别的模型。

为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。 为了解决线性模型的分类问题，我们需要和输出一样多的仿射函数（affine function）。 每个输出对应于它自己的仿射函数：

$$
\mathbf o = \mathbf W \mathbf x + \mathbf b
$$

具体来说，对于任何具有d个输入和q个输出的全连接层， 参数开销为$\mathcal O (dq)$，这个数字在实践中可能高得令人望而却步。 幸运的是，将个d输入转换为q个输出的成本可以减少到$\mathcal O (\frac{dq}{n})$， 其中超参数可以由我们灵活指定，以在实际应用中平衡参数节约和模型有效性。

引入softmax函数：

$$
\hat {\mathbf y} = \mathrm{softmax}(\mathbf o) \ \ \mathrm {where} \ \ \hat y_j = \frac{\exp(o_j)}{\sum_k\exp(o_k)}
$$

这里，对于所有的$j$总有$0\leq \hat y_j \leq 1$。由于softmax运算不会改变为规范化的预测$\mathbf o$之间的大小次序，只会确定分配给每个类别的概率。因此，在预测过程中，我们仍然可以用下式来选择最有可能的类别：

$$
\arg\max_j \hat y_j = \arg\max_j o_j
$$

尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。 因此，softmax回归是一个线性模型（linear model）。

现在我们需要一个损失函数来度量预测的效果，使用最大似然估计：

$$
-\log P(\mathbf Y|\mathbf X)=\sum_{i=1}^{n}-\log P(\mathbf y^{(i)}|\mathbf x^{(i)})=\sum_{i=1}^{n}l(\mathbf y^{(i)},\hat{\mathbf y}^{(i)}),
$$

其中，对于任何标签$\mathbf y$和模型预测$\hat{\mathbf y}$，损失函数为：

$$
l(\mathbf y, \hat{\mathbf y}) = -\sum^q_{j=1}y_i\log\hat{y}_j
$$

这样的损失函数被称为交叉熵损失，利用softmax的定义，我们得到：

$$
\begin{aligned}
l(\mathbf y, \hat{\mathbf y}) 
&= -\sum^q_{j=1}y_i\log\frac{\exp(o_j)}{\sum_{k=1}^q\exp(o_k)} \\
&= \sum^q_{j=1}y_i\log\sum^q_{k=1}\exp(o_k)-\sum^q_{j=1}y_j o_j \\
&= \log\sum^q_{k=1}\exp(o_k)-\sum^q_{j=1}y_j o_j
\end{aligned}
$$

考虑相对于任何未规范化的预测$o_j$的导数，我们得到：

$$
\partial_{o_j}l(\mathbf y, \hat{\mathbf y}) = \frac{\exp(o_j)}{\sum^q_{k=1}\exp(o_k)}-y_i = \mathrm {softmax}(\mathbf o)_j-y_j
$$

### 附：从信息论角度理解交叉熵

信息论的核心思想是量化数据中的信息内容。在信息论中，该数值被称为分布$P$的熵，可以通过以下方程得到：

$$
H[P]=\sum_j -P(j)\log P(j)
$$

信息论的基本定理之一指出，为了从分布$p$中随机抽取的数据进行编码，我们至少需要$H[P]$对其编码。我们可以把熵$H(P)$想象为“知道真实概率的人所经历的惊异程度”，则交叉熵从$P$到$Q$，记为$H(P,Q)$。我们可以把交叉熵想象为“主观概率为$Q$的观察者在看到根据概率$P$生成的数据时的预期惊异“。**交叉熵是一个衡量两个概率分布之间差异的很好的度量，它测量给定模型编码数据所需的比特数。**

## 2. 多层感知机

线性神经网络模型通过单个**仿射变换**（带有偏置项的线性变换）将我们的输入直接映射到输出，然后进行softmax操作。但是，仿射变换是一个很强的假设：**任何特征的增大都会导致模型输出的增大或减小**。这在有些情况下是不符合实际的。

我们可以通过在网络中加入一个或多个隐藏层来克服线性模型的限制，使其能处理更普遍的函数关系类型。这种架构通常被称为多层感知机。

<img src="./image/mlp.svg" alt="mlp" style="zoom:100%;" />

### 2.1 从线性到非线性

我们通过$\mathbf X \in \mathbb R^{n \times d}$来表示$n$个样本的小批量，其中每个样本具有$d$个输出特征。对于具有$h$个隐藏单元的单隐藏层多层感知机，用$\mathbf H \in \mathbb R^{n \times h}$表示隐藏层的输出，称为隐藏表示。按如下方式计算单隐藏层多层感知机的输出$\mathbf O \in \mathbb R^{n \times q}$：

$$
\mathbf H = \mathbf X \mathbf W^{(1)} + \mathbf b^{(1)} \\
\mathbf O = \mathbf H \mathbf W^{(2)} + \mathbf b^{(2)}
$$

但是实际上我们从这样的隐藏层中得不到任何好处，我们可以直接建立和两层模型的完全等价的单层模型：

$$
\begin{align}
\mathbf O 
&= (\mathbf X \mathbf W^{(1)} + \mathbf b^{(1)})\mathbf W^{(2)} + \mathbf b^{(2)} \\
&= \mathbf X \mathbf W^{(1)} \mathbf W^{(2)} + \mathbf b^{(1)} \mathbf W^{(2)} + \mathbf b^{(2)} \\
&= \mathbf X \mathbf W + \mathbf b
\end{align}
$$

为了发挥多层架构的潜力，我们需要在仿射变换之后对每个隐藏层单元应用非线性的**激活函数**$\sigma$，激活函数的输出（例如，$\sigma(\cdot)$）被称为活性值。

$$
\mathbf H = \sigma(\mathbf X \mathbf W^{(1)} + \mathbf b^{(1)}) \\
\mathbf O = \mathbf H \mathbf W^{(2)} + \mathbf b^{(2)}
$$

为了构建更通用的多层感知机，我们可以继续堆叠这样的隐藏层。

### 2.3 激活函数

#### 2.3.1 ReLU函数

最受欢迎的激活函数是修正线性单元，因为它实现简单，同时在各种预测任务中表现良好。

$$
\mathrm {ReLU}(x) = \max (x,0)
$$

通俗来讲，ReLU函数通过将对应的活性值设置为0，仅保留正元素丢弃所有负元素。使用ReLU的原因是，它求导表现得特别好，要么让参数消失，要么让参数通过。这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题。

#### 2.3.2 sigmoid函数

该函数将输入变换为区间(0,1)上的输出。因此，sigmoid通常称为挤压函数：他将范围(-inf,inf)中的任意输入压缩到区间(0,1)中的某个值：

$$
\mathrm{sigmoid}(x)=\frac{1}{1+\exp(-x)}
$$

#### 2.3.3 tanh函数

与sigmoid函数类似，tanh(双曲正切)函数也能将其输入压缩转换到区间(-1,1)上，tanh函数的公式如下：

$$
\tanh(x) = \frac{1-\exp(-2x)}{1+\exp(-2x)}
$$

### 2.4 权重衰减

为了缓解过拟合的问题，在训练参数化机器学习模型的时候，权重衰减是最广泛使用的正则化的技术之一，这项技术通过函数与零的距离来衡量函数的复杂度，我们可以用线性函数$f(\mathbf x)=\mathbf w^{\mathrm T}\mathbf x$中的权重向量的某个范数（通常使用$L_2$范数）作为惩罚项加到最小化损失的问题中。

我们使用正则化常数$\lambda$来平衡这个新的额外惩罚的损失：

$$
L(\mathbf w,b)+\frac{\lambda}{2}||\mathbf w||^2
$$

则$L_2$正则化回归的小批量随机梯度下降更新如下式：

$$
\mathbf w \leftarrow 
 (1-\eta\lambda) \mathbf w -\frac{\eta}{|\mathcal B|}\sum_{i \in \mathcal B}\mathbf x^{(i)}(\mathbf w^{\mathrm T}\mathbf x^{(i)}+b-y^{(i)})
$$

### 2.5 暂退法 (Dropout)

经典的泛化理论认为，为了缩小训练和测试之间的差距，应该以简单的模型为目标。在标准暂退法正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差，每个中间活性值$h$以暂退概率$p$由随机变量$h'$替换：

$$
h' = \begin{cases}
0 \ \ \ \ \ \  \ \mathrm {with\ probability} \ p \\
\frac{h}{1-p} \ \ \mathrm {others}
\end{cases}
$$

根据此模型的设计，其期望值保持不变，即$E[h']=h$

## 3. 卷积神经网络

### 3.1 卷积层

在图像领域，对于高维感知数据，多层感知机的参数太多，要训练这个模型将不可实现。但是我们可以利用图像中本就丰富的结构来使用机器学习模型。我们用$i,j$表示输入图像和隐藏表示中位置$(i,j)$处的像素：

$$
\begin{aligned}
\ [\mathbf H]_{i,j}
&=[\mathbf U]_{i,j} + \sum_k \sum_l [W]_{i,j,k,l}[\mathbf X]_{k,l} \\
&=[\mathbf U]_{i,j} + \sum_a \sum_b [W]_{i,j,a,b}[\mathbf X]_{i+a,j+b} \\
\end{aligned}
$$

#### 3.1.1 平移不变性

**不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应。**这意味着检测对象在输入$\mathbf X$中的平移，应该仅导致隐藏表示中$\mathbf H$的平移，$V$和$U$实际不依赖位置$(i,j)$，因此：

$$
[\mathbf H]_{i,j}=u+\sum_a\sum_b[\mathbf V]_{a,b}[\mathbf X]_{i+a,j+b}
$$

#### 3.1.2 局部性

**神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，聚合这些局部特征，以在整个图像级别进行预测。**为了收集用来训练参数$\left [\mathbf H \right]_{i,j}$的相关信息，我们不应该偏离到距$(i,j)$很远的地方，这意味着在$\lvert a \lvert >\Delta$或$\lvert b \lvert>\Delta$的范围之外，我们可以设置$\left [\mathbf V \right] _{a,b}=0$，因此：

$$
[\mathbf H]_{i,j} = u +\sum_{a=-\Delta}^{\Delta}\sum_{b=-\Delta}^{\Delta}
[\mathbf V]_{a,b}[\mathbf X]_{i+a,j+b}
$$

#### 3.1.3 卷积

在数学上，卷积被定义为：

$$
(f*g)(x)=\int f(\mathbf z)g(\mathbf x -\mathbf z)d \mathbf z
$$

也就是说，卷积是把一个函数“翻转”并移位$\mathbf x$时，测量$f$和$g$之间的重叠。当为离散对象时，积分就变成了求和。对于二维张量，我们定义为：

$$
(f*g)(i,j)=\sum_a \sum_bf(a,b)g(i-a,j-b)
$$

这个形式和我们上面推导的公式类似，因此被称为卷积神经网络。

### 3.2 汇聚层

我们的机器学习任务通常会跟全局图像的问题有关（例如，图像中是否包含一只猫？），所以我们最后一层的神经元应该对整个输入的全局敏感。通过逐渐聚合信息，生成越来越粗糙的映射，最终实现学习全局表示的目标，同时将卷积图层的所有优势保留在中间层。那么汇聚层具有双重目的为**：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。**汇聚层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输出的所有区域上滑动，计算窗口中子张量的**最大值或平均值**。

### 3.3 残差网络*（ResNet）

在深层网络中，随着层数的增加，训练往往会出现退化问题（更深的网络反而训练误差更大），ResNet的核心思想是：即使新加的层什么都没有学到，它至少可以退化为恒等映射$f(\mathbf x)= \mathbf x$，不会让性能变差。

## 4. 循环神经网络

### 4.1 









## Transformer

### 1. 注意力汇聚：Nadaraya-Watson核回归

注意力机制：**查询（自主提示）和键（非自主提示）之间的交互形成了注意力汇聚； 注意力汇聚有选择地聚合了值（感官输入）以生成最终的输出。**

根据输入位置对输出的输出$y_i$进行加权：

$$
f(x)=\sum^n_{i=1}\frac{K(x-x_i)}{\sum_{j=1}^nK(x-x_j)}y_i
$$

其中$K$是核。考虑一个高斯核，其定义为：

$$
K(u)=\frac{1}{\sqrt{2\pi}}\exp(-\frac{u^2}{2})
$$

代入可以得到：

$$
f(x) = \sum^n_{i=1} \mathrm{softmax}(-\frac{1}{2}(x-x_i)^2)y_i
$$

如果一个键$x_i$越是接近给定的查询$x$，那么分配给这个键对应值$y_i$的注意力权重就会越大，也就“获得了更多的注意力”。

接下来，我们将可学习的参数集成到注意力汇聚中：

$$
f(x) = \sum_{i=1}^n\mathrm{softmax}(-\frac{1}{2}((x-x_i)\omega)^2)y_i
$$

### 2. 注意力评分函数

**高斯核指数部分可以视为注意力评分函数**

#### 2.1 加性注意力

给定查询$\mathbf q \in \mathbb R^q$和$\mathbf k \in \mathbb R^k$，加性注意力的评分函数为

$$
a(\mathbf q, \mathbf k) = \mathbf w_v^{\mathrm T}\tanh (\mathbf W_q\mathbf q+\mathbf W_k\mathbf k)\in \mathbb R
$$

其中的参数可以学习，将查询和键连结起来后输入到一个多层感知机（MLP）中，感知机包含一个隐藏层，其隐藏单元数是一个超参数$h$。通过使用$\tanh$作为激活函数，并且禁用偏置项。

#### 2.2 缩放点积注意力

使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度$d$。 假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差， 那么两个向量的点积的均值为0，方差为$d$。 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是1， 我们再将点积除以， 则*缩放点积注意力*（scaled dot-product attention）评分函数为：

$$
a(\mathbf q,\mathbf k) = \mathbf q^{\mathrm T}\mathbf k/\sqrt{d}
$$

最终的缩放点积注意力为：

$$
\mathrm {Attention}(\mathbf Q,\mathbf K,\mathbf V)=\mathrm{softmax}(\frac{\mathbf Q\mathbf K^{\mathrm T}}{\sqrt{d}})\mathbf V \in \mathbb R^{n \times v}
$$

其中，查询$\mathbf Q \in \mathbb R^{n \times d}$、键$\mathbf K \in \mathbb R^{m \times d}$和值$\mathbf V \in \mathbb R^{m \times v}$
