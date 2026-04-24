# Mathematics

## 1. Laplace Transform

The Laplace Transform is a mathematical tool that maps a signal from the **Time Domain** to the **Complex Frequency Domain**, also known as the **$s$-plane**.
$$
t \xrightarrow{\mathcal L} s=\sigma + j\omega
$$
To a linear differential equation, we assume a solution of the form:
$$
x(t) = e^{st}
$$
 where **the real part**  $\sigma$ tells how much its decays:

- Left-Half Plane ($\sigma < 0$): The system is **Stable**. The signal decays over time like $e^{-|\sigma|t}$. The further left the pole is, the faster it disappears (faster decay).
- Right-Half Plane ($\sigma > 0$): The system is **Unstable**. The signal grows or "blows up" 
- On the Origin ($\sigma = 0$): The signal is a constant.

and **the imaginary part** $j\omega$ tells how much it oscillates:

- On the Real Axis ($j\omega = 0$): There is **No Oscillation**. The system just slides down or climbs up exponentially (**Overdamped**).
- Away from the Real Axis ($j\omega \neq 0$): The system **Oscillates**. The further the pole is from the horizontal axis, the higher the frequency of the "ringing" or vibration.

The Laplace Transform of a function $f(t)$ is defined by the following interal:
$$
\mathcal L\{f(t)\} = F(s) = \int^{\infin}_{0}f(t)e^{-st}dt
$$
Using the example:
$$
\mathcal L^{-1} \left\{ \frac{1}{s-a} \right\} = e^{at}
$$
We can see a crucial insight: the **Poles** in the transformed function expose the **exponential pieces of the original.** 

Consider a second-order linear differential equation representing a forced mass-spring-damper system:
$$
mx''(t)+\mu x'(t)+kx(t)=F_0 \cos(\omega t)
$$
To solve this equation in the frequency domain, we apply the Laplace Transform to both sides. Recall the differentiation properties of the Laplace Transform:
$$
\begin{aligned}
x''(t) &\xrightarrow{\mathcal L} s^2X(s)-sx(0)-x'(0) \\
x'(t) &\xrightarrow{\mathcal L} sX(s)-x(0) \\
x(t) &\xrightarrow{\mathcal L} X(s)
\end{aligned}
$$
For simplicity, we assume the system starts from rest with zero initial conditions:
$$
x(0)=0 \ \ x'(0)=0
$$
Additionally, the transform of the harmonic input is given by:
$$
\cos(\omega t) \xrightarrow{\mathcal L} \frac{s}{s^2+\omega^2}
$$
Substituting these into our original differential equation, we obtain the algebraic representation in the $s$-domain:
$$
X(s)(ms^2+\mu s+k) = \frac{F_0 s}{s^2+\omega ^2}
$$
Solving for $X(s)$, we find the total response of the system:
$$
X(s)=\frac{F_0 s}{(s^2+\omega ^2)(ms^2+\mu s+k)}
$$

- The **Transient Response** (Natural Response) is steaming from the $(ms^2+\mu s+k)$, these are the "natural vibrations" of the system. If $\mu > 0$, these terms involve $e^{-\alpha t}$ and will eventually die out. This represents how the system behaves immediately after the force is applied before it settles down.
- The **Steady-State Response** (Forced Response) is steaming from the $(s^2+\omega ^2)$ term, this is the vibration maintained by the external force, regardless of its own natural frequency.

Then we can use **Partial Fraction Decomposition**:
$$
X(s)=\frac{F_0s/m}{(s-r_1)(s-r_2)(s-r_3)(s-r_4)}
$$
We assume $\mu = 0$ to simplify the case:
$$
X(s) = \frac{F_0}{m(k/m-\omega^2)}(\frac{s}{s^2+\omega^2}-\frac{s}{s^2+k/m})
$$
Then  $\xrightarrow{\mathcal L^{-1}}$
$$
x(t) = \frac{F_0}{m(k/m-\omega^2)}\left(\cos(\omega t)-\cos(\sqrt{\frac{k}{m}}t) \right)
$$
Now we defined $\omega_n = \sqrt{k/m}$ as the natural frequency. If we drive an undamped system at its exact natural frequency $\omega = \omega_n$, the partial fraction form above fails (driven by zero). Instead, the tranform becomes:
$$
X(s) = \frac{F_0/m \cdot s}{(s^2+\omega_n ^2)^2}
$$
Then  $\xrightarrow{\mathcal L^{-1}}$
$$
x(t) = \frac{F_0}{2m\omega_n}t\sin(\omega_nt)
$$
Notice the $t$ factor outside the sine function. This means the amplitude increase linearly with time to infinity. This is why soldiers break step when crossing bridges—to avoid matching the natural frequency and causing structural failure. This phenomenon is called **Resonance**.







## 2. Divergence and Curl

### 2.1 梯度

定义
$$
\mathbf{grad} \ u = \frac{\partial u}{\partial x} \mathbf{i} +\frac{\partial u}{\partial y} \mathbf{j} + \frac{\partial u}{\partial z} \mathbf{k} = \left(\frac{\partial u}{\partial x},\frac{\partial u}{\partial y},\frac{\partial u}{\partial z}\right)
$$
而函数$u$在点$P_0(x_0,y_0,z_0)$处沿任一方向$\mathbf{e}_{l}=(\cos \alpha,\cos \beta,\cos \gamma)$的方向导数为
$$
\left. \frac{\partial u}{\partial \mathbf{l}} \right|_{P_o} = \left. \frac{\partial u}{\partial x} \right|_{P_o} \cos \alpha + \left. \frac{\partial u}{\partial y} \right|_{P_o} \cos \beta + \left. \frac{\partial u}{\partial z} \right|_{P_o} \cos \gamma=\mathbf{grad} \ u(P_0) \cdot \mathbf{e}
$$
由此可知，

1. $u$在点$P_0$处沿方向$\mathbf{l}$的方向导数，等于梯度在方向$\mathbf{l}$上的投影
2. 当两者的夹角为0，即$\mathbf{e}_l$的方向和梯度方向$\mathbf{grad} \ u(P_0)$一致时，函数$u(P)$在点$P_0$沿梯度方向的方向导数${\partial u}/{\partial \mathbf{l}}$取到最大值，最大值等于梯度的模$\left| \mathbf{grad} \ u(P_0) \right|$

这就是说，$u$在点$P_0$的梯度方向是$u$值增长得最快的方向。

### 2.2 散度

设$\mathbf{A}(x,y,z)=(P(x,y,z),Q(x,y,z),R(x,y,z))$为空间区域$V$上的向量函数，对$V$上的一点$M(x,y,z)$，

定义
$$
\mathrm{div}\ \mathbf{A} = \frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}
$$
为向量函数$A$在$M(x,y,z)$处的散度。

引入高斯公式
$$
\iiint\limits_{V} \left(\frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}\right)dxdydz=\oiint\limits_{S}Pdydz+Qdzdx+Pdxdy
$$
设$\mathbf{e}_n=(\cos\alpha,\cos\beta,\cos\gamma)$为曲面的单位法向量，则$\mathbf{S}=\mathbf{e}_ndS$称为曲面的面积元素，得到
$$
\iiint\limits_{V}\mathrm{div}\mathbf{A}dV=\oiint\limits_{S} \mathbf{A} \cdot d\mathbf{S}
$$
在$V$中任取一点$M_0$，对三重积分应用中值定理，得
$$
\iiint\limits_{V}\mathrm{div}\mathbf{A}dV = \mathrm{div}\mathbf{A}(M^*)\cdot\Delta V
$$
 其中$M^*$为$V$中的某一点，于是有
$$
\mathrm{div}\mathbf{A}(M^*)=\frac{\displaystyle \oiint\limits_{S} \mathbf{A} \cdot d\mathbf{S}}{\Delta V}
$$
令$V$收缩到点$M_0$（记为$V\to M_0$），则$M^*$也趋向点$M_0$，因此
$$
\mathrm{div}\mathbf{A}(M_0)=\lim_{V \to M_0}\frac{\displaystyle \oiint\limits_{S} \mathbf{A} \cdot d\mathbf{S}}{\Delta V}
$$
根据第二类曲面积分的定义，当流速为$\mathbf{A}$的不可压缩流体，经过封闭曲面$S$的流量是$\displaystyle\oiint\limits_{S} \mathbf{A} \cdot d\mathbf{S}$，于是上式表明$\mathrm{div} \mathbf{A}$是流量对体积$V$的变化率，并称它为$\mathbf{A}$在点$M_0$的流量密度。若$\mathrm{div} \mathbf{A}(M_0)>0$，说明在每一个单位时间内有一点数量的流体流出这一点，则称这一点为源；相反，若$\mathrm{div} \mathbf{A}(M_0)<0$，说明流体在这一点吸收，则称这点为汇。若在向量场$\mathbf{A}$中每一点皆有$\mathrm{div} \mathbf{A}=0$，则称$\mathbf{A}$为无源场。

### 2.3 旋度

设$\mathbf{A}(x,y,z)=(P(x,y,z),Q(x,y,z),R(x,y,z))$为空间区域$V$上的向量函数，对$V$上的一点$M(x,y,z)$，

定义
$$
\mathbf{rot \ A} = 
\left|
\begin{matrix}
	\mathbf{i} & \mathbf{j} & \mathbf{k} \\
	\displaystyle \frac{\partial}{\partial x} & \displaystyle \frac{\partial}{\partial y} & \displaystyle \frac{\partial}{\partial z} \\
	P & Q & R
\end{matrix}
\right|
$$
也可以记为向量函数
$$
\mathbf{rot \ A} =  \left( \frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z}, \frac{\partial P}{\partial z}-\frac{\partial R}{\partial x}, \frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y} \right)
$$
为向量函数$A$在$M(x,y,z)$处的旋度。

引入斯托克斯公式
$$
\iint\limits_S
\left|
\begin{matrix}
	dydz & dzdx & dxdy \\
	\displaystyle \frac{\partial}{\partial x} & \displaystyle \frac{\partial}{\partial y} & \displaystyle \frac{\partial}{\partial z} \\
	P & Q & R
\end{matrix}
\right|
=\oint\limits_L Pdx+Qdy+Rdz
$$
设$\mathbf{e}_T$是曲线$L$上在点$M(x,y,z)$处与指定方向一致的单位切向量，向量$d\mathbf{s}=\mathbf{e}_Tds$称为弧长元素向量，得到
$$
\iint\limits_S
\mathbf{rot \ A} \cdot d \mathbf{S}
=\oint\limits_L \mathbf{A} \cdot d \mathbf{s}
$$
在场$V$中任取一点$M_0$，通过$M_0$点作一平面，在该平面上围绕$M_0$作一任一封闭曲线$L$，记$L$所围区域为$D$，面积也记作$D$，则
$$
\iint\limits_S
\mathbf{rot \ A} \cdot d \mathbf{S}
=\iint\limits_S
\mathbf{rot \ A} \cdot \mathbf{e}_n dS
=\oint\limits_L \mathbf{A} \cdot d \mathbf{s}
=\oint\limits_L (\mathbf{A} \cdot \mathbf{e}_T)ds
$$
该式可以说明：流体的速度场的旋度的法线投影在曲面上对面积的曲面积分等于流体在曲面边界上的环流量。对左端二重积分应用中值定理可得
$$
\iint\limits_S
\mathbf{rot \ A} \cdot \mathbf{e}_n dS = (\mathbf{rot \ A} \cdot \mathbf{e}_n)_{M^*}\cdot D=\oint\limits_L \mathbf{A} \cdot d \mathbf{s}
$$
即
$$
(\mathbf{rot \ A} \cdot \mathbf{e}_n)_{M_0}=\lim_{D \to M_0} \frac{\displaystyle\oint\limits_L \mathbf{A} \cdot d \mathbf{s}}{D}
$$
在流量问题中，我们称$\displaystyle\oint\limits_L \mathbf{A} \cdot d \mathbf{s}$为沿闭曲线$L$的环流量，它表示流速为$A$的不可压缩流体在单位时间内沿闭曲线$L$的流体总量，反映了流体沿$L$流动时的旋转强弱程度，当$\mathrm{rot \ A}=0$时，沿任意封闭曲线的环流量为零，即流体流动时不形成漩涡，称为无旋场。

















