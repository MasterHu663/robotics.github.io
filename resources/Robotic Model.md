---
layout: resource-layout
title: Robotc Model
permalink: /resources/robotc-model
mathjax: true
---

# Robotc Model

## 1. 机器人运动学

### 1.1 空间描述

#### 1.1.1 位置

- 笛卡尔坐标系
  $$
  P=\left[
   \begin{matrix}
     p_x  \\
     p_y  \\
     p_z 
    \end{matrix}
    \right]
  $$

#### 1.1.2 姿态

绕$\mathbf X$轴旋转角$\phi$，旋转矩阵为：

$$
R_\phi=\left[
 \begin{matrix}
   1 & 0 & 0 \\
   0 & \cos\phi & -\sin\phi \\
   0 & \sin\phi & \cos\phi  
  \end{matrix}
  \right]
$$

绕$\mathbf Y$轴旋转角$\theta$，旋转矩阵为：

$$
R_\theta=\left[
 \begin{matrix}
   \cos\theta & 0 & \sin\theta \\
   0 & 1 & 0 \\
   -\sin\theta & 0 & \cos\theta  
  \end{matrix}
  \right]
$$

绕$\mathbf Z$轴旋转角$\psi$，旋转矩阵为：

$$
R_\psi=\left[
 \begin{matrix}
   \cos\psi & -\sin\psi & 0 \\
   \sin\psi & \cos\psi & 0 \\
   0 & 0 & 1  
  \end{matrix}
  \right]
$$

一个坐标系$\{B\}$相对于另一个坐标系$\{A\}$，用旋转矩阵表达：

$$
^A_BR=[\hat X,\hat Y,\hat Z]=\left[
 \begin{matrix}
   r_{11} & r_{12} & r_{13} \\
   r_{21} & r_{22} & r_{23} \\
   r_{31} & r_{32} & r_{33}  
  \end{matrix}
  \right]
$$

9个变量，需要6个约束方程，才能获得3个方向的自由度
$$
|\hat X|=1 \\
|\hat Y|=1 \\
|\hat Z|=1 \\
\hat X \cdot \hat Y=0 \\
\hat X \cdot \hat Z=0 \\
\hat Y \cdot \hat Z=0 \\
$$

但是这样的描述过于复杂，我们可以用三个参数来简便的表示出姿态（旋转一般不满足交换律给姿态的描述带来了困难）:

**a. 固定角**

1. $\mathbf{X-Y-Z}$ 固定角（RPY角）

| Name             | 中文名           | 表达式            | 所绕轴      |
| ---------------- | ---------------- | ----------------- | ----------- |
| $\mathbf{Roll}$  | 回转角（横滚角） | $\phi$、$\phi$  | 绕$X$轴旋转 |
| $\mathbf{Pitch}$ | 俯仰角           | $\theta$、$\theta$ | 绕$Y$轴旋转 |
| $\mathbf{Yaw}$   | 偏转角（航向角） | $\psi$、$\psi$    | 绕$Z$轴旋转 |

先绕$\mathbf{X_A}$轴旋转，再绕$\mathbf{Y_A}$轴旋转，最后$\mathbf{Z_A}$轴旋转

$$
\begin{aligned}
{}^A_B R &= R_Z(\psi)R_Y(\theta)R_X(\phi) \\
&= \begin{bmatrix}
   \cos\psi & -\sin\psi & 0 \\
   \sin\psi & \cos\psi & 0 \\
   0 & 0 & 1  
  \end{bmatrix}
  \begin{bmatrix}
   \cos\theta & 0 & \sin\theta \\
   0 & 1 & 0 \\
   -\sin\theta & 0 & \cos\theta  
  \end{bmatrix}
  \begin{bmatrix}
   1 & 0 & 0 \\
   0 & \cos\phi & -\sin\phi \\
   0 & \sin\phi & \cos\phi  
  \end{bmatrix} \\
&= \begin{bmatrix}
   \cos\psi \cos\theta & \cos\psi \sin\theta \sin\phi - \sin\psi \cos\phi & \cos\psi \sin\theta \cos\phi + \sin\psi \sin\phi \\
   \sin\psi \cos\theta & \sin\psi \sin\theta \sin\phi + \cos\psi \cos\phi & \sin\psi \sin\theta \cos\phi - \cos\psi \sin\phi \\
   -\sin\theta & \cos\theta \sin\phi & \cos\theta \cos\phi
  \end{bmatrix}
\end{aligned}
$$

**b. 欧拉角**

1. $\mathbf{Z-Y-X}$ 欧拉角：先绕$\mathbf{X_B}$轴旋转，再绕$\mathbf{Y_B}$轴旋转，最后$\mathbf{Z_B}$轴旋转（三次绕固定轴旋转的最终姿态和以相反顺序绕运动坐标轴转动的最终姿态相同！）

   $$
   \begin{aligned}
   ^A_BR
   &=R_Z(\psi)R_Y(\theta)R_X(\phi)\\
   &= \left[
   \begin{matrix}
     \cos\psi \cos\theta & \cos\psi \sin\theta \sin\phi - \sin\psi \cos\phi & \cos\psi \sin\theta \cos\phi + \sin\psi \sin\phi \\
     \sin\psi \cos\theta & \sin\psi \sin\theta \sin\phi + \cos\psi \cos\phi & \sin\psi \sin\theta \cos\phi - \cos\psi \sin\phi \\
     -\sin\theta & \cos\theta \sin\phi & \cos\theta \cos\phi
     \end{matrix}
     \right]
   \end{aligned}
   $$

2. $\mathbf{Z-Y-Z}$ 欧拉角：先绕$\mathbf{Z_B}$轴旋转，再绕$\mathbf{Y_B}$轴旋转，最后$\mathbf{Z_B}$轴旋转

   $$
   \begin{aligned}
   ^A_BR
   &=R_Z(\psi)R_Y(\theta)R_Z(\phi)\\
   &= \left[
   \begin{matrix}
     \cos\psi \cos\theta \cos\phi - \sin\psi \sin\phi & -\cos\psi \cos\theta \sin\phi - \sin\psi \cos\phi & \cos\psi \sin\theta\\
     \sin\psi \cos\theta \cos\phi + \cos\psi \sin\phi & -\sin\psi \cos\theta \sin\phi - \cos\psi \cos\phi & \sin\psi \sin\theta\\
     -\sin\theta \cos\phi & \sin\theta \sin\phi & \cos\theta
     \end{matrix}
     \right]
   \end{aligned}
   $$

### 1.2 一般变换

经常有这种情况，我们已知矢量相对于某坐标系$\{B\}$的描述，并且想求出它相对另一个坐标系$\{A\}$的描述。考虑映射的一般情况，坐标系$\{B\}$的原点和坐标系$\{A\}$的原点不重合，有一个偏移量。确定$\{B\}$原点的矢量记为$^AP_{BORG}$。同时$\{B\}$相对$\{A\}$的旋转用$^A_BR$描述，则：
$$
\left[
 \begin{matrix}
   ^AP \\ 1
  \end{matrix}
 \right]
 =
 \left[
 \begin{matrix}
   ^A_BR & ^AP_{BORG} \\
   0 & 1
  \end{matrix}
 \right]
  \left[
 \begin{matrix}
  ^BP \\
  1
  \end{matrix}
 \right]
$$

### 1.3  机械臂运动学

**解决连杆之间的运动学关系**：Denavit-Hartenberg方法（DH法）

表示方法：

$$
a_{i-1} = 沿 \hat X_{i-1} 轴，从\hat Z_{i-1}移动到\hat Z_{i}的距离 \ (连杆长度)
$$

$$
\psi_{i-1} = 沿 \hat X_{i-1} 轴，从\hat Z_{i-1}移动到\hat Z_{i}的角度\ (连杆转角)
$$

$$
d_i = 沿 \hat Z_i 轴，从\hat X_{i-1}移动到\hat X_{i}的距离\ (连杆偏距)
$$

$$
\theta_i = 沿 \hat Z_i 轴，从\hat X_{i-1}移动到\hat X_{i}的距离\ (关节角)
$$

可以推得：

$$
\begin{aligned}
^{i-1}_{i}T
&=R_X(\psi_{i-1})D_X(a_{i-1})R_Z(\theta_{i})D_Z(d_{i}) \\
&=\left[
 \begin{matrix}
   \cos\theta_i & - \sin\theta_i & 0 & a_{i-1}\\
   \sin\theta_i \cos\psi_{i-1} & \cos\theta_i \cos\psi_{i-1} & -\sin\psi_{i-1} & -\sin\psi_{i-1}d_i\\
   \sin\theta_i \sin\psi_{i-1} & \cos\theta_i \sin\psi_{i-1} & \cos\psi_{i-1} & \cos\psi_{i-1}d_i\\
   0 & 0 & 0 & 1
  \end{matrix}
  \right]
\end{aligned}
$$

### 1.4 四元数

各种姿态的表示方式虽然形式不同，但是从本质上来说，它们的描述矩阵是完全一致的，只是表达矩阵的方式有区别而已。另外，这种表达方式往往存在万向节死锁，例如针对$\mathbf{Z-Y-X}$欧拉角，当$\mathbf{Pitch}$处于$\pm 90 ^\circ$（$\theta = \pm 90 ^\circ$）时：（取$\theta = 90 ^\circ$）

$$
\begin{aligned}
^A_BR
&= \left[
\begin{matrix}
  0 & \cos\psi \sin\phi - \sin\psi \cos\phi & \cos\psi \cos\phi + \sin\psi \sin\phi \\
  0 & \sin\psi \sin\phi + \cos\psi \cos\phi & \sin\psi  \cos\phi - \cos\psi \sin\phi \\
  -1 & 0 & 0
  \end{matrix}
  \right] \\
&= \left[
\begin{matrix}
  0 & \sin(\phi-\psi) & \cos(\phi-\psi) \\
  0 & \cos(\phi-\psi) & -\sin(\phi-\psi)\\
  -1 & 0 & 0
  \end{matrix}
  \right]
\end{aligned}
$$

这是所有的项都取决于$(\phi-\psi)$，无法区分两者的独立贡献，它们从两个自由度变成了一个自由度。

我们引入四元数：

$$
q = \omega \mathbf 1 + x \mathbf i + y\mathbf j + z\mathbf k
$$

其中 $\mathbf{i}^2=\mathbf{j}^2=\mathbf{k}^2=\mathbf{i}\mathbf{j}\mathbf{k} = 1$

单位四元数需要满足：

$$
\begin{cases}
  q^\dagger q = 1 \\
  \det(q) = 1
\end{cases}
$$

于是：

$$
q^{\dagger}q = (\omega \mathbf 1 + x \mathbf i + y\mathbf j + z\mathbf k)(\omega \mathbf 1 - x \mathbf i - y\mathbf j - z\mathbf k) = \omega^2+x^2+y^2+z^2=1
$$

用矩阵来表示四元数：

$$
\mathbf 1 =
\left(
\begin{matrix}
  1 & 0 \\
  0 & 1 \\
\end{matrix}
\right),
\mathbf i =
\left(
\begin{matrix}
  0 & -1 \\
  1 & 0 \\
\end{matrix}
\right),
\mathbf j =
\left(
\begin{matrix}
  0 & -i \\
  -i & 0 \\
\end{matrix}
\right),
\mathbf k =
\left(
\begin{matrix}
  i & 0 \\
  0 & -i \\
\end{matrix}
\right)
$$

则：

$$
q =\left(
\begin{matrix}
  \omega+zi & -x-yi \\
  x-yi & \omega-zi \\
\end{matrix}
\right) \Rightarrow \det(q)=\omega^2+x^2+y^2+z^2=1
$$

因此单位四元数需要满足：

$$
\omega^2+x^2+y^2+z^2=1
$$

令：

$$
\omega = \cos \frac{\theta}{2}, \sqrt{x^2+y^2+z^2} = \sin\frac{\theta}{2}
$$

那么

$$
(x,y,z)=\sin\frac{\theta}{2} \cdot \left( \frac{x}{\sin\frac{\theta}{2}}, \frac{y}{\sin\frac{\theta}{2}}, \frac{z}{\sin\frac{\theta}{2}} \right)=\sin\frac{\theta}{2} \mathbf{u}
$$

其中$u$为一个单位向量，于是我们可以写成

$$
q=\cos \frac{\theta}{2} + \sin \frac{\theta}{2} \mathbf u
$$

对于一个三维向量$\mathbf v = (a,b,c) \rightarrow \mathbf v = a\mathbf i + b\mathbf j + z\mathbf k$:

$$
v'=qvq^{-1}
$$

表示绕轴 $\mathbf u$ 旋转了 $\theta$ 角度。

**证明**：

------

引理 **四元数乘法**：对任意四元数$q_1=[s,\mathbf{v}]$，$q_2=[t,\mathbf{u}]$

$$
q_1q_2 = q_1 \otimes q_2=[st-\mathbf{v}\cdot\mathbf{u},s\mathbf{u}+t\mathbf{v}+\mathbf{v}\times\mathbf{u}]
$$

对于该引理，若$s=t=0$，四元数可视为三元向量，$\mathbf{v}\mathbf{u} = -\vec{v} \cdot \vec{u} + \vec{v} \times \vec{u}$为向量的**几何积**。

------

对于与 $u$ 平行的分量 $\mathbf{v_{\parallel}}$ :

$$
\mathbf{v_{\parallel}} \mathbf{u} = = [-\mathbf{v_{\parallel}} \cdot \mathbf{u}, \mathbf{v_{\parallel}} \times \mathbf{u}] = [-\mathbf{v_{\parallel}} \cdot \mathbf{u},0] = \mathbf{u}\mathbf{v_{\parallel}}
$$

进而：

$$
q \mathbf{v_{\parallel}} = \mathbf{v_{\parallel}} q
$$

因此：

$$
\mathbf{v_{\parallel}}'=q\mathbf{v_{\parallel}}q^{-1}=\mathbf{v_{\parallel}}qq^{-1}=\mathbf{v_{\parallel}}
$$

同理我们可以得到：

$$
\mathbf{v_{\perp}} \mathbf{u} = [-\mathbf{v}_{\perp} \cdot \mathbf{u}, \mathbf{v}_{\perp} \times \mathbf{u}]= [0, \mathbf{v}_{\perp} \times \mathbf{u}]=-\mathbf{u} \mathbf{v_{\perp}}
$$

进而：

$$
\begin{aligned}
\mathbf{v_{\perp}} q &= \mathbf{v_{\perp}} (\cos\frac{\theta}{2} + \sin\frac{\theta}{2} \cdot \mathbf{u}) \\
&= \cos\frac{\theta}{2} \cdot \mathbf{v_{\perp}} + \sin\frac{\theta}{2} \cdot (\mathbf{v_{\perp}} \mathbf{u}) \\
&= (\cos\frac{\theta}{2} - \sin\frac{\theta}{2} \cdot \mathbf{u}) \mathbf{v_{\perp}} \\
&= q^* \mathbf{v_{\perp}}
\end{aligned}
$$

又因为$q^*=q^{-1}$，因此：

$$
\mathbf{v_{\perp}}' = q \mathbf{v_{\perp}} q^{-1}=q(q^{-1})^* \mathbf{v_{\perp}}=q^2 \mathbf{v_{\perp}}
$$

而

$$
q^2 = (\cos^2\frac{\theta}{2} - \sin^2\frac{\theta}{2}) + (2\sin\frac{\theta}{2}\cos\frac{\theta}{2}) \cdot \mathbf{u} = \cos\theta + \sin\theta \cdot \mathbf{u}
$$

总结可知：

- **平行部分** $\mathbf{v_{\parallel}}$：变换后仍是 $\mathbf{v_{\parallel}}$（保持不变）

- **垂直部分** $\mathbf{v_{\perp}}$：变换后变成了 $(\cos\theta + \sin\theta \cdot \mathbf{u}) \mathbf{v_{\perp}}$

------

引理：**罗德里格旋转公式** 给定旋转轴 $\mathbf{u}$ 和旋转角度 $\theta$ 后，向量$\mathbf{v}$旋转后

$$
\mathbf{v}' = \mathbf{v} \cos\theta + (\mathbf{u} \times \mathbf{v}) \sin\theta + \mathbf{u}(\mathbf{u} \cdot \mathbf{v})(1 - \cos\theta)
$$

------

令$\mathbf{v} = \mathbf{v_{\perp}}$ 垂直于 $\mathbf u$ ，则 $\mathbf{v_{\perp}}' = \mathbf{v_{\perp}} \cos\theta + (\mathbf{u} \times \mathbf{v_{\perp}}) \sin\theta = (\cos\theta + \sin\theta \cdot \mathbf{u}) \mathbf{v_{\perp}}$，因此平行部分保持不变，垂直部分绕 $\mathbf u$ 旋转 $\theta$ 角度，原命题得证。

由于GPU的发展，矩阵乘法的计算效率极高，所以我们常常将四元数转换为旋转矩阵：

$$
R = \begin{bmatrix} 1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\ 2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\ 2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2 \end{bmatrix}
$$

值得注意的是，**每一个四元数都对应唯一的旋转矩阵，但每一个旋转矩阵却对应两个四元数**。如果我们有一个单位四元数 $q$，它代表某种旋转。如果我们取它的相反数 $-q$（即把 $w, x, y, z$ 全部变号），你会发现 $-q$ 代表的是**完全相同的旋转姿态**。

另外，欧拉角也可以转换为四元数

$$
\mathbf{q} = \begin{bmatrix} w \\ x \\ y \\ z \end{bmatrix} = 
\begin{bmatrix}
\cos\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2} \\
\sin\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} - \cos\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2} \\
\cos\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} \\
\cos\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} - \sin\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2}
\end{bmatrix}
$$

四元数有一个重要的作用，也是比欧拉角要优越的点，就是能够很好（不产生万向节死锁）地让一个物体从姿态 A（四元数 $q_1$）平滑地转动到姿态 B（四元数 $q_2$），平滑的过程被称为四元数插值，其中运用最广泛的插值方法为 **球面线性插值 (Slerp - Spherical Linear Interpolation, Slerp)**

先对两个四元数进行点乘 $d = q_1 \cdot q_2=\cos(\Omega)$，$\Omega$为两种姿态向量的夹角：

1. **$d > 0$**：两个四元数的夹角是锐角（小于 $90^\circ$），它们处于同一个半球。这时候直接插值，路径就是最短的。
2. **$d < 0$**：夹角是钝角（大于 $90^\circ$）。这意味着它们虽然指向同一个旋转姿态，但如果你直接插值，物体会绕着远路旋转。

如果我们发现点乘结果 $d < 0$，我们只需要把其中一个四元数（比如 $q_2$）的所有分量取反，即使用 $-q_2$。

接着，通过在平面利用分量分解（画图作垂线，具体过程略）可得：

$$
q_{t} = \frac{\sin((1-t)\Omega)}{\sin\Omega}q_1 + \frac{\sin(t\Omega)}{\sin\Omega}q_2
$$

<div style="background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 12px; padding: 20px; text-align: center; margin: 20px 0;">
    <img src="{{ '/assets/images/slerp-preview.png' | relative_url }}" alt="Slerp Demo Preview" style="width: 100%; max-width: 500px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <p style="margin-top: 15px; font-weight: bold; color: #333;">PaperPlane Attitude Visualization (Three.js)</p>
    <a href="{{ '/projects/paperplane-slerp.html' | relative_url }}" target="_blank" style="display: inline-block; background: #10b981; color: white; padding: 10px 24px; border-radius: 6px; text-decoration: none; font-weight: bold; transition: 0.3s;">
        进入全屏交互演示
    </a>
</div>

### 1.4 刚体姿态运动学

角速度（绕随体坐标系的角速度）和欧拉角速度（由于RPY角和ZYX角实质上是等价旋转，这里直接用欧拉角来表述回转、俯仰、偏转角）之间的变换：

$$
\left[
 \begin{matrix}
   \omega_{bx} \\
   \omega_{by} \\
   \omega_{bz}
 \end{matrix}
\right] = 
R^T_X(\phi)R^T_Y(\theta)\left[
 \begin{matrix}
   0 \\
   0 \\
   \dot \psi
 \end{matrix}
\right]+
R^T_X(\phi)\left[
 \begin{matrix}
   0 \\
   \dot \theta \\
   0
 \end{matrix}
\right]
+\left[
 \begin{matrix}
   \dot \phi \\
   0 \\
   0
 \end{matrix}
\right]\\
=\left[
 \begin{matrix}
   1 & 0 & -\sin\theta \\
   0 & \cos\phi & \cos\theta\sin\phi  \\
   0 & -\sin\phi & \cos\theta\cos\phi
 \end{matrix}
\right]\left[
 \begin{matrix}
   \dot \phi \\
   \dot \theta \\
   \dot \psi
 \end{matrix}
\right]
$$

值得注意的是，欧拉角速度本质上是在**表述世界坐标系下，随体坐标系各轴的的角速度，是刚体绕世界坐标系转动的表述。**同时，发过来可以说明，**欧拉角实质上是刚体的姿态是随体坐标系各轴在世界坐标系下的描述。**

从而，我们可以得到：

$$
\left[
 \begin{matrix}
   \dot \phi \\
   \dot \theta \\
   \dot \psi
 \end{matrix}
\right]
=\left[
 \begin{matrix}
   1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\
   0 & \cos\phi & -\sin\phi  \\
   0 & \sin\phi\cos\theta & \cos\phi/\cos\theta
 \end{matrix}
\right]\left[
 \begin{matrix}
   \omega_{bx} \\
   \omega_{by} \\
   \omega_{bz}
 \end{matrix}
\right]
$$

当 $\tan\theta = \infty$ 时 ，形成了奇异点，也就是刚体经过了万向节死锁的点，在计算机仿真中表现为数值溢出。

我们用四元数来描述刚体姿态动力学：

假设在$t$时刻，刚体的姿态为$q(t)$。在极短的时间$\Delta t$内，刚体绕着瞬时角速度轴$\mathbf{u}$ 旋转了角度 $\Delta \theta = \|\mathbf{u}\| \Delta t$​。这个**微小旋转**可以用一个增量四元数 $\Delta q$ 来表示。根据四元数定义：

$$
\Delta q = \begin{bmatrix} \cos(\frac{\Delta \theta}{2}) \\ \mathbf{n} \sin(\frac{\Delta \theta}{2}) \end{bmatrix}
$$

其中 $\mathbf{n} = {\mathbf{u}}/{\|\mathbf{u}\|}$ 是旋转轴单位向量。使用泰勒展开：

$$
\Delta q \approx \begin{bmatrix} 1 \\ \|\mathbf{u}\| \Delta t/2 \end{bmatrix} = [1, 0, 0, 0]^{\top} + \frac{1}{2} \begin{bmatrix} 0 \\ \mathbf{u} \end{bmatrix} \Delta t
$$

在体坐标系下，旋转是右乘

$$
q(t + \Delta t) = q(t) \otimes \Delta =q(t) \otimes \left( [1, \mathbf{0}]^\top + \frac{1}{2} \mathbf{u}_q \Delta t \right)  = q(t) + \frac{1}{2} (q(t) \otimes \mathbf{u}_q) \Delta t
$$

其中，$\mathbf{u}_q = [ 0, \mathbf{u} ]^\top$，根据导数的定义：

$$
\dot{q} = \lim_{\Delta t \to 0} \frac{q(t + \Delta t) - q(t)}{\Delta t} =\frac{1}{2} q \otimes \mathbf{u}_q
$$

这个瞬时角速度轴 $\mathbf{u}$ 可以为随体坐标 $\mathbf{\omega_b} = [\omega_{bx} , \omega_{by} , \omega_{bz} ]^\top$，则 $\mathbf{u}_q = [0, \omega_x, \omega_y, \omega_z]^\top$定义为$\mathbf{\omega_q}$，因此

$$
\dot{q} = \frac{1}{2} q \otimes \mathbf{\omega_q}, \mathbf{\omega_q} = [0, \omega_{bx} , \omega_{by} , \omega_{bz} ]^\top
$$

## 2. 机器人动力学

### 2.1 欧拉第一定律

刚体的线动量$\mathbf P$的变化率等于所有外力的合数$\mathbf F_{ext}$作用于刚体

$$
\mathbf F_{ext} = \frac{d\mathbf p}{dt}
$$

其中刚体的线性动量是刚体质量与其质心速度的乘积

$$
\mathbf p = m\mathbf v_c
$$

### 2.2 欧拉第二定律

设定某惯性参考系的固定点O（例如，原点）为参考点，施加于刚体的净外力矩，等于角动量的时间变化率：

$$
\mathbf M_O^{(ext)}=\frac{d \mathbf L_O}{dt}
$$

其中，$\mathbf M_O^{(ext)}$是对于点O合外力矩，$\mathbf L_O$是对于点O的角动量（$\mathbf L=\mathbf r\times \mathbf p$）。

假设施加于系统的合外力矩为零，则系统的角动量的时间变化率为零，系统的角动量守恒。

**相对于质心的欧拉第二运动定律**

无论质心参考系是否为惯性参考系（即不论质心是否呈加速度运动），以质心为参考点，合外力矩等于角动量的时间变化率：

$$
\mathbf M_{cm}=\frac{d \mathbf L_{cm}}{dt}
$$

### 2.3 欧拉方程（刚体运动）

我们可以选取相对于惯量的主轴坐标为体坐标轴系，这使得计算得以简化，因为我们现在可以将角动量的变化分别描述$L$的大小变化和方向变化的部分，并进一步将惯量对角化，方程为：

$$
\mathbf M = (I\frac{d\mathbf\omega}{dt})+(\mathbf\omega)\times I\mathbf\omega
$$

证明如下：在惯性系中，

$$
\mathbf M_{in}=\frac{d\mathbf L_{in}}{dt}
$$

引入一条重要的结论：**对任意矢量A在惯性系S中求导，等于其在转动系S′中求导，加上转动系相对惯性系的角速度ω叉乘这一矢量A**，所以

$$
\mathbf M = (\frac{d\mathbf L}{dt})_{relative}+\mathbf \omega \times\mathbf L
$$

由于在转动系（质心坐标系/体坐标系）中，$\mathbf I$不随时间改变，代入$\mathbf L=\mathbf I \mathbf \omega$得：

$$
(\frac{d\mathbf L}{dt})_{relative}=\mathbf I\frac{d\mathbf\omega}{dt}+\mathbf \omega\frac{d\mathbf I}{dt}=\mathbf I\frac{d\mathbf \omega}{dt}
$$

继而：

$$
\mathbf M=\mathbf I \cdot \mathbf{\dot \omega} + \mathbf \omega \times (\mathbf I\mathbf \cdot \omega)
$$

在体坐标系中，我们通常选取主轴（惯性主轴），使得惯性张量为对角形式：

$$
\mathbf I =    \left[
 \begin{matrix}
   I_1 & 0 & 0 \\
   0 & I_2 & 0 \\
   0 & 0 & I_3
  \end{matrix}
  \right] \tag{3}
$$

角速度在体坐标系中的表示为：

$$
\mathbf \omega =    \left[
 \begin{matrix}
   \omega_1 \\
   \omega_2   \\
   \omega_3 
  \end{matrix}
  \right] \tag{3}
$$

代入可得欧拉方程的分量形式：

$$
M_1=I_1\dot\omega_1+(I_3-I_2)\omega_2\omega_3 \\
M_2=I_2\dot\omega_2+(I_1-I_3)\omega_3\omega_1 \\
M_3=I_3\dot\omega_3+(I_2-I_1)\omega_1\omega_2
$$