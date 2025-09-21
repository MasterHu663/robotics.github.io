# 机器人学

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

绕$\bold X$轴旋转角$\gamma$，旋转矩阵为：
$$
R_\gamma=\left[
 \begin{matrix}
   1 & 0 & 0 \\
   0 & \cos\gamma & -\sin\gamma \\
   0 & \sin\gamma & \cos\gamma  
  \end{matrix}
  \right]
$$
绕$\bold Y$轴旋转角$\beta$，旋转矩阵为：
$$
R_\beta=\left[
 \begin{matrix}
   \cos\beta & 0 & \sin\beta \\
   0 & 1 & 0 \\
   -\sin\beta & 0 & \cos\beta  
  \end{matrix}
  \right]
$$
绕$\bold Z$轴旋转角$\alpha$，旋转矩阵为：
$$
R_\alpha=\left[
 \begin{matrix}
   \cos\alpha & -\sin\alpha & 0 \\
   \sin\alpha & \cos\alpha & 0 \\
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

##### **a. 固定角**

**$\bold{X-Y-Z}$ 固定角（RPY角）**

| Name           | 中文名           | 表达式            | 所绕轴      |
| -------------- | ---------------- | ----------------- | ----------- |
| $\bold{Roll}$  | 回转角（横滚角） | $\gamma$、$\phi$  | 绕$X$轴旋转 |
| $\bold{Pitch}$ | 俯仰角           | $\beta$、$\theta$ | 绕$Y$轴旋转 |
| $\bold{Yaw}$   | 偏转角（航向角） | $\alpha$、$\psi$  | 绕$Z$轴旋转 |

先绕$\bold{X_A}$轴旋转，再绕$\bold{Y_A}$轴旋转，最后$\bold{Z_A}$轴旋转
$$
^A_BR=R_Z(\alpha)R_Y(\beta)R_X(\gamma)\\
=
\left[
 \begin{matrix}
   \cos\alpha & -\sin\alpha & 0 \\
   \sin\alpha & \cos\alpha & 0 \\
   0 & 0 & 1  
  \end{matrix}
  \right]
\left[
 \begin{matrix}
   \cos\beta & 0 & \sin\beta \\
   0 & 1 & 0 \\
   -\sin\beta & 0 & \cos\beta  
  \end{matrix}
  \right]
\left[
 \begin{matrix}
   1 & 0 & 0 \\
   0 & \cos\gamma & -\sin\gamma \\
   0 & \sin\gamma & \cos\gamma  
  \end{matrix}
  \right] \\
  =
  \left[
 \begin{matrix}
   c\alpha c\beta & c\alpha s\beta s\gamma - s\alpha c\gamma & c\alpha s\beta c\gamma + s\alpha s\gamma \\
   s\alpha c\beta & s\alpha s\beta s\gamma + c\alpha c\gamma & s\alpha s\beta c\gamma - c\alpha s\gamma \\
   -s\beta & c\beta s\gamma & c\beta c\gamma
  \end{matrix}
  \right]
$$

##### **b. 欧拉角**

**1. $\bold{Z-Y-X}$ 欧拉角**

先绕$\bold{X_B}$轴旋转，再绕$\bold{Y_B}$轴旋转，最后$\bold{Z_B}$轴旋转
$$
^A_BR=R_Z(\alpha)R_Y(\beta)R_X(\gamma)\\
= \left[
 \begin{matrix}
   c\alpha c\beta & c\alpha s\beta s\gamma - s\alpha c\gamma & c\alpha s\beta c\gamma + s\alpha s\gamma \\
   s\alpha c\beta & s\alpha s\beta s\gamma + c\alpha c\gamma & s\alpha s\beta c\gamma - c\alpha s\gamma \\
   -s\beta & c\beta s\gamma & c\beta c\gamma
  \end{matrix}
  \right]
$$

> 三次绕固定轴旋转的最终姿态和以相反顺序绕运动坐标轴转动的最终姿态相同！

**2. $\bold{Z-Y-Z}$ 欧拉角**

先绕$\bold{Z_B}$轴旋转，再绕$\bold{Y_B}$轴旋转，最后$\bold{Z_B}$轴旋转
$$
^A_BR=R_Z(\alpha)R_Y(\beta)R_Z(\gamma)\\
= \left[
 \begin{matrix}
   c\alpha c\beta c\gamma - s\alpha s\gamma & -c\alpha c\beta s\gamma - s\alpha c\gamma & c\alpha s\beta\\
   s\alpha c\beta c\gamma + c\alpha s\gamma & -s\alpha c\beta s\gamma - c\alpha c\gamma & s\alpha s\beta\\
   -s\beta c\gamma & s\beta s\gamma & c\beta
  \end{matrix}
  \right]
$$

### 1.2 一般变换

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
a_{i-1} = 沿 \hat X_{i-1} 轴，从\hat Z_{i-1}移动到\hat Z_{i}的距离 \ (连杆长度)\\
\alpha_{i-1} = 沿 \hat X_{i-1} 轴，从\hat Z_{i-1}移动到\hat Z_{i}的角度\ (连杆转角) \\
d_i = 沿 \hat Z_i 轴，从\hat X_{i-1}移动到\hat X_{i}的距离\ (连杆偏距) \\
\theta_i = 沿 \hat Z_i 轴，从\hat X_{i-1}移动到\hat X_{i}的距离\ (关节角) \\
$$
可以推得：
$$
^{i-1}_{i}T=R_X(\alpha_{i-1})D_X(a_{i-1})R_Z(\theta_{i})D_Z(d_{i}) \\
=\left[
 \begin{matrix}
   c\theta_i & - s\theta_i & 0 & a_{i-1}\\
   s\theta_i c\alpha_{i-1} & c\theta_i c\alpha_{i-1} & -s\alpha_{i-1} & -s\alpha_{i-1}d_i\\
   s\theta_i s\alpha_{i-1} & c\theta_i s\alpha_{i-1} & c\alpha_{i-1} & c\alpha_{i-1}d_i\\
   0 & 0 & 0 & 1
  \end{matrix}
  \right]
$$

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

## 2. 机器人动力学

### 2.1 欧拉第一定律

刚体的线动量$\bold P$的变化率等于所有外力的合数$\bold F_{ext}$作用于刚体
$$
\bold F_{ext} = \frac{d\bold p}{dt}
$$
其中刚体的线性动量是刚体质量与其质心速度的乘积
$$
\bold p = m\bold v_c
$$

### 2.2 欧拉第二定律

设定某惯性参考系的固定点O（例如，原点）为参考点，施加于刚体的净外力矩，等于角动量的时间变化率：
$$
\bold M_O^{(ext)}=\frac{d \bold L_O}{dt}
$$
其中，$\bold M_O^{(ext)}$是对于点O合外力矩，$\bold L_O$是对于点O的角动量（$\bold L=\bold r\times \bold p$）。

假设施加于系统的合外力矩为零，则系统的角动量的时间变化率为零，系统的角动量守恒。

**相对于质心的欧拉第二运动定律**

无论质心参考系是否为惯性参考系（即不论质心是否呈加速度运动），以质心为参考点，合外力矩等于角动量的时间变化率：
$$
\bold M_{cm}=\frac{d \bold L_{cm}}{dt}
$$

### 2.3 欧拉方程（刚体运动）

我们可以选取相对于惯量的主轴坐标为体坐标轴系，这使得计算得以简化，因为我们现在可以将角动量的变化分别描述$L$的大小变化和方向变化的部分，并进一步将惯量对角化，方程为：
$$
\bold M = (I\frac{d\bold\omega}{dt})+(\bold\omega)\times I\bold\omega
$$
证明如下：在惯性系中，
$$
\bold M_{in}=\frac{d\bold L_{in}}{dt}
$$
引入一条重要的结论：**对任意矢量A在惯性系S中求导，等于其在转动系S′中求导，加上转动系相对惯性系的角速度ω叉乘这一矢量A**，所以
$$
\bold M = (\frac{d\bold L}{dt})_{relative}+\bold \omega \times\bold L
$$
由于在转动系（质心坐标系/体坐标系）中，$\bold I$不随时间改变，代入$\bold L=\bold I \bold \omega$得：
$$
(\frac{d\bold L}{dt})_{relative}=\bold I\frac{d\bold\omega}{dt}+\bold \omega\frac{d\bold I}{dt}=\bold I\frac{d\bold \omega}{dt}
$$
继而：
$$
\bold M=\bold I \cdot \bold{\dot \omega} + \bold \omega \times (\bold I\bold \cdot \omega)
$$
在体坐标系中，我们通常选取主轴（惯性主轴），使得惯性张量为对角形式：
$$
\bold I =    \left[
 \begin{matrix}
   I_1 & 0 & 0 \\
   0 & I_2 & 0 \\
   0 & 0 & I_3
  \end{matrix}
  \right] \tag{3}
$$
角速度在体坐标系中的表示为：
$$
\bold \omega =    \left[
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

