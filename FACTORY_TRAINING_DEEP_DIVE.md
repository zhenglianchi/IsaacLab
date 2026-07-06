# Factory PegInsert 训练深度解析

> 聚焦三个核心问题：RL 策略到底输出什么、阻抗控制怎么配置、三阶段关键点奖励如何工作。

---

## 目录

1. [整体数据流](#1-整体数据流)
2. [策略输出详解](#2-策略输出详解)
3. [阻抗控制全链路](#3-阻抗控制全链路)
4. [三阶段关键点奖励](#4-三阶段关键点奖励)
5. [完整参数速查表](#5-完整参数速查表)

---

## 1. 整体数据流

```
┌──────────────────────────────────────────────────────────────────┐
│                        一个 step (15Hz)                          │
│                                                                  │
│  观测 O_t (19维)                                                 │
│     │                                                            │
│     ▼                                                            │
│  ┌──────────┐     ┌───────────┐                                  │
│  │  LSTM    │────▶│   MLP     │────▶ 动作 a_t (6维)              │
│  │ 2×1024   │     │ 512→128→64│     [dx, dy, dz, da, db, dc]    │
│  └──────────┘     └───────────┘          │                        │
│                                          │ EMA 平滑               │
│                                          ▼                        │
│                                   ┌─────────────┐                │
│                                   │ 增量 → 目标位姿 │              │
│                                   │ (pos + rot)   │              │
│                                   └──────┬──────┘                │
│                                          │                        │
│                                          ▼                        │
│                                   ┌─────────────┐                │
│                                   │ 阻抗控制器    │  8次 @ 120Hz  │
│                                   │ Task PD +    │──────────────▶│
│                                   │ Nullspace    │  关节力矩 τ   │
│                                   └─────────────┘                │
│                                          │                        │
│                                          ▼                        │
│                                   ┌─────────────┐                │
│                                   │   PhysX 仿真  │               │
│                                   │   (120Hz)    │               │
│                                   └──────┬──────┘                │
│                                          │                        │
│                                          ▼                        │
│                                   下一帧观测 + 奖励               │
└──────────────────────────────────────────────────────────────────┘

策略频率: 120Hz / 8 (decimation) = 15Hz
每个策略动作之间：阻抗控制器跑 8 次，PhysX 跑 8 步
```

---

## 2. 策略输出详解

### 2.1 动作空间：6 维增量位移

策略输出一个 **6 维向量**，每个分量都在 `[-1, 1]` 范围内（由 PPO 的 `clip_actions: 1.0` 裁剪）：

```
a_t = [dx,  dy,  dz,  droll,  dpitch,  dyaw]
       └───平移───┘  └──────旋转─────────┘
       范围: [-1,1]  范围: [-1,1]
```

**策略不直接输出目标位姿，而是输出"从当前位置偏移多少"的增量**。

### 2.2 从策略输出到目标位姿的完整转换

```python
# factory_env.py:253-296 — 每一步的核心转换逻辑

# ──── 步骤 1：缩放 ────
# 策略输出 [-1, 1] → 实际物理位移/旋转
pos_actions = actions[:, 0:3] * pos_threshold  # 每步最多移动 ±2cm
rot_actions = actions[:, 3:6] * rot_threshold  # 每步最多旋转 ±0.097rad (≈5.5°)

# 具体数值：
# pos_threshold  = [0.02,  0.02,  0.02]   # 米
# rot_threshold  = [0.097, 0.097, 0.097]  # 弧度

# ──── 步骤 2：当前位置 + 增量 → 目标位置 ────
ctrl_target_pos = current_fingertip_pos + pos_actions

# ──── 步骤 3：位置裁剪 — 不能偏离固定件太远 ────
delta = ctrl_target_pos - fixed_asset_pos
clipped_delta = clamp(delta, -0.05, 0.05)   # 限制在 ±5cm 范围内
ctrl_target_pos = fixed_asset_pos + clipped_delta

# ──── 步骤 4：增量旋转 → 目标四元数 ────
angle = norm(rot_actions)                        # 旋转角度
axis  = rot_actions / angle                      # 旋转轴
delta_quat = quat_from_angle_axis(angle, axis)  # 增量旋转 → 四元数
ctrl_target_quat = delta_quat * current_quat     # 叠加到当前姿态

# ──── 步骤 5：强制指尖朝下 ────
euler = to_euler(ctrl_target_quat)
euler.roll  = π          # 180° — 指尖始终朝下
euler.pitch = 0          # 0°   — 不能倾斜
# yaw 保持策略的原始输出（控制绕 Z 轴的旋转）
ctrl_target_quat = quat_from_euler(π, 0, euler.yaw)
```

> **关键约束**：策略只有 **yaw（绕 Z 轴旋转）** 一个旋转自由度，roll 和 pitch 被强制锁死。这意味着策略学的是"在哪里插入 + 绕 Z 轴转多少"，而不是全姿态控制。

### 2.3 EMA 平滑

动作不是直接执行的，而是经过指数移动平均：

```python
# factory_env.py:213
self.actions = 0.2 * new_action + 0.8 * old_action
#              ↑ 新动作权重 20%    ↑ 旧动作权重 80%
```

**为什么需要 EMA？** 防止策略输出高频抖振。装配任务的间隙只有几十微米，突然的大幅动作会导致插入失败。EMA 让动作变化平滑，物理上表现为柔顺的运动。

### 2.4 策略网络的输入（观测）

```python
# Actor 观测 — 仅含末端执行器层面的信息（19 维）
obs = [
    ee_pos - fixed_pos,    # 3D  末端相对固定件的 XYZ 位置
    ee_quat,               # 4D  末端姿态（四元数 wxyz）
    ee_linear_vel,         # 3D  末端线速度（有限差分）
    ee_angular_vel,        # 3D  末端角速度（有限差分）
    prev_action,           # 6D  上一帧的动作
]
# 总计 3+4+3+3+6 = 19 维
```

**策略看不到什么？**

- 看不到关节角度（7 个关节）
- 看不到手持件（销）的精确位姿
- 看不到固定件（孔）的精确位姿
- 看不到当前的控制增益

这些信息仅 Critic 可见。策略必须从有限差分速度和相对位置**推断**接触状态。

---

## 3. 阻抗控制全链路

### 3.1 为什么用阻抗控制而不是位置控制？

| 控制方式                          | 装配任务表现               |
| ----------------------------- | -------------------- |
| **位置控制** (关节 PD)              | 销碰到孔壁 → 刚性对抗 → 卡死或弹飞 |
| **阻抗控制** (Task-space PD + 力矩) | 销碰到孔壁 → 柔顺退让 → 沿壁滑入  |

装配需要的是**柔顺性（compliance）**：当销接触到孔壁时，机械臂应该"顺从地"调整位置，而不是用力硬怼。阻抗控制模拟了弹簧-阻尼系统，天然适配这个需求。

### 3.2 数学模型

```
τ = J^T · F_task  +  N · τ_null     (总力矩 = 任务空间力矩 + 零空间力矩)

其中:

F_task = Kp · (x_desired - x_current)  +  Kd · (0 - ẋ)
         └──────────位置误差─────────┘     └────速度阻尼────┘

N = I - J^T · (J · M⁻¹ · J^T)⁻¹ · J · M⁻¹    (零空间投影矩阵)

τ_null = M · (Kp_null · (q_default - q_current)  +  Kd_null · (0 - q̇))
```

### 3.3 代码实现分步讲解

**Step 1: 计算末端位姿误差**

```python
# factory_control.py:104-145
def get_pose_error(fingertip_pos, fingertip_quat, target_pos, target_quat, ...):
    # 位置误差（3D 向量）
    pos_error = target_pos - fingertip_pos        # 简单减法

    # 旋转误差（axis-angle 表示）
    quat_dot = dot(target_quat, fingertip_quat)   # 判断最短路径
    if quat_dot < 0: target_quat = -target_quat   # 确保走短弧

    quat_error = target_quat * inv(fingertip_quat)  # 四元数误差
    axis_angle_error = axis_angle_from_quat(quat_error)  # 转为轴角

    # 最终误差是 6D 向量：[dx, dy, dz, dax, day, daz]
    return [pos_error, axis_angle_error]
```

**Step 2: 任务空间 PD — 将位姿误差转换为力/力矩（Wrench）**

```python
# factory_control.py:188-206
def _apply_task_space_gains(delta_pose, linvel, angvel, Kp, Kd):
    wrench = zeros(6)  # [Fx, Fy, Fz, Tx, Ty, Tz]

    # 线性部分：F = Kp_lin · pos_error  +  Kd_lin · (0 - velocity)
    wrench[0:3] = Kp[0:3] * delta_pose[0:3]  +  Kd[0:3] * (-linvel)

    # 旋转部分：T = Kp_rot · rot_error  +  Kd_rot · (0 - angvel)
    wrench[3:6] = Kp[3:6] * delta_pose[3:6]  +  Kd[3:6] * (-angvel)

    return wrench
```

**Step 3: 映射到关节空间**

```python
# factory_control.py:77
# τ_task = J^T · F
dof_torque = J^T @ wrench    # 6×N Jacobian 转置 × 6D wrench → N 维关节力矩
```

**Step 4: 零空间力矩 — 保持关节姿态自然**

```python
# factory_control.py:81-97
# 计算任务空间质量矩阵 Λ = (J · M⁻¹ · J^T)⁻¹
Lambda = inv(J @ inv(M) @ J^T)

# 动态一致的伪逆 J̄ = Λ · J · M⁻¹
J_dyn_inv = Lambda @ J @ inv(M)

# 零空间投影 N = I - J^T · J̄
N = I - J^T @ J_dyn_inv

# 零空间 PD：让关节保持在自然位置
q_error = q_default - q_current
u_null = Kd_null * (-q_vel) + Kp_null * wrap_to_pi(q_error)

# 零空间力矩：τ_null = N · M · u_null
tau_null = N @ M @ u_null

# 最终力矩
tau_final = tau_task + tau_null
tau_final = clamp(tau_final, -100, 100)  # 力矩限制 ±100 N·m
```

> **零空间做什么？** UR5 有 6 个关节，末端位姿只需要 6 个自由度。但 Franka 有 7 个关节——多出的 1 个自由度构成"零空间"。零空间投影确保：在满足末端位姿目标的前提下，让关节角度尽可能接近默认的自然姿态，避免"肘部飞上天"之类的奇怪姿态。

### 3.4 增益参数详解

```python
# factory_env_cfg.py:62
default_task_prop_gains = [100, 100, 100,  30, 30, 30]
#                          └──平移──┘   └──旋转──┘
#                          单位: N/m        单位: N·m/rad

# 临界阻尼微分增益
# factory_utils.py:19-23
deriv_gains = 2 * sqrt(prop_gains)
# 结果约: [20, 20, 20, 11, 11, 11]
deriv_gains[3:6] /= 10.0
# 最终:   [20, 20, 20, 1.1, 1.1, 1.1]
```

| 参数              | 值             | 物理含义                               |
| --------------- | ------------- | ---------------------------------- |
| `Kp_lin` (平移刚度) | 100 N/m       | 偏离目标 1cm → 产生 1N 的回复力              |
| `Kp_rot` (旋转刚度) | 30 N·m/rad    | 偏离目标 0.1rad(≈5.7°) → 产生 3 N·m 回复力矩 |
| `Kd_lin` (平移阻尼) | 20 N·s/m      | 末端 0.1 m/s → 产生 2N 阻尼力             |
| `Kd_rot` (旋转阻尼) | 1.1 N·m·s/rad | 旋转阻尼缩小 10 倍 → 旋转更柔顺                |

> **为什么旋转阻尼缩小 10 倍？** 装配时旋转方向的柔顺性比平移更重要。销插入时如果旋转方向太"硬"，微小的角度偏差会产生大力矩把销弹出来。旋转方向 1/10 的阻尼让旋转更"软"，允许销在孔中自适应旋转。

### 3.5 重置时的高增益

```python
# factory_env_cfg.py:60-61
reset_task_prop_gains = [300, 300, 300, 20, 20, 20]  # 平移刚度 3x
reset_rot_deriv_scale = 10.0
```

重置阶段（夹爪抓取销时）用 3 倍平移刚度，确保抓取动作快速准确。Episode 开始后切回正常增益 `[100, 100, 100, 30, 30, 30]`。

---

## 4. 三阶段关键点奖励

### 4.1 关键点是什么？

在被操作物体（销）和目标位置（孔上方）上各定义 **4 个沿 Z 轴均匀分布的关键点**：

```
         ┌───●───┐  z = +0.075     (销顶部的关键点)
         │       │
  held   │   ●   │  z = +0.025     (销中上部的关键点)
  asset  │       │
  (销)    │   ●   │  z = -0.025     (销中下部的关键点)
         │       │
         └───●───┘  z = -0.075     (销底部的关键点)

         ┌───●───┐  z = +0.075     (孔上方对应位置)
         │       │
  target │   ●   │  z = +0.025
   pose  │       │
         │   ●   │  z = -0.025
         │       │
         └───●───┘  z = -0.075     (孔内部对应位置)


  奖励 = mean(‖kp_held_i - kp_target_i‖₂)  → 通过 squashing 函数
```

```python
# factory_utils.py:12-16
def get_keypoint_offsets(num_keypoints, device):
    offsets = zeros(num_keypoints, 3)
    offsets[:, -1] = linspace(0, 1, num_keypoints) - 0.5
    # 4 个关键点在 Z 轴: [-0.375, -0.125, +0.125, +0.375]
    return offsets

# 使用时乘以 keypoint_scale = 0.15，得到实际偏移：
# [-0.056, -0.019, +0.019, +0.056] 米
# 即关键点分布在销的 11.2cm 高度范围内
```

### 4.2 Squashing 函数

```
r(x) = 1 / (e^(ax) + b + e^(-ax))
```

```python
# factory_utils.py:105-107
def squashing_fn(x, a, b):
    return 1 / (exp(a*x) + b + exp(-a*x))
```

**函数图像**（以三组参数为例）：

```
r(x)
1.0 ┤
    │        ┌────────────  fine (a=100, b=0)
0.8 ┤       ╱│╲                  最陡峭，只在距离<1cm时给高奖励
    │      ╱ │ ╲
0.6 ┤     ╱  │  ╲  ────  coarse (a=50, b=2)
    │    ╱   │   ╲              中等陡峭，1-5cm范围有效
0.4 ┤   ╱    │    ╲
    │  ╱     │     ╲──  baseline (a=5, b=4)
0.2 ┤ ╱      │      ╲            最平缓，远距离也有梯度
    │╱       │       ╲
0.0 ┼────────┼────────┼──────▶ x (关键点距离, 米)
    0       0.05     0.10
```

### 4.3 三组参数的具体行为

| 阶段           | a   | b   | 距离 0.1m | 距离 0.05m | 距离 0.01m | 距离 0.001m |
| ------------ | --- | --- | ------- | -------- | -------- | --------- |
| **baseline** | 5   | 4   | 0.10    | 0.14     | 0.21     | 0.24      |
| **coarse**   | 50  | 2   | ~0      | 0.26     | 0.59     | 0.87      |
| **fine**     | 100 | 0   | ~0      | ~0       | 0.23     | 0.81      |

**三个阶段如何协同工作：**

```
训练早期（销离孔很远，距离 ~10cm）:
  baseline 奖励 ≈ 0.10  ← 提供主要的学习信号
  coarse   奖励 ≈ 0.00  ← 太陡，几乎无梯度
  fine     奖励 ≈ 0.00  ← 完全无梯度
  → 策略学到："往目标方向移动"

训练中期（销接近孔，距离 ~2-3cm）:
  baseline 奖励 ≈ 0.17  ← 逐步升高
  coarse   奖励 ≈ 0.35  ← 开始提供有意义的梯度
  fine     奖励 ≈ 0.00  ← 仍然无梯度
  → 策略学到："不仅要靠近，还要精确对齐"

训练后期（销开始插入，距离 <5mm）:
  baseline 奖励 ≈ 0.22  ← 接近饱和
  coarse   奖励 ≈ 0.65  ← 提供强烈的梯度引导
  fine     奖励 ≈ 0.30  ← 终于开始"看到"梯度
  → 策略学到："毫米级精确插入"
```

### 4.4 PegInsert 的完整奖励构成

```python
# factory_env.py:468-485 (PegInsert 的奖励权重均为默认值)
reward = (
    1.0 * kp_baseline          # 远距离引导
  + 1.0 * kp_coarse            # 中距离对齐
  + 1.0 * kp_fine              # 近距离精细
  - 0.0 * action_penalty        # PegInsert: 动作惩罚权重为 0
  - 0.0 * action_grad_penalty   # PegInsert: 动作变化惩罚权重为 0
  + 1.0 * curr_engaged          # 到达 90% 深度的密集奖励
  + 1.0 * curr_success          # 到达 4% 深度 (success) 的奖励
)
```

> **注意**：PegInsert 的 `action_penalty_ee_scale = 0.0` 和 `action_grad_penalty_scale = 0.0`——不惩罚动作幅度和变化量。这是因为销插入任务本身对精度要求极高，加上动作惩罚会抑制策略采取必要的大幅度动作去完成对齐。

---

## 5. 完整参数速查表

### 5.1 控制参数

| 参数                        | 值                             | 说明                                 |
| ------------------------- | ----------------------------- | ---------------------------------- |
| `action_space`            | 6                             | `[dx, dy, dz, dax, day, daz]` 增量动作 |
| `decimation`              | 8                             | 策略 15Hz，物理 120Hz                   |
| `ema_factor`              | 0.2                           | 新动作 20% + 旧动作 80%                  |
| `pos_action_threshold`    | `[0.02, 0.02, 0.02]`          | 每步最大平移 ±2cm                        |
| `rot_action_threshold`    | `[0.097, 0.097, 0.097]`       | 每步最大旋转 ±5.5°                       |
| `pos_action_bounds`       | `[0.05, 0.05, 0.05]`          | 相对固定件 ±5cm 硬限制                     |
| `default_task_prop_gains` | `[100, 100, 100, 30, 30, 30]` | 正常运行时的阻抗增益                         |
| `reset_task_prop_gains`   | `[300, 300, 300, 20, 20, 20]` | 重置抓取时的阻抗增益 (3x)                    |
| `rot_deriv_scale`         | 10.0                          | 旋转微分增益缩小 10 倍                      |

### 5.2 奖励参数 (PegInsert)

| 参数                       | 值          | 说明              |
| ------------------------ | ---------- | --------------- |
| `num_keypoints`          | 4          | 关键点数量           |
| `keypoint_scale`         | 0.15       | 关键点分布范围 (m)     |
| `keypoint_coef_baseline` | `[5, 4]`   | 远距离阶段 (a, b)    |
| `keypoint_coef_coarse`   | `[50, 2]`  | 中距离阶段 (a, b)    |
| `keypoint_coef_fine`     | `[100, 0]` | 近距离阶段 (a, b)    |
| `success_threshold`      | 0.04       | 孔深 4% = 1mm     |
| `engage_threshold`       | 0.9        | 孔深 90% = 22.5mm |

### 5.3 观测参数

| 参数        | 维度     | 说明                                                                         |
| --------- | ------ | -------------------------------------------------------------------------- |
| Actor 观测  | **19** | `pos_rel(3) + quat(4) + linvel(3) + angvel(3) + prev_action(6)`            |
| Critic 状态 | **56** | Actor 全部 + `joint_pos(7) + held(10) + fixed(7) + gains(6) + thresholds(6)` |

### 5.4 物理参数

| 参数                             | 值             | 说明          |
| ------------------------------ | ------------- | ----------- |
| `dt`                           | 1/120         | 仿真步长 8.33ms |
| `max_position_iteration_count` | 192           | 防止穿透的关键参数   |
| `solver_type`                  | 1             | TGS 求解器     |
| `gpu_max_num_partitions`       | 1             | 单 GPU 分区    |
| `static_friction`              | 1.0           | 高摩擦力        |
| `gravity`                      | (0, 0, -9.81) | 标准重力        |

### 5.5 训练参数

| 参数               | 值     | 说明                    |
| ---------------- | ----- | --------------------- |
| `num_actors`     | 128   | 并行环境数                 |
| `horizon_length` | 128   | 每轮 16,384 transitions |
| `gamma`          | 0.995 | 需要长期规划                |
| `lr`             | 1e-4  | 自适应调整 (KL 阈值)         |
| `entropy_coef`   | 0.0   | 不加熵正则                 |
| `max_epochs`     | 200   | 最多 200 轮              |

---

*文档生成时间: 2026-07-03*
