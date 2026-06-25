# ORU 装配任务 — 强化学习配置文档

> **任务**: ORU (Orbital Replacement Unit) 装配
> **目标**: UR5 机械臂将已预夹持的 ORU 模块精确插入到对接面（Ground）上
> **架构**: 独立 DirectRLEnv，场景和控制来自 `scripts/tutorials/force/`

---

## 目录

1. [任务概述](#1-任务概述)
2. [场景结构](#2-场景结构)
3. [观测空间与状态空间](#3-观测空间与状态空间)
4. [动作空间](#4-动作空间)
5. [控制系统](#5-控制系统)
6. [奖励函数](#6-奖励函数)
7. [成功条件](#7-成功条件)
8. [RL 训练配置](#8-rl-训练配置)
9. [域随机化](#9-域随机化)
10. [关键代码来源](#10-关键代码来源)

---

## 1. 任务概述

| 属性 | 值 |
|------|-----|
| 任务名称 | `oru_assembly` |
| Gym ID | `Isaac-Oru-Direct-v0` |
| 机器人 | **UR5** (6-DOF arm, 无手指) |
| 夹持方式 | ORU 已通过固定关节链预夹持在末端 |
| 固定资产 | 对接面 Ground (`g1.usd`)，对接面高度 0.05m |
| 手持资产 | ORU 装配体 (`ORU.usd`)，ORU 高度 0.15m |
| 每回合时长 | 15 秒 |
| 物理频率 | 120 Hz |
| 策略频率 | 120 / 8 = 15 Hz (decimation=8) |

**装配流程**: ORU 已由 Gripper 预夹持 → UR5 将 ORU 移至对接面上方 → 精确插入使 ORU 底部接触对接面顶部。

**关键特点**: 不需要抓取阶段。Gripper 已经抓住 ORU，ORU 通过固定关节链连接到 UR5 末端，只需要控制 UR5 完成插入动作。

---

## 2. 场景结构

### 2.1 固定关节链

```
UR5 wrist_3_link (articulation link)
  │ FixedJoint (identity)
  ▼
Bridge (dynamic rigid body)
  │ FixedJoint (offset: z=+0.062, rot Y=π)
  ▼
SixForce / Force Sensor (dynamic rigid body)
  │ FixedJoint (offset: z=-0.0253, rot Y=π)
  ▼
Gripper (dynamic rigid body)
  │ FixedJoint (offset: z=-0.257, rot Z=π)
  ▼
ORU 装配体 (kinematic rigid body)
```

ORU 和 Ground 均为 **kinematic** 刚体（不参与动力学计算但参与碰撞检测）。

### 2.2 UR5 机器人

| 属性 | 值 |
|------|-----|
| USD 路径 | `{NUCLEUS}/Robots/UniversalRobots/ur5/ur5.usd` |
| Prim 路径 | `/World/envs/env_.*/Dofbot` |
| 初始关节位置 | shoulder_pan=0, shoulder_lift=-π/2, elbow=π/6, wrist_1=-π/6, wrist_2=-π/2, wrist_3=π/2 |
| 自碰撞 | 启用 |
| 重力 | 对机器人禁用 |
| 控制方式 | 阻抗控制（力矩控制） |

**致动器配置**: 所有 6 个关节 `stiffness=0, damping=0`，effort_limit=87 Nm。

### 2.3 对接面 (Ground)

| 属性 | 值 |
|------|-----|
| USD 路径 | `assets/USD/g1/g1.usd` |
| Prim 路径 | `/World/envs/env_.*/Ground` |
| 初始位置 | (0.4, 0.0, 0.05) |
| 类型 | Kinematic RigidBody（固定不动） |

### 2.4 ORU 装配体

| 属性 | 值 |
|------|-----|
| USD 路径 | `assets/USD/o7/ORU.usd` |
| Prim 路径 | `/World/envs/env_.*/ORU` |
| 类型 | Kinematic RigidBody |
| 连接方式 | 通过固定关节链连接到 UR5 wrist_3_link |
| ORU 半高度 | ≈0.075m (用于计算 ORU 底部位置) |

---

## 3. 观测空间与状态空间

### 3.1 策略观测 (Policy) — 25 维

| 索引 | 观测项 | 维度 | 说明 |
|------|--------|------|------|
| 0-2 | `ee_pos_rel_ground` | 3 | UR5 末端相对对接面位置 |
| 3-6 | `ee_quat` | 4 | UR5 末端四元数姿态 |
| 7-9 | `ee_linvel` | 3 | 末端线速度（有限差分） |
| 10-12 | `ee_angvel` | 3 | 末端角速度（有限差分） |
| 13-18 | `joint_pos` | 6 | UR5 6 个关节角度 |
| 19-24 | `prev_actions` | 6 | 上一时刻动作 |

### 3.2 Critic 状态 (State) — 41 维

| 索引 | 状态项 | 维度 | 说明 |
|------|--------|------|------|
| 0-2 | `ee_pos_rel_ground` | 3 | |
| 3-6 | `ee_quat` | 4 | |
| 7-9 | `ee_linvel` | 3 | |
| 10-12 | `ee_angvel` | 3 | |
| 13-18 | `joint_pos` | 6 | |
| 19-21 | `ground_pos` | 3 | 对接面世界位置 |
| 22-25 | `ground_quat` | 4 | 对接面世界姿态 |
| 26-31 | `task_prop_gains` | 6 | 当前阻抗控制比例增益 |
| 32-34 | `pos_threshold` | 3 | 位置动作阈值 |
| 35-37 | `rot_threshold` | 3 | 旋转动作阈值 |
| 38-43 | `prev_actions` | 6 | |

---

## 4. 动作空间

| 属性 | 值 |
|------|-----|
| 维度 | 6 |
| 范围 | [-1, 1] |
| 解释 | `[dx, dy, dz, drx, dry, drz]` — 末端位姿增量目标 |

### 动作处理流程

```
1. EMA 平滑:
   action[t] = 0.2 * raw_action + 0.8 * action[t-1]

2. 位置增量:
   pos_delta = action[0:3] * [0.02, 0.02, 0.02] m
   target_ee_pos = current_ee_pos + pos_delta
   裁剪到距对接面 ±0.05m 范围内

3. 旋转增量:
   angle = ||action[3:6]||
   axis  = action[3:6] / angle
   rot_quat = quat_from_angle_axis(angle, axis)
   rot_delta_scaled = angle * [0.097, 0.097, 0.097] rad
   target_ee_quat = rot_quat * current_ee_quat
   强制 roll=π, pitch=0 (末端保持竖直向下)

4. 阻抗控制 → 关节力矩
```

---

## 5. 控制系统

### 5.1 阻抗控制 (来自 `scripts/tutorials/force/force1_control.py`)

```
τ = J^T · F_task + τ_null

其中:
  F_task   = Kp · (x_des - x) - Kd · ẋ        (任务空间 PD)
  J        = wrist_3_link 的几何 Jacobian (6×6)
  τ_null   = (I - J^T·J̄^T) · M · (Kp_null·Δq - Kd_null·q̇)
```

### 5.2 阻抗控制增益

| 参数 | 值 | 说明 |
|------|-----|------|
| `default_task_prop_gains` | [100, 100, 100, 30, 30, 30] | Kp [xyz, rpy] |
| `deriv_gains` | `2√(Kp)`，rot_deriv_scale=10 | 临界阻尼 |
| XY 方向权重 | **2.0×** | 水平对齐优先级 |
| 零空间 Kp | 1.0 | |
| 零空间 Kd | 0.1 | |
| 力矩限制 | ±100 Nm | |

---

## 6. 奖励函数

### 6.1 总奖励

```
R_total = kp_baseline + kp_coarse + kp_fine
          - action_penalty - action_grad_penalty
          + success_bonus + engage_bonus
```

### 6.2 核心: Keypoint 距离奖励（三尺度 Squashing）

关键点: 4 个均匀分布在竖直线上（scale=0.2m），位于 ORU 底部和对接面顶部之间。

**Squashing 函数**: `r(x) = 1 / (exp(a·x) + b + exp(-a·x))`

| 奖励项 | a | b | 距离范围 | 作用 |
|--------|---|---|---------|------|
| `kp_baseline` | 5 | 4 | > 5cm | 粗略引导 ORU 向目标移动 |
| `kp_coarse` | 50 | 2 | 1-5cm | 中等精度对齐 |
| `kp_fine` | 100 | 0 | < 1cm | 精确插入 |

### 6.3 动作惩罚

| 奖励项 | 公式 | 权重 |
|--------|------|------|
| `action_penalty` | `||action||₂` | -0.01 |
| `action_grad_penalty` | `||action[t] - action[t-1]||₂` | -0.001 |

### 6.4 阶段奖励

| 条件 | 值 |
|------|-----|
| ORU 进入对接面 90% 高度 (engage) | +1.0 |
| ORU 满足成功条件 (success) | +1.0 |

---

## 7. 成功条件

| 条件 | 判定 | 阈值 |
|------|------|------|
| **XY 对齐** | `||ORU底部_xy - 对接面顶部_xy||₂` | < 5 mm |
| **Z 就位** | `ORU底部_z - 对接面顶部_z` | < 0.05 × 0.05 = 2.5 mm |

其中:
- ORU 底部 = ORU root position − (0, 0, 0.075) — ORU 半高度
- 对接面顶部 = Ground root position + (0, 0, 0.05) — 对接面高度

---

## 8. RL 训练配置

> 使用与 Factory 相同的 PPO 配置。

### 8.1 PPO 超参数

| 参数 | 值 |
|------|-----|
| 算法 | PPO (a2c_continuous) |
| 并行环境数 | 128 |
| Horizon | 128 |
| Mini-batch | 512 |
| Mini-epochs | 4 |
| 学习率 | 1e-4 (adaptive, kl_threshold=0.008) |
| γ (gamma) | 0.995 |
| τ (GAE λ) | 0.95 |
| ε_clip | 0.2 |
| 熵系数 | 0.0 |
| Grad norm clip | 1.0 |
| Critic coef | 2.0 |
| Max epochs | 200 |
| Mixed precision | True |

### 8.2 网络架构

```
策略网络 (Actor):
  MLP: [512, 128, 64] — ELU
  LSTM: 1024 units, 2 layers, LayerNorm
  输出: μ (linear) + σ (learnable, initial=0)

Critic 网络:
  MLP: [512, 128, 64] — ELU
  LSTM: 1024 units, 2 layers, LayerNorm
  输入归一化: True
```

---

## 9. 域随机化

| 参数 | 范围 |
|------|------|
| 对接面初始位置噪声 | ±0.05m (x, y, z), Uniform |
| UR5 初始关节位置 | 固定默认值 |
| EMA 平滑系数 | 固定 0.2 |

---

## 10. 关键代码来源

| 组件 | 来源 |
|------|------|
| 场景配置 (UR5, Bridge, Force_Six, Gripper, ORU, Ground) | `scripts/tutorials/force/force1_SceneCfg.py` |
| 阻抗控制 (`compute_dof_torque`) | `scripts/tutorials/force/force1_control.py` |
| 固定关节创建 | `scripts/tutorials/force/force1_SceneCfg.py` → `oru_utils.py` |
| RL 训练配置 | `factory/agents/rl_games_ppo_cfg.yaml` |
| 奖励函数结构 | `factory/factory_env.py` (keypoint squashing 模式) |

### 文件索引

```
source/isaaclab_tasks/isaaclab_tasks/direct/oru/
├── __init__.py              # Gym 注册: Isaac-Oru-Direct-v0
├── oru_env.py               # OruEnv (DirectRLEnv 子类)
├── oru_env_cfg.py           # OruEnvCfg → 场景 + UR5 + 资产
├── oru_tasks_cfg.py         # OruTaskCfg → 成功/奖励参数
├── oru_control.py           # → 导入 force1_control
├── oru_utils.py             # 固定关节创建 + 奖励辅助
└── agents/
    └── rl_games_ppo_cfg.yaml

scripts/tutorials/force/
├── force1_SceneCfg.py       # UR5 + 资产配置 + 固定关节 (上游源)
├── force1_control.py        # 阻抗控制 compute_dof_torque (上游源)
└── main_force1.py           # 手动控制 Demo

assets/USD/
├── g1/g1.usd                # 对接面
├── o7/ORU.usd               # ORU 装配体
├── bridge/bridge.usd        # 桥接件
├── force/force.usd          # 力传感器
└── gripper/gripper.usd      # 夹爪
```
