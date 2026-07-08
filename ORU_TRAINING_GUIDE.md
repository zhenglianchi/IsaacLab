# ORU 变阻抗装配训练指南

> 当前 ORU 任务架构：策略输出 12D 变阻抗参数 (Kp + Kd)，目标位姿固定，域随机化在初始状态。

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                     一个 step (15Hz)                         │
│                                                              │
│  观测 (43D)                                                  │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────┐    ┌───────────┐                               │
│  │  LSTM    │───▶│   MLP     │───▶ 动作 a_t (12D)            │
│  │ 2×1024   │    │ 512→128→64│    [Kp_scale(6), Kd_scale(6)] │
│  └──────────┘    └───────────┘         │                      │
│                                        │ EMA 平滑             │
│                                        ▼                      │
│                                 ┌─────────────┐              │
│                                 │ Kp = base×scale │            │
│                                 │ Kd = base×scale │            │
│                                 └──────┬──────┘              │
│                                        │                      │
│                                        ▼                      │
│                                 ┌─────────────┐              │
│ 固定目标                         │ 阻抗控制器    │ 8次 @ 120Hz │
│ [gnd_x,gnd_y,0.4,0,0,1,0] ────▶│ Task PD +    │────────────▶│
│                                 │ Nullspace    │  关节力矩 τ  │
│                                 └─────────────┘              │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 策略输出：12D 变阻抗参数

### 动作空间

```
a_t = [Kp_x, Kp_y, Kp_z, Kp_rx, Kp_ry, Kp_rz,    Kd_x, Kd_y, Kd_z, Kd_rx, Kd_ry, Kd_rz]
       └────────── Kp 缩放 (6D) ──────────┘       └────────── Kd 缩放 (6D) ──────────┘
       范围: [-1, 1]                               范围: [-1, 1]
```

### 从策略输出到阻抗增益

```python
# oru_env.py _apply_action

# Kp = base_Kp × (1 + a[:6] × 2.0),  clamp [0.05, 5.0]
scale_kp = 1.0 + actions[:, 0:6] * 2.0
scale_kp = clamp(scale_kp, 0.05, 5.0)
task_prop_gains = base_gains * scale_kp

# Kd = base_Kd × (1 + a[6:] × 2.0),  clamp [0.05, 5.0]
scale_kd = 1.0 + actions[:, 6:12] * 2.0
scale_kd = clamp(scale_kd, 0.05, 5.0)
task_deriv_gains = base_deriv * scale_kd
```

### 基础增益

| 方向         | base_Kp | base_Kd |
| ---------- | ------- | ------- |
| X, Y, Z    | 2.0     | 2.828   |
| Rx, Ry, Rz | 2.0     | 0.943   |

### 增益范围 (action ∈ [-1, 1])

|        | Min (action=-1) | Default (action=0) | Max (action=+1) |
| ------ | --------------- | ------------------ | --------------- |
| Kp_XYZ | 0.1             | 2.0                | 10.0            |
| Kp_rot | 0.1             | 2.0                | 10.0            |
| Kd_XYZ | 0.14            | 2.83               | 14.1            |
| Kd_rot | 0.05            | 0.94               | 4.7             |

---

## 3. 目标位姿

**固定不变**，策略不控制"往哪走"：

```python
# oru_env.py _apply_action
ctrl_target_ee_pos = [ground_x, ground_y, 0.4]    # XY 跟 ground，Z 固定 0.4
ctrl_target_ee_quat = [0, 0, 1, 0]               # wxyz, 180° Y 轴旋转
```

---

## 4. 阻抗控制器

### 数学模型

```
τ = Jᵀ · F_task  +  N · τ_null

F_task = Kp · (x_desired − x_current) + Kd · (0 − ẋ)
N = I − Jᵀ · (J·M⁻¹·Jᵀ)⁻¹ · J·M⁻¹
τ_null = N · M · (kp_null·(q_default−q) + kd_null·(−q̇))
```

### 关键参数

| 参数          | 值                            | 说明                   |
| ----------- | ---------------------------- | -------------------- |
| `Kp`        | 策略控制                         | 6D 可变比例刚度            |
| `Kd`        | 策略控制                         | 6D 可变微分阻尼            |
| `kp_null`   | 1.0                          | 零空间比例增益              |
| `kd_null`   | 2.0                          | 零空间临界阻尼              |
| `dead_zone` | `[0.5,0.5,0.5, 0.2,0.2,0.2]` | ＜0.5N/0.2Nm 力归零，防极限环 |

### 振荡修复（2026-07 调整）

| 修复点   | 之前              | 之后                           |
| ----- | --------------- | ---------------------------- |
| XY 权重 | `xy_weight=2.0` | 移除，统一 1.0                    |
| 零空间阻尼 | `kd_null=0.1`   | `kd_null=2.0` (临界阻尼)         |
| 死区    | 未使用             | `[0.5,0.5,0.5, 0.2,0.2,0.2]` |

---

## 5. 奖励函数

### 关键点奖励

```
ORU 侧参考点: ee_pos (末端位置)
目标侧参考点: [ground_x, ground_y, 0.4] + quat [0,0,1,0]

4 个关键点沿各自 Z 轴均匀分布: [-0.03, -0.01, +0.01, +0.03]m
```

```python
# 关键点距离 = 4 对关键点的平均 L2 距离
kp_dist = mean(‖kp_oru[i] − kp_target[i]‖₂)

# 三阶段 squashing: r(x) = 1 / (e^(ax) + b + e^(−ax))
reward = squashing(kp_dist, a=5,  b=4)     # baseline: 远距离粗引导
       + squashing(kp_dist, a=50, b=2)     # coarse:   中距离对齐
       + squashing(kp_dist, a=100, b=0)    # fine:     近距离精细
       − 0.01 × ‖actions‖₂                 # 动作幅度惩罚
       − 0.001 × ‖actions_diff‖₂            # 动作变化惩罚
       + success.float()                    # 距离 < 5mm 时 +1
       + engaged.float()                    # Z 误差 < 10% 时 +1
```

### 奖励参数速查

| 参数                          | 值          | 说明            |
| --------------------------- | ---------- | ------------- |
| `num_keypoints`             | 4          | 关键点数量         |
| `keypoint_scale`            | 0.08       | ±3cm 分布范围     |
| `keypoint_coef_baseline`    | `[5, 4]`   | 远距离 squashing |
| `keypoint_coef_coarse`      | `[50, 2]`  | 中距离 squashing |
| `keypoint_coef_fine`        | `[100, 0]` | 近距离 squashing |
| `action_penalty_scale`      | 0.01       | 动作 L2 惩罚      |
| `action_grad_penalty_scale` | 0.001      | 动作变化惩罚        |

---

## 6. 观测空间

### Policy 观测 (43D)

| 分量                  | 维度  | 说明               |
| ------------------- | --- | ---------------- |
| `ee_pos_rel_ground` | 3   | EE 相对 ground 的位置 |
| `ee_quat`           | 4   | EE 姿态 (wxyz)     |
| `ee_linvel`         | 3   | EE 线速度 (有限差分)    |
| `ee_angvel`         | 3   | EE 角速度 (有限差分)    |
| `joint_pos`         | 6   | 6 个关节角           |
| `task_prop_gains`   | 6   | 当前 Kp（策略看到自己的刚度） |
| `task_deriv_gains`  | 6   | 当前 Kd（策略看到自己的阻尼） |
| `prev_actions`      | 12  | 上一帧动作            |

### Critic 状态 (56D)

Policy 全部 + `ground_pos(3)` + `ground_quat(4)` + `pos_threshold(3)` + `rot_threshold(3)`

---

## 7. 域随机化

### IK 初始位姿随机化（reset 时）

```
1. 设置 UR5 到默认关节角 → FK 得到 home EE pose
2. 加入随机偏移:
   - 位置: ±12cm (XYZ)
   - 姿态: ±3° (各轴)
3. DLS IK → 随机初始关节角
4. 写入 UR5 初始状态
```

```python
# oru_tasks_cfg.py
ik_rand_pos_noise = (0.12, 0.12, 0.12)     # ±12cm
ik_rand_rot_noise = (0.052, 0.052, 0.052)  # ±3° (0.052 rad)
```

### 目标不随机化

目标位姿始终是 `[ground_x, ground_y, 0.4, 0,0,1,0]`，每个 env 的 ground XY 固定，所以目标也固定。策略从不同起点学会到达同一个目标。

---

## 8. 训练配置

### PPO 参数

| 参数                 | 值     | 说明                |
| ------------------ | ----- | ----------------- |
| `num_envs`         | 16    | 场景数量              |
| `num_actors`       | 128   | PPO 配置中的并行数       |
| `episode_length_s` | 40    | 600 步策略动作/episode |
| `decimation`       | 8     | 策略 15Hz           |
| `ema_factor`       | 0.2   | 动作平滑              |
| `gamma`            | 0.995 | 长期规划              |
| `lr`               | 1e-4  | 自适应               |
| `entropy_coef`     | 0.0   | 无熵正则              |

### 网络架构

```
LSTM (2层, 1024) → MLP (512→128→64) → 12D 输出
```

### 场景设置

```
clone_in_fabric=False   ← FixedJoint 链条必须在每个 env 独立创建
num_envs=16
env_spacing=2.0
```

---

## 9. 成功判定

```python
# oru_env.py _get_curr_successes
xy_dist = ‖target_pos[:2] − ee_pos[:2]‖₂
z_diff  = ee_pos[2] − target_pos[2]

is_centered  = xy_dist < 0.005           # XY < 5mm
is_close     = z_diff < -0.05 × threshold  # Z 低于目标一定比例

success  = is_centered AND is_close
engaged  = z_diff < -0.05 × 0.9          # 90% threshold
```

### 成功阈值

| 参数                  | 值     | 说明       |
| ------------------- | ----- | -------- |
| `xy_tolerance`      | 0.005 | 5mm 中心对准 |
| `success_threshold` | 0.05  | 5%       |
| `engage_threshold`  | 0.90  | 90%      |

---

## 10. 物理仿真参数

| 参数                             | 值     | 说明       |
| ------------------------------ | ----- | -------- |
| `dt`                           | 1/120 | 8.33ms   |
| `max_position_iteration_count` | 192   | 防穿透      |
| `solver_type`                  | 1     | TGS      |
| `gpu_max_num_partitions`       | 1     | 单 GPU 分区 |
| `static_friction`              | 1.0   |          |
| `dynamic_friction`             | 1.0   |          |

---

*最后更新: 2026-07-08*
