# ORU 两阶段奖励设计

> 基于文献调研的创新方案：参考 [Chen 2023], [Yuan 2026], [Habibian 2024], [Ji 2024], [Guangxi 2023]
> 阶段 1 = 多点路径引导 (参考 Factory + Waypoint RL)
> 阶段 2 = 力柔顺插入 (参考 Reach→Insert + SRL-VIC)
> 切换机制 = 接触力检测 + 模糊软过渡 (参考 Guangxi + Dalian)

---

## 一、总体框架

```
Episode 开始时
     │
     ▼
┌─────────────────────────────┐
│  阶段 1: 自由空间对准        │
│  ┌───────────────────────┐  │
│  │ 多点路径关键点奖励     │  │ ← 参考 Habibian 2024 + Factory keypoint squashing
│  │ 姿态对齐奖励           │  │
│  │ 直线偏离惩罚           │  │
│  └───────────────────────┘  │
│  目标: 沿直线到达接触点上方  │
└──────────┬──────────────────┘
           │  Fz > contact_threshold  (接触检测)
           │  + 滞后窗口 (Hystheres)
           ▼
┌─────────────────────────────┐
│  阶段 2: 接触柔顺插入        │
│  ┌───────────────────────┐  │
│  │ Z 推进 + 姿态保持奖励  │  │ ← 参考 Yuan 2026
│  │ 力平滑 + 力峰值惩罚    │  │ ← 参考 Zhang 2024 (SRL-VIC)
│  │ 横向力惩罚 (防蹭边)    │  │
│  │ 精调收敛奖励           │  │ ← 针对"差一点"设计
│  └───────────────────────┘  │
│  目标: 平稳下压完成对接      │
└─────────────────────────────┘
```

---

## 二、阶段 1：多点路径引导（自由空间对准）

### 2.1 参考来源

| 论文 | 借鉴思路 |
|------|---------|
| **Habibian 2024** [5] | Waypoint-based RL：轨迹分解为路径点序列 |
| **Factory (Isaac Lab)** | 多尺度关键点 + squashing 函数 |
| **Chen 2023** [1] | 姿态误差 + 插入深度作为稠密奖励 |

### 2.2 关键点生成

在 EE 当前位姿 → 目标位姿的**直线连线上**，生成 N 个等距关键点。与 Factory 不同：Factory 的关键点在物体 Z 轴上（与姿态耦合），我们的是在世界坐标系直线上（纯几何约束）。

```python
def get_path_keypoints(ee_start, target_pos, num_kp, device):
    """
    Factory 关键点: 在物体自身 Z 轴上 (姿态耦合)
    我们的关键点:   在世界坐标 EE→target 直线上 (几何约束)
    """
    t = torch.linspace(0, 1, num_kp, device=device)
    # 排除 t=0 (起点) 和 t=1 (终点) 避免重复计算
    t = t[1:-1]  # N-2 个中间点
    keypoints = ee_start.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * (target_pos - ee_start).unsqueeze(1)
    return keypoints  # (N_envs, N-2, 3)
```

### 2.3 阶段 1 奖励函数

```python
# ── 阶段 1 奖励 ──────────────────────────────────────
# 1. 多点路径奖励 (核心)
# EE 到每条关键点的距离 → squashing 压缩
# 参考 Factory 的三组 (a,b) 参数，适配不同距离范围

path_keypoints = get_path_keypoints(ee_start, target_pos, num_kp=6)
dist_to_path = torch.norm(ee_pos.unsqueeze(1) - path_keypoints, dim=-1).mean(dim=-1)

# 多尺度 squashing (与 Factory 同结构，参数重新标定)
R_path = squashing(dist_to_path, a=5,  b=4)    # 粗引导 (基线)
       + squashing(dist_to_path, a=50,  b=2)    # 中距离 (对齐)
       + squashing(dist_to_path, a=100, b=0)    # 精细引导

# 2. 终点奖励 (target 位置处额外加一个关键点，权重最高)
dist_to_target = torch.norm(ee_pos - target_pos, dim=-1)
R_target = squashing(dist_to_target, a=150, b=0)  # 终点最高优先级

# 3. 姿态对齐 (quat dot product 的绝对值)
# quat 越接近 [0,0,1,0]，alignment 越高
target_quat = torch.tensor([0,0,1,0])
alignment = torch.abs(torch.sum(ee_quat * target_quat, dim=-1))  # [0, 1]
R_align = alignment  # 姿态对齐度奖励

# 4. 直线偏离惩罚
# 计算 EE 到理想直线的垂直距离
ideal_line_dir = target_pos - ee_start
line_len = torch.norm(ideal_line_dir, dim=-1, keepdim=True).clamp(min=1e-8)
to_ee = ee_pos - ee_start
projection = torch.sum(to_ee * ideal_line_dir, dim=-1, keepdim=True) / line_len
closest_point = ee_start + projection * ideal_line_dir / line_len
deviation = torch.norm(ee_pos - closest_point, dim=-1)
R_deviation = -0.5 * deviation  # 偏离惩罚

# ── 阶段 1 总奖励 ──
R_stage1 = R_path + R_target + 2.0 * R_align + R_deviation \
           - 0.01 * action_penalty - 0.001 * action_grad_penalty
```

### 2.4 借鉴 Factory 但改进点

| Factory 原版 | 我们的改进 |
|-------------|----------|
| 关键点在物体 Z 轴上 | 关键点在**世界直线**上（不依赖姿态） |
| 关键点只在 held/target 之间 | 关键点沿**整条路径**分布 |
| 仅用于奖励 | 同时约束**轨迹形状** |
| 3 阶段 squashing a=5/50/100 | 增加终点专用 a=150 加强终点引导 |

---

## 三、阶段 2：力柔顺插入（接触后）

### 3.1 问题分析

阶段 2 的距离很短（~2cm 以内），但当前的 `exp(-20x)` 在此范围内梯度太浅：

| 距目标 | exp(-20x) | 相邻步差异 |
|--------|-----------|-----------|
| 2cm | 0.670 | — |
| 1cm | 0.819 | +0.149 |
| 5mm | 0.905 | +0.086 |
| 2mm | 0.961 | +0.056 |
| 1mm | 0.980 | +0.019 |
| 0.5mm | 0.990 | +0.010 |

**根因**：梯度太小，RL 很难感知 1mm→0.5mm 的进步 → "差一点总是到不了"

### 3.2 参考来源

| 论文 | 借鉴思路 |
|------|---------|
| **Yuan 2026** [3] | 插入阶段独立 RL 策略 + 力平滑指标 |
| **Zhang 2024** (SRL-VIC) [2] | 安全约束 + 变刚度柔顺 |
| **Ji 2024** [6] | 连续变刚度无需切换控制器 |
| **Chen 2023** [1] | 插入深度作为稠密奖励 |

### 3.3 针对"差一点"的精调收敛奖励

解决 `exp(-20x)` 在 <1cm 时梯度弱的问题，串联两个函数：

```python
# ── 阶段 2 位置精调 ──────────────────────────────────
dist_to_target = torch.norm(ee_pos - target_pos, dim=-1)

# (a) exp(-50x): 在 0~2cm 范围提供更陡梯度
R_close = torch.exp(-50.0 * dist_to_target)

# (b) 对数奖励: 在 <5mm 时梯度放大
R_log = 1.0 / (1.0 - torch.log(dist_to_target.clamp(min=1e-6)) * 0.1)

# (c) Z 方向进度奖励: 只看 Z 方向，因为此时 XY 基本已对齐
z_progress = (ee_z_prev - ee_z) / dt  # Z 方向瞬时速度 (向下为正)
R_z_progress = 10.0 * torch.clamp(z_progress, min=0.0)  # 只奖励向下运动

# 精调组合
R_precision = R_close + 0.5 * R_log + R_z_progress
```

### 3.4 力柔顺奖励

```python
# ── 阶段 2 力柔顺 ────────────────────────────────────
# 参考 Zhang 2024 (SRL-VIC): 安全约束 + 变刚度
# 参考 Yuan 2026: 接触力降低 60%

F = self.applied_wrench[:, :3]    # Fx, Fy, Fz
T = self.applied_wrench[:, 3:6]   # Tx, Ty, Tz
F_xy = torch.norm(F[:, :2], dim=-1)  # 横向力大小

# 1. 力平滑惩罚 (力不应该剧烈变化)
R_force_smooth = -0.005 * torch.norm(F - self.prev_wrench[:, :3], dim=-1)

# 2. 力峰值惩罚 (参考 SRL-VIC 的安全约束)
F_mag = torch.norm(F, dim=-1)
R_force_peak = -0.01 * torch.relu(F_mag - 5.0)**2  # 超过 5N 重罚

# 3. 横向力惩罚 (防蹭边)
# 参考 Guangxi 2023: 模糊规则检测异常接触
R_lateral = -0.5 * F_xy  # XY 方向力越小越好

# 4. Z 方向力奖励 — 保持适度下压力 (不能软到推不动)
R_z_force = -torch.abs(F[:, 2] - 2.0) * 0.001  # 奖励 Z 力接近 2N

# 力柔顺总奖励
R_compliance = R_force_smooth + R_force_peak + R_lateral + R_z_force
```

### 3.5 阶段 2 总奖励

```python
R_stage2 = R_precision + R_compliance
```

---

## 四、阶段切换机制

### 4.1 参考来源

| 论文 | 借鉴思路 |
|------|---------|
| **Yuan 2026** [3] | 两阶段 IL→RL，接触力触发切换 |
| **Guangxi 2023** [7] | 模糊规则避免硬切换导致的奖励跳变 |
| **Dalian 2024** [8] | 模糊奖励 + 粗精两阶段 |

### 4.2 接触检测

```python
# 多种判据综合，避免误触发
Fz = abs(self.applied_wrench[:, 2])

# 判据 1: Z 方向力超过阈值
contact_force = Fz > 0.3  # 0.3N 视为接触

# 判据 2: EE 高度接近目标 (物理约束)
contact_height = (ee_pos[:, 2] - target_z) < 0.05  # 5cm 以内

# 判据 3: 力变化率突增 (首次接触的特征)
force_surge = (Fz - self.prev_wrench[:, 2].abs()) > 0.2  # 力突然增大
contact_surge = force_surge & (Fz > 0.5)

# 综合判据
in_contact = (contact_force & contact_height) | contact_surge
```

### 4.3 软过渡（核心创新点）

参考 Guangxi 2023 和 Dalian 2024 的模糊思想，不用硬 if-else，而是用**平滑混合权重**：

```python
# ── 平滑过渡权重 ──────────────────────────────────
# 参考 Guangxi 2023: 模糊规则 → sigmoid 平滑过渡

# 用 Fz 的 sigmoid 函数计算"接触程度" (0→1)
contact_degree = torch.sigmoid((Fz - 0.3) * 10.0)  # 0.3N 附近过渡

# Hysteresis: 进入容易、退出难 (防止在阈值附近振荡)
# 一旦进入过 stage2，需要 Fz < 0.1 才退回 stage1
was_in_contact = self._was_in_contact
entering = contact_degree > 0.5
leaving = (Fz < 0.1)
self._was_in_contact = torch.where(entering, True, torch.where(leaving, False, was_in_contact))
contact_degree = torch.where(self._was_in_contact, contact_degree.clamp(min=0.3), contact_degree)

# ── 最终奖励混合 ──
R_total = (1.0 - contact_degree) * R_stage1 + contact_degree * R_stage2 \
          - 0.01 * action_penalty - 0.001 * action_grad_penalty \
          + success.float()
```

### 4.4 过渡可视化

```
R_stage1 权重  ▲
          1.0 ├───────╲
              │        ╲
          0.5 │         ╲──── 平滑过渡区 (Fz ≈ 0.2~0.4N)
              │          ╲
          0.0 ├───────────┼──────────▶ Fz (N)
              0     0.2  0.3  0.4
                     │
                    接触判定阈值
                    
R_stage2 权重 ↑
          1.0 ├           ╱───────
              │          ╱
          0.5 │         ╱
              │        ╱
          0.0 ├───────╱───────────▶ Fz (N)
              └── sigmoid 软过渡 → 无奖励跳变
```

---

## 五、观测增强

为支持两阶段感知，观测新增：

```python
obs_dict = {
    ...现有 31D...,
    "applied_wrench": self.applied_wrench,     # 6D 力/力矩反馈
    "contact_degree": contact_degree,           # 1D 接触程度 (sigmoid)
    "pos_error": target_pos - ee_pos,           # 3D 位置误差
}
# 共 41D (+10D) → policy 观测 53D
```

> 参考 Yuan 2026: 触觉传感器增强接触感知。我们用 force/torque 估计替代触觉传感器。

---

## 六、与文献的对比

| | Yuan 2026 | Habibian 2024 | Chen 2023 | 我们的方案 |
|---|---|---|---|---|
| 阶段数 | 2 (IL+RL) | N 路径点 | 1 | **2 (奖励切换)** |
| 轨迹约束 | 无 | MAB 路径点 | 无 | **几何直线关键点** |
| 力平滑 | 有 (触觉) | 无 | 无 | **ΔF + peak + lateral** |
| 切换方式 | 硬切换 (IL→RL) | 离散路径点 | 无 | **Sigmoid 软过渡 + Hysteresis** |
| 传感器 | 触觉 + 视觉 | — | F/T | **F/T (力/力矩估计)** |
| 收敛问题 | — | — | 96%@0.4mm | **对数+指数混合精调** |

---

## 七、实现计划

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `oru_env.py` | 新增 `_get_path_keypoints()` 生成阶段1关键点 |
| 2 | `oru_env.py` | 重写 `_get_rewards()` 为两阶段结构 |
| 3 | `oru_env.py` | 新增 `_get_contact_state()` 接触检测 + 滞后 |
| 4 | `oru_env.py` | 新增 `_get_observations()` 增强 (加力 + 接触标志) |
| 5 | `oru_tasks_cfg.py` | 新增阶段 2 精调参数 (a_fine=50, force_threshold, 等) |
| 6 | `oru_env_cfg.py` | 更新 OBS_DIM_CFG + obs_order |
| 7 | 实验 | Baseline vs +路径 vs +分段 vs 完整 (4 组对照) |

---

*创建时间: 2026-07-23*
