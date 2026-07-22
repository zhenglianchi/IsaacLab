# ORU 变阻抗装配 — 问题分析与创新方案

> 当前 baseline：策略输出 12D 变阻抗参数 (Kp+Kd)，固定目标，IK 域随机化。
> 目标：在 baseline 上提出创新优化，解决装配过程中的两个核心问题。

---

## 1. 当前场景问题诊断

### 问题 1：无法直接插入 — 蹭表面入孔

```
理想路径:                         实际路径:
  ╲                                ╲
   ╲  直接对准插入                   ╲  碰到边缘
    ╲                                ╲╱╲╱╲
     ▼                                ▼  │  ▼
  ┌─────┐                           ┌─────┐
  │ 孔  │                           │ 孔  │
  └─────┘                           └─────┘
                                    ↑ 蹭表面滑入
```

**根因分析**：
- 阻抗参数调节滞后：策略 15Hz 输出 Kp/Kd，但高频接触动态远超此频率
- XY 方向刚度不足或过大：太软→飘移蹭边，太硬→不柔顺无法自适应
- 缺少"预对准"阶段：策略不知道当前 EE 离目标多远、姿态偏差多大
- 旋转柔顺性不足：微小角度偏差导致 ORU 边缘先接触，然后蹭着滑入

**后果**：
- 物理磨损设备表面
- 任务失败（卡住无法插入）
- 装配时间长、效率低

### 问题 2：冲击力过大 — 磨损设备

**根因分析**：
- ORU 接触到孔壁时产生冲击力峰值
- 策略未能提前降低刚度以吸收冲击
- 力反馈延迟：策略通过观测感知接触状态有滞后
- 阻尼参数 Kd 未起足够作用

**后果**：
- 瞬时力过大损坏 ORU 或对接面
- 长期磨损降低设备寿命

---

## 2. 创新方案：两阶段奖励 + 力平滑引导

### 阶段 1：对准阶段（Approach）— 奖励直接插入轨迹

**目标**：让 ORU 沿最优路径直接对准孔口，避免蹭边

```
观测增强：
  + pos_error_xy:  EE 与目标 XY 偏差 (2D)
  + pos_error_z:   EE 与目标 Z 偏差 (1D)
  + alignment_score: 姿态对齐度（quat dot product）

奖励设计：
  R_align = w1 × exp(-α × pos_error_xy²)     # XY 越近奖励越高
          + w2 × exp(-β × pos_error_z²)      # Z 高度合适
          + w3 × alignment_score              # 姿态对齐
          - w4 × 蹭边惩罚                      # 检测到侧面接触时扣分

蹭边检测：
  F_lateral = sqrt(Fx² + Fy²)     # XY 方向合力
  F_normal  = |Fz|                 # Z 方向力
  当 F_lateral > threshold_ratio × F_normal → 判定为蹭边
  每步检测到蹭边：扣 R_rub_penalty × count
```

### 阶段 2：插入阶段（Insertion）— 力平滑优化

**目标**：接触后平滑调节力，避免冲击

```
奖励设计：
  R_force = - w5 × ||ΔF||²                    # 力变化惩罚（平滑性）
           - w6 × max(0, |F| - F_safe)²        # 超安全阈值惩罚
           - w7 × jerk                          # 力导数（加加速度）
           + w8 × success                       # 最终成功插入

力平滑指标：
  force_smoothness = Σ||F_t - F_{t-1}||² / N   # 越小越平滑
  force_peak       = max(|F|)                    # 力峰值

插入进度奖励：
  R_progress = w9 × (z_current - z_start) / (z_target - z_start)
  # 奖励每一步向目标靠近，防止蹭边时"原地踏步"
```

### 两阶段切换机制

```
状态判定（基于 Z 位置或接触力）：
  if z_error > approach_threshold AND Fz < contact_threshold:
      → 阶段 1 (对准)
      奖励权重: R_align 高, R_force 低
  else:
      → 阶段 2 (插入)
      奖励权重: R_force 高, R_align 低
```

---

## 3. 域随机化偏移分析框架

### 分析维度

| 维度 | 指标 | 目的 |
|------|------|------|
| 偏移 X (左右) | 成功率、平均力 | 横向偏移的影响 |
| 偏移 Y (前后) | 成功率、平均力 | 纵向偏移的影响 |
| 偏移 Z (高度) | 插入时间 | 起始高度的影响 |
| 偏移 R (旋转) | 蹭边概率 | 姿态偏差的影响 |
| 综合偏移量 | `‖offset‖` | 偏移大小与性能的关系 |

### 分析代码框架

```python
# 伪代码：批量评估不同偏移下的性能
offsets = [
    (0.00, 0.00, 0.00,  0.0, 0.0, 0.0),   # 基准
    (0.03, 0.00, 0.00,  0.0, 0.0, 0.0),   # X 偏移
    (0.00, 0.03, 0.00,  0.0, 0.0, 0.0),   # Y 偏移
    (0.00, 0.00, 0.00,  0.0, 0.0, 0.05),  # Yaw 偏移
    (0.06, 0.06, 0.00,  0.0, 0.0, 0.05),  # 综合偏移
    ...
]

for offset in offsets:
    result = evaluate(offset)
    record: success, avg_force, peak_force, rub_count, insertion_time
    plot: force trace over time
```

---

## 4. 提出的改动计划（待实施）

### 代码改动

| 文件 | 改动 |
|------|------|
| `oru_env.py` | 新增蹭边检测逻辑、两阶段奖励、力平滑奖励 |
| `oru_tasks_cfg.py` | 新增奖励权重参数、安全力阈值、蹭边检测阈值 |
| `play_force.py` | 扩展为批量评估模式（多偏移自动测试） |

### 新奖励函数骨架

```python
def _get_rewards(self):
    # 现有关键点奖励（保留）
    kp_dist = ...
    rew_kp = squashing_fn(kp_dist, ...)  # 保持引导作用

    # ── 新增 ──
    # 1. 蹭边检测
    F_xy = torch.norm(self.applied_wrench[:, :2], dim=-1)
    F_z  = self.applied_wrench[:, 2].abs()
    is_rubbing = F_xy > 0.3 * F_z  # XY 力超过 Z 力 30%

    # 2. 力平滑
    force_jerk = torch.norm(
        self.applied_wrench - self.prev_wrench, dim=-1
    )

    # 3. 两阶段切换
    z_error = torch.abs(self.ee_pos[:, 2] - self.fixed_target_z)
    in_approach = (z_error > 0.05) & (F_z < 1.0)

    # 4. 组合奖励
    rew_align = exp(-20.0 * pos_error_xy_pow2) * in_approach.float()
    rew_force_smooth = -0.01 * force_jerk
    rew_rub_penalty = -0.5 * is_rubbing.float()
    rew_progress = z_error * 0.1  # 奖励靠近目标

    rew = rew_kp + rew_align + rew_force_smooth + rew_rub_penalty + rew_progress
    return rew
```

### 观测增强

```python
# Policy 观测新增
obs_dict = {
    ...现有...
    "applied_wrench": self.applied_wrench,   # 6D 当前力/力矩
    "pos_error": target_pos - ee_pos,         # 3D 距目标位置误差
}
# 策略看到力反馈 + 位置误差 → 更好地决策阻抗参数
```

---

## 5. 实验设计

### 实验组

| 组 | 说明 |
|----|------|
| A | Baseline（当前） |
| B | Baseline + 蹭边惩罚 |
| C | Baseline + 力平滑奖励 |
| D | Baseline + 两阶段（完整方案） |

### 评价指标

| 指标 | 计算方式 |
|------|---------|
| 成功率 | 成功 episode / 总 episode |
| 平均插入时间 | 从接触开始到成功 |
| 平均力峰值 | max(|F|) over episode |
| 力平滑度 | Σ||ΔF||² / N |
| 蹭边比例 | 检测到蹭边的步数 / 总步数 |
| 偏移鲁棒性 | 成功率 vs ‖offset‖ 曲线 |

---

## 6. 待解决的问题（设计决策）

1. **蹭边判定阈值**：`F_xy / F_z` 的比值设多少合适？需要从 force 曲线上标定
2. **两阶段切换条件**：用 Z 位置还是接触力？还是两者结合？
3. **力平滑 vs 快速插入的权衡**：过度平滑会拖慢插入速度
4. **观测是否加力信息**：加力信息会让策略"看到"接触，但增加观测维度
5. **是否需要 Curriculum Learning**：先学宽松偏移 → 逐步放大偏移

---

*创建时间: 2026-07-08*
*状态: 方案设计阶段，待实施*
