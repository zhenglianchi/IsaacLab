# 下一轮会话待办（2026-10-06 晚记录，用户要求次日开始评估）

## 用户指令
次日用户发消息后，**直接开始评估**（不必再确认方案）。

## 一、当前配置（已定稿，勿随意更改）
- `max_epochs = 60`（用户要求：以后所有训练统一 60 epoch）
- `episode_length_s = 30.0`（450 步）；**同步复位**（成功不再单独终止回合）
- `reference_mode = "setpoint"`（单目标点，无轨迹/无限速）
- 起点高度原值（`start_z_offset = 0`，EE 0.7388 m / ORU 0.3465 m，就位面 0.0375 m）
- 域随机化：x∈[−12,0] cm、y∈±12 cm、z 不随机、姿态各轴 ±3°（按可达性裁剪：腕高处水平可达半径 0.42 m vs 基座到插槽 0.40 m）
- RL：`gamma = 0.999`、`learning_rate = 1e-4`、`mini_epochs = 4`、`gain_range = 2.0`
- 奖励：`success_bonus = 3000`、`hover_penalty = 1.0`、`timeout_penalty = 500`、
  `insertion_time_penalty = 0.05`、`insertion_xy_weight = insertion_angle_weight = 0.15`、
  `force_peak_weight = 0.01`（阈值 60 N）、`force_peak_budget = 40 N`、`force_peak_penalty = 20 /N`（**新增的尾部惩罚**）
- `target_quat` 已按无载荷实测标定；固定关节无驱动（已知"绕工具轴残留扭转"建模缺陷，暂不修）

## 二、已测基线（统一口径：100 环境 / 450 步 / seed 1234 / 同步复位 / 单目标点）
| 组 | 成功率 | 接触力峰值 中位 / P90 / 最大 | >50N | 翻转中位 | 腕部力矩中位/P90 Nm | 横向力中位 N | 绕轴力矩中位 Nm |
|---|---|---|---|---|---|---|---|
| **C0（fixed，零动作）** | **91.0%** | 83.9 / 177.4 / 450.9 N | 70 | 57 | 6.04 / 7.16 | 9.96 | 5.74 |
| **Ours v14**（`ours_v14_s0`，旧奖励）| **95.0%** | **47.1** / 195.1 / 533.5 | 48 | 41 | — | — | — |
| Ours v15 **epoch32**（回报最优副本）| 73.0% | 61.9 / **149.1** / **297.0** | 61 | 55 | 6.68 / 7.25 | 12.25 | 4.99 |

## 三、明日第一步：评估 v15 的 epoch 60 策略
```
检查点：logs/rl_games/OruAssembly/ours_v15_s0/nn/last_OruAssembly_ep_60_rew__1599.6842_.pth
命令：python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Oru-Direct-v0 \
        --num_envs 100 --headless --checkpoint <上面的路径> \
        env.task.experiment_method=full agent.params.config.player.deterministic=True
```
- 环境会自动写 `logs/oru_episode_metrics.csv`（11 列）→ 与 `logs/oru_episode_metrics_c0.csv` 对比即可；
- 建议跑约 330 秒后停止（覆盖 1~2 个 450 步回合批次）。

**判据**：若 ep60 成功率 ≈ 0.98 且 P90/最大 ≈ 149/297 N → **v15-ep60 就是论文的"本文方法"**；
若 ep60 尾部回到 195/533 N → **以 v14 为主结果**，尾部如实报告并注明成因（子步级求解器尖峰）。

## 四、之后
1. 填表 3-9「本文方法」列 + 更新 3.6.7 + 推送；
2. 建议设 `save_frequency: 10`（因**回报最优 ≠ 成功率最优**，模型应按评估指标挑选，与 3.6.6 的原则一致）；
3. 若还要压尾部：物理限幅（指令轴向力上限 60 → 20 N、穿透容差 −0.5 → −0.2 mm）。

---

# 追加（用户 2026-10-06 晚指示）

## 评估口径：**同时用腕部六维力**
真机使用的是**腕部六维力/力矩传感器**，因此评估时除了接触传感器口径，还要按**腕部力**再评一次：
- 腕部力/力矩取自 `robot.data.body_incoming_joint_wrench_b[:, _ee_frame_idx]`（前 3 维力、后 3 维力矩，体坐标系），用 `ee_quat` 旋到世界系；
- 逐回合记录器（`logs/oru_episode_metrics*.csv`）现在共 **13 列**，含腕部量：
  `wrist_tau_peak_Nm`（|τ| 峰值）、`wrist_fxy_peak_N`（横向力）、`wrist_tauz_peak_Nm`（绕工具轴 τ_z）、
  **`wrist_f_peak_N`（|F| 峰值）**、**`wrist_fz_peak_N`（轴向 |F_z| 峰值）**；
- **两种口径都要报**并注明差异：接触传感器为 **1/120 s（子步）**、反映真实接触载荷（C0 中位 83.9 N、最大 450.9 N）；腕部六维为 **15 Hz（策略频率）**、含链条惯性、量级更小（C0 中位约 10~30 N）。
  真机可比性以**腕部口径**为准，物理冲击强度以**接触传感器口径**为准。

## 明日执行顺序（更新）
1. 因记录器新增两列，**先用新记录器重跑一次 C0**（3 分钟，命令见上）→ 覆盖 `logs/oru_episode_metrics_c0.csv`；
2. 评估 **v15 epoch-60** 检查点（`nn/last_OruAssembly_ep_60_rew__1599.6842_.pth`）；
3. 出 **三行对比**（C0 / v14 / v15-ep60），其中**接触力与腕部力两套口径并列**；
4. 按判据定稿主结果 → 填表 3-9 + 更新 3.6.7 + 推送。

---

# 追加 2（用户指示：分量式记录，不看合力）

- 逐回合指标 CSV **不再只记合力模**，而是**把每个方向的力/力矩分量分开记录**：
  - `cf_{x,y,z}_{max,min}`：**接触传感器**力的三分量（世界系，带符号极值）
  - `wf_{x,y,z}_{max,min}`：**腕部**力三分量（世界系）
  - `wt_{x,y,z}_{max,min}`：**腕部**力矩三分量（世界系）
  - `ct_{x,y,z}_{max,min}`：**接触传感器力矩**三分量 —— **仅当该 IsaacLab/PhysX 版本提供时**才有这几列
    （启动时会打印 `[oru] episode metrics columns: N | contact torque source: ...`；
     现有代码路径只用 `net_forces_w`/`force_matrix_w`，即**多数版本只提供力**，此时无 `ct_*` 列）
- 原有的合力模列保留（`force_peak_N`、`wrist_f_peak_N`、`wrist_fxy_peak_N`、`wrist_fz_peak_N`、
  `wrist_tau_peak_Nm`、`wrist_tauz_peak_Nm`），便于出总表；分量列用于看方向与符号。
- **注意**：表头变了，旧的 `logs/oru_episode_metrics*.csv` 已删除 ✓，明天评估前必须重跑 C0 生成新表头。
- 评估时两套口径都要报：**接触传感器**（1/120 s 子步，真实接触载荷）与**腕部六维**（15 Hz，含链条惯性；真机可比性以此为准）。
