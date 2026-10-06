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

---

# 追加 3（用户指示：C0 / v14 / v15 全部重评）

明天评估必须**三组都用新记录器（分量式表头）重跑一遍**，不能沿用旧数字：

| 组 | 命令 | 检查点 / 方式 | 输出 |
|---|---|---|---|
| **C0** | `python tools/diagnose_oru_v2.py --num-envs 100 --steps 450 --seed 1234 --output .installation/c0_m3` | 零动作（`experiment_method=fixed`）| `logs/oru_episode_metrics_c0.csv` |
| **v14** | `python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Oru-Direct-v0 --num_envs 100 --headless --checkpoint logs/rl_games/OruAssembly/ours_v14_s0/nn/OruAssembly.pth env.task.experiment_method=full agent.params.config.player.deterministic=True` | 旧奖励下训练的最优副本（此前测得 95.0% / 中位 47.1 N，但那是**旧表头**，必须重测）| `logs/oru_episode_metrics_v14.csv` |
| **v15** | 同上，`--checkpoint logs/rl_games/OruAssembly/ours_v15_s0/nn/last_OruAssembly_ep_60_rew__1599.6842_.pth` | 新尾部惩罚的末轮策略（训练日志成功率 0.984）| `logs/oru_episode_metrics_v15ep60.csv` |

**关键操作细节**：
- 每次 play.py 运行**前**把 `logs/oru_episode_metrics.csv` **改名或删除**（环境只在文件不存在时写表头，否则会沿用上一次的表头 ✗）；
- 每次运行约 330 秒后停止（覆盖 1~2 个 450 步回合批次，约 100~200 个回合）；
- 最后出**三行对比**：成功率、接触力各分量峰值、腕部力/力矩各分量峰值、翻转、姿态/角速度 RMS；
  另附此前的汇总口径（接触力合力中位/P90/最大）。

---

# 追加 4（紧急：环境在分量式记录器补丁后无法初始化）

## 现状（2026-10-06 深夜）
- `action_space 12->13` + `switch_mode="learned"`（学 α）**已实现并推送** ✓（`58c37b2` ✓）。
- 但**环境初始化后立即静默退出** ✗：训练 `ours_v16/v16b/v16c` 全部 `exit code 1` ✗；
  1 环境 20 步诊断也**在 PhysX 初始化后无输出退出** ✗，且**没有 Python traceback** ✗。
- 已修的两处（都不是根因 ✗）：
  1. `_mp` 在 `_init_tensors` 里未定义 → `UnboundLocalError`（已在使用点就地定义 ✓ `23fde36` ✓）；
  2. 修完 `_mp` 后仍静默退出 ✗ → 说明**还有别的问题** ✗（非 Python 异常 ✓ 很可能是原生层/形状不匹配 ✓）。

## 已确认的"最后已知可用"版本
- **`2fbe57c`**（13 列记录器，C0 在 `pwsh-250` 上成功跑出 100 行 ✓）。
- **`10b7be5`**（分量式记录器 cf_/wf_/wt_/ct_ ✗）与 **`58c37b2`**（+ α 学习 ✓）之后**未成功运行过** ✗。

## 下一步：用二分定位（不要继续盲改 ✗）
1. **先修正命令行守卫的自身 bug** ✗：`... | Select-String | ...` 之后 `$LASTEXITCODE` 取的是 `Select-String` 的退出码 ✓✗
   → 必须写成 `& $py ... 2>&1 | Tee-Object -FilePath $log; $code = $LASTEXITCODE` 或先重定向再单独取码 ✓。
2. **逐个 hunk 回放**：`git diff 2fbe57c..HEAD -- source/isaaclab_tasks/isaaclab_tasks/direct/oru/oru_env.py`
   → 依次只保留一个 hunk（分量累计器 ✓ / 行写入 ✓ / 重置 ✓ / α 块 ✓），每次用 `--num-envs 1 --steps 20 --nominal` 验证（约 1 分钟 ✓），
   通过再叠加下一个 ✓。
3. 若单靠"13 列版"（`2fbe57c`）也失败 ✗，则根因在 env 之外（cfg / 启动环境 ✓），再看 `--num-envs 1` 的完整输出 ✓。
4. 定位后：**补 `mean_switch_alpha` 日志** ✓（证明策略学会切换 ✓，这是论文主张的关键证据 ✗ 目前缺失），再按 60 epoch 重训 ✓。

## 训练与评估的既有结论（仍然有效 ✓）
- C0（fixed）：成功率 **91.0%** ✓、接触力中位 **83.9 N** / P90 177.4 / 最大 450.9 ✓、翻转中位 57 ✓。
- v14（旧奖励最优副本）：成功率 **95.0%** ✓、接触力中位 **47.1 N** ✓（旧表头 ✓，需按新表头重测 ✓）。
- v15-epoch32（"回报最优"副本）：73.0% ✗、P90 **149.1** / 最大 **297.0 N** ✓（尾部惩罚有效但成功率被牺牲 ✗）。
- v15-epoch60（末轮，训练日志成功率 0.984 ✓）：**尚未评估** ✗（当时的 play.py 被中途停止 ✓）。
- 结论：rl_games 的 `<name>.pth` 是**回报最优** ✓，在新奖励下**回报最优 ≠ 成功率最优** ✗ → 评估应以**末轮检查点 + 评估集指标**为准 ✓，建议 `save_frequency: 10` ✓。

---

# 追加 5（用户 2026-10-07 指示）

## ① 查明 v17 的失败原因
问题：失败是**悬空（未接触）**还是**最后没有完全插入（压着/抖动但未坐实）**？
- 先用已有评估 CSV 做了**免费判读**（依据：`force_peak_N` 与 `contact_flips` 的组合）：
  几乎无力且无翻转 → 悬空；有力或有抖动 → 已接触但未坐实。结果见运行输出。
- 若要**精确**的末状态（就位间隙、横向误差、姿态、接触步数），需要给逐回合指标再加列
  （`final_gap_mm`, `final_xy_mm`, `final_tilt_rad`, `contact_steps`），然后重跑一次 v17 评估。
  **注意**：加列必须**先读代码再改**，并过 1 环境 20 步验证闸门，再提交。

## ② v14 的"最坏情况力"太高：目标是**所有情况的峰值 ≈ 40 N**
现状：C0 最大 450.9 N、v14 最大 533.5 N、v17 最大 980~1455 N —— 都不满足"最坏 ≈40 N"。
判读：这些尖峰是**接触求解器在啮合瞬间的单子步响应**（穿透容差 `rest_offset=-0.0005` + 接触刚度），
**不是策略能调节的量**，所以"最坏 ≤40 N"本质是**接触/力参数**问题，不是奖励问题。

可行路线（按性价比排序）：
1. **降低指令轴向力上限**：`max_task_force_z` 60 → 40 / 20 / 10 N，用 **C0 扫描**（每个 3 分钟、无需策略）测出峰值；
2. **减小穿透容差**：`rest_offset` −0.5 mm → −0.2 / −0.1 mm，同样用 C0 扫描；
3. 提高接触段阻尼（`scale_kd` 在接触段的上限）或降低接近速度（ramp 模式，此前测得无差异，作为补充）；
4. 在 1~3 选出"C0 最大力 ≈40 N"的配置后，再在该配置下**重训/评估 Ours**（阈值切换版），保证与 C0 同口径可比。

**论文口径提醒**：接触传感器为 1/120 s 子步口径、含求解器离散成分；腕部为 15 Hz、含链条惯性。
用户已定：**力以接触传感器为准，力矩以腕部为准**。
