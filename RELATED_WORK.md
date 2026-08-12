# 变阻抗装配 RL — 相关文献综述

> 对应创新方向：(1) 多点路径奖励 (2) 分段接触奖励 (3) 变阻抗策略学习

---

## 一、核心相关论文速查表

| # | 论文 | 发表 | 核心方法 | 关联创新点 |
|---|------|------|---------|-----------|
| 1 | SDQN+PDQN 变阻抗装配 | Chen et al., Applied Intelligence 2023 | DQN 双层 + 位姿误差奖励 | ③ |
| 2 | SRL-VIC 安全变阻抗 | Zhang et al., IEEE RA-L 2024 | 安全 Critic + 恢复策略 | ③ |
| 3 | Reach→Insert 两阶段 | Yuan et al., arXiv 2026 | IL 靠近 + RL 插入 | ② |
| 4 | DA-VIL 双臂变阻抗 | Fu et al., 2024 | RL 在线调节阻抗 | ③ |
| 5 | Waypoint RL 路径点 | Habibian et al., IEEE RA-L 2024 | MAB 路径点序列 | ① |
| 6 | 变刚度 DRL 装配 | Ji et al., IJPR 2024 | 变刚度在线自适应 | ②+③ |
| 7 | DDPG+模糊奖励装配 | Guangxi Univ., Robot 2023 | 模糊规则防止局部最优 | ② |
| 8 | RLVAC 两阶段 TD3 | Dalian Maritime Univ., 2024 | 粗调+精调模糊奖励 | ②+③ |
| 9 | 力柔顺扩散模型 | Okada et al., IROS 2024 | 去噪扩散 VIC | ② |

---

## 二、按创新点分类

### 创新一：多点路径奖励 — 直线轨迹约束

#### [5] Waypoint-Based Reinforcement Learning for Robot Manipulation Tasks
**Habibian et al., IEEE RA-L, 2024**

> 将轨迹分解为多个路径点，每个路径点作为 Multi-Armed Bandit 独立学习。路径点间插值得到连续轨迹。

**对接你的方法**：
- 你的"多点奖励"也是在起点→终点连线上布 N 个等距关键点
- Habibian 的关键点是**学习出来的**（MAB），你的关键点是**固定生成的**（几何直线）
- 可以结合：先用几何关键点预训练 → 再让 RL 自适应调整关键点位置

**可引用作为**：多点路径约束的可行性依据，说明"多点引导优于单点引导"

---

### 创新二：分段奖励 — 接触前后分离

#### [3] From Reach to Insert: Tactile-Augmented Precision Assembly under Sub-Millimeter Tolerances
**Yuan et al., arXiv 2605.04649, 2026**

> 两阶段分解：IL 负责靠近目标（Reach），RL 负责插入（Insert）。接触力降低 60%，0.05mm 间隙 67% 成功率。

**对接你的方法**：
- 最直接的参考论文！"Reach→Insert"就是你的"对准→插入"
- 差异：Yuan 用 IL+RL 两个策略，你用**一个策略 + 奖励切换**
- 差异：Yuan 用触觉传感器，你用**力/力矩估计**

**可引用作为**：两阶段分解有效性的直接证据

#### [6] Deep Reinforcement Learning on Variable Stiffness Compliant Control
**Ji et al., Int. J. Production Research, 2024**

> 变刚度处理变化接触状态，**无需暂停或切换控制器模式**。

**对接你的方法**：
- 和你一样：不切换控制器，只切换**奖励函数权重**
- 策略持续输出 12D 阻抗参数，奖励函数告诉它"现在该硬还是该软"

**可引用作为**：连续控制优于模式切换的依据

#### [7] DDPG with Fuzzy Reward for Peg-in-Hole
**Guangxi University, Robot, 2023**

> 模糊规则奖励函数防止策略陷入局部最优。5 种不同孔径下 10 步内完成装配。

**对接你的方法**：
- 你的阶段切换可以用**模糊隶属度**替代硬阈值，实现平滑过渡
- `fade_weight` 其实就是简化版模糊逻辑

**可引用作为**：奖励函数设计中模糊/渐进切换优于硬切换

#### [8] RLVAC: Two-Stage TD3 with Fuzzy Reward
**Dalian Maritime University, 2024**

> 粗调（MLP 姿态分类）+ 精调（RL 导纳参数搜索），模糊奖励机制。

**对接你的方法**：
- 粗调 ≈ 你的自由空间阶段，精调 ≈ 你的接触插入阶段
- 差异：RLVAC 用两个不同模块，你用**同一个策略 + 切换奖励**

**可引用作为**：二阶段方法的另一个变体

---

### 创新三：变阻抗策略学习 (12D Kp+Kd)

#### [1] Active Compliance Control Based on Combined Reinforcement Learning
**Chen et al., Applied Intelligence, 2023**

> SDQN 判断接触状态 + PDQN 优化变阻抗参数。奖励 = 姿态误差 + 插入深度。96% 成功率 (0.40mm)。

**对接你的方法**：
- 最接近的 baseline！用 RL 输出变阻抗参数
- 差异：Chen 用 DQN（离散动作），你用 PPO+LSTM（连续动作 12D）
- 差异：Chen 6 维动作，你 **12 维（Kp+Kd 独立）**

**可引用作为**：RL 输出变阻抗参数可行性的直接证明

#### [2] SRL-VIC: Safe RL with Variable Impedance Control
**Zhang et al., IEEE RA-L, 2024**

> 安全 Critic + 恢复策略。策略同时学任务执行和在线阻抗调节。Sim-to-Real 无需微调。

**对接你的方法**：
- 安全 Critic 思路可借鉴：在阶段 2 中，力峰值惩罚本质上是一个"安全约束"
- 可扩展：增加一个安全 Critic 专门评估力风险

**可引用作为**：变阻抗 RL 中安全性保障的方法

#### [4] DA-VIL: Adaptive Dual-Arm Manipulation with RL and VIC
**Fu et al., 2024**

> 策略学习 + 梯度优化结合，在线动态调制阻抗。双臂场景。

**对接你的方法**：
- 证明 RL 输出变阻抗参数在线调节是可行的
- 你的是 PPO 直接输出，DA-VIL 是 RL + 梯度优化混合

**可引用作为**：RL 在线阻抗调节的有效性支撑

---

## 三、你的方法与文献的对比矩阵

| | 你的方法 | Chen 2023 [1] | Yuan 2026 [3] | Habibian 2024 [5] | Ji 2024 [6] |
|---|---|---|---|---|---|
| 阻抗维度 | **12D (Kp+Kd)** | 6D | — | — | 6D |
| 轨迹约束 | **多点几何直线** | 无 | 无 | MAB 路径点 | 无 |
| 阶段切换 | **力检测 + 软过渡** | 无 | IL→RL 硬切换 | 无 | 无切换 |
| 力平滑 | **ΔF + peak + lateral** | 无 | 有 (触觉) | 无 | 无 |
| 算法 | PPO + LSTM | DQN | IL+SAC | MAB | DRL |
| 动作空间 | 连续 12D | 离散 | 连续 | 离散 | 连续 |

---

## 四、可引用的论述逻辑

```
1. 背景：RL 变阻抗装配是有效方法 [1][4][6]
   └─ Chen 2023: 96%成功率证明 RL 输出阻抗参数可行
   └─ Ji 2024: 变刚度不需要切换控制器模式

2. 动机：单点奖励 + 无阶段区分 → 轨迹弯曲 + 力冲击
   └─ Yuan 2026: 两阶段分解显著降低接触力
   └─ Habibian 2024: 多点路径引导优于单点

3. 方法：多点路径奖励 + 分段接触奖励
   └─ 创新①: 多点奖励 ≈ Habibian + 几何约束
   └─ 创新②: 分段奖励 ≈ Yuan + Ji + 软过渡

4. 实验：对比 Baseline vs. +多点 vs. +分段 vs. 完整
   └─ 参考 [3][5][7] 的评价指标体系
```

---

## 五、参考文献

1. Chen et al., "Active compliance control of robot peg-in-hole assembly based on combined reinforcement learning," *Applied Intelligence*, vol. 53, pp. 30677–30690, 2023. [DOI](https://link-hkg.springer.com/article/10.1007/s10489-023-05156-5)

2. Zhang et al., "SRL-VIC: A Variable Stiffness-Based Safe Reinforcement Learning for Contact-Rich Robotic Tasks," *IEEE RA-L*, June 2024. [IEEE](https://ieeexplore.ieee.org/abstract/document/10517611)

3. Yuan et al., "From Reach to Insert: Tactile-Augmented Precision Assembly under Sub-Millimeter Tolerances," *arXiv:2605.04649*, May 2026. [arXiv](https://arxiv.gg/abs/2605.04649)

4. Fu et al., "DA-VIL: Adaptive Dual-Arm Manipulation with Reinforcement Learning and Variable Impedance Control," *arXiv:2410.19712*, Oct. 2024. [arXiv](https://ui.adsabs.harvard.edu/abs/2024arXiv241019712F/abstract)

5. Habibian et al., "Waypoint-Based Reinforcement Learning for Robot Manipulation Tasks," *IEEE RA-L*, 2024. [IEEE](https://ieeexplore.ieee.org/abstract/document/10802681)

6. Ji et al., "Deep reinforcement learning on variable stiffness compliant control for programming-free robotic assembly in smart manufacturing," *Int. J. Production Research*, 2024. [Semantic Scholar](https://www.semanticscholar.org/paper/Deep-reinforcement-learning-on-variable-stiffness-Ji-Liu/569d460d2eccc31b2e410da83a93aa24a17b5f94)

7. Guangxi University, "Robotic Peg-in-hole Assembly Algorithm Based on Reinforcement Learning," *Robot*, vol. 45, no. 3, pp. 321–332, 2023. [DOI](https://robot.sia.cn/en/article/doi/10.13973/j.cnki.robot.220011)

8. Dalian Maritime University, "Reinforcement learning guided variable admittance control for fuzzy pose estimation and precise adjustment," *Chinese Journal of Scientific Instrument*, vol. 45, no. 11, pp. 170–177, 2024. [DOI](https://cdn.sciengine.com/doi/10.19650/j.cnki.cjsi.J2413206)

9. Okada et al., "A Contact Model based on Denoising Diffusion to Learn Variable Impedance Control for Contact-rich Manipulation," *IROS*, 2024. [Semantic Scholar](https://www.semanticscholar.org/paper/A-Contact-Model-based-on-Denoising-Diffusion-to-for-Okada-Komatsu/82705205268c7d8a85563a42704276c23d2e37d9/figure/4)

---

*创建时间: 2026-07-23*
