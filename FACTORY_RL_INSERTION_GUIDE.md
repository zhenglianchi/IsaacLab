# Factory 装配插入任务强化学习详解

> 基于 IsaacLab 的 Factory PegInsert 任务（Isaac-Factory-PegInsert-Direct-v0）的完整 RL 实现分析，以及基于它改编的 ORU 装配任务（Isaac-Oru-Direct-v0）。

---

## 目录

1. [整体架构](#1-整体架构)
2. [Gym 环境注册](#2-gym-环境注册)
3. [场景搭建](#3-场景搭建)
4. [动作空间与控制](#4-动作空间与控制)
5. [观测空间（非对称 Actor-Critic）](#5-观测空间非对称-actor-critic)
6. [奖励函数设计（核心）](#6-奖励函数设计核心)
7. [成功判定](#7-成功判定)
8. [重置与域随机化](#8-重置与域随机化)
9. [PPO 训练配置](#9-ppo-训练配置)
10. [仿真物理参数](#10-仿真物理参数)
11. [ORU 适配说明](#11-oru-适配说明)

---

## 1. 整体架构

### 类继承关系

```
DirectRLEnv (isaaclab.envs)
  ├── FactoryEnv (factory_env.py)     # Franka Panda + PegInsert 销插入任务
  └── OruEnv (oru_env.py)             # UR5 + ORU 装配任务
```

### 文件结构与职责

| 文件                             | 职责                                      |
| ------------------------------ | --------------------------------------- |
| `*_env.py`                     | RL 环境主逻辑：step、reset、reward、observations |
| `*_env_cfg.py`                 | 环境配置：仿真参数、观测/状态维度、机器人模型                 |
| `*_tasks_cfg.py`               | 任务配置：资产规格、奖励系数、成功阈值                     |
| `*_control.py`                 | 控制器：阻抗控制（任务空间 PD + 零空间投影）               |
| `*_utils.py`                   | 工具函数：关键点偏移、奖励压缩函数、摩擦力设置                 |
| `__init__.py`                  | Gym 环境注册                                |
| `agents/rl_games_ppo_cfg.yaml` | PPO 超参数                                 |

### FactoryEnv 初始化流程

```python
# factory_env.py:26-38
class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # 1. 根据 obs_order / state_order 动态计算观测和状态维度
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        # 2. 加上上一帧动作（作为观测/状态的一部分）
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space

        super().__init__(cfg, render_mode, **kwargs)

        # 3. 设置刚体惯量修正（补 IGE armature 差异）
        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        # 4. 初始化张量
        self._init_tensors()
        # 5. 设置默认动力学参数（增益、摩擦力等）
        self._set_default_dynamics_parameters()
```

---

## 2. Gym 环境注册

每个任务通过 `gym.register()` 注册，绑定环境类、配置类和 PPO 配置：

```python
# factory/__init__.py
gym.register(
    id="Isaac-Factory-PegInsert-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_env_cfg:FactoryTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
```

配置类的层级结构：

```python
# factory_env_cfg.py:191-195
@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0

# factory_tasks_cfg.py:104-108
@configclass
class PegInsert(FactoryTask):
    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()    # 固定件：8mm 孔
    held_asset_cfg = Peg8mm()      # 手持件：8mm 销
    asset_size = 8.0
    duration_s = 10.0
```

---

## 3. 场景搭建

### 3.1 资产定义

场景包含三类核心资产：**机器人**、**固定件（目标）**、**手持件（被操作物体）**。

```python
# factory_env_cfg.py:121-188 (机器人 - Franka Panda)
robot = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/franka_mimic.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,          # 机器人不受重力
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.00871,
            "panda_joint2": -0.10368,
            # ... 7个关节 + 2个手指关节
            "panda_finger_joint2": 0.04,  # 手指张开
        },
    ),
    actuators={
        "panda_arm1": ImplicitActuatorCfg(   # 关节1-4
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=0.0, damping=0.0,      # 力控模式 (零刚度/阻尼)
            effort_limit_sim=87,
        ),
        "panda_arm2": ImplicitActuatorCfg(   # 关节5-7
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=0.0, damping=0.0,
            effort_limit_sim=12,
        ),
        "panda_hand": ImplicitActuatorCfg(   # 手爪
            joint_names_expr=["panda_finger_joint[1-2]"],
            stiffness=7500.0, damping=173.0,  # 位置控制模式
            effort_limit_sim=40.0,
        ),
    },
)
```

> **关键设计**：机械臂使用 **力矩控制** (`stiffness=0, damping=0`)，而手爪使用 **位置控制** (`stiffness=7500`)。这使得 RL 策略可以直接输出力矩指令。

### 3.2 固定件与手持件配置

```python
# factory_tasks_cfg.py:88-101 (PegInsert 任务的资产)
@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986      # 8mm 销的直径
    height = 0.050
    mass = 0.019

@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081         # 8mm 孔（略大于销，有间隙）
    height = 0.025
    base_height = 0.0
```

### 3.3 场景组装

```python
# factory_env.py:85-116
def _setup_scene(self):
    # 地面
    spawn_ground_plane(prim_path="/World/ground",
                       cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
    # 桌子
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/envs/env_.*/Table", cfg,
             translation=(0.55, 0.0, 0.0))

    # 三个核心 Articulation
    self._robot = Articulation(self.cfg.robot)
    self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
    self._held_asset = Articulation(self.cfg_task.held_asset)

    # 克隆环境到所有 parallel envs
    self.scene.clone_environments(copy_from_source=False)
```

---

## 4. 动作空间与控制

### 4.1 动作空间定义

动作空间为 **6 维连续空间**（3 平移 + 3 旋转）：

```python
# factory_env_cfg.py:73
action_space = 6  # [dx, dy, dz, droll, dpitch, dyaw]
```

### 4.2 EMA 动作平滑

策略输出的原始动作经过指数移动平均 (EMA) 平滑处理，防止高频抖振：

```python
# factory_env.py:207-213
def _pre_physics_step(self, action):
    self.actions = (
        self.cfg.ctrl.ema_factor * action.clone().to(self.device)
        + (1 - self.cfg.ctrl.ema_factor) * self.actions
    )
    # ema_factor = 0.2 → 新动作权重 20%，旧动作 80%
```

### 4.3 动作到控制信号的转换

策略输出的增量动作（delta）经过缩放到世界坐标系的目标位姿，再通过阻抗控制器映射为关节力矩：

```python
# factory_env.py:253-302 (_apply_action 核心流程)
def _apply_action(self):
    # 1. 增量位移 → 缩放 (阈值乘子)
    pos_actions = self.actions[:, 0:3] * self.pos_threshold  # [0.02, 0.02, 0.02]
    rot_actions = self.actions[:, 3:6] * self.rot_threshold  # [0.097, 0.097, 0.097]

    # 2. 目标末端位姿 = 当前位置 + 增量
    ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

    # 3. 位置裁剪：策略不能移动超过固定件 5cm 以外
    delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
    pos_error_clipped = torch.clip(
        delta_pos,
        -self.cfg.ctrl.pos_action_bounds[0],   # [-0.05, -0.05, -0.05]
        self.cfg.ctrl.pos_action_bounds[1],     # [0.05, 0.05, 0.05]
    )
    ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

    # 4. 增量旋转 → 四元数
    angle = torch.norm(rot_actions, p=2, dim=-1)
    axis = rot_actions / angle.unsqueeze(-1)
    rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
    ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
        rot_actions_quat, self.fingertip_midpoint_quat)

    # 5. 强制末端保持朝下 (roll=π, pitch=0)  — 装配任务约束
    target_euler_xyz[:, 0] = 3.14159  # roll = pi (指尖朝下)
    target_euler_xyz[:, 1] = 0.0      # pitch = 0

    # 6. 调用控制器生成关节力矩
    self.generate_ctrl_signals(
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos=0.0,  # 手指夹紧
    )
```

### 4.4 阻抗控制器（Operational Space Control）

这是整个控制链路的核心。采用 **Task-Space PD + 零空间投影** 架构：

```python
# factory_control.py:20-101
def compute_dof_torque(cfg, dof_pos, dof_vel, ...):
    """
    计算 Franka 关节力矩以驱动末端到目标位姿。

    数学原理：
      τ = J^T · F_task + N · τ_null
    其中:
      F_task = Kp · Δx - Kd · ẋ           (任务空间 PD)
      N = I - J^T · J̄^T                    (零空间投影矩阵)
      τ_null = M · (Kp_null · Δq - Kd_null · q̇)  (零空间力矩)
    """

    # Step 1: 计算位姿误差（位置 + 旋转 axis-angle）
    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos, fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric", rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Step 2: 计算任务空间力/力矩 (wrench)
    # F_task = Kp * Δx - Kd * ẋ
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose, fingertip_midpoint_linvel,
        fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains,
    )
    task_wrench += task_wrench_motion

    # Step 3: Dead zone — 模拟低力不可靠区
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # Step 4: 映射到关节空间 τ = J^T · F
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Step 5: 零空间力矩 — 保持关节角度在自然位置附近
    arm_mass_matrix_task = torch.inverse(
        jacobian @ arm_mass_matrix_inv @ jacobian_T)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    # 零空间投影 N = I - J^T · J̄^T
    u_null = kd_null * (-dof_vel[:, :7]) + kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (I - J^T @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # Step 6: 力矩裁剪
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)

    # 手指使用 PhysX 内置 PD 控制器（位置控制）
    self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
    self.joint_torque[:, 7:9] = 0.0
    self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
    self._robot.set_joint_effort_target(self.joint_torque)
```

**任务空间 PD 增益设计**：

```python
# factory_env_cfg.py:62
default_task_prop_gains = [100, 100, 100, 30, 30, 30]
#                           ↑平移增益      ↑旋转增益
# 平移方向增益更高（100 vs 30），因为装配需要高精度的位置控制

# factory_utils.py:19-23 — 临界阻尼下的微分增益
def get_deriv_gains(prop_gains, rot_deriv_scale=1.0):
    """Kd = 2 * sqrt(Kp)，旋转方向额外缩小 10 倍"""
    deriv_gains = 2 * torch.sqrt(prop_gains)
    deriv_gains[:, 3:6] /= rot_deriv_scale  # 旋转阻尼更小 → 更柔顺
    return deriv_gains
```

---

## 5. 观测空间（非对称 Actor-Critic）

### 5.1 设计原理

Factory 使用 **非对称 Actor-Critic** 架构：

- **Policy (Actor)** 接收简化观测（模拟真实传感器）
- **Critic (Value)** 接收完整状态（训练时可用，模拟中获取不到的信息）

```python
# factory_env_cfg.py:17-42 — 各观测/状态组件的维度
OBS_DIM_CFG = {
    "fingertip_pos": 3,              # 指尖位置
    "fingertip_pos_rel_fixed": 3,    # 指尖相对固定件位置
    "fingertip_quat": 4,             # 指尖姿态 (四元数)
    "ee_linvel": 3,                  # 末端线速度
    "ee_angvel": 3,                  # 末端角速度
}

STATE_DIM_CFG = {
    # ... Actor 的所有观测 ...
    "joint_pos": 7,                  # 7 个关节位置
    "held_pos": 3,                   # 手持件位置
    "held_pos_rel_fixed": 3,         # 手持件相对固定件位置
    "held_quat": 4,                  # 手持件姿态
    "fixed_pos": 3,                  # 固定件位置
    "fixed_quat": 4,                 # 固定件姿态
    "task_prop_gains": 6,            # 当前任务空间增益
    "pos_threshold": 3,              # 位置动作阈值
    "rot_threshold": 3,              # 旋转动作阈值
}
```

### 5.2 观测构建

```python
# factory_env.py:160-192
def _get_factory_obs_state_dict(self):
    """构建观测和状态字典。"""

    # Actor 观测 — 仅含末端执行器信息（模拟实际传感器可获取的）
    obs_dict = {
        "fingertip_pos": self.fingertip_midpoint_pos,
        "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
        "fingertip_quat": self.fingertip_midpoint_quat,
        "ee_linvel": self.ee_linvel_fd,      # 有限差分速度（更稳定）
        "ee_angvel": self.ee_angvel_fd,
        "prev_actions": prev_actions,         # 上一帧动作
    }

    # Critic 状态 — 包含完整世界信息
    state_dict = {
        **obs_dict,
        "joint_pos": self.joint_pos[:, 0:7],
        "held_pos": self.held_pos,
        "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
        "held_quat": self.held_quat,
        "fixed_pos": self.fixed_pos,
        "fixed_quat": self.fixed_quat,
        "task_prop_gains": self.task_prop_gains,
        "pos_threshold": self.pos_threshold,
        "rot_threshold": self.rot_threshold,
    }
    return obs_dict, state_dict
```

> **关键细节**：
> 
> - Actor 的线速度和角速度使用 **有限差分 (finite-differenced)** 而非物理引擎直接输出，因为更可靠
> - 固定件位置加了 **观测噪声** `noisy_fixed_pos`，模拟传感器不完美
> - 上一帧动作作为观测一部分，提供时序信息

### 5.3 观测顺序和最终维度

```python
# factory_env_cfg.py:77-89
obs_order = [
    "fingertip_pos_rel_fixed",  # 3
    "fingertip_quat",           # 4
    "ee_linvel",                # 3
    "ee_angvel",                # 3
]  # = 13 维 + 6 (prev_actions) = 19 维

state_order = [
    "fingertip_pos",            # 3
    "fingertip_quat",           # 4
    "ee_linvel",                # 3
    "ee_angvel",                # 3
    "joint_pos",                # 7
    "held_pos",                 # 3
    "held_pos_rel_fixed",       # 3
    "held_quat",                # 4
    "fixed_pos",                # 3
    "fixed_quat",               # 4
]  # = 37 维 + 13 (增益/阈值) + 6 (prev_actions) = 56 维
```

---

## 6. 奖励函数设计（核心）

### 6.1 多尺度关键点奖励

Factory 任务的奖励函数核心是**多尺度关键点匹配**。在被操作物体（held）和目标位置（target/fixed）上各定义 N 个关键点，计算它们之间的距离，通过不同陡峭度的 **squashing function** 来引导不同阶段的行为。

```
         关键点 (keypoints)
    ┌───●───┐          ┌───●───┐
    │   ●   │  ←→     │   ●   │   计算 L2 距离
    │   ●   │          │   ●   │
    └───●───┘          └───●───┘
   held asset       target pose

   奖励 = Σ squashing_fn(距离, a_i, b_i)
```

### 6.2 关键点生成

关键点沿 Z 轴（垂直方向）均匀分布：

```python
# factory_utils.py:12-16
def get_keypoint_offsets(num_keypoints, device):
    """沿单位长度的线均匀分布关键点，中心在原点"""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    # 4 个关键点：z = [-0.375, -0.125, 0.125, 0.375] * keypoint_scale
    return keypoint_offsets
```

在计算奖励时，关键点被变换到世界坐标系：

```python
# factory_env.py:441-458
for idx, keypoint_offset in enumerate(keypoint_offsets):
    # 手持件侧的关键点
    keypoints_held[:, idx] = tf_combine(
        held_base_quat, held_base_pos,
        [1,0,0,0], keypoint_offset * keypoint_scale,
    )[1]
    # 目标侧的关键点
    keypoints_fixed[:, idx] = tf_combine(
        target_held_base_quat, target_held_base_pos,
        [1,0,0,0], keypoint_offset * keypoint_scale,
    )[1]

# 平均 L2 距离
keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)
```

### 6.3 Squashing Function（压缩函数）

三阶段压缩函数的数学形式：

```
r(x) = 1 / (exp(a·x) + b + exp(-a·x))
```

```python
# factory_utils.py:105-107
def squashing_fn(x, a, b):
    """bounded reward: r(x) = 1 / (exp(a*x) + b + exp(-a*x))
    - a: 控制斜率（越大越陡）
    - b: 控制最大值（b=0时最大为无穷，b越大峰值越低）
    """
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))
```

**三组参数覆盖不同距离范围**：

| 阶段       | a   | b   | 有效距离范围      | 作用           |
| -------- | --- | --- | ----------- | ------------ |
| baseline | 5   | 4   | 远距离 (>5cm)  | 引导向目标点粗略移动   |
| coarse   | 50  | 2   | 中距离 (1-5cm) | 对齐两个资产       |
| fine     | 100 | 0   | 近距离 (<1cm)  | 精细插入的"最后一英寸" |

### 6.4 完整奖励函数

```python
# factory_env.py:424-486
def _get_factory_rew_dict(self, curr_successes):
    rew_dict = {
        # 三阶段关键点奖励
        "kp_baseline": squashing_fn(keypoint_dist, a=5, b=4),
        "kp_coarse":   squashing_fn(keypoint_dist, a=50, b=2),
        "kp_fine":     squashing_fn(keypoint_dist, a=100, b=0),

        # 动作惩罚
        "action_penalty_ee": torch.norm(self.actions, p=2),       # L2 动作幅度
        "action_grad_penalty": torch.norm(                        # 动作变化量
            self.actions - self.prev_actions, p=2, dim=-1),

        # 阶段性奖励
        "curr_engaged": curr_engaged.float(),  # 到达 engage 阈值 (90%)
        "curr_success": curr_successes.float(), # 到达 success 阈值
    }

    rew_scales = {
        "kp_baseline": 1.0,
        "kp_coarse": 1.0,
        "kp_fine": 1.0,
        "action_penalty_ee": -action_penalty_ee_scale,     # 负奖励
        "action_grad_penalty": -action_grad_penalty_scale, # 负奖励
        "curr_engaged": 1.0,   # 密集奖励，引导到接近目标
        "curr_success": 1.0,   # 稀疏奖励，成功时才给
    }

    # 总奖励 = Σ (reward_i * scale_i)
    rew_buf = sum(rew_dict[k] * rew_scales[k] for k in rew_dict)
    return rew_buf
```

> **设计思想**：baseline + coarse + fine 三阶段确保（1）远距离有梯度引导（2）中距离有足够的对齐精度（3）近距离有极高的精度。动作惩罚防止抖振和过大动作。

---

## 7. 成功判定

### 7.1 判定逻辑

成功由两个条件同时满足决定：

```python
# factory_env.py:343-382
def _get_curr_successes(self, success_threshold):
    # 1. XY 方向对齐：手持件底部与目标位置的 XY 距离 < 2.5mm
    xy_dist = torch.linalg.vector_norm(
        target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
    is_centered = xy_dist < 0.0025  # 2.5 mm

    # 2. Z 方向深度：销底部低于孔顶面一定比例即为成功
    height_threshold = fixed_cfg.height * success_threshold  # 孔深 × 4% = 1mm

    z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
    is_close_or_below = z_disp < height_threshold

    curr_successes = torch.logical_and(is_centered, is_close_or_below)
```

### 7.2 成功指标记录

```python
# factory_env.py:384-403
def _log_factory_metrics(self, rew_dict, curr_successes):
    # Episode 结束时的成功率
    if torch.any(self.reset_buf):
        self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

    # 首次成功所需步数
    first_success = torch.logical_and(curr_successes,
                                      torch.logical_not(self.ep_succeeded))
    self.ep_succeeded[curr_successes] = 1
    self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
```

---

## 8. 重置与域随机化

### 8.1 重置流程

```
_reset_idx(env_ids)
  │
  ├── 1. 移动资产到默认姿态
  │     _set_assets_to_default_pose()
  │
  ├── 2. 移动 Franka 到默认关节位置
  │     _set_franka_to_default_pose()
  │
  ├── 3. 仿真若干步（无动作）
  │     step_sim_no_action()
  │
  └── 4. 随机化初始状态
        randomize_initial_state()
          ├── 临时禁用重力
          ├── 随机化固定件位置/朝向
          ├── 随机化夹爪初始位置（IK求解）
          ├── 随机化手持件在夹爪中的位姿
          ├── 夹爪闭合抓取
          └── 恢复重力
```

### 8.2 固定件随机化

```python
# factory_env.py:614-651
def randomize_initial_state(self, env_ids):
    # 临时禁用重力（防止重置时物体掉落）
    physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

    # (1) 固定件位置随机化
    rand_sample = torch.rand((len(env_ids), 3))  # [0, 1]
    fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
    fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(
        [0.05, 0.05, 0.05])  # 缩放到 ±5cm

    # (2) 固定件朝向随机化
    fixed_orn_yaw_range = np.deg2rad(360.0)  # PegInsert: 全方位旋转
    rand_sample = torch.rand((len(env_ids), 3))
    fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
    fixed_orn_euler[:, 0:2] = 0.0  # 仅改变 yaw

    # (3) 固定件观测噪声（模拟传感器误差）
    fixed_asset_pos_noise = torch.randn((len(env_ids), 3)) * [0.001, 0.001, 0.001]
    self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise
```

### 8.3 夹爪初始位姿（IK 迭代求解）

```python
# factory_env.py:671-722
# 不断尝试 IK 直到所有环境都求解成功
bad_envs = env_ids.clone()
while True:
    # 在固定件上方随机位置
    above_fixed_pos = fixed_tip_pos.clone()
    above_fixed_pos[:, 2] += hand_init_pos[2]  # 上方一定高度
    # 添加随机偏移
    above_fixed_pos[bad_envs] += above_fixed_pos_rand

    # IK 求解
    pos_error, aa_error = self.set_pos_inverse_kinematics(
        ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
        ctrl_target_fingertip_midpoint_quat=hand_down_quat,
        env_ids=bad_envs,
    )

    # 检查是否有 IK 失败的 env，重试
    bad_envs = bad_envs[any_error_ids]
    if bad_envs.shape[0] == 0:
        break
```

### 8.4 手持件在夹爪中的随机化

```python
# factory_env.py:742-783
# 在夹爪坐标系中随机偏移手持件
rand_sample = torch.rand((N, 3))
held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
held_asset_pos_noise_level = [0.003, 0.0, 0.003]  # X 和 Z 方向 ±3mm
held_asset_pos_noise *= held_asset_pos_noise_level
```

### 8.5 夹爪闭合抓取

```python
# factory_env.py:785-820
# 重置阶段使用高增益快速闭合
reset_task_prop_gains = [300, 300, 300, 20, 20, 20]  # 3x 正常值
grasp_time = 0.0
while grasp_time < 0.25:  # 250ms
    self.ctrl_target_joint_pos[:, 7:] = 0.0  # 手指闭合
    self.close_gripper_in_place()
    grasp_time += self.sim.get_physics_dt()

# 然后切换到正常增益开始 episode
self.task_prop_gains = self.default_gains  # [100, 100, 100, 30, 30, 30]

# 恢复重力
physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
```

---

## 9. PPO 训练配置

### 9.1 RL Games PPO 参数

```yaml
# factory/agents/rl_games_ppo_cfg.yaml
params:
  algo:
    name: a2c_continuous

  config:
    name: Factory
    device: cuda:0
    ppo: True
    mixed_precision: True          # 混合精度训练加速
    normalize_input: True           # 输入归一化
    normalize_value: True           # Value 归一化 (PopArt)

    # 训练规模
    num_actors: 128                 # 128 个并行环境
    horizon_length: 128             # 每个 rollout 收集 128 步
    minibatch_size: 512             # 小批量大小
    mini_epochs: 4                  # 每个 rollout 数据训练 4 轮

    # PPO 超参数
    gamma: 0.995                    # 折扣因子 (装配任务需要长期规划)
    tau: 0.95                       # GAE lambda
    learning_rate: 1.0e-4           # 学习率
    lr_schedule: adaptive           # 自适应学习率 (KL 阈值)
    kl_threshold: 0.008             # KL 散度阈值
    e_clip: 0.2                     # PPO clip 范围
    grad_norm: 1.0                  # 梯度裁剪
    entropy_coef: 0.0               # 无熵正则（装配任务不需要探索多样性）

    # 训练长度
    max_epochs: 200                 # 最多 200 轮

  network:
    # Actor-Critic 共享网络
    name: actor_critic
    separate: False                 # 共享特征提取

    # MLP 主干
    mlp:
      units: [512, 128, 64]        # 3 层 MLP
      activation: elu
      d2rl: False

    # LSTM 记忆网络（关键设计）
    rnn:
      name: lstm
      units: 1024                   # 隐层 1024 维
      layers: 2                     # 2 层 LSTM
      before_mlp: True              # LSTM 在 MLP 之前
      concat_input: True            # 上一帧输出拼接当前输入
      layer_norm: True              # 层归一化稳定训练
```

> **关键设计**：
> 
> - **LSTM 网络**：因为装配任务是部分可观测的（Actor 看不到手持件和固定件的精确位姿），需要 LSTM 来推断接触状态和隐形变量
> - **PopArt** (`normalize_value: True`)：稳定 Value 预测，因为装配任务奖励从远距离到成功跨越多个数量级
> - **无熵正则** (`entropy_coef: 0.0`)：装配是确定性任务，不需要探索多样性
> - **128 并行环境 × 128 步** = 每轮 16,384 个 transition

### 9.2 训练架构图

```
┌─────────────────────────────────────────────────┐
│                    PPO Training                   │
│                                                  │
│  128 并行环境 (GPU 仿真)                         │
│      ↓                                          │
│  Rollout: 128 steps × 128 envs = 16,384 trans   │
│      ↓                                          │
│  GAE Advantage + Value Bootstrap                │
│      ↓                                          │
│  4 epochs × minibatch 512                       │
│      ↓                                          │
│  PPO Clip Loss + Value Loss + (no entropy)      │
│      ↓                                          │
│  Adaptive LR (KL threshold = 0.008)             │
└─────────────────────────────────────────────────┘
```

---

## 10. 仿真物理参数

装配任务对物理仿真精度要求极高，因为销与孔的配合间隙在亚毫米级别。

```python
# factory_env_cfg.py:117-117
sim = SimulationCfg(
    device="cuda:0",
    dt=1/120,                      # 120Hz 仿真频率 (8.33ms)
    gravity=(0.0, 0.0, -9.81),
    physx=PhysxCfg(
        solver_type=1,             # TGS (Temporal Gauss-Seidel) — 比 PGS 更稳定

        # 位置迭代数（最重要！）
        max_position_iteration_count=192,  # 极高，防止物体穿透
        max_velocity_iteration_count=1,

        # 接触参数
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.00625,  # 6.25mm

        # GPU 内存配置
        gpu_max_rigid_contact_count=2**23,      # ~8.4M contacts
        gpu_max_rigid_patch_count=2**23,
        gpu_collision_stack_size=2**28,         # ~268M
        gpu_max_num_partitions=1,               # 关键：单分区确保稳定性
    ),
    physics_material=RigidBodyMaterialCfg(
        static_friction=1.0,        # 高摩擦 — 抓取和插入需要
        dynamic_friction=1.0,
    ),
)

# 控制频率
decimation = 8  # 策略运行在 120/8 = 15Hz
```

> **关键参数解读**：
> 
> - `gpu_max_num_partitions=1`：使用单个 GPU 分区，避免跨分区物理破碎
> - `max_position_iteration_count=192`：非常高的迭代数，防止 peg-in-hole 紧密配合时的穿透
> - `decimation=8`：策略 15Hz，物理 120Hz。让阻抗控制器（task-space PD）在每帧策略输出之间跑 8 步物理，实现平滑控制

---

## 11. ORU 适配说明

ORU (Orbital Replacement Unit) 任务从 Factory 的 PegInsert 任务改编而来，核心变化如下：

### 11.1 机器人更换：Franka Panda → UR5

| 特性    | Factory              | ORU                |
| ----- | -------------------- | ------------------ |
| 机器人   | Franka Panda (7 DOF) | UR5 (6 DOF)        |
| 末端执行器 | 两指夹爪                 | 固定连接链 (no gripper) |
| 自由度   | 9 (7 arm + 2 finger) | 6                  |
| 动作空间  | 6D (增量位姿)            | 6D (增量位姿)          |

### 11.2 资产连接方式：Grasping → FixedJoint Chain

Factory 使用夹爪抓取，ORU 使用 PhysX FixedJoint 将资产固定连接：

```python
# oru_env.py:412-440 — 通过 FixedJoint 链连接 UR5 → Bridge → Sensor → Gripper → ORU
def _create_fixed_joints(stage, env_idx: int):
    ns = f"/World/envs/env_{env_idx}"

    # UR5 wrist_3_link → Bridge（桥接件）
    create_one_fixed_joint(stage, f"{ns}/Dofbot/wrist_3_link/bridge_joint",
        f"{ns}/Dofbot/wrist_3_link", f"{ns}/Bridge/base_link")

    # Bridge → SixForce（六维力传感器）
    create_one_fixed_joint(stage, f"{ns}/Bridge/base_link/force_joint",
        f"{ns}/Bridge/base_link", f"{ns}/SixForce/base_link",
        child_offset_pos=(0, 0, 0.062), child_offset_axis=(0, 1, 0),
        child_offset_angle=math.pi)

    # SixForce → Gripper（夹爪）
    create_one_fixed_joint(stage, f"{ns}/SixForce/base_link/gripper_joint",
        f"{ns}/SixForce/base_link", f"{ns}/Gripper/base_link",
        child_offset_pos=(0, 0, -0.0253), child_offset_axis=(0, 1, 0),
        child_offset_angle=math.pi)

    # Gripper → ORU（目标装配件）
    create_one_fixed_joint(stage, f"{ns}/Gripper/base_link/oru_joint",
        f"{ns}/Gripper/base_link", f"{ns}/ORU/base_link",
        child_offset_pos=(0, 0, -0.257), child_offset_axis=(0, 0, 1),
        child_offset_angle=math.pi)
```

> **关键差异**：使用 `replicate_physics=True`，FixedJoint 只在 env_0 上创建，PhysX 自动共享到所有虚拟环境。

### 11.3 场景资产差异

| 组件   | Factory          | ORU                            |
| ---- | ---------------- | ------------------------------ |
| 地面   | `GroundPlaneCfg` | `assets/USD/g1/g1.usd` (自定义表面) |
| 桌子   | SeattleLabTable  | 无                              |
| 力传感器 | 无                | 六维力传感器 (force.usd)             |
| 桥接件  | 无                | bridge.usd                     |

### 11.4 奖励设计简化

ORU 不需要夹爪相关的逻辑（不需 grasp/reset grasping），奖励采用虚拟 ORU 底面与地面顶面的关键点距离：

```python
# oru_env.py:288-330
def _get_rewards(self):
    # 通过 FK 计算 ORU 底面位置（而非从物理引擎读取）
    oru_bottom = self._virtual_oru_bottom()
    ground_top = self._ground_top()

    # 关键点距离计算
    kp_dist = torch.norm(kp_oru - kp_target, p=2, dim=-1).mean(-1)

    # 三阶段 squashing 奖励 + 动作惩罚 + 成功/engage 奖励
    rew = (
        squashing_fn(kp_dist, a0, b0)    # baseline (5, 4)
        + squashing_fn(kp_dist, a1, b1)  # coarse (50, 2)
        + squashing_fn(kp_dist, a2, b2)  # fine (100, 0)
    )
    rew -= 0.01 * torch.norm(actions, p=2)       # 动作幅度惩罚
    rew -= 0.001 * torch.norm(actions_diff, p=2)  # 动作变化惩罚
    rew += curr_success.float() + curr_engaged.float()
```

### 11.5 ORU 成功判定差异

```python
# oru_env.py:332-337 — 简化的判定条件
def _get_curr_successes(self, threshold):
    xy = torch.norm(ground_top[:, :2] - oru_bottom[:, :2], dim=-1)
    z = oru_bottom[:, 2] - ground_top[:, 2]
    return (xy < 0.005) & (z < 0.05 * threshold)
    #      XY 容差 5mm         Z 容差: 5% of ground height
```

### 11.6 观测空间差异

|      | Factory Actor             | ORU Actor                  |
| ---- | ------------------------- | -------------------------- |
| 位置特征 | fingertip_pos + rel_fixed | ee_pos_rel_ground          |
| 姿态特征 | fingertip_quat            | ee_quat                    |
| 速度特征 | ee_linvel, ee_angvel      | ee_linvel_fd, ee_angvel_fd |
| 关节   | 无 (Actor 不可见)             | joint_pos (6D)             |
| 维度   | 13 + 6(action) = 19       | 19 + 6(action) = 25        |

### 11.7 阻抗控制适配

ORU 的控制器新增了 **XY 方向加倍权重**，因为 ORU 装配需要在 XY 平面精确对齐：

```python
# oru_control.py:162-171
xy_weight = 2.0  # XY 方向 2x 权重
task_wrench[:, 0:2] = (
    xy_weight * task_prop_gains[:, 0:2] * lin_error[:, 0:2]
    + xy_weight * task_deriv_gains[:, 0:2] * (0.0 - ee_linvel[:, 0:2])
)
```

---

## 总结：Factory 装配 RL 的核心设计原则

| 设计维度     | 核心思想                                                         |
| -------- | ------------------------------------------------------------ |
| **控制模式** | 阻抗控制 (task-space PD) — 策略学习增量位姿，底层控制器转换为力矩                   |
| **动作平滑** | EMA 平滑 + 位置裁剪 + 姿态强制 (roll=π 朝下)                             |
| **观测设计** | 非对称 Actor-Critic — Actor 仅见末端传感器数据，Critic 见完整状态              |
| **奖励设计** | 三阶段多尺度关键点距离 + squashing function — 从粗到精的连续引导                 |
| **网络架构** | LSTM(2层, 1024) + MLP(512→128→64) — 记忆能力解决部分可观测               |
| **仿真保真** | 192 次位置迭代 + TGS 求解器 + 单 GPU 分区 — 确保亚毫米接触稳定                   |
| **域随机化** | 固定件位姿 + 手持件在夹爪中的偏移 + 观测噪声                                    |
| **训练规模** | 128 envs × 128 steps × 200 epochs，PPO + PopArt + adaptive LR |

---

*文档生成时间: 2026-07-03*
