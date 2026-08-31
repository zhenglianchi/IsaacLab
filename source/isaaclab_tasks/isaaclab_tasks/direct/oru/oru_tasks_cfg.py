# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ORU Assembly: task configuration.

Defines ORU-specific physical parameters and reward shaping for the
UR5 insertion task (ORU already attached via fixed-joint chain).
"""

from isaaclab.utils import configclass


@configclass
class OruTaskCfg:
    """ORU assembly task parameters — physical properties and reward shaping."""

    # ── Task identity ──────────────────────────────────────────────
    name: str = "oru_assembly"
    duration_s: float = 15.0

    # ── Fixed target pose (world frame) ────────────────────────────
    # Position [0.4, 0, 0] + Quaternion [0, 0, 1, 0] (wxyz, 180° around Y)
    # This matches the Ground config pos + rot. The policy always targets
    # this same pose — domain randomization is on the START, not the GOAL.
    #
    # Z = 0.4297: EE height when the ORU is fully docked.
    # ORU seated Z = 0.03742 + chain offset -0.39230 → EE Z = 0.42972.
    target_pos: tuple = (0.4, 0.0, 0.4297)
    target_quat: tuple = (0.0, 0.0, 1.0, 0.0)
    # Success height (EE Z, world): ORU fully docked.
    success_z: float = 0.42972

    # ── Domain randomization: IK noise (reset-time only) ───────────
    # At each reset, we:
    #   1. Set UR5 to default joints → read default EE pose via FK
    #   2. Add random offset (within these bounds) to that EE pose
    #   3. Use DLS IK to solve for joint angles that reach the perturbed pose
    #   4. Write those joints as the initial state
    # The target pose NEVER changes — policy learns to reach the same goal
    # from different starting configurations.
    ik_rand_pos_noise: tuple = (0.12, 0.12, 0.12)          # ±12cm EE position noise
    ik_rand_rot_noise: tuple = (0.052, 0.052, 0.052)       # ±3° (0.052rad)

    # ── Fixed IK offset for single-case evaluation ──────────────────
    # When set (not None), overrides random noise. Used by play_force.py.
    # Format: (dx, dy, dz) meters, (drx, dry, drz) radians
    fixed_ik_offset_pos: tuple | None = None
    fixed_ik_offset_rot: tuple | None = None

    # ── Success thresholds ─────────────────────────────────────────
    # XY centering on docking surface
    xy_tolerance: float = 0.002       # 2 mm
    # Z height fraction of ground-surface height for success
    success_threshold: float = 0.05   # 5 % of ground height
    engage_threshold: float = 0.90    # 90 % of ground height → engaged
    # Completion bonus (per step while success holds). Must dominate the
    # per-step income, or the policy parks near the target instead of
    # finishing the insertion.
    success_reward: float = 40.0

    # ── Two-stage reward: stage 1 path keypoints (free space) ──────
    # N keypoints evenly spaced on the straight line start→target
    num_path_keypoints: int = 5
    # Multi-scale squashing coefficients (same structure as Factory)
    keypoint_coef_baseline: list = (5, 4)     # far → coarse approach
    keypoint_coef_coarse: list = (50, 2)      # medium → alignment
    keypoint_coef_fine: list = (100, 0)        # near → fine insertion
    # Target-point reward: extra steep squashing at the goal
    target_squash_a: float = 150.0
    # Pose alignment reward weight (quat dot with target quat)
    align_weight: float = 2.0
    # Alignment pays only when close to aligned: relu(dot − threshold)
    # scaled to [0, align_weight]. A flat per-step payment would reward
    # hovering over completing the task.
    align_threshold: float = 0.9
    # Straight-line deviation penalty weight (m → reward)
    deviation_weight: float = 2.0

    # ── Two-stage reward: stage 2 precision + force compliance ─────
    precision_a: float = 150.0                 # exp(-150·d): reward fires only within
                                               # the last cm of the insertion
    log_reward_coef: float = 0.8               # -coef*log(dist): steepest gradient at
                                               # dist→0 (∂r/∂d = -coef/d)
    z_progress_weight: float = 40.0            # reward downward motion
    force_smooth_weight: float = 0.005         # ΔF penalty
    force_peak_threshold: float = 60.0         # force safety limit (N) — aligned with
                                               # max_task_force_z so the required
                                               # >50N insertion force is not taxed
    force_peak_weight: float = 0.01            # squared penalty above limit
    lateral_force_weight: float = 0.1          # XY force penalty (anti-rubbing)
    z_force_target: float = 2.0                # desired downward force (N)
    z_force_weight: float = 0.001              # deviation penalty around target
    # Depth-gap pressure for the last cm (position-level, complements the
    # velocity-level z_progress): keeps a gradient when contact stalls the
    # EE — "close but parked" is worse than "pushing in".
    z_depth_weight: float = 50.0               # 1cm above tolerance → -0.5/step
    z_gap_tolerance: float = 0.005             # 5mm residual gap is fine (slack)

    # ── Stage 1 approach: potential-based distance shaping ─────────
    # r_approach = approach_weight * (d_{t-1} - d_t): rewards EVERY cm of
    # closing distance so the long middle of the descent carries a gradient.
    approach_weight: float = 5.0

    # ── Stage switch: contact detection + soft transition ──────────
    contact_force_threshold: float = 0.5       # F_mag > this → contact (N).
                                               # Above the free-space
                                               # chain-inertia band (>0.5N)
    contact_height_threshold: float = 0.05     # within 5cm of target height
    contact_surge_delta: float = 0.2           # force surge for first contact
    contact_surge_force: float = 0.5           # surge must exceed this
    contact_sigmoid_slope: float = 10.0        # soft-transition slope
    contact_leave_threshold: float = 0.1       # hysteresis: Fz must drop below to leave

    # ── Action penalties ───────────────────────────────────────────
    action_penalty_scale: float = 0.01         # L2 norm of action
    action_grad_penalty_scale: float = 0.001   # action change penalty

    # ── EE target bounds (relative to ground) ──────────────────────
    ee_pos_action_bounds: tuple = (0.05, 0.05, 0.05)    # ± m
    ee_rot_action_bounds: tuple = (0.3, 0.3, 0.3)       # ± rad

    # ── Default impedance gains ────────────────────────────────────
    # Kp: [X, Y, Z, Rx, Ry, Rz] — baseline proportional stiffness
    # Kd: critical damping 2*sqrt(Kp)
    default_task_prop_gains: tuple = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    default_task_deriv_gains: tuple = (40.0, 40.0, 40.0, 40.0, 40.0, 40.0)

    # ── Commanded wrench limits (anti-overshoot) ───────────────────
    # Clamps task_wrench in the controller. Free-space approach stays soft
    # (8N): at the wrist singularity m_eff≈0.2kg, 20N+ launches the EE at
    # ~100 m/s² and the path overshoots. Z is raised to max_task_force_z
    # by oru_env once contact is detected (stage 2).
    max_task_force: float = 8.0
    max_task_force_z: float = 60.0   # stage-2 Z cap: the last cm of seating
                                     # needs >50N down-force
    max_task_torque: float = 6.0     # Nm, per rotational axis

    # ── Chain gravity + compensation (force1 parity) ────────────────
    # RL default: weightless chain (disable_gravity=True everywhere).
    # Diagnostics may enable gravity on the chain + joint-space gravity
    # compensation (oru_control gravity_comp) to tension the chain.
    enable_chain_gravity: bool = False
    gravity_comp_enable: bool = True

    # ── FixedJoint chain stiffness (PhysX joint drive) ────────────────
    # Optional stiff joint drive (spring-damper on the locked DOFs) to
    # resist FixedJoint yield under dynamic loads (chain whip).
    # None = no drive (default PhysX behavior).
    joint_drive_stiffness: float | None = None
    joint_drive_damping: float | None = None

    # ── Variable impedance: policy controls Kp + Kd (12D action) ───
    # Action[:6]  → Kp = base_Kp * (1 + a * gain_range)
    # Action[6:]  → Kd = base_Kd * (1 + a * gain_range)
    # Clamped to [5%, 500%] of base.
    gain_range: float = 2.0
