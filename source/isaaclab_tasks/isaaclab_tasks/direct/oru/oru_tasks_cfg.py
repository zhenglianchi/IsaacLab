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
    target_pos: tuple = (0.4, 0.0, 0.436)
    target_quat: tuple = (0.0, 0.0, 1.0, 0.0)

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
    xy_tolerance: float = 0.005       # 5 mm
    # Z height fraction of ground-surface height for success
    success_threshold: float = 0.05   # 5 % of ground height
    engage_threshold: float = 0.90    # 90 % of ground height → engaged

    # ── Reward: keypoint-based (same structure as Factory) ─────────
    num_keypoints: int = 4
    keypoint_scale: float = 0.08      # m — ±3cm keypoint spread

    # Three-stage squashing-function parameters  r(x)=1/(e^{ax}+b+e^{-ax})
    keypoint_coef_baseline: list = (5, 4)     # far → coarse approach
    keypoint_coef_coarse: list = (50, 2)      # medium → alignment
    keypoint_coef_fine: list = (100, 0)        # near → fine insertion

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
    default_task_deriv_gains: tuple = (20.0, 20.0, 20.0, 20.0, 20.0, 20.0)

    # ── Variable impedance: policy controls Kp + Kd (12D action) ───
    # Action[:6]  → Kp = base_Kp * (1 + a * gain_range)
    # Action[6:]  → Kd = base_Kd * (1 + a * gain_range)
    # Clamped to [5%, 500%] of base.
    gain_range: float = 2.0
