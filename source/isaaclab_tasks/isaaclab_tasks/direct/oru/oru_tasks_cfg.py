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

    # ── Success thresholds ─────────────────────────────────────────
    # XY centering on docking surface
    xy_tolerance: float = 0.005       # 5 mm
    # Z height fraction of ground-surface height for success
    success_threshold: float = 0.05   # 5 % of ground height
    engage_threshold: float = 0.90    # 90 % of ground height → engaged

    # ── Reward: keypoint-based (same structure as Factory) ─────────
    num_keypoints: int = 4
    keypoint_scale: float = 0.2       # m — scale of keypoint spread

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
    default_task_prop_gains: tuple = (100.0, 100.0, 100.0, 30.0, 30.0, 30.0)
    rot_deriv_scale: float = 10.0
