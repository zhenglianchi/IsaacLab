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
    duration_s: float = 90.0  # unused; kept in sync with OruEnvCfg.episode_length_s

    # ── Fixed target pose (world frame) ────────────────────────────
    # Position [0.4, 0, 0] + Quaternion [0, 0, 1, 0] (wxyz, 180° around Y)
    # This matches the Ground config pos + rot. The policy always targets
    # this same pose — domain randomization is on the START, not the GOAL.
    #
    # EE control target: the impedance anchor aims here. The SUCCESS criterion is
    # measured on the ORU body instead (oru_seat_z below) - the ORU is the part
    # that enters the slot.
    target_pos: tuple = (0.4, 0.0, 0.4298)
    target_quat: tuple = (0.0, 0.0, 1.0, 0.0)
    # EE approach/control height (world). NOT the success reference.
    success_z: float = 0.4298

    # v2 pilot parameters, to be calibrated on validation runs before the suite.
    experiment_method: str = "full"  # full/single/no_path/no_stage/hard_switch/fixed
    preinsert_height: float = 0.08  # above the physical seat, NOT a new success height
    entry_xy_tolerance: float = 0.003
    entry_angle_tolerance: float = 0.035  # radians, approximately 2 degrees
    entry_height_tolerance: float = 0.005
    entry_confirm_steps: int = 3
    switch_duration_s: float = 0.3
    reference_speed: float = 0.02  # m/s, maximum axial reference motion
    insertion_bias: float = 0.01  # virtual equilibrium below seat, contact-gated
    fixed_stage_weight: float = 0.5  # no_stage ablation
    # Steady downward preload (N) applied once the ORU is geometrically seated and
    # the feedback gains are frozen. Zeroing the gains alone is not enough: the ORU
    # was only held in the seat by the controller's downward push, and with the
    # chain weightless nothing pulls it back down, so it floats up until the
    # contact penetration allowance is consumed. A constant force holds it seated
    # without position feedback, so it cannot pump the spring/contact limit cycle
    # that the bouncing came from.
    hold_force: float = 3.0
    torque_scale: float = 1.0  # no assumed inverse-decimation compensation
    joint_torque_limit: float = 100.0
    oru_seat_z: float = 0.03750  # = success_z 0.4298 - measured chain offset 0.39230
    # (2026-10-05: was the legacy 0.03742; re-derived from the EE height the
    #  penetration-calibrated run actually reaches, so the ORU criterion matches the
    #  commanded target. Measured: ee_z - oru_z is 0.39228-0.39230 over 14 runs, so
    #  the two references describe the same seat to 0.02 mm.)
    oru_seat_quat: tuple = (0.0, 1.0, 0.0, 0.0)
    seat_z_tolerance: float = 0.002
    seat_angle_tolerance: float = 0.035
    success_hold_s: float = 0.3
    success_speed_tolerance: float = 0.01
    success_angular_speed_tolerance: float = 0.05  # rad/s
    ik_iterations: int = 20
    ik_position_tolerance: float = 0.002
    ik_angle_tolerance: float = 0.0175

    # ── Domain randomization: IK noise (reset-time only) ───────────
    # At each reset, we:
    #   1. Set UR5 to default joints → read default EE pose via FK
    #   2. Add random offset (within these bounds) to that EE pose
    #   3. Use DLS IK to solve for joint angles that reach the perturbed pose
    #   4. Write those joints as the initial state
    # The target pose NEVER changes — policy learns to reach the same goal
    # from different starting configurations.
    ik_rand_pos_noise: tuple = (0.01, 0.01, 0.01)  # pilot: ±1cm; enlarge after validation
    ik_rand_rot_noise: tuple = (0.01745, 0.01745, 0.01745)  # pilot: ±1 degree per axis

    # ── Fixed IK offset for single-case evaluation ──────────────────
    # When set (not None), overrides random noise. Used by play_force.py.
    # Format: (dx, dy, dz) meters, (drx, dry, drz) radians
    fixed_ik_offset_pos: tuple | None = None
    fixed_ik_offset_rot: tuple | None = None

    # ── Global reward scale ────────────────────────────────────────
    # Applied INSIDE the environment, as the last step of _get_rewards. It has to
    # live here rather than in rl_games' reward_shaper: rl_games accumulates the
    # LOGGED/printed episode reward from the UNSHAPED reward
    # (a2c_common.py:782 `current_rewards += rewards` then :789
    #  `game_rewards.update(current_rewards[...])`), so a shaper-only change would
    # silently scale the training signal while still printing tens of thousands.
    # With the scale here, the logged reward, the printed 'saving next best
    # rewards' and the trained signal are all the same number.
    # Why 0.01: the completion bonus is paid every step the seat holds (40/step)
    # over an episode of up to 1350 steps, so an early success otherwise yields
    # O(3e4). It is a uniform factor, so the relative ordering of all reward
    # terms - and therefore the optimal policy - is unchanged.
    reward_scale: float = 0.01

    # ── Success thresholds (measured on the ORU body) ─────────────
    # The ORU is the part that enters the slot, so the seat test reads the ORU pose
    # (oru_env._oru_pose_errors). The EE reading is only 0.2-0.7 mm away laterally
    # and 0.02 mm vertically, but "EE in place" does not by itself prove the part
    # is: _ee_pose_errors() is kept for the diagnostic comparison.
    # XY centering on docking surface
    xy_tolerance: float = 0.002       # 2 mm
    # Z height fraction of ground-surface height for success
    success_threshold: float = 0.05   # 5 % of ground height — UNUSED (legacy): it is
                                      # passed to _get_curr_successes() but the body
                                      # uses seat_z_tolerance / xy_tolerance /
                                      # seat_angle_tolerance instead.
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
    # Reward units per meter of NEW best insertion depth (1mm -> +0.5).
    z_progress_weight: float = 500.0
    insertion_time_penalty: float = 2.5  # per control step before stable success
    insertion_xy_weight: float = 0.5    # cost at one XY tolerance
    insertion_angle_weight: float = 0.5 # cost at one angle tolerance
    force_smooth_weight: float = 0.005         # ΔF penalty
    force_peak_threshold: float = 60.0         # force safety limit (N) — aligned with
                                               # max_task_force_z so the required
                                               # >50N insertion force is not taxed
    force_peak_weight: float = 0.01            # squared penalty above limit
    lateral_force_weight: float = 0.1          # XY force penalty (anti-rubbing)
    z_force_target: float = -2.0               # world -Z is downward
    z_force_weight: float = 0.0                # disable uncalibrated commanded-force target
    # Actual ORU height error outside the SAME +/-2mm success band.
    z_depth_weight: float = 100.0  # 1cm residual gap -> -0.8 per control step

    # ── Stage 1 approach: distance-progress shaping ────────────────
    # r_approach = approach_weight * (d_{t-1} - d_t): rewards EVERY cm of
    # closing distance so the long middle of the descent carries a gradient.
    approach_weight: float = 5.0

    # ── Stage switch: contact detection + soft transition ──────────
    # Applied to the FILTERED ORU<->Ground contact force from the ContactSensor
    # (OruSceneCfg.Contact), which reads ~0 while airborne. Do NOT feed the wrist
    # reaction here: free-space chain inertia is 3.9-5.9 N and the seated bounce
    # 2.2-10.3 N, so a 0.5 N threshold "confirms contact" 82 mm above the seat.
    contact_force_threshold: float = 0.5       # contact force > this -> touching (N)
    contact_leave_threshold: float = 0.1       # hysteresis: must drop below to leave
    # UNUSED legacy knobs (kept only so old config dumps still load). The wrist-force
    # surge/sigmoid scheme they belong to was replaced by the contact sensor above;
    # nothing reads these - do not tune them expecting an effect.
    contact_height_threshold: float = 0.05
    contact_surge_delta: float = 0.2
    contact_surge_force: float = 0.5
    contact_sigmoid_slope: float = 10.0

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
