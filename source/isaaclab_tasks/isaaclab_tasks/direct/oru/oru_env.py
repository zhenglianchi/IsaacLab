# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ORU Assembly environment — UR5 + impedance control for insertion.

Scene: UR5(Dofbot) → Bridge → SixForce → Gripper → ORU  +  Ground
clone_in_fabric=False → each env is independent (own USD + physics).

Domain randomization (reset-time):
  1. Set UR5 to default joints → read default EE pose via FK
  2. Add random offset (pos + rot) to that EE pose
  3. DLS IK → joint angles for perturbed EE pose
  4. Write those joints as the initial state
  → Target pose is FIXED ([0.4, 0, 0, 0, 0, 1, 0])
  → Policy learns to reach the same target from randomized starting configs
"""

from __future__ import annotations

import math
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import axis_angle_from_quat

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply

from . import oru_control, oru_utils, oru_logic
from .oru_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, OruEnvCfg


class OruEnv(DirectRLEnv):
    """ORU Assembly — UR5 inserts pre-grasped ORU onto docking surface."""

    cfg: OruEnvCfg

    # ==================================================================
    # Init
    # ==================================================================

    def __init__(self, cfg: OruEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = sum(OBS_DIM_CFG[k] for k in cfg.obs_order) + cfg.action_space
        cfg.state_space = sum(STATE_DIM_CFG[k] for k in cfg.state_order) + cfg.action_space

        super().__init__(cfg, render_mode, **kwargs)
        # super().__init__ → InteractiveScene(clone) → _setup_scene(FixedJoints)
        self._init_tensors()

    # ==================================================================
    # Setup Scene — FixedJoints on env_0 (shared via replicate_physics)
    # ==================================================================

    def _setup_scene(self):
        """Assets already loaded by InteractiveScene(cfg.scene).

        clone_in_fabric=False means each environment is a fully independent
        USD branch — no fabric sharing, no physics replication. FixedJoints
        MUST be created on EVERY environment before sim.play().
        """
        # Grab asset handles (loaded by InteractiveScene from OruSceneCfg)
        self.robot: Articulation = self.scene["Dofbot"]
        self.bridge: RigidObject = self.scene["Bridge"]
        self.force_sensor: RigidObject = self.scene["SixForce"]
        self.gripper: RigidObject = self.scene["Gripper"]
        self.oru: RigidObject = self.scene["ORU"]
        self.ground: RigidObject = self.scene["Ground"]
        # filtered ORU<->Ground contact force (see OruSceneCfg.Contact)
        self.contact_sensor: ContactSensor = self.scene["Contact"]

        # FixedJoints on ALL envs — each env is independent (clone_in_fabric=False)
        stage = sim_utils.SimulationContext.instance().stage
        for env_idx in range(self.scene.cfg.num_envs):
            _create_fixed_joints(
                stage, env_idx,
                drive_stiffness=self.cfg.task.joint_drive_stiffness,
                drive_damping=self.cfg.task.joint_drive_damping,
            )

    # ==================================================================
    # Tensors
    # ==================================================================

    def _init_tensors(self):
        self.ee_frame_name = "wrist_3_link"
        self.arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self._ee_frame_idx: int | None = None
        self._arm_joint_ids: torch.Tensor | None = None

        N = self.num_envs
        self.prev_ee_pos = torch.zeros((N, 3), device=self.device)
        self.prev_ee_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).unsqueeze(0).repeat(N, 1)
        self.prev_joint_pos = torch.zeros((N, 6), device=self.device)

        self.actions = torch.zeros((N, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        prop = self.cfg.task.default_task_prop_gains
        deriv = self.cfg.task.default_task_deriv_gains
        self.base_gains = torch.tensor(prop, device=self.device).repeat(N, 1)
        self.base_deriv = torch.tensor(deriv, device=self.device).repeat(N, 1)
        self.task_prop_gains = self.base_gains.clone()
        self.task_deriv_gains = self.base_deriv.clone()
        self.gain_range = self.cfg.task.gain_range

        # Kept for state dict compatibility (not used in _apply_action)
        self.pos_threshold = torch.zeros((N, 3), device=self.device)
        self.rot_threshold = torch.zeros((N, 3), device=self.device)

        # IK controller — used only at reset time for domain randomization
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", ik_method="dls", use_relative_mode=False,
        )
        self._ik = DifferentialIKController(ik_cfg, N, device=self.device)

        # Fixed target: XY follows ground, Z=0.4, quat=[0,0,1,0]
        tq = self.cfg.task.target_quat
        self.fixed_target_quat = torch.tensor(tq, device=self.device).repeat(N, 1)
        self.fixed_target_z = self.cfg.task.target_pos[2]  # 0.4

        self.ep_succeeded = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.ep_success_times = torch.zeros(N, dtype=torch.long, device=self.device)
        self.last_update_timestamp = 0.0

        # Two-stage reward state
        self.applied_wrench = torch.zeros((N, 6), device=self.device)
        self.prev_F_mag = torch.zeros(N, device=self.device)  # measured force magnitude (last step)
        self._was_in_contact = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.ep_ee_start = torch.zeros((N, 3), device=self.device)  # episode start pos (path keypoints base)
        self.prev_reward_z = torch.zeros(N, device=self.device)
        self.prev_dist_target = torch.zeros(N, device=self.device)  # last-step dist to target (approach shaping)
        self._insertion_phase = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._entry_count = torch.zeros(N, dtype=torch.long, device=self.device)
        self._contact_count = torch.zeros_like(self._entry_count)
        self._success_count = torch.zeros_like(self._entry_count)
        self._stage_alpha = torch.zeros(N, device=self.device)
        self._contact_alpha = torch.zeros(N, device=self.device)
        self._control_z = torch.full((N,), self.fixed_target_z + self.cfg.task.preinsert_height, device=self.device)
        self._stable_success = torch.zeros(N, dtype=torch.bool, device=self.device)
        # Geometry-only seat latch (no velocity gate). Drives the controller freeze:
        # freezing on the full seat test is circular, because the bounce that the
        # control force causes is exactly what keeps the velocity gates unsatisfied.
        self._seat_geom_count = torch.zeros_like(self._entry_count)
        self._seat_latched = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._can_advance = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._best_insertion_gap = torch.full((N,), self.cfg.task.preinsert_height, device=self.device)
        self._last_logic_step = -1
        if self.cfg.task.experiment_method not in {"full", "single", "no_path", "no_stage", "hard_switch", "fixed"}:
            raise ValueError(f"Unknown ORU method: {self.cfg.task.experiment_method}")

    # ==================================================================
    # Frame indices (lazy)
    # ==================================================================

    def _ensure_frame_indices(self):
        if self._ee_frame_idx is None:
            self._ee_frame_idx = self.robot.find_bodies(self.ee_frame_name)[0][0]
            self._arm_joint_ids = self.robot.find_joints(self.arm_joint_names)[0]

    # ==================================================================
    # Intermediate values
    # ==================================================================

    def _compute_intermediate_values(self, dt: float):
        self._ensure_frame_indices()

        self.ee_pos = self.robot.data.body_pos_w[:, self._ee_frame_idx]
        self.ee_quat = self.robot.data.body_quat_w[:, self._ee_frame_idx]

        self.joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        self.joint_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]

        jac_idx = self._ee_frame_idx - 1
        self.jacobian = self.robot.root_physx_view.get_jacobians()[
            :, jac_idx, :, self._arm_joint_ids
        ]

        # EE-ORIGIN twist via the Jacobian: J @ q_dot — the correct EE
        # velocity (body_lin_vel_w is the wrist COM velocity, wrong near the
        # singularity). Used by both the controller damping and the policy.
        self.ee_twist = (self.jacobian @ self.joint_vel.unsqueeze(-1)).squeeze(-1)
        self.ee_linvel = self.ee_twist[:, :3]
        self.ee_angvel = self.ee_twist[:, 3:6]

        # Finite-difference velocities (kept for state-dict compatibility;
        # the Jacobian twist above is the live source)
        pos_diff = self.ee_pos - self.prev_ee_pos
        self.ee_linvel_fd = pos_diff / dt
        self.prev_ee_pos = self.ee_pos.clone()

        rot_diff = torch_utils.quat_mul(
            self.ee_quat, torch_utils.quat_conjugate(self.prev_ee_quat)
        )
        rot_diff *= torch.sign(rot_diff[:, 0]).unsqueeze(-1)
        self.ee_angvel_fd = axis_angle_from_quat(rot_diff) / dt
        self.prev_ee_quat = self.ee_quat.clone()

        joint_diff = self.joint_pos - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos.clone()

        self.last_update_timestamp = self.robot._data._sim_timestamp

    # ==================================================================
    # Pre-physics — EMA smoothing
    # ==================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        # Reset exactly once in _reset_idx, preserving the initial distance baseline.
        actions = actions.clamp(-1.0, 1.0)
        if self.cfg.task.experiment_method == "fixed":
            actions = torch.zeros_like(actions)
        self.actions = (
            self.cfg.ema_factor * actions.clone().to(self.device)
            + (1.0 - self.cfg.ema_factor) * self.actions
        )

    # ==================================================================
    # Apply action
    # ==================================================================

    def _apply_action(self):
        dt = self.physics_dt
        # Refresh controller inputs from the LATEST physics state every
        # substep. _compute_intermediate_values is timestamp-gated and only
        # re-fires once per env step (not per substep) — feeding its stale
        # velocity to the Kd term aliases the wrench at step rate.
        self._ensure_frame_indices()
        self.ee_pos = self.robot.data.body_pos_w[:, self._ee_frame_idx]
        self.ee_quat = self.robot.data.body_quat_w[:, self._ee_frame_idx]
        self.joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        self.joint_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]
        jac_idx = self._ee_frame_idx - 1
        self.jacobian = self.robot.root_physx_view.get_jacobians()[
            :, jac_idx, :, self._arm_joint_ids
        ]
        self.ee_twist = (self.jacobian @ self.joint_vel.unsqueeze(-1)).squeeze(-1)
        self.ee_linvel = self.ee_twist[:, :3]
        self.ee_angvel = self.ee_twist[:, 3:6]

        # ── Actions: [:6]=Kp scale, [6:]=Kd scale ──
        scale_kp = 1.0 + self.actions[:, 0:6] * self.gain_range
        scale_kp = torch.clamp(scale_kp, min=0.05, max=5.0)
        scale_kd = 1.0 + self.actions[:, 6:12] * self.gain_range
        scale_kd = torch.clamp(scale_kd, min=0.05, max=5.0)

        self.task_prop_gains = self.base_gains * scale_kp
        self.task_deriv_gains = self.base_deriv * scale_kd

        # ── Freeze the feedback, keep a steady preload, once seated ────
        # Any residual position feedback keeps pushing the ORU off the seat, so it
        # bounces and the hold requirement (speed < 0.01 m/s, angular speed <
        # 0.05 rad/s) never stays satisfied - measured: the seat was reached (gap
        # +0.18 mm, tilt 1.2 deg) yet the candidate streak peaked at 3 of the
        # required 5, with speed 0.015 m/s and angular speed 0.65 rad/s. Waiting for
        # the full seat test would be circular, so the freeze triggers on the
        # geometry-only latch (xy / depth / tilt inside tolerance for
        # entry_confirm_steps).
        # Zeroing BOTH gains alone is also wrong: the ORU was only held down by the
        # controller's push and the chain is weightless, so with no force at all it
        # floats up (measured). Instead the feedback is zeroed and a CONSTANT
        # downward task force (hold_force) is applied, which holds the part in the
        # seat without any position feedback to pump the limit cycle.
        # Not latched permanently: if the ORU leaves the geometric seat the counter
        # resets and normal control resumes by itself.
        task_force_ff = None
        if torch.any(self._seat_latched):
            frozen = self._seat_latched.unsqueeze(-1)
            self.task_prop_gains = torch.where(
                frozen, torch.zeros_like(self.task_prop_gains), self.task_prop_gains
            )
            self.task_deriv_gains = torch.where(
                frozen, torch.zeros_like(self.task_deriv_gains), self.task_deriv_gains
            )
            task_force_ff = torch.zeros((self.num_envs, 6), device=self.device)
            task_force_ff[:, 2] = torch.where(
                self._seat_latched,
                torch.full_like(self._seat_latched, -float(self.cfg.task.hold_force), dtype=torch.float),
                torch.zeros_like(self._seat_latched, dtype=torch.float),
            )

        # ── Target: XY = ground XY, Z = 0.4, quat = [0,0,1,0] ──
        ground_pos = self.ground.data.root_pos_w
        ctrl_target_ee_pos = ground_pos.clone()
        task = self.cfg.task
        self._control_z = oru_logic.axial_reference(
            self._control_z, self.ee_pos[:, 2], self._insertion_phase, self._can_advance,
            self._contact_alpha, seat_z=self.fixed_target_z, cfg=task, dt=dt,
        )
        ctrl_target_ee_pos[:, 2] = self._control_z
        ctrl_target_ee_quat = self.fixed_target_quat

        # ── Chain gravity compensation (force1 parity) ──
        # With cfg.task.enable_chain_gravity, the chain bodies + ORU feel
        # gravity and hang from the wrist. Compensate the weight so the
        # wrist can hover (and the wrench clamp is not consumed by the
        # load): F_comp = (0,0,+m_total*g) in world, tau = Jᵀ F_comp.
        gravity_comp = None
        if self.cfg.task.enable_chain_gravity and self.cfg.task.gravity_comp_enable:
            m_total = (
                self.bridge.data.default_mass
                + self.force_sensor.data.default_mass
                + self.gripper.data.default_mass
                + self.oru.data.default_mass
            )  # (num_envs, 1)
            f_comp = torch.zeros((self.num_envs, 6), device=self.device)
            f_comp[:, 2] = m_total.squeeze(-1) * self.cfg.sim.gravity[2] * -1.0
            jac_T = torch.transpose(self.jacobian, dim0=1, dim1=2)
            gravity_comp = (jac_T @ f_comp.unsqueeze(-1)).squeeze(-1)

        ee_linvel_now = self.ee_linvel
        ee_angvel_now = self.ee_angvel

        # Contact-confirmed force cap ramps for every method, including baselines.
        task = self.cfg.task
        z_force_limit = task.max_task_force + (task.max_task_force_z - task.max_task_force) * self._contact_alpha

        joint_torque, self.applied_wrench = oru_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            ee_pos=self.ee_pos,
            ee_quat=self.ee_quat,
            ee_linvel=ee_linvel_now,
            ee_angvel=ee_angvel_now,
            jacobian=self.jacobian,
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            gravity_comp=gravity_comp,
            z_force_limit=z_force_limit,
            task_force_ff=task_force_ff,
        )

        self.robot.set_joint_effort_target(joint_torque, joint_ids=self._arm_joint_ids)

    # ==================================================================
    # Observations
    # ==================================================================

    def _get_observations(self) -> dict:
        self._compute_intermediate_values(self.physics_dt)

        ground_pos = self.ground.data.root_pos_w
        ground_quat = self.ground.data.root_quat_w

        # EE quaternion relative to ground: ground_q⁻¹ * ee_q
        ground_quat_inv = torch_utils.quat_conjugate(ground_quat)
        ee_quat_rel_ground = torch_utils.quat_mul(ground_quat_inv, self.ee_quat)

        obs_dict = {
            "ee_pos_rel_ground": self.ee_pos - ground_pos,
            "ee_quat": ee_quat_rel_ground,
            "ee_linvel": self.ee_linvel,      # Jacobian twist: true EE-origin velocity
            "ee_angvel": self.ee_angvel,
            "joint_pos": self.joint_pos,
            "task_prop_gains": self.task_prop_gains,
            "task_deriv_gains": self.task_deriv_gains,
            "applied_wrench": self.applied_wrench,
            # TRUE wrist reaction force — applied_wrench is the commanded PD
            # output, not a contact signal.
            "measured_force": self.robot.data.body_incoming_joint_wrench_b[
                :, self._ee_frame_idx, :3
            ],
            "stage_state": torch.stack(
                (self._insertion_phase.float(), self._stage_alpha, self._contact_alpha,
                 self._control_z - self.fixed_target_z, self._best_insertion_gap), dim=-1
            ),
        }
        state_dict = {
            **obs_dict,
            "ground_pos": ground_pos,
            "ground_quat": ground_quat,
            "task_prop_gains": self.task_prop_gains,
            "task_deriv_gains": self.task_deriv_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
        }

        policy_obs = oru_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order)
        policy_obs = torch.cat([policy_obs, self.actions], dim=-1)

        critic_obs = oru_utils.collapse_obs_dict(state_dict, self.cfg.state_order)
        critic_obs = torch.cat([critic_obs, self.actions], dim=-1)

        return {"policy": policy_obs, "critic": critic_obs}

    # ==================================================================
    # Rewards — keypoints from EE to fixed target
    # ==================================================================

    def _get_target_ref(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed target: XY=ground XY, Z=0.4, quat=[0,0,1,0]."""
        pos = self.ground.data.root_pos_w.clone()
        pos[:, 2] = self.fixed_target_z
        return pos, self.fixed_target_quat

    def _get_measured_force_mag(self) -> torch.Tensor:
        """Measured EE joint reaction force magnitude (frame-invariant).

        body_incoming_joint_wrench is the real transmitted force: ~0 in
        free space, rises on contact (applied_wrench is the commanded
        task-space force — wrong for contact detection).
        """
        self._ensure_frame_indices()
        F_meas = self.robot.data.body_incoming_joint_wrench_b[:, self._ee_frame_idx, :3]
        return torch.norm(F_meas, dim=-1)

    def _get_contact_force_vec(self) -> torch.Tensor:
        """Filtered ORU<->Ground contact force vector of the strongest reported pair.

        Shape (N, 3), world frame. This is the pair's real contact force, not the
        wrist reaction: it is ~0 in free space (where the wrist channel carries
        3.9-5.9 N of chain inertia) and only rises when the ORU touches the docking
        surface. The vector form lets diagnostics check the contact *direction*.
        """
        f = self.contact_sensor.data.force_matrix_w
        if f is None or f.numel() == 0:
            f = self.contact_sensor.data.net_forces_w
        if f is None:
            return torch.zeros((self.num_envs, 3), device=self.device)
        flat = f.reshape(self.num_envs, -1, 3)
        idx = torch.linalg.vector_norm(flat, dim=-1).argmax(dim=-1)
        return flat[torch.arange(self.num_envs, device=self.device), idx]

    def _get_contact_force_mag(self) -> torch.Tensor:
        """Filtered ORU<->Ground contact force magnitude from the ContactSensor."""
        return torch.linalg.vector_norm(self._get_contact_force_vec(), dim=-1)

    def _update_stage_state(self):
        """Once per policy step: guidance by geometry, contact by real force.

        The wrist reaction is NOT used for contact: it carries inertial loads that
        overlap the contact band (measured 3.9-5.9 N airborne vs 2-10 N seated), so
        the old gate confirmed contact while the ORU was still 82 mm up.
        """
        if self._last_logic_step == self.common_step_counter:
            return
        self._last_logic_step = self.common_step_counter
        task = self.cfg.task
        # Task geometry is read from the EE frame (see _ee_pose_errors); the ORU is
        # rigidly linked to it, so this matches the previous ORU-based gates to
        # within 0.02 mm vertically and 0.2-0.7 mm laterally.
        xy, gap, angle = self._ee_pose_errors()
        (self._insertion_phase, self._entry_count, self._was_in_contact,
         self._contact_count, self._stage_alpha, self._contact_alpha,
         self._can_advance) = oru_logic.update_phase(
            self._insertion_phase, self._entry_count, self._was_in_contact,
            self._contact_count, self._stage_alpha, self._contact_alpha,
            xy, angle, gap, self._get_contact_force_mag(), cfg=task, dt=self.step_dt,
        )

        # Geometry-only seat latch, held for entry_confirm_steps. This is what lets
        # _apply_action stop controlling before the velocity gates can be met.
        geom = oru_logic.seat_geometry_ok(xy, gap, angle, cfg=task)
        self._seat_geom_count = torch.where(geom, self._seat_geom_count + 1, torch.zeros_like(self._seat_geom_count))
        self._seat_latched = self._seat_geom_count >= task.entry_confirm_steps

    def _get_contact_degree(self) -> torch.Tensor:
        # Historical name retained for tools; this now means reward phase blend.
        return self._stage_alpha

    def _alignment_reward(self):
        dot = (self.ee_quat * self.fixed_target_quat).sum(-1).abs().clamp(0, 1)
        return self.cfg.task.align_weight * torch.exp(-((2 * torch.acos(dot)) / 0.1) ** 2)

    def _get_path_reward(self, target_ref_pos: torch.Tensor, dist_target: torch.Tensor) -> torch.Tensor:
        """Stage 1: continuous projection progress + keypoint corridor deviation.

        Progress t ∈ [0,1] is the EE projection along the start→target line —
        a continuous number, so closely-spaced keypoints cannot cause
        double-trigger issues. Distance-progress shaping on 3D
        distance carries a gradient through the long middle of the descent.
        """
        task = self.cfg.task

        path_end = target_ref_pos.clone()
        path_end[:, 2] += task.preinsert_height
        line = path_end - self.ep_ee_start
        line_len2 = (line**2).sum(-1, keepdim=True).clamp(min=1e-8)

        # 1. Continuous projection progress
        t = torch.sum((self.ee_pos - self.ep_ee_start) * line, dim=-1, keepdim=True) / line_len2
        t = t.clamp(0.0, 1.0).squeeze(-1)
        dist_to_goal = 1.0 - t  # remaining fraction of path

        a0, b0 = task.keypoint_coef_baseline
        a1, b1 = task.keypoint_coef_coarse
        a2, b2 = task.keypoint_coef_fine
        r_progress = (
            oru_utils.squashing_fn(dist_to_goal, a0, b0)
            + oru_utils.squashing_fn(dist_to_goal, a1, b1)
            + oru_utils.squashing_fn(dist_to_goal, a2, b2)
        )

        # 2. Perpendicular deviation to the straight line (zero ON the line
        #    everywhere, including the goal — no centroid pull-back)
        closest = self.ep_ee_start + t.unsqueeze(-1) * line  # t clamped [0,1]
        deviation = torch.norm(self.ee_pos - closest, dim=-1)
        r_deviation = -task.deviation_weight * deviation

        # 3. Terminal target reward: steep squashing at the goal
        r_target = oru_utils.squashing_fn(dist_target, task.target_squash_a, 0.0)

        # 4. Distance progress shaping (no policy-invariance claim).
        r_approach = task.approach_weight * (self.prev_dist_target - dist_target)

        if task.experiment_method in {"no_path", "single"}:
            r_progress = torch.zeros_like(r_progress)
            r_deviation = torch.zeros_like(r_deviation)
        return r_progress + r_deviation + r_target + r_approach

    def _get_insertion_reward(self, target_ref_pos: torch.Tensor) -> torch.Tensor:
        """Actual depth progress, residual pose cost, and contact compliance."""
        task = self.cfg.task
        F_mag = self._get_measured_force_mag()

        # Reward the actual task geometry (EE frame, same source as the seat test),
        # never the virtual preload target.
        xy, gap, angle = self._ee_pose_errors()
        remaining = gap.clamp(min=0)
        aligned = (xy < task.entry_xy_tolerance) & (angle < task.entry_angle_tolerance)
        valid_progress = aligned & (gap >= -task.seat_z_tolerance)
        # Only a new best depth earns progress: retracting and reinserting to
        # the same depth cannot earn it again, even across stage transitions.
        improvement = (self._best_insertion_gap - remaining).clamp(min=0, max=0.01)
        r_z_progress = task.z_progress_weight * improvement * valid_progress.float()
        self._best_insertion_gap = torch.where(
            valid_progress, torch.minimum(self._best_insertion_gap, remaining), self._best_insertion_gap
        )
        # No 5mm reward dead band: use exactly the physical success tolerance.
        # Penalize excessive depth too, rather than rewarding penetration.
        r_z_depth = -task.z_depth_weight * torch.relu(gap.abs() - task.seat_z_tolerance)
        r_xy = -task.insertion_xy_weight * xy / task.xy_tolerance
        r_angle = -task.insertion_angle_weight * angle / task.seat_angle_tolerance
        r_unfinished = -task.insertion_time_penalty * (~self._stable_success).float()

        # Force smoothness: penalize abrupt changes in measured force
        r_force_smooth = -task.force_smooth_weight * (F_mag - self.prev_F_mag).abs()

        # Lateral wrist reaction in world coordinates; not isolated contact force.
        F = quat_apply(self.ee_quat, self.robot.data.body_incoming_joint_wrench_b[:, self._ee_frame_idx, :3])
        F_xy = torch.norm(F[:, :2], dim=-1)
        r_lateral = -task.lateral_force_weight * F_xy

        # Z force target: keep moderate downward commanded force
        r_z_force = -task.z_force_weight * torch.abs(self.applied_wrench[:, 2] - task.z_force_target)

        return (
            r_z_progress + r_z_depth + r_xy + r_angle + r_unfinished
            + r_force_smooth + r_lateral + r_z_force
        )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values(self.physics_dt)

        target_ref_pos, _ = self._get_target_ref()
        dist_target = torch.norm(self.ee_pos - target_ref_pos, dim=-1)

        # ── Two-stage reward with rate-limited soft transition ───────
        r_stage1 = self._get_path_reward(target_ref_pos, dist_target)
        r_stage2 = self._get_insertion_reward(target_ref_pos)
        contact_degree = self._get_contact_degree()  # [0,1] soft switch
        # Exposed for diagnostics (tools/diagnose_oru_v2.py -> tools/check_stage_switch.py):
        # the stage-2 term must be exactly zero until real contact, otherwise the
        # insertion costs are taxed while the part is still airborne.
        self.stage1_reward = r_stage1
        self.stage2_reward = r_stage2
        self.stage_blend = contact_degree

        method = self.cfg.task.experiment_method
        if method == "single":
            rew = r_stage1
        elif method == "no_stage":
            weight = self.cfg.task.fixed_stage_weight
            rew = (1.0 - weight) * r_stage1 + weight * r_stage2
        else:
            if method == "hard_switch":
                contact_degree = self._insertion_phase.float()
            rew = (1.0 - contact_degree) * r_stage1 + contact_degree * r_stage2
        rew += self._alignment_reward()  # common term, paid exactly once

        task = self.cfg.task
        rew -= task.force_peak_weight * torch.relu(self._get_measured_force_mag() - task.force_peak_threshold) ** 2

        # Action penalties (all methods)
        rew -= self.cfg.task.action_penalty_scale * torch.norm(self.actions, p=2, dim=-1)
        rew -= self.cfg.task.action_grad_penalty_scale * torch.norm(
            self.actions - self.prev_actions, p=2, dim=-1
        )
        curr_s = self._stable_success
        # Completion bonus — must dominate the per-step income or the policy
        # settles for "hover near target" (success +1 vs align +1.8/step was
        # net-negative to finish). Paid every step success holds, so keeping
        # the seat is also rewarded.
        rew += curr_s.float() * self.cfg.task.success_reward

        # Logging + state update
        self.prev_actions = self.actions.clone()
        self.prev_F_mag = self._get_measured_force_mag().clone()
        self.prev_dist_target = dist_target.clone()
        self.prev_reward_z = self.ee_pos[:, 2].clone()
        if torch.any(self.reset_buf):
            self.extras["log"] = {"episode_success_rate": self.ep_succeeded[self.reset_buf].float().mean()}
        self.extras["rew_pos_error"] = torch.norm(self.ee_pos - target_ref_pos, dim=-1).mean()
        self.extras["rew_contact_degree"] = contact_degree.mean()
        return rew

    def _ee_pose_errors(self):
        """Task geometry measured on the EE frame (the criterion used everywhere).

        The ORU is bolted to the EE through the fixed-joint chain, and that link is
        effectively rigid here: measured over 14 runs, ``ee_z - oru_z`` is
        0.39228-0.39230 m (0.02 mm spread) and the lateral EE/ORU offset at the seat
        is 0.30-0.90 mm vs 0.07-0.44 mm for the ORU (so the EE criterion is
        0.2-0.7 mm stricter laterally, i.e. ~1/3 of the 2 mm tolerance); the
        implied EE/ORU angular difference over the 392 mm link is <= 0.001 rad
        against a 0.035 rad tolerance. Hence the seat test is expressed on the EE,
        which is also the pose the policy actually commands.

        Returns (xy vs docking surface, z gap vs EE seat height, tilt vs target quat).
        """
        task = self.cfg.task
        target = self.ground.data.root_pos_w
        xy = torch.linalg.vector_norm(self.ee_pos[:, :2] - target[:, :2], dim=-1)
        gap = self.ee_pos[:, 2] - self.scene.env_origins[:, 2] - task.success_z
        quat = torch.tensor(task.target_quat, device=self.device)
        angle = 2 * torch.acos((self.ee_quat * quat).sum(-1).abs().clamp(0, 1))
        return xy, gap, angle

    def _oru_pose_errors(self):
        """Same errors measured on the ORU body; kept for diagnostics and for the
        geometric ground truth (see tools/check_stage_switch.py)."""
        task = self.cfg.task
        pos = self.oru.data.root_pos_w
        target = self.ground.data.root_pos_w
        xy = torch.linalg.vector_norm(pos[:, :2] - target[:, :2], dim=-1)
        gap = pos[:, 2] - self.scene.env_origins[:, 2] - task.oru_seat_z
        quat = torch.tensor(task.oru_seat_quat, device=self.device)
        angle = 2 * torch.acos((self.oru.data.root_quat_w * quat).sum(-1).abs().clamp(0, 1))
        return xy, gap, angle

    def _get_curr_successes(self, threshold: float) -> torch.Tensor:
        """EE seat pose (see _ee_pose_errors), bounded depth, orientation and low velocity."""
        xy, gap, angle = self._ee_pose_errors()
        speed = torch.linalg.vector_norm(self.ee_linvel, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.ee_angvel, dim=-1)
        return oru_logic.seat_candidate(xy, gap, angle, speed, angular_speed, cfg=self.cfg.task)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values(self.physics_dt)
        self._update_stage_state()
        candidate = self._get_curr_successes(self.cfg.task.success_threshold)
        self._success_count = torch.where(candidate, self._success_count + 1, 0)
        hold_steps = math.ceil(self.cfg.task.success_hold_s / self.step_dt)
        self._stable_success = self._success_count >= hold_steps
        first = self._stable_success & ~self.ep_succeeded
        self.ep_success_times[first] = self.episode_length_buf[first]
        self.ep_succeeded |= self._stable_success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    # ==================================================================
    # Reset
    # ==================================================================

    def _reset_buffers(self, env_ids: torch.Tensor):
        self.ep_succeeded[env_ids] = False
        self.ep_success_times[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_F_mag[env_ids] = 0.0
        self._was_in_contact[env_ids] = False
        self.prev_dist_target[env_ids] = 0.0
        self._best_insertion_gap[env_ids] = self.cfg.task.preinsert_height
        self._insertion_phase[env_ids] = False
        self._can_advance[env_ids] = False
        self._entry_count[env_ids] = 0
        self._contact_count[env_ids] = 0
        self._success_count[env_ids] = 0
        self._seat_geom_count[env_ids] = 0
        self._seat_latched[env_ids] = False
        self._stable_success[env_ids] = False
        self._stage_alpha[env_ids] = 0
        self._contact_alpha[env_ids] = 0
        self._control_z[env_ids] = self.fixed_target_z + self.cfg.task.preinsert_height
        self.applied_wrench[env_ids] = 0
        self.task_prop_gains[env_ids] = self.base_gains[env_ids]
        self.task_deriv_gains[env_ids] = self.base_deriv[env_ids]

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._reset_buffers(env_ids)
        n = len(env_ids)
        if n == 0:
            return
        self._ensure_frame_indices()
        task = self.cfg.task
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=env_ids)
        joints = self.robot.data.default_joint_pos[env_ids].clone()
        zero_vel = torch.zeros_like(joints)
        self.robot.write_joint_state_to_sim(joints, zero_vel, env_ids=env_ids)
        self.robot.set_joint_effort_target(zero_vel, env_ids=env_ids)
        # FK only: never step the entire simulator while resetting a subset.
        self.sim.forward()
        home = self.robot.data.body_pose_w[env_ids, self._ee_frame_idx].clone()
        if task.fixed_ik_offset_pos is not None:
            pos_noise = torch.tensor(task.fixed_ik_offset_pos, device=self.device).expand(n, -1)
            rot_noise = torch.tensor(task.fixed_ik_offset_rot or (0., 0., 0.), device=self.device).expand(n, -1)
        else:
            pos_noise = (2 * torch.rand(n, 3, device=self.device) - 1) * torch.tensor(task.ik_rand_pos_noise, device=self.device)
            rot_noise = (2 * torch.rand(n, 3, device=self.device) - 1) * torch.tensor(task.ik_rand_rot_noise, device=self.device)
        target_pos = home[:, :3] + pos_noise
        dq = torch_utils.quat_from_euler_xyz(rot_noise[:, 0], rot_noise[:, 1], rot_noise[:, 2])
        target_quat = torch_utils.quat_mul(dq, home[:, 3:])
        # Subset-sized controller and world-frame poses/Jacobian agree.
        ik = DifferentialIKController(self._ik.cfg, n, device=self.device)
        ik.set_command(torch.cat((target_pos, target_quat), dim=-1))
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        for _ in range(task.ik_iterations):
            current = self.robot.data.body_pose_w[env_ids, self._ee_frame_idx]
            jac = self.robot.root_physx_view.get_jacobians()[env_ids, self._ee_frame_idx - 1][:, :, self._arm_joint_ids]
            old = joints[:, self._arm_joint_ids]
            solved = ik.compute(current[:, :3], current[:, 3:], jac, old)
            joints[:, self._arm_joint_ids] = old + (solved - old).clamp(-0.1, 0.1)
            joints = joints.clamp(limits[:, :, 0], limits[:, :, 1])
            self.robot.write_joint_state_to_sim(joints, zero_vel, env_ids=env_ids)
            self.sim.forward()
        pose = self.robot.data.body_pose_w[env_ids, self._ee_frame_idx].clone()
        residual = torch.linalg.vector_norm(pose[:, :3] - target_pos, dim=-1)
        angle = 2 * torch.acos((pose[:, 3:] * target_quat).sum(-1).abs().clamp(0, 1))
        if torch.any(residual > task.ik_position_tolerance) or torch.any(angle > task.ik_angle_tolerance):
            raise RuntimeError(f"ORU reset IK failed: max position error={residual.max().item():.5f}m, angle={angle.max().item():.5f}rad. Reduce/validate reset bounds.")
        # PhysX merges this fixed-joint chain into the UR5 articulation.
        # FK updates every attached link; writing RigidObject transforms here
        # is rejected for non-root articulation links and would poison caches.
        self.prev_ee_pos[env_ids] = pose[:, :3]
        self.prev_ee_quat[env_ids] = pose[:, 3:]
        self.ep_ee_start[env_ids] = pose[:, :3]
        self.prev_dist_target[env_ids] = torch.linalg.vector_norm(pose[:, :3] - self._get_target_ref()[0][env_ids], dim=-1)
        self.prev_reward_z[env_ids] = pose[:, 2]
        self.prev_joint_pos[env_ids] = joints[:, self._arm_joint_ids]
        if not hasattr(self, "ee_linvel_fd"):
            self.ee_linvel_fd = torch.zeros_like(self.prev_ee_pos)
            self.ee_angvel_fd = torch.zeros_like(self.prev_ee_pos)
        self.ee_linvel_fd[env_ids] = 0
        self.ee_angvel_fd[env_ids] = 0
        self._compute_intermediate_values(self.physics_dt)


# ==================================================================
# Fixed joint helper — env_0 only (replicate_physics=True shares to all)
# ==================================================================

def _create_fixed_joints(stage, env_idx: int, *, drive_stiffness=None, drive_damping=None):
    """Create a single FixedJoint chain on one environment.

    Called for EVERY environment (clone_in_fabric=False) so each env
    gets its own independent USD hierarchy + physics for the
    UR5→Bridge→SixForce→Gripper→ORU FixedJoint chain.
    """
    ns = f"/World/envs/env_{env_idx}"
    ou = oru_utils

    ou.create_one_fixed_joint(
        stage, f"{ns}/Dofbot/wrist_3_link/bridge_joint",
        f"{ns}/Dofbot/wrist_3_link", f"{ns}/Bridge/base_link",
        drive_stiffness=drive_stiffness, drive_damping=drive_damping,
    )
    ou.create_one_fixed_joint(
        stage, f"{ns}/Bridge/base_link/force_joint",
        f"{ns}/Bridge/base_link", f"{ns}/SixForce/base_link",
        child_offset_axis=(0, 1, 0), child_offset_angle=math.pi,
        child_offset_pos=(0, 0, 0.062),
        drive_stiffness=drive_stiffness, drive_damping=drive_damping,
    )
    ou.create_one_fixed_joint(
        stage, f"{ns}/SixForce/base_link/gripper_joint",
        f"{ns}/SixForce/base_link", f"{ns}/Gripper/base_link",
        child_offset_pos=(0, 0, -0.0253), child_offset_axis=(0, 1, 0), child_offset_angle=math.pi,
        drive_stiffness=drive_stiffness, drive_damping=drive_damping,
    )
    ou.create_one_fixed_joint(
        stage, f"{ns}/Gripper/base_link/oru_joint",
        f"{ns}/Gripper/base_link", f"{ns}/ORU/base_link",
        child_offset_pos=(0, 0, -0.305), child_offset_axis=(0, 0, 1), child_offset_angle=math.pi,
        drive_stiffness=drive_stiffness, drive_damping=drive_damping,
    )
