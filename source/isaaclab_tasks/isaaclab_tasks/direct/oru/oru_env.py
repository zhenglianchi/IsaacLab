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
from isaaclab.utils.math import axis_angle_from_quat

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from . import oru_control, oru_utils
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
        self.prev_dist_target = torch.zeros(N, device=self.device)  # last-step dist to target (approach shaping)

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
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)
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

        # ── Target: XY = ground XY, Z = 0.4, quat = [0,0,1,0] ──
        ground_pos = self.ground.data.root_pos_w
        ctrl_target_ee_pos = ground_pos.clone()
        ctrl_target_ee_pos[:, 2] = self.fixed_target_z
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

        # ── Stage-2 Z force cap ──
        # Contact (stage 2) or success → Z force cap max_task_force_z (>50N
        # for the last cm). Success latches it too: a seated ORU's weightless
        # chain unloads the wrist (F_mag→0), which would otherwise drop the
        # cap back to the 8N free-space limit. Free space stays at
        # max_task_force so stage-1 approach cannot slingshot.
        task = self.cfg.task
        z_force_limit = torch.where(
            self._was_in_contact | self.ep_succeeded,
            torch.full((self.num_envs,), task.max_task_force_z, device=self.device),
            torch.full((self.num_envs,), task.max_task_force, device=self.device),
        )

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

    def _get_contact_degree(self) -> torch.Tensor:
        """Contact detection (measured force) with hysteresis → sigmoid [0,1]."""
        task = self.cfg.task
        F_mag = self._get_measured_force_mag()
        height_ok = (self.ee_pos[:, 2] - self.fixed_target_z) < task.contact_height_threshold

        # Multi-criteria contact detection
        contact_force = F_mag > task.contact_force_threshold
        force_surge = (F_mag - self.prev_F_mag) > task.contact_surge_delta
        contact_surge = force_surge & (F_mag > task.contact_surge_force)

        # BOTH paths gated by height: the measured wrist wrench rises with
        # chain inertia during ANY free-space acceleration (>>0.5N), so an
        # ungated surge would flip to stage-2 mid-approach and eat the
        # stage-2 force penalties for no reason.
        entering = (contact_force & height_ok) | (contact_surge & height_ok)
        leaving = F_mag < task.contact_leave_threshold

        # Hysteresis: once in contact, stay until force drops well below threshold
        self._was_in_contact = torch.where(
            entering, True, torch.where(leaving, False, self._was_in_contact)
        )

        # Sigmoid soft transition on measured force magnitude. While NOT in
        # contact the degree is also height-gated: free-space chain inertia
        # (>0.5N) would otherwise bleed stage-2 rewards into stage-1. Once
        # latched in contact, the gate is bypassed so the blend cannot reset.
        degree = torch.sigmoid((F_mag - task.contact_force_threshold) * task.contact_sigmoid_slope)
        degree = torch.where(self._was_in_contact, degree, torch.where(height_ok, degree, torch.zeros_like(degree)))
        # Keep a floor of 30% once in contact (hysteresis blend)
        degree = torch.where(self._was_in_contact, degree.clamp(min=0.3), degree)
        return degree

    def _get_path_reward(self, target_ref_pos: torch.Tensor, dist_target: torch.Tensor) -> torch.Tensor:
        """Stage 1: continuous projection progress + keypoint corridor deviation.

        Progress t ∈ [0,1] is the EE projection along the start→target line —
        a continuous number, so closely-spaced keypoints cannot cause
        double-trigger issues. Potential-based approach shaping on 3D
        distance carries a gradient through the long middle of the descent.
        """
        task = self.cfg.task

        line = target_ref_pos - self.ep_ee_start
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

        # 4. Potential-based approach: reward each cm of closing distance.
        #    Telescopes over the episode → no local trap, pure gradient.
        r_approach = task.approach_weight * (self.prev_dist_target - dist_target)

        # 5. Pose alignment: quat dot with target quat, threshold form —
        #    only pays once aligned (relu(dot−threshold), scaled to
        #    [0, align_weight]). Flat payment was a per-step stipend.
        alignment = torch.abs(torch.sum(self.ee_quat * self.fixed_target_quat, dim=-1))
        r_align = (
            task.align_weight
            * torch.relu(alignment - task.align_threshold)
            / (1.0 - task.align_threshold)
        )

        return r_progress + r_deviation + r_target + r_align + r_approach

    def _get_insertion_reward(self, target_ref_pos: torch.Tensor) -> torch.Tensor:
        """Stage 2: precision convergence + force compliance (Yuan + SRL-VIC style)."""
        task = self.cfg.task
        F_mag = self._get_measured_force_mag()

        # Precision convergence: steep exp + log (increasing as distance→0)
        dist_target = torch.norm(self.ee_pos - target_ref_pos, dim=-1)
        r_close = torch.exp(-task.precision_a * dist_target)
        r_log = -task.log_reward_coef * torch.log(dist_target.clamp(min=1e-4))

        # Z progress: reward downward motion toward docking height
        z_vel_down = torch.clamp(self.ee_linvel[:, 2], max=0.0)  # negative = downward
        r_z_progress = task.z_progress_weight * (-z_vel_down)

        # Depth-gap pressure: penalize the residual z gap above tolerance.
        # Position-level signal — keeps a gradient when contact stalls the EE.
        z_gap = self.ee_pos[:, 2] - target_ref_pos[:, 2] - task.z_gap_tolerance
        r_z_depth = -task.z_depth_weight * torch.relu(z_gap)

        # Force smoothness: penalize abrupt changes in measured force
        r_force_smooth = -task.force_smooth_weight * (F_mag - self.prev_F_mag).abs()

        # Force peak: squared penalty above safety limit
        r_force_peak = -task.force_peak_weight * torch.relu(F_mag - task.force_peak_threshold) ** 2

        # Lateral force: commanded XY task force (frame-safe, drives XY alignment)
        F = self.applied_wrench[:, :3]
        F_xy = torch.norm(F[:, :2], dim=-1)
        r_lateral = -task.lateral_force_weight * F_xy

        # Z force target: keep moderate downward commanded force
        r_z_force = -task.z_force_weight * torch.abs(F[:, 2] - task.z_force_target)

        # Alignment kept alive in stage 2 — otherwise entering the contact
        # zone would blend away the align income. Threshold form: pays only
        # when aligned (see stage-1 comment).
        alignment = torch.abs(torch.sum(self.ee_quat * self.fixed_target_quat, dim=-1))
        r_align = (
            task.align_weight
            * torch.relu(alignment - task.align_threshold)
            / (1.0 - task.align_threshold)
        )

        return (
            r_close + r_log + r_z_progress + r_z_depth
            + r_force_smooth + r_force_peak + r_lateral + r_z_force + r_align
        )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values(self.physics_dt)

        target_ref_pos, _ = self._get_target_ref()
        dist_target = torch.norm(self.ee_pos - target_ref_pos, dim=-1)

        # ── Two-stage reward with sigmoid soft transition ────────────
        r_stage1 = self._get_path_reward(target_ref_pos, dist_target)
        r_stage2 = self._get_insertion_reward(target_ref_pos)
        contact_degree = self._get_contact_degree()  # [0,1] soft switch

        rew = (1.0 - contact_degree) * r_stage1 + contact_degree * r_stage2

        # Action penalties (both stages)
        rew -= self.cfg.task.action_penalty_scale * torch.norm(self.actions, p=2, dim=-1)
        rew -= self.cfg.task.action_grad_penalty_scale * torch.norm(
            self.actions - self.prev_actions, p=2, dim=-1
        )
        curr_s = self._get_curr_successes(self.cfg.task.success_threshold)
        # Completion bonus — must dominate the per-step income or the policy
        # settles for "hover near target" (success +1 vs align +1.8/step was
        # net-negative to finish). Paid every step success holds, so keeping
        # the seat is also rewarded.
        rew += curr_s.float() * self.cfg.task.success_reward

        # Logging + state update
        self.prev_actions = self.actions.clone()
        self.prev_F_mag = self._get_measured_force_mag().clone()
        self.prev_dist_target = dist_target.clone()
        if torch.any(self.reset_buf):
            self.extras["success_rate"] = curr_s.float().mean()
        self.extras["rew_pos_error"] = torch.norm(self.ee_pos - target_ref_pos, dim=-1).mean()
        self.extras["rew_contact_degree"] = contact_degree.mean()
        return rew

    def _get_curr_successes(self, threshold: float) -> torch.Tensor:
        oru_pos = self.ee_pos
        target_pos, _ = self._get_target_ref()
        xy = torch.norm(target_pos[:, :2] - oru_pos[:, :2], dim=-1)
        # Success: EE reaches the true docking height (success_z), XY centered
        z_ok = oru_pos[:, 2] <= self.cfg.task.success_z
        return (xy < self.cfg.task.xy_tolerance) & z_ok

    # ==================================================================
    # Done
    # ==================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        successes = self._get_curr_successes(self.cfg.task.success_threshold)
        first = torch.logical_and(successes, ~self.ep_succeeded)
        self.ep_succeeded[successes] = True
        first_ids = first.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_ids] = self.episode_length_buf[first_ids]
        return time_out, time_out

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
        self.prev_dist_target[env_ids] = 0.0  # overwritten in _reset_idx with the true start distance

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # ── Step 1: reset UR5 root + default joints ────────────────
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        jpos = self.robot.data.default_joint_pos[env_ids]
        jvel = torch.zeros_like(jpos)
        self.robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
        self.robot.set_joint_effort_target(
            torch.zeros((n, self.robot.num_joints), device=self.device), env_ids=env_ids,
        )

        # Step sim once so FK gives us the "home" EE pose
        self._step_sim_no_action()

        # ── Step 2: FK → default EE pose in world frame ─────────────
        self._ensure_frame_indices()
        ee_pose_w = self.robot.data.body_pose_w[:, self._ee_frame_idx]
        home_ee_pos_w = ee_pose_w[env_ids, 0:3].clone()
        home_ee_quat_w = ee_pose_w[env_ids, 3:7].clone()

        # ── Step 3: offset EE pose (fixed case or randomized) ────────
        if self.cfg.task.fixed_ik_offset_pos is not None:
            # Fixed offset for single-case evaluation (play_force.py)
            rand_pos = torch.tensor(self.cfg.task.fixed_ik_offset_pos, device=self.device).unsqueeze(0).repeat(n, 1)
            rand_rot = torch.tensor(self.cfg.task.fixed_ik_offset_rot, device=self.device).unsqueeze(0).repeat(n, 1)
        else:
            pos_std = torch.tensor(self.cfg.task.ik_rand_pos_noise, device=self.device)
            rot_std = torch.tensor(self.cfg.task.ik_rand_rot_noise, device=self.device)
            rand_pos = (torch.rand((n, 3), device=self.device) - 0.5) * 2 * pos_std
            rand_rot = (torch.rand((n, 3), device=self.device) - 0.5) * 2 * rot_std

        target_ee_pos_w = home_ee_pos_w + rand_pos

        # Apply rotation noise as euler delta around current orientation
        rand_quat = torch_utils.quat_from_euler_xyz(
            rand_rot[:, 0], rand_rot[:, 1], rand_rot[:, 2],
        )
        target_ee_quat_w = torch_utils.quat_mul(rand_quat, home_ee_quat_w)

        # ── Step 4: convert target to robot base frame (IK needs base frame) ──
        root_pose_w = self.robot.data.root_pose_w[env_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_ee_pos_w, target_ee_quat_w,
        )

        # ── Step 5: DLS IK → joint angles ───────────────────────────
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._ee_frame_idx - 1, :, :
        ][env_ids][:, :, self._arm_joint_ids]
        current_joints = self.robot.data.joint_pos[env_ids][:, self._arm_joint_ids]

        ik_command = torch.cat([ee_pos_b, ee_quat_b], dim=1)
        self._ik.set_command(ik_command)
        ik_joints = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, current_joints)

        # ── Step 6: write IK joint angles as initial state ──────────
        full_jpos = self.robot.data.default_joint_pos[env_ids].clone()
        full_jpos[:, self._arm_joint_ids] = ik_joints
        self.robot.write_joint_state_to_sim(full_jpos, jvel, env_ids=env_ids)
        self.robot.set_joint_effort_target(
            torch.zeros((n, self.robot.num_joints), device=self.device), env_ids=env_ids,
        )

        # ── Step 7: step sim to settle FixedJoint chain ─────────────
        self._step_sim_no_action()

        # update tracking buffers
        self._ensure_frame_indices()
        self.prev_ee_pos[env_ids] = self.robot.data.body_pos_w[:, self._ee_frame_idx][env_ids].clone()
        self.prev_ee_quat[env_ids] = self.robot.data.body_quat_w[:, self._ee_frame_idx][env_ids].clone()
        # Episode start position — base for stage-1 path keypoints
        self.ep_ee_start[env_ids] = self.robot.data.body_pos_w[:, self._ee_frame_idx][env_ids].clone()
        # Approach-shaping baseline: distance from the (IK-randomized) start
        self.prev_dist_target[env_ids] = torch.norm(
            self.robot.data.body_pos_w[:, self._ee_frame_idx][env_ids]
            - self._get_target_ref()[0][env_ids],
            dim=-1,
        )
        self.prev_joint_pos[env_ids] = self.robot.data.joint_pos[:, self._arm_joint_ids][env_ids].clone()
        self.ee_linvel_fd[env_ids] = 0.0
        self.ee_angvel_fd[env_ids] = 0.0

    def _step_sim_no_action(self):
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
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
