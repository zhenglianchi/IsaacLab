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
            _create_fixed_joints(stage, env_idx)

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
        self.base_gains = torch.tensor(prop, device=self.device).repeat(N, 1)
        self.task_prop_gains = self.base_gains.clone()
        self.task_deriv_gains = oru_utils.get_deriv_gains(
            self.task_prop_gains, self.cfg.task.rot_deriv_scale
        )
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
        self.ee_linvel = self.robot.data.body_lin_vel_w[:, self._ee_frame_idx]
        self.ee_angvel = self.robot.data.body_ang_vel_w[:, self._ee_frame_idx]

        self.joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        self.joint_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]

        jac_idx = self._ee_frame_idx - 1
        self.jacobian = self.robot.root_physx_view.get_jacobians()[
            :, jac_idx, :, self._arm_joint_ids
        ]
        self.mass_matrix = (
            self.robot.root_physx_view.get_generalized_mass_matrices()
            [:, self._arm_joint_ids, :]
            [:, :, self._arm_joint_ids]
        )

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
        if self.last_update_timestamp < self.robot._data._sim_timestamp:
            self._compute_intermediate_values(dt)

        # ── Actions are gain scaling factors: [-1,1] → Kp = base * (1 + a * range) ──
        scale = 1.0 + self.actions[:, 0:6] * self.gain_range
        scale = torch.clamp(scale, min=0.05, max=5.0)
        self.task_prop_gains = self.base_gains * scale
        self.task_deriv_gains = oru_utils.get_deriv_gains(
            self.task_prop_gains, self.cfg.task.rot_deriv_scale,
        )

        # ── Target: XY = ground XY, Z = 0.4, quat = [0,0,1,0] ──
        ground_pos = self.ground.data.root_pos_w
        ctrl_target_ee_pos = ground_pos.clone()
        ctrl_target_ee_pos[:, 2] = self.fixed_target_z
        ctrl_target_ee_quat = self.fixed_target_quat

        joint_torque, self.applied_wrench = oru_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            ee_pos=self.ee_pos,
            ee_quat=self.ee_quat,
            ee_linvel=self.ee_linvel_fd,
            ee_angvel=self.ee_angvel_fd,
            jacobian=self.jacobian,
            mass_matrix=self.mass_matrix,
            ctrl_target_ee_pos=ctrl_target_ee_pos,
            ctrl_target_ee_quat=ctrl_target_ee_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        self.robot.set_joint_effort_target(joint_torque, joint_ids=self._arm_joint_ids)

    # ==================================================================
    # Observations
    # ==================================================================

    def _get_observations(self) -> dict:
        self._compute_intermediate_values(self.physics_dt)

        ground_pos = self.ground.data.root_pos_w
        ground_quat = self.ground.data.root_quat_w

        obs_dict = {
            "ee_pos_rel_ground": self.ee_pos - ground_pos,
            "ee_quat": self.ee_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "joint_pos": self.joint_pos,
            "task_prop_gains": self.task_prop_gains,   # policy sees its variable stiffness
        }
        state_dict = {
            **obs_dict,
            "ground_pos": ground_pos,
            "ground_quat": ground_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
        }

        policy_obs = oru_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order)
        policy_obs = torch.cat([policy_obs, self.actions], dim=-1)

        critic_obs = oru_utils.collapse_obs_dict(state_dict, self.cfg.state_order)
        critic_obs = torch.cat([critic_obs, self.actions], dim=-1)

        return {"policy": policy_obs, "critic": critic_obs}

    # ==================================================================
    # Rewards — virtual ORU pose from UR5 FK + chain offset
    # ==================================================================

    # offset from UR5 wrist_3_link to ORU bottom (sum of force1 joints − half height)
    ORU_BOTTOM_Z = 0.062 - 0.0253 - 0.257 - 0.075  # ≈ −0.2953
    GROUND_H = 0.05
    ORU_HALF_H = 0.075

    def _virtual_oru_bottom(self) -> torch.Tensor:
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = self.ORU_BOTTOM_Z
        id_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        _, w = torch_utils.tf_combine(self.ee_quat, self.ee_pos, id_q, local)
        return w

    def _ground_top(self) -> torch.Tensor:
        ground_pos = self.ground.data.root_pos_w
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = self.GROUND_H
        id_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        _, w = torch_utils.tf_combine(self.ground.data.root_quat_w, ground_pos, id_q, local)
        return w

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values(self.physics_dt)

        oru_bottom = self._virtual_oru_bottom()
        ground_top = self._ground_top()

        num_kp = self.cfg.task.num_keypoints
        scale = self.cfg.task.keypoint_scale
        offsets = oru_utils.get_keypoint_offsets(num_kp, self.device) * scale
        id_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        oru_q = self.ee_quat
        ground_q = self.ground.data.root_quat_w

        kp_oru = torch.zeros((self.num_envs, num_kp, 3), device=self.device)
        kp_target = torch.zeros((self.num_envs, num_kp, 3), device=self.device)
        for i, off in enumerate(offsets):
            kp_oru[:, i] = torch_utils.tf_combine(oru_q, oru_bottom, id_q, off.repeat(self.num_envs, 1))[1]
            kp_target[:, i] = torch_utils.tf_combine(ground_q, ground_top, id_q, off.repeat(self.num_envs, 1))[1]

        kp_dist = torch.norm(kp_oru - kp_target, p=2, dim=-1).mean(-1)

        a0, b0 = self.cfg.task.keypoint_coef_baseline
        a1, b1 = self.cfg.task.keypoint_coef_coarse
        a2, b2 = self.cfg.task.keypoint_coef_fine

        rew = (
            oru_utils.squashing_fn(kp_dist, a0, b0)
            + oru_utils.squashing_fn(kp_dist, a1, b1)
            + oru_utils.squashing_fn(kp_dist, a2, b2)
        )
        rew -= self.cfg.task.action_penalty_scale * torch.norm(self.actions, p=2, dim=-1)
        rew -= self.cfg.task.action_grad_penalty_scale * torch.norm(
            self.actions - self.prev_actions, p=2, dim=-1
        )
        curr_s = self._get_curr_successes(self.cfg.task.success_threshold)
        curr_e = self._get_curr_successes(self.cfg.task.engage_threshold)
        rew += curr_s.float() + curr_e.float()

        self.prev_actions = self.actions.clone()
        if torch.any(self.reset_buf):
            self.extras["success_rate"] = curr_s.float().mean()
        self.extras["rew_kp_dist"] = kp_dist.mean()
        return rew

    def _get_curr_successes(self, threshold: float) -> torch.Tensor:
        ob = self._virtual_oru_bottom()
        gt = self._ground_top()
        xy = torch.norm(gt[:, :2] - ob[:, :2], dim=-1)
        z = ob[:, 2] - gt[:, 2]
        return (xy < self.cfg.task.xy_tolerance) & (z < self.GROUND_H * threshold)

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

        # ── Step 3: add random offset to EE pose ────────────────────
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

def _create_fixed_joints(stage, env_idx: int):
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
    )
    ou.create_one_fixed_joint(
        stage, f"{ns}/Bridge/base_link/force_joint",
        f"{ns}/Bridge/base_link", f"{ns}/SixForce/base_link",
        child_offset_axis=(0, 1, 0), child_offset_angle=math.pi,
        child_offset_pos=(0, 0, 0.062),
    )
    ou.create_one_fixed_joint(
        stage, f"{ns}/SixForce/base_link/gripper_joint",
        f"{ns}/SixForce/base_link", f"{ns}/Gripper/base_link",
        child_offset_pos=(0, 0, -0.0253), child_offset_axis=(0, 1, 0), child_offset_angle=math.pi,
    )
    ou.create_one_fixed_joint(
        stage, f"{ns}/Gripper/base_link/oru_joint",
        f"{ns}/Gripper/base_link", f"{ns}/ORU/base_link",
        child_offset_pos=(0, 0, -0.257), child_offset_axis=(0, 0, 1), child_offset_angle=math.pi,
    )
