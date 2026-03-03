import torch

from isaaclab.controllers import OperationalSpaceController
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

class UR5HybridOSCController:

    def __init__(self, scene, sim,
                 robot_name="Dofbot",
                 end_effector_link="wrist_3_link",
                 contact_forces="contact_forces"):

        self.scene = scene
        self.sim = sim
        self.robot = scene[robot_name]
        self.contact_forces = scene[contact_forces]

        self.device = self.robot.data.device
        self.num_envs = scene.num_envs
        

        # --------------------------------------------------
        # Scene Entity
        # --------------------------------------------------
        self.entity_cfg = SceneEntityCfg(
            robot_name,
            joint_names=[".*"],
            body_names=[end_effector_link],
        )
        self.entity_cfg.resolve(scene)

        self.ee_frame_name = "wrist_3_link"
        arm_joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"]
        self.ee_frame_idx = self.robot.find_bodies(self.ee_frame_name)[0][0]
        self.arm_joint_ids = self.robot.find_joints(arm_joint_names)[0]

        # --------------------------------------------------
        # 官方示例 OSC 配置（完全一致）
        # --------------------------------------------------
        self.osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs", "wrench_abs"],
            impedance_mode="variable_kp",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_damping_ratio_task=1.0,
            contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            motion_control_axes_task=[1, 1, 0, 1, 1, 1],
            contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
            nullspace_control="none",
        )

        self.osc = OperationalSpaceController(
            self.osc_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

    def get_tcp(self):
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_frame_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_frame_idx]
        return torch.cat([ee_pos_w, ee_quat_w], dim=-1)

    def update_states(self):
        """Update the robot states.
        Returns:
            jacobian_b (torch.tensor): Jacobian in the body frame.
            mass_matrix (torch.tensor): Mass matrix.
            gravity (torch.tensor): Gravity vector.
            ee_pose_b (torch.tensor): End-effector pose in the body frame.
            ee_vel_b (torch.tensor): End-effector velocity in the body frame.
            root_pose_w (torch.tensor): Root pose in the world frame.
            ee_pose_w (torch.tensor): End-effector pose in the world frame.
            ee_force_b (torch.tensor): End-effector force in the body frame.
            joint_pos (torch.tensor): The joint positions.
            joint_vel (torch.tensor): The joint velocities.

        Raises:
            ValueError: Undefined target_type.
        """
        # obtain dynamics related quantities from simulation
        ee_jacobi_idx = self.ee_frame_idx - 1
        jacobian_w = self.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, self.arm_joint_ids]
        mass_matrix = self.robot.root_physx_view.get_generalized_mass_matrices()[:, self.arm_joint_ids, :][:, :, self.arm_joint_ids]
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[:, self.arm_joint_ids]
        # Convert the Jacobian from world to root frame
        jacobian_b = jacobian_w.clone()
        root_rot_matrix = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

        # Compute current pose of the end-effector
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_frame_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_frame_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
        ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        # Compute the current velocity of the end-effector
        ee_vel_w = self.robot.data.body_vel_w[:, self.ee_frame_idx, :]  # Extract end-effector velocity in the world frame
        root_vel_w = self.robot.data.root_vel_w  # Extract root velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_apply_inverse(self.robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
        ee_ang_vel_b = quat_apply_inverse(self.robot.data.root_quat_w, relative_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

        # Calculate the contact force
        ee_force_w = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)
        sim_dt = self.sim.get_physics_dt()
        self.contact_forces.update(sim_dt)  # update contact sensor
        # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
        # taking the max of three surfaces as only one should be the contact of interest
        ee_force_w, _ = torch.max(torch.mean(self.contact_forces.data.net_forces_w_history, dim=1), dim=1)

        # This is a simplification, only for the sake of testing.
        ee_force_b = ee_force_w

        # Get joint positions and velocities
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]

        return (
            jacobian_b,
            mass_matrix,
            gravity,
            ee_pose_b,
            ee_vel_b,
            root_pose_w,
            ee_pose_w,
            ee_force_b,
            joint_pos,
            joint_vel,
        )