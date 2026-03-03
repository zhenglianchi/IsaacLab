# ur5_controller.py
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


class UR5Controller:

    def __init__(self, scene, args_cli, robot_name="Dofbot", end_effector_link="wrist_3_link"):
        self.scene = scene
        self.robot = scene[robot_name]
        self.robot_name = robot_name
        self.ground = scene["Ground"]
        self.bridge = scene["Bridge"]
        self.force = scene["Froce_Six"]
        self.gripper = scene["Gripper"]
        self.oru = scene["ORU"]

        # Differential IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            ik_method="dls",
            use_relative_mode=False,
        )
        self.diff_ik = DifferentialIKController(diff_ik_cfg, scene.num_envs, device=self.robot.data.device)

        # EE markers
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.base_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/robot_base"))
        self.ee_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/ee_goal"))
        self.bridge_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/bridge"))
        self.Froce_Six_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/force"))
        self.gripper_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/gripper"))
        self.oru_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/oru"))
        self.ground_marker = VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/ground"))

        # Robot entity config
        self.robot_entity_cfg = SceneEntityCfg(
            robot_name,
            joint_names=[".*"],
            body_names=[end_effector_link],
        )
        self.robot_entity_cfg.resolve(scene)

        self.ee_jacobian_idx = (
            self.robot_entity_cfg.body_ids[0] - 1
            if self.robot.is_fixed_base
            else self.robot_entity_cfg.body_ids[0]
        )

        self.joint_pos_des = self.robot.data.default_joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
        self.sim_dt = scene.sim.get_physics_dt() if hasattr(scene, "sim") else 0.01  # fallback
        self.args_cli = args_cli

    # -------------------
    # Utilities
    # -------------------
    def get_current_joint_positions(self):
        """Return current joint positions (rad)"""
        return self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].clone()

    def get_end_effector_pose(self):
        """Return EE position and quaternion in world frame"""
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        pos = ee_pose_w[:, 0:3].clone()
        quat = ee_pose_w[:, 3:7].clone()
        return pos, quat

    def move_ee_to(self, target_pos, target_quat):
        """
        Move end-effector to target position.
        target_pos: tensor of shape [num_envs, 3] or [3]
        fixed_orientation: if True, keep current EE orientation
        """
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]

        ee_quat = target_quat

        # prepare IK command
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0).repeat(self.args_cli.num_envs, 1)
        ik_command = torch.cat([target_pos, ee_quat.repeat(self.args_cli.num_envs, 1)], dim=1)
        self.diff_ik.set_command(ik_command)

        # Compute Jacobian
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobian_idx, :, self.robot_entity_cfg.joint_ids
        ]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            self.robot.data.root_pose_w[:, 0:3],
            self.robot.data.root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        self.joint_pos_des = self.diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(self.joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        
        """Step simulation and update EE markers"""
        self.scene.write_data_to_sim()
        self.scene.sim.step()
        self.scene.update(self.sim_dt)

        # visualize EE
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        pos, quat = ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        self.ee_marker.visualize(pos, quat)
        self.goal_marker.visualize(target_pos + self.scene.env_origins, target_quat)

        self.base_marker.visualize(self.robot.data.root_pose_w[:, 0:3],self.robot.data.root_pose_w[:, 3:7])

        bridge_pose = self.bridge.data.root_pose_w
        self.bridge_marker.visualize(bridge_pose[:, 0:3],bridge_pose[:, 3:7])

        force_pose = self.force.data.root_pose_w
        self.Froce_Six_marker.visualize(force_pose[:, 0:3],force_pose[:, 3:7])

        gripper_pose = self.gripper.data.root_pose_w
        self.gripper_marker.visualize(gripper_pose[:, 0:3],gripper_pose[:, 3:7])

        oru_pose = self.oru.data.root_pose_w
        self.oru_marker.visualize(oru_pose[:, 0:3],oru_pose[:, 3:7])

        ground_pose = self.ground.data.root_pose_w
        self.ground_marker.visualize(ground_pose[:, 0:3],ground_pose[:, 3:7])
