import torch

from isaaclab.controllers import OperationalSpaceController
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


class UR5HybridOSCController:

    def __init__(self, scene, args_cli,
                 robot_name="Dofbot",
                 end_effector_link="wrist_3_link"):

        self.scene = scene
        self.robot = scene[robot_name]

        self.robot.actuators["shoulder_pan_joint"].stiffness = 0.0
        self.robot.actuators["shoulder_pan_joint"].damping = 0.0
        self.robot.actuators["shoulder_lift_joint"].stiffness = 0.0
        self.robot.actuators["shoulder_lift_joint"].damping = 0.0
        self.robot.actuators["elbow_joint"].stiffness = 0.0
        self.robot.actuators["elbow_joint"].damping = 0.0
        self.robot.actuators["wrist_1_joint"].stiffness = 0.0
        self.robot.actuators["wrist_1_joint"].damping = 0.0
        self.robot.actuators["wrist_2_joint"].stiffness = 0.0
        self.robot.actuators["wrist_2_joint"].damping = 0.0
        self.robot.actuators["wrist_3_joint"].stiffness = 0.0
        self.robot.actuators["wrist_3_joint"].damping = 0.0


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

        self.ee_body_id = self.entity_cfg.body_ids[0]
        self.joint_ids = self.robot.find_joints(
            ["shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"]
        )[0]

        # --------------------------------------------------
        # 官方示例 OSC 配置（完全一致）
        # --------------------------------------------------
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs", "wrench_abs"],
            impedance_mode="variable_kp",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_damping_ratio_task=1.0,
            contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            motion_control_axes_task=[1, 1, 0, 1, 1, 1],
            contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
            nullspace_control="position",
        )

        self.osc = OperationalSpaceController(
            osc_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

    # ------------------------------------------------------
    # 获取 EE 位姿
    # ------------------------------------------------------
    def get_end_effector_pose(self):
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        return ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]

    # ------------------------------------------------------
    # Step（完全官方逻辑）
    # ------------------------------------------------------
    def step(self, command):
        if command.dim() == 1:
            command = command.unsqueeze(0).repeat(self.num_envs, 1)

        # command = [pose(7), wrench(6), kp(6)]
        target_pose_w = command[:, 0:7]
        target_wrench = command[:, 7:13]
        kp = command[:, 13:19]

        target_pos_w = target_pose_w[:, 0:3]
        target_quat_w = target_pose_w[:, 3:7]

        # =====================================================
        # 2️⃣ 当前 EE 位姿 (world → base)
        # =====================================================
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        root_pose_w = self.robot.data.root_pose_w

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        # 拼成 7维
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=1)

        # =====================================================
        # 3️⃣ 目标位姿转换到 base frame
        # =====================================================
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            target_pos_w,
            target_quat_w,
        )

        command_b = torch.cat(
            [
                target_pos_b,
                target_quat_b,
                target_wrench,
                kp,
            ],
            dim=1,
        )

        self.osc.set_command(command_b)

        # =====================================================
        # 4️⃣ Jacobian（安全两步 slice）
        # =====================================================
        jacobian_all = self.robot.root_physx_view.get_jacobians()
        jacobian = jacobian_all[:, self.ee_body_id, :, :]
        jacobian = jacobian[:, :, self.joint_ids]

        # =====================================================
        # 5️⃣ Mass matrix
        # =====================================================
        mass_matrix_all = self.robot.root_physx_view.get_mass_matrices()
        mass_matrix = mass_matrix_all[:, self.joint_ids, :]
        mass_matrix = mass_matrix[:, :, self.joint_ids]

        # =====================================================
        # 6️⃣ Joint state
        # =====================================================
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]

        # =====================================================
        # 7️⃣ EE 速度 (world → base)
        # =====================================================
        ee_vel_w = self.robot.data.body_vel_w[:, self.ee_body_id]
        root_vel_w = self.robot.data.root_vel_w

        print("ee_pose_b shape:", ee_pose_b.shape)
        print("ee_vel_b shape:", ee_vel_b.shape)

        ee_vel_lin_b, ee_vel_ang_b = subtract_frame_transforms(
            root_vel_w[:, 0:3],
            root_vel_w[:, 3:7],
            ee_vel_w[:, 0:3],
            ee_vel_w[:, 3:7],
        )

        ee_vel_b = torch.cat([ee_vel_lin_b, ee_vel_ang_b], dim=1)

        # =====================================================
        # 8️⃣ 当前接触力（如果没有传感器，用0）
        # =====================================================
        ee_force_b = torch.zeros(
            (self.num_envs, 6),
            device=self.device,
        )

        # =====================================================
        # 9️⃣ 重力
        # =====================================================
        gravity = self.robot.data.gravity

        # =====================================================
        # 🔟 Nullspace 目标
        # =====================================================
        nullspace_target = joint_pos.clone()

        # =====================================================
        # 1️⃣1️⃣ 计算 torque（新版关键字接口）
        # =====================================================
        torques = self.osc.compute(
            jacobian_b=jacobian,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=nullspace_target,
        )

        # =====================================================
        # 1️⃣2️⃣ 发送 torque
        # =====================================================
        self.robot.set_joint_effort_target(
            torques,
            joint_ids=self.joint_ids,
        )

    def set_kp(self, kp):
        self.kp = kp