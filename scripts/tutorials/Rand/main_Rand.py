import argparse
import torch
import numpy as np
import os
import csv
from isaaclab.app import AppLauncher
import time

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Assembly docking demo with UR5 robot"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--nh", type=int, default=4, help="Structure type (1-4).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from Rand_SceneCfg import RandSceneCfg, scene_reset, add_fixed_joint, create_assembly_body, create_docking_surface
from Rand_control import compute_dof_torque
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)
from isaaclab.sim import schemas
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    TriangleMeshPropertiesCfg,
)

# Update the target commands
def update_target(sim, scene, root_pose_w, ee_target_set, current_goal_idx):
    """Update the targets for the impedance controller.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        root_pose_w: (torch.tensor) Root pose in the world frame.
        ee_target_set: (torch.tensor) End-effector target set.
        current_goal_idx: (int) Current goal index.

    Returns:
        command (torch.tensor): Updated target command.
        ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
        ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
        next_goal_idx (int): Next goal index.

    Raises:
        ValueError: Undefined target_type.
    """

    # update the ee desired command
    command = torch.zeros(scene.num_envs, 19, device=sim.device)  # 7 for pose + 6 for kp + 6 for kd
    command[:] = ee_target_set[current_goal_idx]

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:] = command[:, :7]

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


# ==========================================================
# Main
# ==========================================================
def main():

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=2,
            max_velocity_iteration_count=2,
        )
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))
    
    # Design scene
    scene_cfg = RandSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Now we are ready!
    sim = sim_utils.SimulationContext.instance()
    stage = sim.stage

    # Create assembly body and docking surface for each environment
    for env_idx in range(args_cli.num_envs):
        env_ns = f"/World/envs/env_{env_idx}"
        
        # Create docking surface (fixed on ground, holes facing up)
        docking_surface_path = f"{env_ns}/DockingSurface"
        create_docking_surface(docking_surface_path, nh=args_cli.nh)
        
        # Set docking surface position (on ground, z = 0.015 for half thickness)
        docking_xform = UsdGeom.Xform.Get(stage, docking_surface_path)
        docking_xform.AddTranslateOp().Set((0.4, 0, 0.015))
        
        # Create assembly body (to be attached to gripper)
        assembly_path = f"{env_ns}/Assembly"
        create_assembly_body(assembly_path, nh=args_cli.nh)

    # Add fixed joints (including gripper -> assembly)
    add_fixed_joint(stage, args_cli)

    # Play the simulator
    sim.reset()
    scene.reset()
    
    # --------------------------------------------------
    # Controller
    # --------------------------------------------------
    robot = scene["Dofbot"]
    # Define end-effector frame and arm joints
    ee_frame_name = "wrist_3_link"
    arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    # Get end-effector frame index
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    # Get arm joint IDs
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    
    # ==========================================================
    # 目标位置设置为对接面中心坐标
    # ==========================================================
    # 对接面位置（与SceneCfg中设置一致）
    docking_pos = np.array([0.4, 0, 0.015])  # 对接面中心位置
    docking_quat = np.array([1, 0, 0, 0])  # 默认姿态
    
    # 目标位置：对接面中心上方一定高度（例如0.2米）
    target_pos = docking_pos.copy()
    target_pos[2] += 0.2  # 在对接面上方0.2米处
    
    # 定义绕 Z 轴 180° 的旋转四元数 (wxyz 格式)
    quat_z_180 = np.array([0.0, 0.0, 0.0, 1.0])

    # 四元数乘法函数 (Hamilton product)
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])

    # 计算旋转后的四元数（全局 Z 轴旋转 180°）
    rotated_quat = quat_multiply(quat_z_180, docking_quat)
    rotated_quat = rotated_quat / np.linalg.norm(rotated_quat)  # 归一化
    
    # 定义多个目标位置，实现接近和对接动作
    approach_height = 0.2  # 接近高度
    dock_height = 0.04     # 对接高度（定位销插入孔洞）
    
    # 目标序列：初始位置 -> 接近位置 -> 对接位置
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            # 初始位置：对接面上方0.3米
            [target_pos[0], target_pos[1], 0.3,  rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3]],
            # 接近位置：对接面上方0.15米
            [target_pos[0], target_pos[1], 0.15, rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3]],
            # 对接位置：定位销插入孔洞
            [target_pos[0], target_pos[1], dock_height, rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3]],
        ],
        device=sim.device,
    )

    kp_set_task = torch.tensor(
        [
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],   # 初始阶段
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],   # 接近阶段
            [100.0, 100.0, 50.0, 100.0, 100.0, 100.0],  # 对接阶段（增加xy刚度）
        ],
        device=sim.device,
    )

    kd_set_task = torch.tensor(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ],
        device=sim.device,
    )

    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, kp_set_task, kd_set_task], dim=-1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # ==========================================================
    # 初始化力/力矩数据保存文件
    # ==========================================================
    # Create data save directory
    save_dir = "rand_data_logs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate unique filename using timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(save_dir, f"wrench_data_{timestamp}.csv")

    # Initialize CSV file with header
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "Fx(N)", "Fy(N)", "Fz(N)", "Tx(Nm)", "Ty(Nm)", "Tz(Nm)"])

    print(f"Force/torque data will be saved to: {csv_filename}")

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)
    
    # Get robot data
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=1)
    
    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        scene.num_envs, 19, device=sim.device  # 7 pose + 6 kp + 6 kd
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(arm_joint_ids), device=sim.device)

    count = 0
    goal_step_count = 0
    goal_switch_interval = 500  # Switch goal every 500 steps
    
    while simulation_app.is_running():
        # Reset every 1500 steps
        if count % 1500 == 0:
            # Reset joint state to default
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
            robot.write_data_to_sim()
            robot.reset()
            scene.reset()

            # Reset target pose
            robot.update(sim_dt)
            print(f"Robot TCP: {robot.data.body_pos_w[0, ee_frame_idx].cpu().numpy()}")
            current_goal_idx = 0
            command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                sim, scene, root_pose_w, ee_target_set, 0
            )
            goal_step_count = 0
        else:
            # Switch goal every goal_switch_interval steps
            if goal_step_count % goal_switch_interval == 0 and count > 0:
                command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                    sim, scene, root_pose_w, ee_target_set, current_goal_idx
                )
                print(f"Switched to goal {current_goal_idx}")
            
            # Get the updated states
            robot.update(sim_dt)
            
            # Get required data for impedance control
            dof_pos = robot.data.joint_pos[:, arm_joint_ids]
            dof_vel = robot.data.joint_vel[:, arm_joint_ids]
            
            # Get end-effector data
            ee_pos = robot.data.body_pos_w[:, ee_frame_idx]
            ee_quat = robot.data.body_quat_w[:, ee_frame_idx]
            ee_linvel = robot.data.body_lin_vel_w[:, ee_frame_idx]
            ee_angvel = robot.data.body_ang_vel_w[:, ee_frame_idx]
            
            # Get Jacobian and mass matrix using PhysX view
            ee_jacobi_idx = ee_frame_idx - 1
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
            mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
            
            # Extract target pose and gains from command
            ctrl_target_ee_pos = command[:, :3]
            ctrl_target_ee_quat = command[:, 3:7]
            task_prop_gains = command[:, 7:13]
            task_deriv_gains = command[:, 13:19]
            
            # Compute joint torques using impedance control
            dof_torque, task_wrench = compute_dof_torque(
                robot.cfg,
                dof_pos,
                dof_vel,
                ee_pos,
                ee_quat,
                ee_linvel,
                ee_angvel,
                jacobian,
                mass_matrix,
                ctrl_target_ee_pos,
                ctrl_target_ee_quat,
                task_prop_gains,
                task_deriv_gains,
                sim.device
            )
            
            # Apply computed torques
            joint_efforts = dof_torque[:, arm_joint_ids]
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()
            
            print(f"Step: {count}")
            print(f"EE Position: {ee_pos[0].cpu().numpy()}")
            print(f"EE Target: {ctrl_target_ee_pos[0].cpu().numpy()}")
            print(f"Task Wrench: {task_wrench[0].cpu().numpy()}")
            print(f"Joint Torques: {joint_efforts[0].cpu().numpy()}")

            # Print end-effector 6D force/torque
            ee_wrench_b = robot.data.body_incoming_joint_wrench_b[:, ee_frame_idx, :]
            ee_force_b = ee_wrench_b[:, :3]   # 3D force
            ee_torque_b = ee_wrench_b[:, 3:]  # 3D torque
            ee_force_b_np = ee_force_b[0].cpu().numpy()
            ee_torque_b_np = ee_torque_b[0].cpu().numpy()
            print(f"End-effector 6D Force/Torque - F: [{ee_force_b_np[0]:.3f}, {ee_force_b_np[1]:.3f}, {ee_force_b_np[2]:.3f}], "
                    f"T: [{ee_torque_b_np[0]:.3f}, {ee_torque_b_np[1]:.3f}, {ee_torque_b_np[2]:.3f}] (Unit: N, N·m)")

            # Save 6D force/torque data to CSV file
            with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    count,
                    f"{ee_force_b_np[0]:.6f}",
                    f"{ee_force_b_np[1]:.6f}",
                    f"{ee_force_b_np[2]:.6f}",
                    f"{ee_torque_b_np[0]:.6f}",
                    f"{ee_torque_b_np[1]:.6f}",
                    f"{ee_torque_b_np[2]:.6f}",
                ])
            
            goal_step_count += 1
        # Perform step
        sim.step(render=True)
        # Update robot buffers
        robot.update(sim_dt)
        # Update buffers
        scene.update(sim_dt)
        
        # Update sim-time
        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
