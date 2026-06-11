# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from SceneCfg import NewRobotsSceneCfg, add_fixed_joint, scene_reset
from pose_controller import UR5Controller


def run_simulator(sim, scene, ur5_ctrl, ee_target_set, sim_dt):
    count = 0
    current_goal_idx = 0
    while simulation_app.is_running():
        # reset every 1500 steps
        if count % 1500 == 0:
            # reset counters
            count = 0
            scene_reset(scene)
            # reset joint state to default
            default_joint_pos = ur5_ctrl.robot.data.default_joint_pos.clone()
            default_joint_vel = ur5_ctrl.robot.data.default_joint_vel.clone()
            ur5_ctrl.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            ur5_ctrl.robot.write_data_to_sim()
            ur5_ctrl.robot.reset()
            scene.reset()
            # reset target pose
            ur5_ctrl.robot.update(sim_dt)
            current_goal_idx = (current_goal_idx + 1) % len(ee_target_set)
            print(f"Moving to target {current_goal_idx}: {ee_target_set[current_goal_idx]}")
        
        # Get current target
        target_pos = ee_target_set[current_goal_idx][:, :3]
        target_quat = ee_target_set[current_goal_idx][:, 3:7]
        
        # Move end-effector to target position using position control
        ur5_ctrl.move_ee_to(target_pos, target_quat)
        
        count += 1


def main():
    """Main function."""
    # Initialize the simulation context with physics settings matching force
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=2,
            max_velocity_iteration_count=2,
        )
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # Design scene - use the same configuration as force
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Now we are ready!
    sim = SimulationContext.instance()
    stage = sim.stage

    add_fixed_joint(stage, args_cli)

    # Play the simulator
    sim.reset()
    scene.reset()
    
    # Initialize controller
    ur5_ctrl = UR5Controller(scene, args_cli)
    
    # ==========================================================
    # 目标位置设置为Ground中心坐标（与main_force.py一致）
    # ==========================================================
    # 获取Ground的世界坐标位置和姿态
    ground_pos = scene["Ground"].data.root_pos_w[0].cpu().numpy()
    ground_quat = scene["Ground"].data.root_quat_w[0].cpu().numpy()
    print(f"Ground position: {ground_pos}")
    print(f"Ground quaternion: {ground_quat}")
    
    # 目标位置：Ground中心上方不同高度
    target_pos = ground_pos.copy()
    target_quat = ground_quat.copy()
    
    # 定义绕 Z 轴 180° 的旋转四元数 (wxyz 格式)
    # 180° = π rad，cos(π/2)=0, sin(π/2)=1，绕 Z 轴为 [w, x, y, z] = [0, 0, 0, 1]
    quat_z_180 = np.array([0.0, 0.0, 0.0, 1.0])

    # 四元数乘法函数 (Hamilton product)
    def quat_multiply(q1, q2):
        """四元数乘法 (wxyz 格式)"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])

    # 计算旋转后的四元数（全局 Z 轴旋转 180°）
    rotated_quat = quat_multiply(quat_z_180, target_quat)
    rotated_quat = rotated_quat / np.linalg.norm(rotated_quat)  # 归一化
    
    # 定义目标位姿序列（与main_force.py相同）
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [target_pos[0], target_pos[1], 0.2,  rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3]],
            [target_pos[0], target_pos[1], 0.25,  rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3]],
            [target_pos[0], target_pos[1], 0.35,  rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3]],
        ],
        device=sim.device,
    )
    
    # Expand for multiple environments
    ee_target_set = [
        pose.unsqueeze(0).repeat(args_cli.num_envs, 1) 
        for pose in ee_goal_pose_set_tilted_b
    ]
    
    # Get simulation dt
    sim_dt = sim.get_physics_dt()
    
    # Run the simulator
    run_simulator(sim, scene, ur5_ctrl, ee_target_set, sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
