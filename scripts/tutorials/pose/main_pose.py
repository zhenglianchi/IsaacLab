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
            count = 0
            scene_reset(scene)

            # update robot buffers after reset
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
    # Initialize the simulation context
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

    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim = SimulationContext.instance()
    stage = sim.stage

    add_fixed_joint(stage, args_cli)

    # Play the simulator
    sim.reset()
    scene.reset()

    # Initialize controller (pass sim explicitly for physics stepping)
    ur5_ctrl = UR5Controller(scene, sim, args_cli)

    # ==========================================================
    # 目标位姿与 force1 一致
    # ==========================================================
    ee_goal_pose_set = torch.tensor(
        [
            [0.4, 0, 0.4,  0, 0, 1, 0],
        ],
        device=sim.device,
    )

    # Expand for multiple environments
    ee_target_set = [
        pose.unsqueeze(0).repeat(args_cli.num_envs, 1)
        for pose in ee_goal_pose_set
    ]

    sim_dt = sim.get_physics_dt()

    # Run the simulator
    run_simulator(sim, scene, ur5_ctrl, ee_target_set, sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
