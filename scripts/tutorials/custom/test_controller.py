# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# ---------------------------------
# Launch Isaac Sim
# ---------------------------------
parser = argparse.ArgumentParser(description="Differential IK with UR5 (fixed EE orientation)")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------
# Imports
# ---------------------------------
import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets import UR5_CFG

# ======================================================
# Scene configuration
# ======================================================
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ======================================================
# Simulation loop
# ======================================================
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    robot = scene["robot"]

    # Differential IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        ik_method="dls",
        use_relative_mode=False,
    )
    diff_ik = DifferentialIKController(diff_ik_cfg, scene.num_envs, device=sim.device)

    # Markers
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # End-effector goals (positions only, fixed orientation)
    ee_positions = torch.tensor(
        [
            [0.5,  0.4, 0.6],
            [0.5, -0.4, 0.6],
            [0.5,  0.0, 0.5],
        ],
        device=sim.device,
    )
    # Fixed orientation (quaternion)
    fixed_quat = torch.tensor([0, -0.707, 0.707, 0], device=sim.device)

    current_goal = 0
    # Compose full EE goal [pos + quat]
    ik_command = torch.cat(
        [ee_positions[current_goal], fixed_quat]
    ).unsqueeze(0).repeat(scene.num_envs, 1)

    # Robot entity config
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=[".*"],
        body_names=["wrist_3_link"],
    )
    robot_entity_cfg.resolve(scene)

    # Jacobian index
    if robot.is_fixed_base:
        ee_jacobian_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobian_idx = robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    count = 0

    # Initialize joint_pos_des
    joint_pos_des = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids].clone()
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]].clone()

    while simulation_app.is_running():

        if count % 150 == 0:
            count = 0

            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            robot.reset()

            diff_ik.reset()
            diff_ik.set_command(ik_command)

            joint_pos_des = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids].clone()
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]].clone()

            # Switch to next position goal
            current_goal = (current_goal + 1) % len(ee_positions)
            ik_command[:, 0:3] = ee_positions[current_goal].repeat(scene.num_envs, 1)
            ik_command[:, 3:7] = fixed_quat.repeat(scene.num_envs, 1)

        else:
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobian_idx, :, robot_entity_cfg.joint_ids
            ]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )

            joint_pos_des = diff_ik.compute(
                ee_pos_b,
                ee_quat_b,
                jacobian,
                joint_pos,
            )

        robot.set_joint_position_target(
            joint_pos_des,
            joint_ids=robot_entity_cfg.joint_ids,
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        # Markers
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(
            ik_command[:, 0:3] + scene.env_origins,
            ik_command[:, 3:7],
        )


# ======================================================
# Entry
# ======================================================
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.5])

    scene = InteractiveScene(
        TableTopSceneCfg(args_cli.num_envs, env_spacing=2.0)
    )
    sim.reset()

    print("[INFO] UR5 Differential IK ready (fixed EE orientation)")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
