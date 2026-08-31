"""Measure the Z offset between UR5 wrist_3_link (EE) and ORU base_link.

The FixedJoint chain is rigid, so (ORU_Z - EE_Z) is constant for any joint
configuration. Given the ORU docked height Z=0.08265, the EE target height
is:  EE_Z = 0.08265 - offset.

Usage:
    python scripts/tutorials/force/measure_oru_offset.py
"""

import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from force1_SceneCfg import NewRobotsSceneCfg, add_fixed_joint


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    scene = InteractiveScene(NewRobotsSceneCfg(1, env_spacing=2.0))
    sim = SimulationContext.instance()
    stage = sim.stage

    add_fixed_joint(stage, args_cli)

    sim.reset()
    scene.reset()

    robot = scene["Dofbot"]
    oru = scene["ORU"]

    # Settle 1 second (120 steps): FREEZE arm joints every step so the
    # FixedJoint chain settles at its equilibrium while the arm stays put.
    sim_dt = sim.get_physics_dt()
    default_jpos = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(default_jpos)
    zero_effort = torch.zeros(1, robot.num_joints, device=sim.device)
    robot.set_joint_effort_target(zero_effort)
    for _ in range(120):
        # Re-write frozen joint state each step — holds the arm in place
        robot.write_joint_state_to_sim(default_jpos, zero_vel)
        robot.set_joint_effort_target(zero_effort)
        scene.write_data_to_sim()
        sim.step(render=True)   # render so user can visually verify the joint chain
        scene.update(sim_dt)

    ee_frame_name = "wrist_3_link"
    ee_idx = robot.find_bodies(ee_frame_name)[0][0]

    ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
    oru_pos = oru.data.root_pos_w[0].cpu().numpy()

    offset = oru_pos[2] - ee_pos[2]

    print("=" * 60)
    print(f"EE  (wrist_3_link) Z = {ee_pos[2]:.5f}")
    print(f"ORU (base_link)    Z = {oru_pos[2]:.5f}")
    print(f"Offset (ORU_Z - EE_Z) = {offset:+.5f}")
    print("-" * 60)
    docked_oru_z = 0.1878   # ORU coordinate frame height at docking (new ORU USD)
    print(f"给定 ORU 对接高度 Z = {docked_oru_z}")
    print(f"对应 EE 目标高度 = {docked_oru_z} - ({offset:+.5f}) = {docked_oru_z - offset:.5f}")
    print("=" * 60)
    print("[INFO] 窗口保持打开, 目视检查关节链后手动关闭即可")

    # Keep rendering until the user closes the viewport window
    while simulation_app.is_running():
        robot.write_joint_state_to_sim(default_jpos, zero_vel)
        robot.set_joint_effort_target(zero_effort)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
