"""Solve UR5 joint angles for the ORU docking pose via iterative DLS IK.

Target EE pose: pos [0.4, 0, 0.5801], quat [0, 0, 1, 0] (wxyz)
(docked ORU frame Z = 0.1878, measured EE↔ORU offset = -0.39230)

Usage:
    python scripts/tutorials/force/solve_docking_ik.py
"""

import argparse
import math
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
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

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
    sim_dt = sim.get_physics_dt()

    ee_frame_name = "wrist_3_link"
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    ee_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_ids = robot.find_joints(arm_joint_names)[0]

    # ── Target pose ────────────────────────────────────────────────
    target_pos = torch.tensor([[0.4, 0.0, 0.5801]], device=sim.device)
    target_quat = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=sim.device)  # wxyz

    zero_effort = torch.zeros(1, robot.num_joints, device=sim.device)
    robot.set_joint_effort_target(zero_effort)

    # ── Iterative DLS IK ───────────────────────────────────────────
    lam = 0.1
    max_iters = 500
    for it in range(max_iters):
        ee_pose = robot.data.body_pose_w[:, ee_idx]
        ee_pos = ee_pose[:, :3]
        ee_quat = ee_pose[:, 3:7]

        # Pose error
        pos_err = target_pos - ee_pos
        # shortest-path quaternion error → axis-angle
        quat_dot = (target_quat * ee_quat).sum(dim=-1, keepdim=True)
        tq = torch.where(quat_dot >= 0, target_quat, -target_quat)
        q_inv = quat_conjugate(ee_quat)
        q_err = quat_mul(tq, q_inv)
        aa_err = axis_angle_from_quat(q_err)
        delta_pose = torch.cat([pos_err, aa_err], dim=-1)  # (1, 6)

        if delta_pose.abs().max() < 1e-4:
            print(f"[IK] 收敛于迭代 {it}")
            break

        # DLS solve
        jac = robot.root_physx_view.get_jacobians()[:, ee_idx - 1, :, arm_ids]  # (1, 6, 6)
        J = jac[0]  # (6, 6)
        JT = J.T
        lam_mat = (lam**2) * torch.eye(6, device=sim.device)
        delta_q = JT @ torch.linalg.solve(J @ JT + lam_mat, delta_pose[0])  # (6,)

        # Apply joint update + step
        jpos = robot.data.joint_pos.clone()
        jpos[0, arm_ids] += delta_q
        robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos))
        robot.set_joint_effort_target(zero_effort)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)

    # ── Final result ───────────────────────────────────────────────
    ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
    ee_quat = robot.data.body_quat_w[0, ee_idx].cpu().numpy()
    jpos = robot.data.joint_pos[0, arm_ids].cpu().numpy()
    jdeg = jpos * 180.0 / math.pi

    print("=" * 60)
    print(f"EE 最终位置: [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]  (目标 [0.4, 0, 0.5801])")
    print(f"EE 最终姿态: {ee_quat}  (目标 [0, 0, 1, 0])")
    print(f"位置误差: {math.sqrt(((ee_pos - [0.4, 0, 0.5801])**2).sum()):.6f} m")
    print("-" * 60)
    print("目标关节角 (度):")
    for name, deg in zip(arm_joint_names, jdeg):
        print(f"  {name}: {deg:.4f}")
    print("-" * 60)
    print("直接粘贴到配置:")
    print("    \"shoulder_pan_joint\": {:.2f} * math.pi / 180,".format(jdeg[0]))
    print("    \"shoulder_lift_joint\": {:.2f} * math.pi / 180,".format(jdeg[1]))
    print("    \"elbow_joint\": {:.2f} * math.pi / 180,".format(jdeg[2]))
    print("    \"wrist_1_joint\": {:.2f} * math.pi / 180,".format(jdeg[3]))
    print("    \"wrist_2_joint\": {:.2f} * math.pi / 180,".format(jdeg[4]))
    print("    \"wrist_3_joint\": {:.2f} * math.pi / 180,".format(jdeg[5]))
    print("=" * 60)
    print("[INFO] 窗口保持打开, 检查后手动关闭")

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
