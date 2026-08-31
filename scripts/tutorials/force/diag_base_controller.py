"""Diagnostic: does the BASE impedance controller descend straight down?

No policy is used — actions are always ZERO (base gains Kp=100, Kd=20).
The controller pulls the EE to the fixed target (ground XY, Z=0.4295) from
the exact home pose (30cm above, IK offset 0,0,0).

Answers the question: "does going down really require drifting outward?"
  - If EE descends straight (x stays ~0.4): the controller/physics are fine,
    the failure is in the trained policy/reward.
  - If EE drifts +X significantly: the controller or scene has a bug.

Usage:
  python scripts/tutorials/force/diag_base_controller.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action base controller diagnostic.")
parser.add_argument("--task", type=str, default="Isaac-Oru-Direct-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=1600, help="Sim steps (1 step = 1/15 s).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    # Exact home pose — start straight above the target, no randomization
    env_cfg.task.fixed_ik_offset_pos = (0.0, 0.0, 0.0)
    env_cfg.task.fixed_ik_offset_rot = (0.0, 0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped: DirectRLEnv = env.unwrapped

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    ee_idx = env_unwrapped._ee_frame_idx
    target_z = env_unwrapped.fixed_target_z
    target_xy = env_unwrapped.ground.data.root_pos_w[0, :2].cpu().numpy()

    action = torch.zeros(env_unwrapped.num_envs, env_unwrapped.cfg.action_space, device=env_unwrapped.device)

    print(f"[INFO] Target: XY=({target_xy[0]:.4f},{target_xy[1]:.4f}) Z={target_z:.4f}")
    print(f"[INFO] Start EE: {env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()}")

    for step in range(args_cli.steps):
        obs, rew, terminated, truncated, info = env.step(action)

        if step % 20 == 0 or step == args_cli.steps - 1:
            ee = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
            F = env_unwrapped.applied_wrench[0, :3].cpu().numpy()          # commanded task force
            Fm = env_unwrapped._get_measured_force_mag()[0].item()         # measured wrist wrench
            cd = env_unwrapped._get_contact_degree()[0].item()
            print(
                f"Step {step:5d} | EE=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f}) "
                f"dX={ee[0]-target_xy[0]:+.4f} dZ={ee[2]-target_z:+.4f} | "
                f"F_cmd=({F[0]:+6.2f},{F[1]:+6.2f},{F[2]:+6.2f})N "
                f"F_meas={Fm:6.2f}N contact={cd:4.2f}"
            )

    ee = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
    print(f"\n[RESULT] Final EE: ({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}) "
          f"-> dX={ee[0]-target_xy[0]:+.4f}, dZ={ee[2]-target_z:+.4f}")
    print("[RESULT] Straight descent = |dX| stays ~0. Outward drift = |dX| grows >> 0.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
