"""Diagnostic: pure-Z descent under the base impedance controller.

The XY target is FROZEN at the EE's starting position — the controller only
commands motion along Z (straight down toward the ORU). Shows:

  - does the arm press straight down, or does it bow outward?
    (dXY tracks any lateral drift the impedance must fight)
  - the commanded force / measured wrist wrench / contact state

No policy — actions are always ZERO (base gains Kp=100, Kd=20).
Requires the x8 tensors compensation (oru_control.py).

Usage:
  python scripts/tutorials/force/diag_pure_z.py            # GUI
  python scripts/tutorials/force/diag_pure_z.py --headless --steps 200
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pure-Z descent diagnostic.")
parser.add_argument("--task", type=str, default="Isaac-Oru-Direct-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=200, help="Sim steps (1 step = 1/15 s).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--kp_xy", type=float, default=None,
                    help="Override XY proportional gain (default: env base). Kd_xy = 2*sqrt(Kp_xy).")
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

    # Optional XY stiffness override (fight the inertial bow harder)
    if args_cli.kp_xy is not None:
        kp = args_cli.kp_xy
        kd = 2.0 * kp ** 0.5  # critical damping
        base_g = env_unwrapped.base_gains.clone()
        base_g[0, 0:2] = kp
        env_unwrapped.base_gains = base_g
        base_d = env_unwrapped.base_deriv.clone()
        base_d[0, 0:2] = kd
        env_unwrapped.base_deriv = base_d
        print(f"[INFO] XY stiffness override: Kp_xy={kp}, Kd_xy={kd:.1f} (Z unchanged)")

    # FROZEN XY target = the EE's starting XY (pure-Z command, no XY alignment)
    ee_start = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy().copy()
    target_xy = ee_start[:2]
    print(f"[INFO] Start EE: {ee_start}")
    print(f"[INFO] Target: XY frozen at ({target_xy[0]:.4f},{target_xy[1]:.4f}) Z={target_z:.4f}")
    print(f"[INFO] Pure-Z command: only the Z error drives the wrench.")

    action = torch.zeros(env_unwrapped.num_envs, env_unwrapped.cfg.action_space, device=env_unwrapped.device)

    for step in range(args_cli.steps):
        obs, rew, terminated, truncated, info = env.step(action)

        if step % 10 == 0 or step == args_cli.steps - 1:
            ee = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
            F = env_unwrapped.applied_wrench[0, :3].cpu().numpy()          # commanded task force
            Fm = env_unwrapped._get_measured_force_mag()[0].item()         # measured wrist wrench
            cd = env_unwrapped._get_contact_degree()[0].item()
            dxy = ((ee[0]-target_xy[0])**2 + (ee[1]-target_xy[1])**2) ** 0.5
            print(
                f"Step {step:5d} | EE=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f}) "
                f"dXY={dxy:+.4f} dZ={ee[2]-target_z:+.4f} | "
                f"F_cmd=({F[0]:+6.2f},{F[1]:+6.2f},{F[2]:+6.2f})N "
                f"F_meas={Fm:6.2f}N contact={cd:4.2f}"
            )

    ee = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
    dxy = ((ee[0]-target_xy[0])**2 + (ee[1]-target_xy[1])**2) ** 0.5
    print(f"\n[RESULT] Final EE: ({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}) "
          f"-> dXY={dxy:+.4f}, dZ={ee[2]-target_z:+.4f}")
    print("[RESULT] Small dXY = the arm presses straight down (no outward bow).")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
