# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play script with fixed-ik-offset case evaluation + wrench data recording.

Usage:
  python scripts/reinforcement_learning/rl_games/play_force.py \
      --task=Isaac-Oru-Direct-v0 \
      --ik-offset-pos "0.05,0.0,-0.02" \
      --ik-offset-rot "0.0,0.0,0.1"

The CSV is saved to force_data_logs/ and can be plotted with:
  python tools/plot_force_data.py force_data_logs/<file>.csv
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
import csv
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with fixed IK offset + wrench recording.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent config entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_last_checkpoint", action="store_true",
    help="Use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ── Fixed IK offset (single-case evaluation) ──────────────────────
parser.add_argument(
    "--ik-offset-pos", type=str, default=None,
    help="Fixed EE position offset (dx,dy,dz) in meters, e.g. '0.05,0.0,-0.02'",
)
parser.add_argument(
    "--ik-offset-rot", type=str, default=None,
    help="Fixed EE rotation offset (drx,dry,drz) in radians, e.g. '0.0,0.0,0.1'",
)
# ── Max steps per episode ─────────────────────────────────────────
parser.add_argument("--max-steps", type=int, default=500, help="Max steps to record before stopping.")
# ── Baseline checkpoint compatibility: drop wrench obs (43D model) ──
parser.add_argument(
    "--no-wrench-obs", action="store_true",
    help="Remove applied_wrench from obs/state (for baseline checkpoints trained with 43D obs).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_offset(s: str | None) -> tuple[float, ...] | None:
    """Parse 'x,y,z' string into tuple of floats."""
    if s is None:
        return None
    return tuple(float(v.strip()) for v in s.split(","))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with fixed IK offset + wrench recording."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Force single env for evaluation — one scene, one wrench trace
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    # ── Apply fixed IK offset if provided ──────────────────────────
    offset_pos = parse_offset(args_cli.ik_offset_pos)
    offset_rot = parse_offset(args_cli.ik_offset_rot)
    if offset_pos is not None:
        env_cfg.task.fixed_ik_offset_pos = offset_pos
        env_cfg.task.fixed_ik_offset_rot = offset_rot
        print(f"[INFO] Fixed IK offset — pos: {offset_pos}, rot: {offset_rot}")
    else:
        print("[INFO] No fixed IK offset provided — using random domain randomization.")

    # ── Baseline compatibility: drop wrench obs (43D checkpoint) ──
    if args_cli.no_wrench_obs:
        env_cfg.obs_order = [k for k in env_cfg.obs_order if k != "applied_wrench"]
        env_cfg.state_order = [k for k in env_cfg.state_order if k != "applied_wrench"]
        print("[INFO] Dropped applied_wrench from obs/state (baseline 43D compatibility)")

    # ── Find checkpoint ────────────────────────────────────────────
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from: {log_root_path}")

    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # ── Prepare CSV output ─────────────────────────────────────────
    save_dir = "force_data_logs"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"wrench_data_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Step", "Fx(N)", "Fy(N)", "Fz(N)", "Tx(Nm)", "Ty(Nm)", "Tz(Nm)",
        "EE_x", "EE_y", "EE_z",
        "Kp_x", "Kp_y", "Kp_z", "Kp_rx", "Kp_ry", "Kp_rz",
        "Kd_x", "Kd_y", "Kd_z", "Kd_rx", "Kd_ry", "Kd_rz",
    ])
    print(f"[INFO] Wrench data → {csv_path}")

    # ── Create environment ─────────────────────────────────────────
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # ── Load model ─────────────────────────────────────────────────
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # ── Run episode ────────────────────────────────────────────────
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # Single env → env 0 is the only one
    step_count = 0
    env_idx = 0

    # Access the OruEnv through the wrapper chain
    _env = env
    while hasattr(_env, "env"):
        _env = _env.env
    oru_env = _env.unwrapped if hasattr(_env, "unwrapped") else _env

    while simulation_app.is_running() and step_count < args_cli.max_steps:
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)

            # ── Print EE pose + impedance gains ───────────────────────
            ee_pos = oru_env.robot.data.body_pos_w[env_idx, oru_env._ee_frame_idx].cpu().numpy()
            target_z = oru_env.fixed_target_z
            kp = oru_env.task_prop_gains[env_idx].cpu().numpy()
            kd = oru_env.task_deriv_gains[env_idx].cpu().numpy()
            print(f"Step {step_count}: EE=[{ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f}] "
                  f"dZ={ee_pos[2]-target_z:+.4f}")
            print(f"         Kp=[{kp[0]:.1f},{kp[1]:.1f},{kp[2]:.1f} | {kp[3]:.1f},{kp[4]:.1f},{kp[5]:.1f}]")
            print(f"         Kd=[{kd[0]:.1f},{kd[1]:.1f},{kd[2]:.1f} | {kd[3]:.1f},{kd[4]:.1f},{kd[5]:.1f}]")

            # ── Record wrench + EE pose + impedance gains ──────────
            ee_wrench_b = oru_env.robot.data.body_incoming_joint_wrench_b
            f = ee_wrench_b[env_idx, oru_env._ee_frame_idx, :3].cpu().numpy()
            t = ee_wrench_b[env_idx, oru_env._ee_frame_idx, 3:6].cpu().numpy()
            csv_writer.writerow([
                step_count,
                f"{f[0]:.6f}", f"{f[1]:.6f}", f"{f[2]:.6f}",
                f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                f"{ee_pos[0]:.6f}", f"{ee_pos[1]:.6f}", f"{ee_pos[2]:.6f}",
                f"{kp[0]:.4f}", f"{kp[1]:.4f}", f"{kp[2]:.4f}",
                f"{kp[3]:.4f}", f"{kp[4]:.4f}", f"{kp[5]:.4f}",
                f"{kd[0]:.4f}", f"{kd[1]:.4f}", f"{kd[2]:.4f}",
                f"{kd[3]:.4f}", f"{kd[4]:.4f}", f"{kd[5]:.4f}",
            ])

            if len(dones) > 0:
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

        step_count += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    csv_file.close()
    print(f"[DONE] {step_count} steps recorded → {csv_path}")
    print(f"  Plot with: python tools/plot_force_data.py {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
