# -*- coding: utf-8 -*-
#Record the C0 baseline (fixed impedance, ZERO action) as an mp4, reusing the same
#rendering path that rl_games play.py uses.
import argparse, os, pathlib
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Oru-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=450)
parser.add_argument("--out", default="outputs/videos/c0_baseline.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import imageio
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
env_cfg.task.experiment_method = "full"
env_cfg.task.switch_mode = "contact"   # C0 = fixed impedance: alpha from the contact logic, actions zero
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
obs, _ = env.reset()
frames = []
frames.append(env.render())
for i in range(args_cli.steps):
    # C0: zero action = fixed impedance, no policy
    action = np.zeros((args_cli.num_envs, env.action_space.shape[-1]), dtype=np.float32)
    obs, rew, term, trunc, info = env.step(action)
    f = env.render()
    if f is not None:
        frames.append(f)
out = pathlib.Path(args_cli.out)
out.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(str(out), frames, fps=15)
print("saved", out, len(frames), "frames")
env.close()
simulation_app.close()
