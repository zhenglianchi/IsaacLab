import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Isaac-Oru-Direct-v0")
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args(); a.headless = True
app = AppLauncher(a).app
import torch, gymnasium as gym, isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=a.num_envs)
env = gym.make(a.task, cfg=cfg)
env.reset()
adim = int(env.unwrapped.cfg.action_space)
for i in range(a.steps):
    act = 0.1 * torch.randn(a.num_envs, adim, device="cuda:0")
    env.step(act)
print("SMOKE_OK action_space=%d steps=%d" % (adim, a.steps), flush=True)
env.close()
app.close()
