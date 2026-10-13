# -*- coding: utf-8 -*-
# Probe: build the env, step 60 times, and report the plate prim world pose + AABB.
import argparse, pathlib, sys
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Isaac-Oru-Direct-v0")
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--out", default="assets/USD/zongzhuang/zongzhuang.usd")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args(); a.headless = True
app = AppLauncher(a).app
import gymnasium as gym, numpy as np
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from pxr import Usd, UsdGeom, Gf

env_cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=a.num_envs)
env_cfg.task.experiment_method = "full"
env = gym.make(a.task, cfg=env_cfg)
env.reset()
for i in range(a.steps):
    import torch
    act = torch.zeros((a.num_envs, env.action_space.shape[-1]), dtype=torch.float32, device="cuda:0")
    env.step(act)
    if i % 20 == 0 or i == a.steps - 1:
        stage = env.unwrapped.sim.stage if hasattr(env.unwrapped, "sim") else None
        st = stage if stage is not None else Usd.Stage.Open(str(pathlib.Path("assets/USD/zongzhuang/zongzhuang.usd")))
        path = "/World/envs/env_0/Zongzhuang"
        prim = st.GetPrimAtPath(path)
        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        m = xc.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = bb.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        print("STEP %3d  prim=%s  origin=(%.4f, %.4f, %.4f)  AABB min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f)"
              % (i, "yes" if prim else "MISSING", t[0], t[1], t[2], mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
                 mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]), flush=True)
env.close(); app.close()
