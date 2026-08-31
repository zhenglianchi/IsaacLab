# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""交互式拖拽场景: 手动把 ORU 拖进 Ground 对接槽, 读出目标位姿.

物理不启动, 资产以纯 USD 形式加载:
  - Ground 放在 (0.4, 0, 0.05), 绕 Y 轴 180°
  - ORU 起始放在对接槽上方 (0.4, 0, 0.35)
  - 选中 ORU 用 W(平移)/E(旋转) gizmo 拖动
  - 每 2 秒打印 ORU 当前世界位姿 (pos + quat wxyz)

Usage:
    python scripts/tutorials/force/interactive_scene.py
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive ORU drag scene (no physics).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from pxr import Gf, Usd, UsdGeom

from force1_SceneCfg import GROUND_CFG, ORU_CFG


@configclass
class DragSceneCfg(InteractiveSceneCfg):
    """只有 ORU + Ground 的最小场景 (无机器人, 无物理)."""

    ground_plane = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000, color=(0.75, 0.75, 0.75)),
    )
    ORU = ORU_CFG.replace(prim_path="{ENV_REGEX_NS}/ORU")
    Ground = GROUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Ground")


def set_prim_transform(stage: Usd.Stage, prim_path: str, pos, quat_wxyz):
    """Set the local transform of a prim in the stage (no physics)."""
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    # quat_wxyz = (w, x, y, z) → Gf.Quatd(real, i, j, k)
    w, x, y, z = quat_wxyz
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), Gf.Vec3d(x, y, z)))


def read_prim_transform(stage: Usd.Stage, prim_path: str):
    """Read world transform of a prim from USD."""
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    r = m.ExtractRotationQuat()  # (real, i, j, k) = (w, x, y, z)
    return (t[0], t[1], t[2]), (r.GetReal(), r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))

    scene_cfg = DragSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim = SimulationContext.instance()
    stage = sim.stage

    # 直接在 USD 层放置 Ground 和 ORU (不启动物理)
    ns = "/World/envs/env_0"
    set_prim_transform(stage, f"{ns}/Ground", (0.4, 0.0, 0.05), (0.0, 0.0, 1.0, 0.0))  # 绕Y 180°
    set_prim_transform(stage, f"{ns}/ORU", (0.4, 0.0, 0.35), (1.0, 0.0, 0.0, 0.0))     # 起始悬于槽上方

    print("=" * 60)
    print("[INFO] 物理未启动 — 可直接用 gizmo 拖动:")
    print("  1. 选中 /World/envs/env_0/ORU")
    print("  2. W = 平移, E = 旋转, 把 ORU 拖进 Ground 对接槽")
    print("  3. 本终端每 2 秒打印 ORU 当前世界位姿")
    print("  4. 对准后记录最后打印的 pos + quat")
    print("=" * 60)

    last_print = time.time()
    last_pose = None
    while simulation_app.is_running():
        sim.render()
        time.sleep(0.02)

        if time.time() - last_print > 2.0:
            pos, quat = read_prim_transform(stage, f"{ns}/ORU")
            changed = "" if (pos, quat) == last_pose else " ← 有变化"
            print(f"ORU pos={[round(p, 4) for p in pos]}  quat(wxyz)={[round(q, 4) for q in quat]}{changed}")
            last_pose = (pos, quat)
            last_print = time.time()

    simulation_app.close()


if __name__ == "__main__":
    main()
