# Copyright (c) 2025.
# 示例：在 Isaac Lab 中生成 1/2/3/4 类定位销结构（定位销 + 导向杆）
#
# 运行方式（在 Isaac Lab 工程根目录）：
#   ./isaaclab.sh -p scripts/tutorials/custom/run_locating_pins_demo.py --nh 4
#
# 其中 --nh 可以为 1 / 2 / 3 / 4，分别对应：
#   1: 仅中心一个定位销
#   2: 两个对称定位销
#   3: 三个构成等边三角形的定位销
#   4: 在 3 的基础上增加中心一个更长的导向杆

import argparse
import math

import numpy as np

from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim import SimulationContext


# ---------- 命令行解析 ----------
parser = argparse.ArgumentParser(description="Demo: spawn locating pin structures (Nh = 1/2/3/4) in Isaac Lab.")
parser.add_argument(
    "--nh",
    type=int,
    choices=[1, 2, 3, 4],
    default=4,
    help="Locating structure type: 1/2/3/4.",
)
# 追加 AppLauncher 的通用参数，例如 --headless, --renderer 等
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------- 几何位置计算函数 ----------

def compute_pin_positions(nh: int, d12: float = 0.04, side_3: float = 0.04, base_z: float = 0.0):
    """根据 Nh = 1/2/3/4 计算每个定位销 / 导向杆的 3D 位置.

    参数:
        nh: 结构类型 1/2/3/4
        d12: Nh=2 时两孔之间的距离 (m)
        side_3: Nh=3/4 时等边三角形边长 (m)
        base_z: 基准 z 高度，一般是撑在基座平面上方一点

    返回:
        pin_positions: list[np.ndarray]，每个为 (3,) 位置，表示定位销位置
        guide_pos: np.ndarray 或 None，Nh=4 时为导向杆位置
    """
    center = np.array([0.0, 0.0, base_z], dtype=float)
    pin_positions = []
    guide_pos = None

    if nh == 1:
        # 单销：在中心
        pin_positions = [center]

    elif nh == 2:
        # 双销：沿 X 方向对称
        offset = np.array([d12 / 2.0, 0.0, 0.0])
        pin_positions = [center + offset, center - offset]

    elif nh in (3, 4):
        # 三销：等边三角形，边长 side_3，对应外接圆半径 R = side_3 / sqrt(3)
        R = side_3 / math.sqrt(3.0)
        angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        for ang in angles:
            x = R * math.cos(ang)
            y = R * math.sin(ang)
            pin_positions.append(center + np.array([x, y, 0.0]))
        if nh == 4:
            # 中心导向杆
            guide_pos = center

    else:
        raise ValueError("nh must be in {1,2,3,4}")

    return pin_positions, guide_pos


# ---------- 场景设计 ----------

def design_scene(nh: int):
    """设计场景：地面 + 光源 + 基座 + Nh 类型的定位销结构."""
    # 1) 地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 2) 平行光
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light, translation=(1.0, 0.0, 10.0))

    # 3) 创建一个 Xform，把所有定位销结构挂在下面
    prim_utils.create_prim("/World/LocatingFixture", "Xform")

    # 4) 基座平板（简单用薄立方体表示）
    base_size = (0.15, 0.15, 0.01)  # (x, y, z)
    base_height = base_size[2]
    cfg_base = sim_utils.CuboidCfg(
        size=base_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic=True),  # 固定不动
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
    )
    cfg_base.func(
        "/World/LocatingFixture/BasePlate",
        cfg_base,
        translation=(0.0, 0.0, base_height / 2.0),
    )

    # 5) 计算定位销 / 导向杆位置（z 坐标放在基座上方）
    pin_base_z = base_height  # 销底部刚好在基座上表面
    pin_positions, guide_pos = compute_pin_positions(
        nh=nh,
        d12=0.04,      # 示例距离，后续可以改为随机采样
        side_3=0.05,   # 示例三角形边长
        base_z=pin_base_z,
    )

    # 6) 定义定位销圆柱几何
    pin_radius = 0.005   # 5 mm
    pin_height = 0.04    # 40 mm
    pin_cfg = sim_utils.CylinderCfg(
        radius=pin_radius,
        height=pin_height,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8)),
    )

    # 7) 依次生成所有定位销
    for i, pos in enumerate(pin_positions):
        prim_path = f"/World/LocatingFixture/Pin_{nh}_{i}"
        # CylinderCfg 的 reference 原点在几何中心，因此 z 方向要抬高一半高度
        spawn_pos = (float(pos[0]), float(pos[1]), float(pos[2] + pin_height / 2.0))
        pin_cfg.func(prim_path, pin_cfg, translation=spawn_pos)

    # 8) 若 Nh=4，则在中心增加一个更长导向杆
    if nh == 4 and guide_pos is not None:
        guide_extra_len = 0.02  # 比定位销多 20 mm
        guide_height = pin_height + guide_extra_len
        guide_cfg = sim_utils.CylinderCfg(
            radius=pin_radius,
            height=guide_height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.1)),
        )
        spawn_pos = (
            float(guide_pos[0]),
            float(guide_pos[1]),
            float(guide_pos[2] + guide_height / 2.0),
        )
        guide_cfg.func("/World/LocatingFixture/GuideRod", guide_cfg, translation=spawn_pos)


# ---------- 主函数 ----------

def main():
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 设置一个合适的观察视角
    # camera_position, camera_target
    sim.set_camera_view(
        eye=[0.3, 0.3, 0.2],
        target=[0.0, 0.0, 0.03],
    )

    # 设计场景
    design_scene(nh=args_cli.nh)

    # 重置仿真
    sim.reset()
    print(f"[INFO] Locating pin demo scene created with Nh = {args_cli.nh}")
    print("[INFO] You should see the locating pins above the square base plate.")

    # 主仿真循环
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
