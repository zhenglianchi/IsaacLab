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
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim import SimulationContext

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
    """设计场景：地面 + 光源 + 对接面（带孔，在下方） + 装配体（带定位销，在上方）."""
    # 1) 地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 2) 平行光
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light, translation=(1.0, 0.0, 10.0))

    # 3) 定义几何参数
    pin_radius = 0.005   # 5 mm
    pin_height = 0.01    # 10 mm（普通定位销长度）
    
    # 计算定位销/孔洞位置
    pin_positions, guide_pos = compute_pin_positions(
        nh=nh,
        d12=0.04,      # 示例距离
        side_3=0.05,   # 示例三角形边长
        base_z=0.0,
    )
    
    # 4) 创建对接面（带孔，在地面，孔朝上）- 固定不动
    # 导向杆比定位销长 0.02，总长度为 0.03
    guide_extra_len = 0.02
    guide_height = pin_height + guide_extra_len  # 导向杆长度 = 0.03
    create_docking_surface(nh, pin_positions, guide_pos, pin_radius, pin_height, guide_height)
    
    # 5) 创建装配体（带定位销，在对接面上方，定位销朝下）- 可移动
    create_assembly_body(nh, pin_positions, guide_pos, pin_radius, pin_height)


def create_docking_surface(nh: int, pin_positions: list, guide_pos: np.ndarray, pin_radius: float, pin_height: float, guide_height: float):
    """创建对接面（带孔洞的平板，固定在地面，孔洞朝上）.
    
    参数:
        nh: 结构类型 1/2/3/4
        pin_positions: 定位销位置列表
        guide_pos: 导向杆位置（Nh=4 时）
        pin_radius: 定位销半径
        pin_height: 定位销高度（对应孔洞深度）
        guide_height: 导向杆高度（对应导向孔洞深度）
    """
    import trimesh
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    
    # 对接面尺寸
    surface_size = (0.12, 0.12, 0.03)  # (x, y, z)
    surface_height = surface_size[2]
    
    # 孔洞半径（比定位销半径稍大，便于插入）
    hole_radius = pin_radius * 1.1  # 孔洞半径是定位销半径的1.1倍
    
    # ========== 使用 trimesh 创建真正带孔洞的网格 ==========
    
    # 1. 创建基础盒子（中心在原点）
    box_mesh = trimesh.creation.box(surface_size)
    
    # 2. 创建定位销孔洞（深度 = pin_height）
    for hole_pos in pin_positions:
        # 创建圆柱体（沿Z轴）
        hole_mesh = trimesh.creation.cylinder(radius=hole_radius, height=pin_height)
        # 圆柱体位置：中心在上表面，向下穿透 pin_height
        # 盒子范围：[-size/2, size/2]，上表面在 z = surface_height/2
        # 圆柱体中心 z = surface_height/2 - pin_height/2，这样圆柱体从 z = surface_height/2 - pin_height 到 z = surface_height/2
        hole_center_z = surface_height / 2.0 - pin_height / 2.0
        hole_mesh.apply_translation([hole_pos[0], hole_pos[1], hole_center_z])
        # 布尔差运算
        box_mesh = trimesh.boolean.difference([box_mesh, hole_mesh])
    
    # 3. 创建导向杆孔洞（深度 = guide_height，比定位销深）
    if nh == 4 and guide_pos is not None:
        # 创建圆柱体（沿Z轴）
        hole_mesh = trimesh.creation.cylinder(radius=hole_radius, height=guide_height)
        # 圆柱体位置：中心在上表面，向下穿透 guide_height
        hole_center_z = surface_height / 2.0 - guide_height / 2.0
        hole_mesh.apply_translation([guide_pos[0], guide_pos[1], hole_center_z])
        # 布尔差运算
        box_mesh = trimesh.boolean.difference([box_mesh, hole_mesh])
    
    # 4. 确保法线朝外
    box_mesh.fix_normals()
    
    # ========== 在 USD 中创建对接面 ==========
    
    # 获取当前 stage
    stage = omni.usd.get_context().get_stage()
    
    # 创建对接面根节点（底部接触地面）
    # box_mesh中心在原点，底部在 z = -surface_height/2，需要向上移动 surface_height/2
    root_pos = (0.0, 0.0, surface_height / 2.0)
    root_prim = prim_utils.create_prim("/World/DockingSurface", "Xform", translation=root_pos)
    
    # 设置为固定刚体（不可移动）
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    root_prim.GetAttribute("physics:kinematicEnabled").Set(True)
    
    # 创建 mesh prim
    mesh_prim = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/DockingSurface/mesh"))
    mesh_prim.GetPointsAttr().Set(box_mesh.vertices.tolist())
    mesh_prim.GetFaceVertexIndicesAttr().Set(box_mesh.faces.flatten().tolist())
    mesh_prim.GetFaceVertexCountsAttr().Set([len(f) for f in box_mesh.faces])
    
    # 设置碰撞属性（使用精确网格碰撞）
    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
    mesh_collision_api.GetApproximationAttr().Set("none")
    
    # 设置材质（绿色）
    material_path = "/World/DockingSurface/material"
    mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.3))
    mat_cfg.func(material_path, mat_cfg)
    from isaaclab.sim.utils import bind_visual_material
    bind_visual_material("/World/DockingSurface/mesh", material_path)
    
    # 启用双面渲染
    mesh_prim.GetDoubleSidedAttr().Set(True)


def create_assembly_body(nh: int, pin_positions: list, guide_pos: np.ndarray, pin_radius: float, pin_height: float):
    """创建装配体（带定位销，在对接面上方，定位销朝下指向孔洞）.
    
    参数:
        nh: 结构类型 1/2/3/4
        pin_positions: 定位销位置列表
        guide_pos: 导向杆位置（Nh=4 时）
        pin_radius: 定位销半径
        pin_height: 定位销高度
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    
    # 基座平板尺寸
    base_size = (0.15, 0.15, 0.06)
    base_height = base_size[2]
    
    # 对接面尺寸
    surface_height = 0.03
    
    # 装配体整体高度（从对接面到装配体顶部）
    # 对接面厚度 + 间隙 + 定位销高度 + 基座厚度
    total_gap = 0.6  # 非常大的间隙，确保能清楚看到对接面的孔洞
    assembly_z = surface_height + total_gap + pin_height + base_height / 2.0
    
    # 获取当前 stage
    stage = omni.usd.get_context().get_stage()
    
    # ========== 创建装配体根节点 ==========
    root_prim = prim_utils.create_prim("/World/AssemblyBody", "Xform", translation=(0.0, 0.0, assembly_z))
    
    # 设置为可移动刚体
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.GetMassAttr().Set(0.5)  # 设置质量
    
    # ========== 创建基座平板（在上方） ==========
    base_cfg = sim_utils.CuboidCfg(
        size=base_size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
    )
    base_cfg.func(
        "/World/AssemblyBody/BasePlate",
        base_cfg,
        translation=(0.0, 0.0, 0.0),  # 相对于根节点居中
    )
    
    # ========== 创建定位销（在基座下方，朝下指向孔洞） ==========
    # 定位销底部位置（相对于基座底部朝下）
    pin_base_z = -base_height / 2.0
    
    pin_cfg = sim_utils.CylinderCfg(
        radius=pin_radius,
        height=pin_height,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8)),
    )
    
    for i, pos in enumerate(pin_positions):
        prim_path = f"/World/AssemblyBody/Pin_{nh}_{i}"
        # 定位销朝下，所以 z 坐标为负方向
        spawn_pos = (float(pos[0]), float(pos[1]), float(pin_base_z - pin_height / 2.0))
        pin_cfg.func(prim_path, pin_cfg, translation=spawn_pos)
    
    # ========== 创建导向杆（Nh=4 时，在基座下方，朝下） ==========
    if nh == 4 and guide_pos is not None:
        guide_extra_len = 0.02  # 比定位销多 20 mm
        guide_height = pin_height + guide_extra_len
        guide_cfg = sim_utils.CylinderCfg(
            radius=pin_radius,
            height=guide_height,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.1)),
        )
        spawn_pos = (
            float(guide_pos[0]),
            float(guide_pos[1]),
            float(pin_base_z - guide_height / 2.0),
        )
        guide_cfg.func("/World/AssemblyBody/GuideRod", guide_cfg, translation=spawn_pos)


# ---------- 主函数 ----------

def main():
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 设置一个合适的观察视角
    # camera_position, camera_target
    sim.set_camera_view(
        eye=[0.3, 0.3, 0.3],
        target=[0.0, 0.0, 0.05],
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
