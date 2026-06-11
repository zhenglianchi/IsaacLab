import math
import torch
import numpy as np
import trimesh
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import UsdPhysics, Gf, Sdf
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import schemas
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    ConvexDecompositionPropertiesCfg,
)

# =====================================================
# Quaternion Utils
# =====================================================

def quat_from_axis_angle(axis, angle):
    half = angle * 0.5
    s = math.sin(half)

    return (
        math.cos(half),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
    )


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


# =====================================================
# Assembly Body Configuration (replaces ORU)
# =====================================================

def compute_pin_positions(nh: int, d12: float, side_3: float, base_z: float = 0.0):
    """Compute pin positions based on structure type.
    
    Args:
        nh: Structure type (1-4)
        d12: Distance between pins 1 and 2
        side_3: Side length for triangular configurations
        base_z: Base z-coordinate
    
    Returns:
        pin_positions: List of pin positions
        guide_pos: Guide pin position (only for nh=4)
    """
    pin_positions = []
    guide_pos = None
    
    if nh == 1:
        pin_positions = [(0.0, 0.0, base_z)]
    elif nh == 2:
        pin_positions = [(-d12/2, 0.0, base_z), (d12/2, 0.0, base_z)]
    elif nh == 3:
        pin_positions = [
            (-side_3/2, -side_3/(2*math.sqrt(3)), base_z),
            (side_3/2, -side_3/(2*math.sqrt(3)), base_z),
            (0.0, side_3/math.sqrt(3), base_z)
        ]
    elif nh == 4:
        pin_positions = [
            (-d12/2, -d12/2, base_z),
            (-d12/2, d12/2, base_z),
            (d12/2, -d12/2, base_z),
            (d12/2, d12/2, base_z)
        ]
        guide_pos = (0.0, 0.0, base_z)
    
    return pin_positions, guide_pos


def create_assembly_body(prim_path: str, nh: int = 4):
    """Create assembly body with locating pins (replaces ORU).
    
    Args:
        prim_path: USD prim path
        nh: Structure type (1-4)
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    
    stage = omni.usd.get_context().get_stage()
    
    # Parameters
    pin_radius = 0.005   # 5 mm
    pin_height = 0.01    # 10 mm
    base_size = (0.12, 0.12, 0.03)  # (x, y, z)
    guide_extra_len = 0.02
    guide_height = pin_height + guide_extra_len  # 0.03 m
    
    # Compute pin positions
    pin_positions, guide_pos = compute_pin_positions(
        nh=nh,
        d12=0.04,
        side_3=0.05,
        base_z=0.0,
    )
    
    # Create root Xform
    root_prim = UsdGeom.Xform.Define(stage, prim_path)
    
    # Create base box
    base_mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/base_link")
    
    # Create box vertices and faces
    half_x, half_y, half_z = base_size[0]/2, base_size[1]/2, base_size[2]/2
    vertices = [
        (-half_x, -half_y, -half_z), (half_x, -half_y, -half_z),
        (half_x, half_y, -half_z), (-half_x, half_y, -half_z),
        (-half_x, -half_y, half_z), (half_x, -half_y, half_z),
        (half_x, half_y, half_z), (-half_x, half_y, half_z)
    ]
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7]   # left
    ]
    
    flat_faces = []
    for face in faces:
        flat_faces.extend(face)
    
    base_mesh.GetPointsAttr().Set(vertices)
    base_mesh.GetFaceVertexIndicesAttr().Set(flat_faces)
    base_mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
    
    # Add pins
    for i, pos in enumerate(pin_positions):
        pin_mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/pin_{i}")
        
        # Create cylinder mesh for pin
        num_segments = 16
        pin_vertices = []
        pin_faces = []
        
        for j in range(num_segments):
            angle = 2 * math.pi * j / num_segments
            x = pos[0] + pin_radius * math.cos(angle)
            y = pos[1] + pin_radius * math.sin(angle)
            pin_vertices.append((x, y, half_z))           # bottom
            pin_vertices.append((x, y, half_z + pin_height))  # top
        
        # Side faces
        for j in range(num_segments):
            bottom_idx = j * 2
            top_idx = j * 2 + 1
            next_bottom_idx = ((j + 1) % num_segments) * 2
            next_top_idx = ((j + 1) % num_segments) * 2 + 1
            pin_faces.append([bottom_idx, top_idx, next_top_idx])
            pin_faces.append([bottom_idx, next_top_idx, next_bottom_idx])
        
        # Cap faces
        bottom_center_idx = len(pin_vertices)
        top_center_idx = bottom_center_idx + 1
        pin_vertices.append((pos[0], pos[1], half_z))
        pin_vertices.append((pos[0], pos[1], half_z + pin_height))
        
        for j in range(num_segments):
            bottom_idx = j * 2
            next_bottom_idx = ((j + 1) % num_segments) * 2
            pin_faces.append([bottom_center_idx, bottom_idx, next_bottom_idx])
            
            top_idx = j * 2 + 1
            next_top_idx = ((j + 1) % num_segments) * 2 + 1
            pin_faces.append([top_center_idx, next_top_idx, top_idx])
        
        pin_flat_faces = []
        for face in pin_faces:
            pin_flat_faces.extend(face)
        
        pin_mesh.GetPointsAttr().Set(pin_vertices)
        pin_mesh.GetFaceVertexIndicesAttr().Set(pin_flat_faces)
        pin_mesh.GetFaceVertexCountsAttr().Set([3] * len(pin_faces))
    
    # Add guide pin (for nh=4)
    if nh == 4 and guide_pos is not None:
        guide_mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/guide_pin")
        
        num_segments = 16
        guide_vertices = []
        guide_faces = []
        
        for j in range(num_segments):
            angle = 2 * math.pi * j / num_segments
            x = guide_pos[0] + pin_radius * math.cos(angle)
            y = guide_pos[1] + pin_radius * math.sin(angle)
            guide_vertices.append((x, y, half_z))              # bottom
            guide_vertices.append((x, y, half_z + guide_height))  # top
        
        # Side faces
        for j in range(num_segments):
            bottom_idx = j * 2
            top_idx = j * 2 + 1
            next_bottom_idx = ((j + 1) % num_segments) * 2
            next_top_idx = ((j + 1) % num_segments) * 2 + 1
            guide_faces.append([bottom_idx, top_idx, next_top_idx])
            guide_faces.append([bottom_idx, next_top_idx, next_bottom_idx])
        
        # Cap faces
        bottom_center_idx = len(guide_vertices)
        top_center_idx = bottom_center_idx + 1
        guide_vertices.append((guide_pos[0], guide_pos[1], half_z))
        guide_vertices.append((guide_pos[0], guide_pos[1], half_z + guide_height))
        
        for j in range(num_segments):
            bottom_idx = j * 2
            next_bottom_idx = ((j + 1) % num_segments) * 2
            guide_faces.append([bottom_center_idx, bottom_idx, next_bottom_idx])
            
            top_idx = j * 2 + 1
            next_top_idx = ((j + 1) % num_segments) * 2 + 1
            guide_faces.append([top_center_idx, next_top_idx, top_idx])
        
        guide_flat_faces = []
        for face in guide_faces:
            guide_flat_faces.extend(face)
        
        guide_mesh.GetPointsAttr().Set(guide_vertices)
        guide_mesh.GetFaceVertexIndicesAttr().Set(guide_flat_faces)
        guide_mesh.GetFaceVertexCountsAttr().Set([3] * len(guide_faces))
    
    # Set rigid body properties
    UsdPhysics.RigidBodyAPI.Apply(root_prim.GetPrim())
    root_prim.GetPrim().GetAttribute("physics:kinematicEnabled").Set(True)
    
    # Set collision properties for base
    UsdPhysics.CollisionAPI.Apply(base_mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(base_mesh.GetPrim())
    
    # Apply material
    material_path = f"{prim_path}/material"
    from isaaclab.sim.utils import bind_visual_material
    mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2))
    mat_cfg.func(material_path, mat_cfg)
    
    for child_prim in root_prim.GetPrim().GetAllChildren():
        bind_visual_material(child_prim.GetPath().pathString, material_path)


def create_docking_surface(prim_path: str, nh: int = 4):
    """Create docking surface with holes (replaces Ground).
    
    Args:
        prim_path: USD prim path
        nh: Structure type (1-4)
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    
    stage = omni.usd.get_context().get_stage()
    
    # Parameters
    pin_radius = 0.005   # 5 mm
    pin_height = 0.01    # 10 mm
    hole_radius = pin_radius * 1.3  # Slightly larger than pin
    surface_size = (0.12, 0.12, 0.03)  # (x, y, z)
    surface_height = surface_size[2]
    guide_extra_len = 0.02
    guide_height = pin_height + guide_extra_len  # 0.03 m
    
    # Compute pin positions (same as assembly)
    pin_positions, guide_pos = compute_pin_positions(
        nh=nh,
        d12=0.04,
        side_3=0.05,
        base_z=0.0,
    )
    
    # Create root Xform
    root_prim = UsdGeom.Xform.Define(stage, prim_path)
    
    # Create box mesh with holes using trimesh
    box_mesh = trimesh.creation.box(surface_size)
    
    # Create holes (depth = pin_height)
    for hole_pos in pin_positions:
        hole_trimesh = trimesh.creation.cylinder(radius=hole_radius, height=pin_height)
        hole_center_z = surface_height / 2.0 - pin_height / 2.0
        hole_trimesh.apply_translation([hole_pos[0], hole_pos[1], hole_center_z])
        box_mesh = trimesh.boolean.difference([box_mesh, hole_trimesh])
    
    # Create guide hole (depth = guide_height)
    if nh == 4 and guide_pos is not None:
        hole_trimesh = trimesh.creation.cylinder(radius=hole_radius, height=guide_height)
        hole_center_z = surface_height / 2.0 - guide_height / 2.0
        hole_trimesh.apply_translation([guide_pos[0], guide_pos[1], hole_center_z])
        box_mesh = trimesh.boolean.difference([box_mesh, hole_trimesh])
    
    box_mesh.fix_normals()
    
    # Create mesh prim
    mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/mesh")
    mesh_prim.GetPointsAttr().Set(box_mesh.vertices.tolist())
    mesh_prim.GetFaceVertexIndicesAttr().Set(box_mesh.faces.flatten().tolist())
    mesh_prim.GetFaceVertexCountsAttr().Set([len(f) for f in box_mesh.faces])
    
    # Set rigid body properties (fixed)
    UsdPhysics.RigidBodyAPI.Apply(root_prim.GetPrim())
    root_prim.GetPrim().GetAttribute("physics:kinematicEnabled").Set(True)
    
    # Set collision properties
    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim.GetPrim())
    mesh_collision_api.GetApproximationAttr().Set("none")
    
    # Apply material
    material_path = f"{prim_path}/material"
    from isaaclab.sim.utils import bind_visual_material
    mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.3))
    mat_cfg.func(material_path, mat_cfg)
    bind_visual_material(mesh_prim.GetPath().pathString, material_path)


# =====================================================
# Assets Config
# =====================================================

BRIDGE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/bridge/bridge.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0, 0, 0.0),
        rot=(1, 0, 0, 0),
    ),
)


FORCE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/force/force.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
    ),
)


GRIPPER_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/gripper/gripper.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
    ),
)


# =====================================================
# UR5 - Force Control Configuration (stiffness=0, damping=0)
# =====================================================

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5/ur5.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -math.pi / 2,
            "elbow_joint": math.pi / 6,
            "wrist_1_joint": -math.pi / 6,
            "wrist_2_joint": -math.pi / 2,
            "wrist_3_joint": math.pi / 2,
        },
        pos=(0, 0, 0),
    ),
    actuators={
        "shoulder_pan_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=0,
            damping=0,
        ),
        "shoulder_lift_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=0,
            damping=0,
        ),
        "elbow_joint": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=0,
            damping=0,
        ),
        "wrist_1_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=0,
            damping=0,
        ),
        "wrist_2_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_2_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=0,
            damping=0,
        ),
        "wrist_3_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_3_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=0,
            damping=0,
        ),
    },
)


# =====================================================
# Scene - Assembly and Docking Surface Scene
# =====================================================

class RandSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000,
            color=(0.75, 0.75, 0.75)
        )
    )

    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")

    Froce_Six = FORCE_CFG.replace(prim_path="{ENV_REGEX_NS}/SixForce")

    Gripper = GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Gripper")

    Bridge = BRIDGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Bridge")


def scene_reset(scene):

    root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
    root_dofbot_state[:, :3] += scene.env_origins

    scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
    scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

    joint_pos, joint_vel = (
        scene["Dofbot"].data.default_joint_pos.clone(),
        scene["Dofbot"].data.default_joint_vel.clone(),
    )

    scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
    scene["Dofbot"].set_joint_effort_target(
        torch.zeros_like(joint_pos)
    )

    scene.write_data_to_sim()


# =====================================================
# Fixed Joint
# =====================================================

def create_fixed_joint(
    stage,
    joint_path,
    parent_path,
    child_path,
    *,
    child_offset_pos=(0, 0, 0),
    child_offset_quat=None,
    child_offset_axis=(0, 0, 1),
    child_offset_angle=0.0,
):

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    joint.CreateBody0Rel().SetTargets([parent_path])
    joint.CreateBody1Rel().SetTargets([child_path])

    # parent frame
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1))

    # child pos
    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(*map(float, child_offset_pos))
    )

    # child rot
    if child_offset_quat is not None:

        w, x, y, z = child_offset_quat

        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(float(w), Gf.Vec3f(x, y, z))
        )

    else:

        ax = Gf.Vec3f(*map(float, child_offset_axis))

        if ax.GetLength() > 0 and abs(child_offset_angle) > 0:

            ax = ax.GetNormalized()
            half = child_offset_angle * 0.5

            qw = math.cos(half)
            qv = ax * math.sin(half)

            joint.CreateLocalRot1Attr().Set(Gf.Quatf(qw, qv))

        else:
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))

    joint.CreateBreakForceAttr().Set(1e10)
    joint.CreateBreakTorqueAttr().Set(1e10)


# =====================================================
# Add Joints
# =====================================================

def add_fixed_joint(stage, args_cli):

    for env_idx in range(args_cli.num_envs):

        env_ns = f"/World/envs/env_{env_idx}"

        # flange -> bridge
        create_fixed_joint(
            stage,
            f"{env_ns}/Dofbot/wrist_3_link/bridge_joint",
            f"{env_ns}/Dofbot/wrist_3_link",
            f"{env_ns}/Bridge/base_link",
        )

        # bridge -> force
        create_fixed_joint(
            stage,
            f"{env_ns}/Bridge/base_link/force_joint",
            f"{env_ns}/Bridge/base_link",
            f"{env_ns}/SixForce/base_link",
            child_offset_axis=(0, 1, 0),
            child_offset_angle=math.pi,
            child_offset_pos=(0, 0, 0.062),
        )

        # force -> gripper
        create_fixed_joint(
            stage,
            f"{env_ns}/SixForce/base_link/gripper_joint",
            f"{env_ns}/SixForce/base_link",
            f"{env_ns}/Gripper/base_link",
            child_offset_pos=(0, 0, -0.0253),
            child_offset_axis=(0, 1, 0),
            child_offset_angle=math.pi,
        )

        # gripper -> assembly body
        create_fixed_joint(
            stage,
            f"{env_ns}/Gripper/base_link/assembly_joint",
            f"{env_ns}/Gripper/base_link",
            f"{env_ns}/Assembly/base_link",
            child_offset_pos=(0, 0, -0.29),
        )
