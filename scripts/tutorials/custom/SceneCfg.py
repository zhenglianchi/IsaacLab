
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import UsdPhysics, Gf
from isaaclab.assets import RigidObjectCfg  # <-- changed from ArticulationCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

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


ORU_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/ORU/ORU.usd",
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


GROUND_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/ground/ground.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0, 0),
        rot=(0, 1, 0, 0),
    ),
)


# =====================================================
# UR5
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
            solver_velocity_iteration_count=0,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -math.pi / 2,
            "elbow_joint": math.pi / 3,
            "wrist_1_joint": -math.pi / 3,
            "wrist_2_joint": -math.pi / 2,
            "wrist_3_joint": 0.0,
        },
        pos=(0, 0, 0),
    ),

    actuators={

        "shoulder_pan_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=800,
            damping=40,
        ),

        "shoulder_lift_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=800,
            damping=40,
        ),

        "elbow_joint": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=800,
            damping=40,
        ),

        "wrist_1_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=800,
            damping=40,
        ),

        "wrist_2_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_2_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=800,
            damping=40,
        ),

        "wrist_3_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_3_joint"],
            effort_limit_sim=87,
            velocity_limit_sim=1,
            stiffness=800,
            damping=40,
        ),
    },
)


# =====================================================
# Scene
# =====================================================

class NewRobotsSceneCfg(InteractiveSceneCfg):

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

    ORU = ORU_CFG.replace(prim_path="{ENV_REGEX_NS}/ORU")

    Ground = GROUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Ground")

    Bridge = BRIDGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Bridge")




def scene_reset(scene):
    root_Ground_state = scene["Ground"].data.default_root_state.clone()
    root_Ground_state[:, :3] += scene.env_origins

    scene["Ground"].write_root_pose_to_sim(root_Ground_state[:, :7])

    root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
    root_dofbot_state[:, :3] += scene.env_origins

    scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
    scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

    joint_pos, joint_vel = (
        scene["Dofbot"].data.default_joint_pos.clone(),
        scene["Dofbot"].data.default_joint_vel.clone(),
    )

    scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
    scene["Dofbot"].set_joint_position_target(joint_pos)

    scene.write_data_to_sim()

    scene.reset()



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

        # flange → bridge
        create_fixed_joint(
            stage,
            f"{env_ns}/Dofbot/wrist_3_link/bridge_joint",
            f"{env_ns}/Dofbot/wrist_3_link",
            f"{env_ns}/Bridge/base_link",
        )

        # bridge → force
        create_fixed_joint(
            stage,
            f"{env_ns}/Bridge/base_link/force_joint",
            f"{env_ns}/Bridge/base_link",
            f"{env_ns}/SixForce/base_link",
            child_offset_axis=(0, 1, 0),
            child_offset_angle=math.pi,
            child_offset_pos=(0, 0, 0.062),
        )

        # force → gripper
        create_fixed_joint(
            stage,
            f"{env_ns}/SixForce/base_link/gripper_joint",
            f"{env_ns}/SixForce/base_link",
            f"{env_ns}/Gripper/base_link",
            child_offset_pos=(0, 0, -0.0253),
            child_offset_axis=(0, 1, 0),
            child_offset_angle=math.pi,
        )

        # gripper → ORU
        create_fixed_joint(
            stage,
            f"{env_ns}/Gripper/base_link/oru_joint",
            f"{env_ns}/Gripper/base_link",
            f"{env_ns}/ORU/base_link",
            child_offset_pos=(0, 0, -0.29),
        )