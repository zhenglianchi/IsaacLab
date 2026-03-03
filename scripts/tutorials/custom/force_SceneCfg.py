
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import UsdPhysics, Gf
from isaaclab.assets import RigidObjectCfg  # <-- changed from ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
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

    Wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.2), rot=(0.9238795325, 0.0, -0.3826834324, 0.0)
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )




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
    scene["Dofbot"].set_joint_position_target(joint_pos)

    scene.write_data_to_sim()

    scene.reset()
