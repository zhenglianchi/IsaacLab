# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ORU Assembly: environment configuration.

UR5 robot + fixed-joint chain (Bridge→Force_Six→Gripper→ORU) + Ground.
Scene matches force1_SceneCfg.NewRobotsSceneCfg exactly.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .oru_tasks_cfg import OruTaskCfg

# ==========================================================================
# Obs / State dims
# ==========================================================================

OBS_DIM_CFG = {
    "ee_pos_rel_ground": 3,
    "ee_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 6,
}

STATE_DIM_CFG = {
    "ee_pos_rel_ground": 3,
    "ee_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 6,
    "ground_pos": 3,
    "ground_quat": 4,
    "task_prop_gains": 6,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

# ==========================================================================
# Asset configs — match force1_SceneCfg field-for-field
# ==========================================================================

UR5_CFG = ArticulationCfg(
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
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder_pan_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            effort_limit_sim=87, velocity_limit_sim=1, stiffness=0, damping=0,
        ),
        "shoulder_lift_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            effort_limit_sim=87, velocity_limit_sim=1, stiffness=0, damping=0,
        ),
        "elbow_joint": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=87, velocity_limit_sim=1, stiffness=0, damping=0,
        ),
        "wrist_1_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint"],
            effort_limit_sim=87, velocity_limit_sim=1, stiffness=0, damping=0,
        ),
        "wrist_2_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_2_joint"],
            effort_limit_sim=87, velocity_limit_sim=1, stiffness=0, damping=0,
        ),
        "wrist_3_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_3_joint"],
            effort_limit_sim=87, velocity_limit_sim=1, stiffness=0, damping=0,
        ),
    },
)

BRIDGE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/bridge/bridge.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=5.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
)

FORCE_SENSOR_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/force/force.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
)

GRIPPER_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/gripper/gripper.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=5.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
)

ORU_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/o7/ORU.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0, rest_offset=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
)

GROUND_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/USD/g1/g1.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.0, rest_offset=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.05),
        rot=(0.0, 1.0, 0.0, 0.0),
    ),
)


# ==========================================================================
# Scene — match NewRobotsSceneCfg field-for-field
# ==========================================================================

@configclass
class OruSceneCfg(InteractiveSceneCfg):
    """Full scene matching force1_SceneCfg.NewRobotsSceneCfg.

    All assets are loaded at once by InteractiveScene, exactly like
    main_force1.py does.
    """
    # ground plane
    ground_plane = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000, color=(0.75, 0.75, 0.75)),
    )
    # assets
    Dofbot: ArticulationCfg = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    Bridge: RigidObjectCfg = BRIDGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Bridge")
    SixForce: RigidObjectCfg = FORCE_SENSOR_CFG.replace(prim_path="{ENV_REGEX_NS}/SixForce")
    Gripper: RigidObjectCfg = GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Gripper")
    ORU: RigidObjectCfg = ORU_CFG.replace(prim_path="{ENV_REGEX_NS}/ORU")
    Ground: RigidObjectCfg = GROUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Ground")


# ==========================================================================
# Environment Config
# ==========================================================================

@configclass
class OruEnvCfg(DirectRLEnvCfg):
    """ORU Assembly env config."""

    decimation: int = 8
    action_space: int = 6
    observation_space: int = 0   # computed at init
    state_space: int = 0

    obs_order: list = [
        "ee_pos_rel_ground", "ee_quat", "ee_linvel", "ee_angvel", "joint_pos",
    ]
    state_order: list = [
        "ee_pos_rel_ground", "ee_quat", "ee_linvel", "ee_angvel", "joint_pos",
        "ground_pos", "ground_quat", "task_prop_gains", "pos_threshold", "rot_threshold",
    ]

    task: OruTaskCfg = OruTaskCfg()
    episode_length_s: float = 15.0

    ema_factor: float = 0.2
    pos_action_threshold: tuple = (0.02, 0.02, 0.02)
    rot_action_threshold: tuple = (0.097, 0.097, 0.097)

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0,
        ),
    )

    # All assets inside the scene config — loaded together by InteractiveScene
    scene: OruSceneCfg = OruSceneCfg(
        num_envs=8, env_spacing=2.0, clone_in_fabric=True,
    )


@configclass
class OruAssemblyCfg(OruEnvCfg):
    pass
