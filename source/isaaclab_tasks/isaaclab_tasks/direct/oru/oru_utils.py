# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ORU Assembly: utility module.

- Fixed joint creation (from force1_SceneCfg pattern)
- Observation collation
- Reward helpers
"""

from __future__ import annotations

import math
import torch

import isaacsim.core.utils.torch as torch_utils
from pxr import UsdPhysics, Gf, Sdf


# ==========================================================================
# Fixed Joint — attach UR5 → Bridge → Force_Six → Gripper → ORU
# ==========================================================================

def create_one_fixed_joint(
    stage,
    joint_path: str,
    parent_path: str,
    child_path: str,
    *,
    child_offset_pos=(0.0, 0.0, 0.0),
    child_offset_quat=None,
    child_offset_axis=(0.0, 0.0, 1.0),
    child_offset_angle=0.0,
    drive_stiffness=None,
    drive_damping=None,
):
    """Create a single PhysX FixedJoint between two prims.

    drive_stiffness/drive_damping: optional PhysX joint drive (spring-damper
    on all locked DOFs). FixedJoints yield under dynamic loads (chain whip);
    a stiff drive makes the chain behave like a rigid rod.
    """
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([parent_path])
    joint.CreateBody1Rel().SetTargets([child_path])

    # parent frame — identity
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))

    # child position offset
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*map(float, child_offset_pos)))

    # child rotation offset
    if child_offset_quat is not None:
        w, x, y, z = child_offset_quat
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(float(w), Gf.Vec3f(x, y, z)))
    else:
        ax = Gf.Vec3f(*map(float, child_offset_axis))
        if ax.GetLength() > 0 and abs(child_offset_angle) > 0:
            ax = ax.GetNormalized()
            half = child_offset_angle * 0.5
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(math.cos(half), ax * math.sin(half)))
        else:
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    joint.CreateBreakForceAttr().Set(1e10)
    joint.CreateBreakTorqueAttr().Set(1e10)

    # Optional PhysX joint drive — stiffen the locked constraint against
    # dynamic yield (chain whip). physxJoint:drive:* is the PhysX schema
    # attribute set (non-custom on the prim, per SchemaRegistry probe).
    if drive_stiffness is not None:
        prim = joint.GetPrim()
        prim.CreateAttribute("physxJoint:drive:stiffness", Sdf.ValueTypeNames.Double).Set(drive_stiffness)
        prim.CreateAttribute("physxJoint:drive:damping", Sdf.ValueTypeNames.Double).Set(
            drive_damping if drive_damping is not None else 2.0 * math.sqrt(drive_stiffness)
        )


def create_oru_fixed_joints(stage, num_envs: int):
    """Create the full fixed-joint chain for all environments.

    Chain: wrist_3_link → Bridge → Force_Six → Gripper → ORU
    (matching force1_SceneCfg.add_fixed_joint exactly)
    """
    for env_idx in range(num_envs):
        env_ns = f"/World/envs/env_{env_idx}"

        # flange → bridge
        create_one_fixed_joint(
            stage,
            f"{env_ns}/Dofbot/wrist_3_link/bridge_joint",
            f"{env_ns}/Dofbot/wrist_3_link",
            f"{env_ns}/Bridge/base_link",
        )

        # bridge → force
        create_one_fixed_joint(
            stage,
            f"{env_ns}/Bridge/base_link/force_joint",
            f"{env_ns}/Bridge/base_link",
            f"{env_ns}/SixForce/base_link",
            child_offset_axis=(0, 1, 0),
            child_offset_angle=math.pi,
            child_offset_pos=(0, 0, 0.062),
        )

        # force → gripper
        create_one_fixed_joint(
            stage,
            f"{env_ns}/SixForce/base_link/gripper_joint",
            f"{env_ns}/SixForce/base_link",
            f"{env_ns}/Gripper/base_link",
            child_offset_pos=(0, 0, -0.0253),
            child_offset_axis=(0, 1, 0),
            child_offset_angle=math.pi,
        )

        # gripper → ORU
        create_one_fixed_joint(
            stage,
            f"{env_ns}/Gripper/base_link/oru_joint",
            f"{env_ns}/Gripper/base_link",
            f"{env_ns}/ORU/base_link",
            child_offset_pos=(0, 0, -0.305),
            child_offset_axis=(0, 0, 1),
            child_offset_angle=math.pi,
        )


# ==========================================================================
# Observation helpers
# ==========================================================================

def collapse_obs_dict(obs_dict: dict, keys: list) -> torch.Tensor:
    """Stack observation tensors in the given key order."""
    obs_list = [obs_dict[k] for k in keys]
    return torch.cat(obs_list, dim=-1)


# ==========================================================================
# Reward helpers — keypoint-based squashing reward
# ==========================================================================

def get_keypoint_offsets(num_keypoints: int, device: str) -> torch.Tensor:
    """Uniformly-spaced keypoints along a vertical line, centered at 0."""
    offsets = torch.zeros((num_keypoints, 3), device=device)
    offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return offsets


def squashing_fn(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Bounded reward: r(x) = 1 / (exp(a*x) + b + exp(-a*x))."""
    return 1.0 / (torch.exp(a * x) + b + torch.exp(-a * x))


def get_deriv_gains(prop_gains: torch.Tensor, rot_deriv_scale: float = 1.0) -> torch.Tensor:
    """Critical damping derivative gains: 2 * sqrt(Kp), rot scaled down."""
    deriv = 2.0 * torch.sqrt(prop_gains)
    deriv[:, 3:6] /= rot_deriv_scale
    return deriv


# ==========================================================================
# Friction
# ==========================================================================

def set_friction(asset, value: float, num_envs: int):
    """Set static + dynamic friction on all bodies of an asset."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value
    materials[..., 1] = value
    env_ids = torch.arange(num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)
