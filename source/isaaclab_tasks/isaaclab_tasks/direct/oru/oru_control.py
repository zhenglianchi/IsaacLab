# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ORU Assembly: impedance control for UR5 (6-DOF).

Task-space PD control: τ = Jᵀ·F_task, with dead zone + wrench clamp.
(UR5 is non-redundant — no nullspace.)
Adapted from scripts/tutorials/force/force1_control.py.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_mul,
    quat_conjugate,
)


# ==========================================================================
# Main entry — compute DOF torque
# ==========================================================================


def compute_dof_torque(
    cfg,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ee_linvel: torch.Tensor,
    ee_angvel: torch.Tensor,
    jacobian: torch.Tensor,
    ctrl_target_ee_pos: torch.Tensor,
    ctrl_target_ee_quat: torch.Tensor,
    task_prop_gains: torch.Tensor,
    task_deriv_gains: torch.Tensor,
    device: str,
    dead_zone_thresholds: torch.Tensor | None = None,
    gravity_comp: torch.Tensor | None = None,
    z_force_limit: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute UR5 DOF torque to move end-effector towards target pose.

    Args:
        gravity_comp: (num_envs, 6) joint-space gravity compensation torques.
            Applied when the chain bodies have gravity enabled
            (cfg.task.enable_chain_gravity): the arm is weightless
            (disable_gravity=True) but the chain+ORU weight hangs on the
            wrist — compensate it so the wrist can hover without fighting
            the load (and so the wrench clamp is not consumed by it).

    Returns:
        dof_torque: (num_envs, num_joints) joint torques.
        task_wrench: (num_envs, 6) task-space wrench [Fx,Fy,Fz,Tx,Ty,Tz].
    """
    num_envs = dof_pos.shape[0]
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)
    if gravity_comp is not None:
        dof_torque[:, :6] += gravity_comp

    # Pose error
    pos_error, axis_angle_error = get_pose_error(
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        ctrl_target_ee_pos=ctrl_target_ee_pos,
        ctrl_target_ee_quat=ctrl_target_ee_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_ee_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Task-space PD
    task_wrench += task_space_pd(
        delta_ee_pose, ee_linvel, ee_angvel, task_prop_gains, task_deriv_gains
    )

    # Dead zone — suppress tiny wrenches to prevent limit-cycle oscillation
    dead_zone = dead_zone_thresholds
    if dead_zone is None:
        dead_zone = torch.tensor([0.1, 0.1, 0.1, 0.05, 0.05, 0.05], device=device)
    task_wrench = torch.where(
        task_wrench.abs() < dead_zone,
        torch.zeros_like(task_wrench),
        task_wrench.sign() * (task_wrench.abs() - dead_zone),
    )

    # Wrench clamp — per-axis limits. XY stays soft (max_task_force) so the
    # ORU does not wedge sideways; Z has its own cap (max_task_force_z, or a
    # per-env z_force_limit raised by the env on contact) for the >50N
    # stage-2 insertion force. Free-space approach stays bounded against
    # Kp×error slingshot at episode start.
    max_force = getattr(cfg.task, "max_task_force", 5.0)
    max_force_z = getattr(cfg.task, "max_task_force_z", max_force)
    max_torque = getattr(cfg.task, "max_task_torque", 2.0)
    wrench_limit = torch.tensor(
        [max_force, max_force, max_force_z, max_torque, max_torque, max_torque], device=device
    ).expand(num_envs, -1).clone()
    if z_force_limit is not None:
        wrench_limit[:, 2] = z_force_limit
    task_wrench = task_wrench.clamp(-wrench_limit, wrench_limit)

    # Map to joint space
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, :6] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # PHYSX TENSORS COMPENSATION (x8): omni.physics.tensors applies the
    # commanded drive effort at 1/decimation strength, so scale the
    # commanded torque by the decimation factor.
    dof_torque = dof_torque * 8.0

    # Clamp (compensated scale: design limit 100 Nm effective = 800 commanded)
    dof_torque = torch.clamp(dof_torque, min=-800.0, max=800.0)
    return dof_torque, task_wrench


# ==========================================================================
# Pose error
# ==========================================================================


def get_pose_error(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    ctrl_target_ee_pos: torch.Tensor,
    ctrl_target_ee_quat: torch.Tensor,
    jacobian_type: str,
    rot_error_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Task-space error between target and current EE pose."""
    pos_error = ctrl_target_ee_pos - ee_pos

    if jacobian_type == "geometric":
        # Shortest-path quaternion error
        quat_dot = (ctrl_target_ee_quat * ee_quat).sum(dim=1, keepdim=True)
        ctrl_target_ee_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0,
            ctrl_target_ee_quat,
            -ctrl_target_ee_quat,
        )
        ee_quat_norm = quat_mul(ee_quat, quat_conjugate(ee_quat))[:, 0]
        ee_quat_inv = quat_conjugate(ee_quat) / ee_quat_norm.unsqueeze(-1)
        quat_error = quat_mul(ctrl_target_ee_quat, ee_quat_inv)
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    else:
        return pos_error, axis_angle_error


# ==========================================================================
# Task-space PD
# ==========================================================================


def task_space_pd(
    delta_ee_pose: torch.Tensor,
    ee_linvel: torch.Tensor,
    ee_angvel: torch.Tensor,
    task_prop_gains: torch.Tensor,
    task_deriv_gains: torch.Tensor,
) -> torch.Tensor:
    """Apply task-space PD gains to compute wrench."""
    task_wrench = torch.zeros_like(delta_ee_pose)

    lin_error = delta_ee_pose[:, 0:3]
    task_wrench[:, 0:3] = (
        task_prop_gains[:, 0:3] * lin_error
        + task_deriv_gains[:, 0:3] * (0.0 - ee_linvel)
    )

    rot_error = delta_ee_pose[:, 3:6]
    task_wrench[:, 3:6] = (
        task_prop_gains[:, 3:6] * rot_error
        + task_deriv_gains[:, 3:6] * (0.0 - ee_angvel)
    )
    return task_wrench
