# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ORU Assembly: impedance control for UR5 (6-DOF).

Operational-space control with task-space PD + nullspace projection.
Adapted from scripts/tutorials/force/force1_control.py.
"""

from __future__ import annotations

import math
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
    mass_matrix: torch.Tensor,
    ctrl_target_ee_pos: torch.Tensor,
    ctrl_target_ee_quat: torch.Tensor,
    task_prop_gains: torch.Tensor,
    task_deriv_gains: torch.Tensor,
    device: str,
    dead_zone_thresholds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute UR5 DOF torque to move end-effector towards target pose.

    Returns:
        dof_torque: (num_envs, num_joints) joint torques.
        task_wrench: (num_envs, 6) task-space wrench [Fx,Fy,Fz,Tx,Ty,Tz].
    """
    num_envs = dof_pos.shape[0]
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

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

    # Dead zone
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # Map to joint space
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, :6] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Nullspace — keep joints near default positions
    default_dof_pos = torch.tensor(
        [0.0, -math.pi / 2, math.pi / 6, -math.pi / 6, -math.pi / 2, math.pi / 2],
        device=device,
    ).repeat(num_envs, 1)

    distance_to_default = default_dof_pos - dof_pos[:, :6]
    distance_to_default = (distance_to_default + torch.pi) % (2 * torch.pi) - torch.pi

    kp_null = 1.0
    kd_null = 0.1
    u_null = kd_null * (-dof_vel[:, :6]) + kp_null * distance_to_default

    arm_mass_matrix_inv = torch.inverse(mass_matrix)
    arm_mass_matrix_task = torch.inverse(jacobian @ arm_mass_matrix_inv @ jacobian_T)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    nullspace_proj = (
        torch.eye(6, device=device).unsqueeze(0) - jacobian_T @ j_eef_inv
    )
    torque_null = nullspace_proj @ (mass_matrix @ u_null.unsqueeze(-1))
    dof_torque[:, :6] += torque_null.squeeze(-1)

    # Clamp
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
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
    # XY direction gets 2x priority for horizontal alignment
    xy_weight = 2.0
    task_wrench[:, 0:2] = (
        xy_weight * task_prop_gains[:, 0:2] * lin_error[:, 0:2]
        + xy_weight * task_deriv_gains[:, 0:2] * (0.0 - ee_linvel[:, 0:2])
    )
    task_wrench[:, 2:3] = (
        task_prop_gains[:, 2:3] * lin_error[:, 2:3]
        + task_deriv_gains[:, 2:3] * (0.0 - ee_linvel[:, 2:3])
    )

    rot_error = delta_ee_pose[:, 3:6]
    task_wrench[:, 3:6] = (
        task_prop_gains[:, 3:6] * rot_error
        + task_deriv_gains[:, 3:6] * (0.0 - ee_angvel)
    )
    return task_wrench
