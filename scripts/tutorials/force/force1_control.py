import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_mul, quat_conjugate


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    ee_pos,
    ee_quat,
    ee_linvel,
    ee_angvel,
    jacobian,
    mass_matrix,
    ctrl_target_ee_pos,
    ctrl_target_ee_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    dead_zone_thresholds=None,
    gravity=None,
):
    """Compute DOF torque to move end-effector towards target pose using impedance control.
    — synced with oru_control.py (RL version) + optional gravity compensation."""

    num_envs = dof_pos.shape[0]
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)
    if gravity is not None:
        dof_torque[:, :6] += gravity  # 重力补偿 (抵消臂杆自重)

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

    # Task-space PD (NO xy_weight — same as oru_control)
    task_wrench += task_space_pd(
        delta_ee_pose, ee_linvel, ee_angvel, task_prop_gains, task_deriv_gains
    )

    # Dead zone (same default as oru_control)
    dead_zone = dead_zone_thresholds
    if dead_zone is None:
        dead_zone = torch.tensor([0.1, 0.1, 0.1, 0.05, 0.05, 0.05], device=device)
    task_wrench = torch.where(
        task_wrench.abs() < dead_zone,
        torch.zeros_like(task_wrench),
        task_wrench.sign() * (task_wrench.abs() - dead_zone),
    )

    # Map to joint space (keep gravity compensation accumulated via +=)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, :6] += (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Nullspace (hardcoded defaults — same as oru_control)
    default_dof_pos = torch.tensor(
        [-15.60 * torch.pi / 180, -82.61 * torch.pi / 180, 29.85 * torch.pi / 180,
         -37.15 * torch.pi / 180, -89.96 * torch.pi / 180, 74.46 * torch.pi / 180],
        device=device,
    ).repeat(num_envs, 1)

    distance_to_default = default_dof_pos - dof_pos[:, :6]
    distance_to_default = (distance_to_default + torch.pi) % (2 * torch.pi) - torch.pi

    kp_null = 20.0  # strong posture hold for vertical-press verification
    kd_null = 9.0   # ~2*sqrt(20) critical damping
    u_null = kd_null * (-dof_vel[:, :6]) + kp_null * distance_to_default

    arm_mass_matrix_inv = torch.inverse(mass_matrix)
    arm_mass_matrix_task = torch.inverse(jacobian @ arm_mass_matrix_inv @ jacobian_T)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
    nullspace_proj = torch.eye(6, device=device).unsqueeze(0) - jacobian_T @ j_eef_inv
    torque_null = nullspace_proj @ (mass_matrix @ u_null.unsqueeze(-1))
    dof_torque[:, :6] += torque_null.squeeze(-1)

    # PHYSX TENSORS COMPENSATION (x8): omni.physics.tensors applies commanded
    # drive effort at 1/decimation strength (verified: x8 = exact full-Newton
    # response). Applies to ALL terms (impedance, gravity comp, nullspace).
    dof_torque = dof_torque * 8.0

    # Clamp (compensated scale: design limit 100 Nm effective = 800 commanded)
    dof_torque = torch.clamp(dof_torque, min=-800.0, max=800.0)
    return dof_torque, task_wrench


def get_pose_error(
    ee_pos,
    ee_quat,
    ctrl_target_ee_pos,
    ctrl_target_ee_quat,
    jacobian_type,
    rot_error_type,
):
    """Task-space error between target and current EE pose."""
    pos_error = ctrl_target_ee_pos - ee_pos

    if jacobian_type == "geometric":
        quat_dot = (ctrl_target_ee_quat * ee_quat).sum(dim=1, keepdim=True)
        ctrl_target_ee_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_ee_quat, -ctrl_target_ee_quat
        )

        ee_quat_norm = quat_mul(ee_quat, quat_conjugate(ee_quat))[:, 0]
        ee_quat_inv = quat_conjugate(ee_quat) / ee_quat_norm.unsqueeze(-1)
        quat_error = quat_mul(ctrl_target_ee_quat, ee_quat_inv)
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def task_space_pd(
    delta_ee_pose, ee_linvel, ee_angvel, task_prop_gains, task_deriv_gains
):
    """Apply task-space PD gains to compute wrench (NO extra xy_weight)."""
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
