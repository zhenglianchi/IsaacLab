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
):
    """Compute DOF torque to move end-effector towards target pose using impedance control.
    
    Args:
        cfg: Configuration object
        dof_pos: Joint positions
        dof_vel: Joint velocities
        ee_pos: End-effector position
        ee_quat: End-effector quaternion
        ee_linvel: End-effector linear velocity
        ee_angvel: End-effector angular velocity
        jacobian: Jacobian matrix
        mass_matrix: Mass matrix
        ctrl_target_ee_pos: Target end-effector position
        ctrl_target_ee_quat: Target end-effector quaternion
        task_prop_gains: Task space proportional gains
        task_deriv_gains: Task space derivative gains
        device: Device to use
        dead_zone_thresholds: Dead zone thresholds for force control
    
    Returns:
        dof_torque: Joint torques
        task_wrench: Task space wrench
    """
    num_envs = dof_pos.shape[0]
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    task_wrench = torch.zeros((num_envs, 6), device=device)

    # Compute pose error
    pos_error, axis_angle_error = get_pose_error(
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        ctrl_target_ee_pos=ctrl_target_ee_pos,
        ctrl_target_ee_quat=ctrl_target_ee_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_ee_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Compute task space wrench using PD control
    task_wrench_motion = _apply_task_space_gains(
        delta_ee_pose=delta_ee_pose,
        ee_linvel=ee_linvel,
        ee_angvel=ee_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    task_wrench += task_wrench_motion

    # Apply dead zone thresholds if provided
    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )

    # Map task space wrench to joint space torque
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, :6] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Nullspace control to keep joints near default positions
    if hasattr(cfg, 'default_dof_pos'):
        default_dof_pos_tensor = torch.tensor(cfg.default_dof_pos, device=device).repeat((num_envs, 1))
        distance_to_default = default_dof_pos_tensor - dof_pos[:, :6]
        # Normalize angle to [-pi, pi]
        distance_to_default = (distance_to_default + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Nullspace control law
        kp_null = 1.0  # Proportional gain for nullspace
        kd_null = 0.1  # Derivative gain for nullspace
        u_null = kd_null * -dof_vel[:, :6] + kp_null * distance_to_default
        
        # Compute nullspace projection matrix
        arm_mass_matrix_inv = torch.inverse(mass_matrix)
        arm_mass_matrix_task = torch.inverse(
            jacobian @ arm_mass_matrix_inv @ jacobian_T
        )
        j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
        nullspace_proj = torch.eye(6, device=device).unsqueeze(0) - jacobian_T @ j_eef_inv
        
        # Apply nullspace torque
        torque_null = nullspace_proj @ (mass_matrix @ u_null.unsqueeze(-1))
        dof_torque[:, :6] += torque_null.squeeze(-1)

    # Clamp torque to safe limits
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench


def get_pose_error(
    ee_pos,
    ee_quat,
    ctrl_target_ee_pos,
    ctrl_target_ee_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between target end-effector pose and current pose.
    
    Args:
        ee_pos: End-effector position
        ee_quat: End-effector quaternion
        ctrl_target_ee_pos: Target end-effector position
        ctrl_target_ee_quat: Target end-effector quaternion
        jacobian_type: Jacobian type ("geometric" or "analytic")
        rot_error_type: Rotation error type ("quat" or "axis_angle")
    
    Returns:
        pos_error: Position error
        rot_error: Rotation error
    """
    # Compute position error
    pos_error = ctrl_target_ee_pos - ee_pos

    # Compute rotation error
    if jacobian_type == "geometric":
        # Compute quaternion error
        quat_dot = (ctrl_target_ee_quat * ee_quat).sum(dim=1, keepdim=True)
        ctrl_target_ee_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_ee_quat, -ctrl_target_ee_quat
        )

        # Compute quaternion inverse
        ee_quat_norm = quat_mul(
            ee_quat, quat_conjugate(ee_quat)
        )[:, 0]  # scalar component
        ee_quat_inv = quat_conjugate(ee_quat) / ee_quat_norm.unsqueeze(-1)
        quat_error = quat_mul(ctrl_target_ee_quat, ee_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def _apply_task_space_gains(
    delta_ee_pose, ee_linvel, ee_angvel, task_prop_gains, task_deriv_gains
):
    """Apply task space gains to compute task space wrench.
    
    Args:
        delta_ee_pose: End-effector pose error
        ee_linvel: End-effector linear velocity
        ee_angvel: End-effector angular velocity
        task_prop_gains: Task space proportional gains
        task_deriv_gains: Task space derivative gains
    
    Returns:
        task_wrench: Task space wrench
    """
    task_wrench = torch.zeros_like(delta_ee_pose)

    # Apply gains to linear error components with priority on xy direction
    lin_error = delta_ee_pose[:, 0:3]
    # 对 xy 方向应用额外的权重
    xy_weight = 2.0  # xy 方向权重
    task_wrench[:, 0:2] = xy_weight * task_prop_gains[:, 0:2] * lin_error[:, 0:2] + xy_weight * task_deriv_gains[:, 0:2] * (
        0.0 - ee_linvel[:, 0:2]
    )
    # z 方向保持原权重
    task_wrench[:, 2:3] = task_prop_gains[:, 2:3] * lin_error[:, 2:3] + task_deriv_gains[:, 2:3] * (
        0.0 - ee_linvel[:, 2:3]
    )

    # Apply gains to rotational error components
    rot_error = delta_ee_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - ee_angvel
    )
    return task_wrench
