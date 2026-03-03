import argparse
import torch
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from force_SceneCfg import NewRobotsSceneCfg, scene_reset
from force_controller import UR5HybridOSCController
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

# Update the target commands
def update_target(sim,scene,osc,root_pose_w,ee_target_set,current_goal_idx):
    """Update the targets for the operational space controller.

    Args:
        sim: (SimulationContext) Simulation context.
        scene: (InteractiveScene) Interactive scene.
        osc: (OperationalSpaceController) Operational space controller.
        root_pose_w: (torch.tensor) Root pose in the world frame.
        ee_target_set: (torch.tensor) End-effector target set.
        current_goal_idx: (int) Current goal index.

    Returns:
        command (torch.tensor): Updated target command.
        ee_target_pose_b (torch.tensor): Updated target pose in the body frame.
        ee_target_pose_w (torch.tensor): Updated target pose in the world frame.
        next_goal_idx (int): Next goal index.

    Raises:
        ValueError: Undefined target_type.
    """

    # update the ee desired command
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:] = ee_target_set[current_goal_idx]

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "wrench_abs":
            pass  # ee_target_pose_b could stay at the root frame for force control, what matters is ee_target_b
        else:
            raise ValueError("Undefined target_type within update_target().")

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(ee_target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


# Convert the target commands to the task frame
def convert_to_task_frame(osc, command, ee_target_pose_b):
    """Converts the target commands to the task frame.

    Args:
        osc: OperationalSpaceController object.
        command: Command to be converted.
        ee_target_pose_b: Target pose in the body frame.

    Returns:
        command (torch.tensor): Target command in the task frame.
        task_frame_pose_b (torch.tensor): Target pose in the task frame.

    Raises:
        ValueError: Undefined target_type.
    """
    command = command.clone()
    task_frame_pose_b = ee_target_pose_b.clone()

    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
            )
            cmd_idx += 7
        elif target_type == "wrench_abs":
            # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
            # easier), so not transforming
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _convert_to_task_frame().")

    return command, task_frame_pose_b

# ==========================================================
# Main
# ==========================================================
def main():

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Now we are ready!
    sim = sim_utils.SimulationContext.instance()
    #stage = sim.stage

    #add_fixed_joint(stage, args_cli)

    # Play the simulator
    sim.reset()
    scene.reset()
    # --------------------------------------------------
    # Controller
    # --------------------------------------------------
    controller = UR5HybridOSCController(scene, sim)
    # ==========================================================
    # 官方目标集（完全一致）
    # ==========================================================
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [0.43435,  0.10915,  0.62798,  0.0, -0.707, 0.707, 0.0],
            [0.43435,  0.10915,  0.62798,  0.0, -0.707, 0.707, 0.0],
            [0.43435,  0.10915,  0.62798,  0.0, -0.707, 0.707, 0.0],
        ],
        device=sim.device,
    )

    ee_goal_wrench_set_tilted_task = torch.tensor(
        [
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )

    kp_set_task = torch.tensor(
        [
            [360.0, 360.0, 360.0, 360.0, 360.0, 360.0],
            [420.0, 420.0, 420.0, 420.0, 420.0, 420.0],
            [320.0, 320.0, 320.0, 320.0, 320.0, 320.0],
        ],
        device=sim.device,
    )

    ee_target_set = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task, kp_set_task], dim=-1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    controller.robot.update(dt=sim_dt)
    joint_centers = torch.mean(controller.robot.data.soft_joint_pos_limits[:, controller.arm_joint_ids, :], dim=-1)
    # get the updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    ) = controller.update_states()

    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        scene.num_envs, controller.osc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(scene.num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(scene.num_envs, controller.robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(scene.num_envs, len(controller.arm_joint_ids), device=sim.device)

    count = 0
    
    while simulation_app.is_running():
        # reset every 500 steps
        if count % 500 == 0:
            # reset joint state to default
            default_joint_pos = controller.robot.data.default_joint_pos.clone()
            default_joint_vel = controller.robot.data.default_joint_vel.clone()
            controller.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            controller.robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
            controller.robot.write_data_to_sim()
            controller.robot.reset()
            # reset contact sensor
            controller.contact_forces.reset()
            # reset target pose
            controller.robot.update(sim_dt)
            print(controller.get_tcp())
            _, _, _, ee_pose_b, _, _, _, _, _, _ = controller.update_states()
            command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = update_target(
                sim, scene, controller.osc, root_pose_w, ee_target_set, current_goal_idx
            )
            # set the osc command
            controller.osc.reset()
            command, task_frame_pose_b = convert_to_task_frame(controller.osc, command=command, ee_target_pose_b=ee_target_pose_b)
            controller.osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
        else:
            # get the updated states
            (
                jacobian_b,
                mass_matrix,
                gravity,
                ee_pose_b,
                ee_vel_b,
                root_pose_w,
                ee_pose_w,
                ee_force_b,
                joint_pos,
                joint_vel,
            ) = controller.update_states()
            # compute the joint commands
            joint_efforts = controller.osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                current_ee_force_b=ee_force_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )
            # apply actions
            controller.robot.set_joint_effort_target(joint_efforts, joint_ids=controller.arm_joint_ids)
            controller.robot.write_data_to_sim()

        # perform step
        sim.step(render=True)
        # update robot buffers
        controller.robot.update(sim_dt)
        # update buffers
        scene.update(sim_dt)
        # update sim-time
        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()

