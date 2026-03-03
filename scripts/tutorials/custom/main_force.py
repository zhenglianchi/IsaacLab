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
from SceneCfg import NewRobotsSceneCfg, add_fixed_joint, scene_reset
from force_controller import UR5HybridOSCController


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
    stage = sim.stage

    add_fixed_joint(stage, args_cli)

    # Play the simulator
    sim.reset()
    scene.reset()
    # --------------------------------------------------
    # Controller
    # --------------------------------------------------
    controller = UR5HybridOSCController(scene, args_cli)

    # ==========================================================
    # 官方目标集（完全一致）
    # ==========================================================
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [0.6, 0.15, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.6, -0.3, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
            [0.8, 0.0, 0.5, 0.0, 0.92387953, 0.0, 0.38268343],
        ],
        device=sim.device,
    )

    ee_goal_wrench_set_tilted_task = torch.tensor(
        [
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
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

    num_goals = ee_goal_pose_set_tilted_b.shape[0]
    current_goal_idx = 0

    sim_dt = sim.get_physics_dt()
    step_count = 0

    # ==========================================================
    # Simulation Loop (与官方一致的目标切换方式)
    # ==========================================================
    while simulation_app.is_running():

        if step_count % 500 == 0:

            current_goal_idx = (current_goal_idx + 1) % num_goals

            target_pose = ee_goal_pose_set_tilted_b[current_goal_idx]
            target_wrench = ee_goal_wrench_set_tilted_task[current_goal_idx]
            kp = kp_set_task[current_goal_idx]

            # 更新 controller 内部参数
            controller.set_kp(kp)

        target_pose = ee_goal_pose_set_tilted_b[current_goal_idx]
        target_wrench = ee_goal_wrench_set_tilted_task[current_goal_idx]
        kp = kp_set_task[current_goal_idx]

        # 拼成 19 维
        command = torch.cat(
            [
                target_pose,
                target_wrench,
                kp,
            ],
            dim=0,
        ).unsqueeze(0)

        controller.step(command)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step_count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()