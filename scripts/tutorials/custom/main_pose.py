# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
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
from isaaclab.sim import SimulationContext
from SceneCfg import NewRobotsSceneCfg, add_fixed_joint, scene_reset
from pose_controller import UR5Controller


def run_simulator(sim, scene, ur5_ctrl):
    count = 0
    index = -1
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            scene_reset(scene)
            ee_positions, ee_quat = ur5_ctrl.get_end_effector_pose()
            ee_positions[0][2] = ee_positions[0][2]-0.1
            index += 1

        ur5_ctrl.move_ee_to(ee_positions, ee_quat)
        count += 1



def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Now we are ready!
    sim = SimulationContext.instance()
    stage = sim.stage

    add_fixed_joint(stage, args_cli)

    # Play the simulator
    sim.reset()
    scene.reset()
    ur5_ctrl = UR5Controller(scene, args_cli)
    # Run the simulator
    run_simulator(sim, scene, ur5_ctrl)


'''if __name__ == "__main__":
    main()
    simulation_app.close()'''
