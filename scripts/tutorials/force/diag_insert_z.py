"""Diagnostic: can FIXED impedance insert the ORU with a pure-Z command?

No policy — fixed gains. The XY target is frozen at the start (or aligned to
the ground XY), and the Z target sits BELOW the contact surface so the
commanded force stays on and keeps pressing the ORU into the dock.

Answers: "why does the RL insertion always stop a few mm short?" — if the
arm reaches success_z with a strong enough Kp_z, the physics allows the
insertion and the failure is in the policy/reward; if it stalls at the
contact surface even at Kp_z=500, the insertion needs more force than a
position impedance can give (dead zone / clamp / geometry).

Requires the x8 tensors compensation (oru_control.py).

Usage:
  python scripts/tutorials/force/diag_insert_z.py --headless
  python scripts/tutorials/force/diag_insert_z.py --kp_z 500 --target_z 0.40
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed-impedance pure-Z insertion diagnostic.")
parser.add_argument("--task", type=str, default="Isaac-Oru-Direct-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=400, help="Sim steps (1 step = 1/15 s).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--kp_z", type=float, default=300.0,
                    help="Z proportional gain (default 300). Kd_z = 2*sqrt(Kp_z).")
parser.add_argument("--kp_xy", type=float, default=100.0,
                    help="XY proportional gain (default 100). Kd_xy = 2*sqrt(Kp_xy).")
parser.add_argument("--kd_xy", type=float, default=None,
                    help="XY derivative gain override (default 2*sqrt(Kp_xy)). 0 isolates "
                         "the vibration-feedback hypothesis: spurious Kd lateral force "
                         "from high-freq wrist jitter -> wedge jam.")
parser.add_argument("--kd_z", type=float, default=None,
                    help="Z derivative gain override (default 2*sqrt(Kp_z)).")
parser.add_argument("--max_force", type=float, default=None,
                    help="Override the Fx/Fy/Fz wrench clamp (default: env 20 N). "
                         "Higher = more insertion force available.")
parser.add_argument("--oru_mass", type=float, default=None,
                    help="Override ORU mass (kg). Chain-load test: lighter ORU -> less "
                         "inertia whipping the chain off-axis during descent.")
parser.add_argument("--no_comp", action="store_true",
                    help="With --oru_gravity: gravity ON but NO joint-space "
                         "compensation (isolate spurious J^T lateral torques).")
parser.add_argument("--oru_gravity", action="store_true",
                    help="Enable gravity on the ORU (and chain bodies). The weight "
                         "tensions the FixedJoint chain straight — like force1, where "
                         "the chain hangs taut and insertion works manually.")
parser.add_argument("--seed_vel_z", type=float, default=0.0,
                    help="Pre-seed the first-step FD velocity estimate (m/s, + = up). "
                         "The env's ee_linvel_fd starts at 0, so step 1 has NO damping "
                         "and the wrist free-falls at full clamp force -> chain whip -> "
                         "ORU lands off-center. Seeding gives step 1 a real damping term.")
parser.add_argument("--phase1_steps", type=int, default=0,
                    help="Two-phase profile: first N steps approach at low gain "
                         "(--phase1_kp/--phase1_target), then switch to the main "
                         "gains/target. Position impedance + wrench clamp = free-fall "
                         "+ hard brake; a slow crawl avoids whipping the chain and "
                         "lets the ORU land centered. --phase1_steps 0 disables.")
parser.add_argument("--phase1_target", type=float, default=0.45,
                    help="Phase-1 Z target (default 0.45, 15mm above contact ~0.4355).")
parser.add_argument("--phase1_kp", type=float, default=100.0,
                    help="Phase-1 Kp_z (default 100: crawl force = 100*err, no clamp hit).")
parser.add_argument("--oru_com", type=float, default=None,
                    help="Set ORU center of mass z-position (local, m). The ORU "
                         "COM is off-axis by 8.9mm X (com_rel_root=(-0.0089,..,"
                         "-0.0570)); with chain gravity on, that creates a "
                         "constant lateral torque. --oru_com 0.057 sets the "
                         "COM to (0,0,-0.057) — centered on the chain axis.")
parser.add_argument("--chain_damp", type=float, default=None,
                    help="Damping multiplier for chain bodies (Bridge/SixForce/"
                         "Gripper/ORU): linear_damping = ang_damp, angular_damping "
                         "= 5x. Suppresses chain pendulum micro-swing that bends "
                         "the FixedJoint chain and throws the ORU off-axis.")
parser.add_argument("--joint_drive", type=float, default=None,
                    help="PhysX joint drive stiffness on all FixedJoints (N/m). "
                         "Stiffens the chain against dynamic yield (whip). "
                         "Damping defaults to 2*sqrt(k).")
parser.add_argument("--penetration", type=float, default=0.0,
                    help="Allow ORU/ground contact penetration (rest_offset=-x, in cm). "
                         "e.g. --penetration 1.0 -> rest_offset=-0.01 on ORU and Ground "
                         "colliders. Also sets max_depenetration_velocity=5 on the ORU.")
parser.add_argument("--target_z", type=float, default=0.40,
                    help="Z target below the contact surface (default 0.40; success_z=0.42972).")
parser.add_argument("--xy", type=str, default="freeze", choices=["freeze", "align"],
                    help="freeze = XY target at EE start; align = XY target at ground XY.")
parser.add_argument("--show_pose", action="store_true",
                    help="Also print ORU pose vs ground and joint positions (find the stall contact).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    # Exact home pose — start straight above the target, no randomization
    env_cfg.task.fixed_ik_offset_pos = (0.0, 0.0, 0.0)
    env_cfg.task.fixed_ik_offset_rot = (0.0, 0.0, 0.0)
    # Long episode so the insertion isn't cut short by the 15 s reset
    env_cfg.episode_length_s = 100.0
    if args_cli.max_force is not None:
        env_cfg.task.max_task_force = args_cli.max_force
        env_cfg.task.max_task_torque = max(6.0, args_cli.max_force * 0.3)
    if args_cli.oru_mass is not None:
        import isaaclab.sim as sim_utils
        env_cfg.scene.ORU.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=args_cli.oru_mass)
        print(f"[INFO] ORU mass override: {args_cli.oru_mass} kg")
    if args_cli.joint_drive is not None:
        env_cfg.task.joint_drive_stiffness = args_cli.joint_drive
        env_cfg.task.joint_drive_damping = 2.0 * args_cli.joint_drive ** 0.5
        print(f"[INFO] Joint drive: k={args_cli.joint_drive} N/m, "
              f"d={2.0 * args_cli.joint_drive ** 0.5:.0f} N*s/m on all FixedJoints")
    if args_cli.chain_damp is not None:
        for name in ["Bridge", "SixForce", "Gripper", "ORU"]:
            rp = getattr(env_cfg.scene, name).spawn.rigid_props
            rp.linear_damping = args_cli.chain_damp
            rp.angular_damping = args_cli.chain_damp * 5.0
        print(f"[INFO] Chain damping: lin={args_cli.chain_damp} ang={args_cli.chain_damp * 5.0}")
    if args_cli.oru_gravity:
        # Chain bodies feel gravity again (force1 parity): weight tensions the
        # FixedJoint chain straight so the ORU stays centered under the wrist.
        # The env adds joint-space gravity compensation via
        # cfg.task.enable_chain_gravity (oru_env._apply_action).
        env_cfg.task.enable_chain_gravity = True
        if args_cli.no_comp:
            env_cfg.task.gravity_comp_enable = False
        for name in ["Bridge", "SixForce", "Gripper", "ORU"]:
            getattr(env_cfg.scene, name).spawn.rigid_props.disable_gravity = False
        comp_s = "NO compensation" if args_cli.no_comp else "with gravity compensation"
        print(f"[INFO] Chain gravity ON ({comp_s}, force1 parity)")
    if args_cli.penetration > 0.0:
        ro = -args_cli.penetration / 100.0
        env_cfg.scene.ORU.spawn.rigid_props.max_depenetration_velocity = 5.0
        env_cfg.scene.ORU.spawn.collision_props.rest_offset = ro
        env_cfg.scene.Ground.spawn.collision_props.rest_offset = ro
        print(f"[INFO] Penetration allowed: rest_offset={ro} on ORU+Ground colliders.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped: DirectRLEnv = env.unwrapped

    if args_cli.oru_com is not None:
        # Set ORU center of mass via UsdPhysics.MassAPI (local frame).
        # Reset writes pose/vel, not mass properties — safe before reset.
        from pxr import UsdPhysics, Gf
        stage = env_unwrapped.sim.stage
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/ORU")
        if prim.IsValid() and UsdPhysics.MassAPI(prim):
            api = UsdPhysics.MassAPI(prim)
            api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, -args_cli.oru_com))
            print(f"[INFO] ORU COM override: (0, 0, {-args_cli.oru_com}) on {prim.GetPath()}")
        else:
            print(f"[WARN] MassAPI not found on /World/envs/env_0/ORU")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    ee_idx = env_unwrapped._ee_frame_idx
    success_z = env_unwrapped.cfg.task.success_z  # fully-inserted EE height

    # ── Fixed impedance gains (no policy) ──
    kp_z, kd_z = args_cli.kp_z, 2.0 * args_cli.kp_z ** 0.5
    kp_xy, kd_xy = args_cli.kp_xy, 2.0 * args_cli.kp_xy ** 0.5
    if args_cli.kd_xy is not None:
        kd_xy = args_cli.kd_xy
    if args_cli.kd_z is not None:
        kd_z = args_cli.kd_z
    g = env_unwrapped.base_gains.clone()
    g[0, 0:2] = kp_xy
    g[0, 2] = kp_z
    env_unwrapped.base_gains = g
    d = env_unwrapped.base_deriv.clone()
    d[0, 0:2] = kd_xy
    d[0, 2] = kd_z
    env_unwrapped.base_deriv = d
    print(f"[INFO] Fixed gains: Kp_xy={kp_xy} Kd_xy={kd_xy:.1f} | Kp_z={kp_z} Kd_z={kd_z:.1f}")

    # ── Target: XY frozen at start (or aligned), Z below contact ──
    ee_start = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy().copy()
    if args_cli.xy == "align":
        target_xy = env_unwrapped.ground.data.root_pos_w[0, :2].cpu().numpy()
        print(f"[INFO] XY target: align to ground {target_xy}")
    else:
        target_xy = ee_start[:2]
        print(f"[INFO] XY target: frozen at EE start ({target_xy[0]:.4f},{target_xy[1]:.4f})")
    target_z = args_cli.target_z
    # CRITICAL: the env's fixed target Z (cfg.task.target_pos[2] = 0.4295) is
    # what the controller actually tracks. Override it so --target_z really
    # commands a lower setpoint (otherwise the error stays ~1cm and the
    # commanded force stays ~20N no matter what gains/clamp we use).
    env_unwrapped.fixed_target_z = target_z
    if args_cli.seed_vel_z != 0.0:
        # First-step damping: the env's FD velocity starts at 0, so step 1 has
        # no damping and the wrist free-falls at full clamp force. Pretend the
        # wrist was already descending at --seed_vel_z so step 1 damps.
        env_unwrapped.prev_ee_pos[:, 2] += args_cli.seed_vel_z / 15.0
        print(f"[INFO] Seeded first-step FD velocity: {args_cli.seed_vel_z:+.2f} m/s")
    print(f"[INFO] Z target: {target_z:.4f} (success_z={success_z:.4f}, contact ~0.4355)")
    print(f"[INFO] Start EE: {ee_start}")
    if args_cli.show_pose:
        oru0 = env_unwrapped.oru.data.root_pos_w[0].cpu().numpy()
        gnd0 = env_unwrapped.ground.data.root_pos_w[0].cpu().numpy()
        oru_q0 = env_unwrapped.oru.data.root_quat_w[0].cpu().numpy()
        print(f"[INFO] Start ORU rel ground: ({oru0[0]-gnd0[0]:+.4f},{oru0[1]-gnd0[1]:+.4f},"
              f"{oru0[2]-gnd0[2]:+.4f}) | ORU quat=({oru_q0[0]:+.3f},{oru_q0[1]:+.3f},"
              f"{oru_q0[2]:+.3f},{oru_q0[3]:+.3f}) | ground root: {gnd0}")
        # Chain geometry: each body's offset from the wrist (chain straightness)
        wrist = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        attr_map = {"Bridge": "bridge", "SixForce": "force_sensor", "Gripper": "gripper", "ORU": "oru"}
        for name, attr in attr_map.items():
            obj = getattr(env_unwrapped, attr)
            b = obj.data.root_pos_w[0].cpu().numpy()
            rel = b - wrist
            m = float(obj.data.default_mass[0].item())
            print(f"[INFO]   {name:9s} mass={m:.3f}kg", end="")
            com_b = None
            try:
                com_b = obj.data.body_com_pose_b
                if com_b is not None:
                    com_b = com_b[0].cpu().numpy().reshape(-1)[:3]
            except Exception:
                com_b = None
            com_s = ""
            if com_b is not None:
                com_s = f" | com_rel_root=({float(com_b[0]):+.4f},{float(com_b[1]):+.4f},{float(com_b[2]):+.4f})"
            print(f" | rel wrist: ({rel[0]:+.4f},{rel[1]:+.4f},{rel[2]:+.4f}){com_s}")

    action = torch.zeros(env_unwrapped.num_envs, env_unwrapped.cfg.action_space, device=env_unwrapped.device)

    # Two-phase: slow crawl near the surface (low Kp so the clamp never hits,
    # the Kd damps, descent is gentle) -> then main gains/target for the push.
    phase1_active = args_cli.phase1_steps > 0
    if phase1_active:
        g_p1 = g.clone()
        g_p1[0, 2] = args_cli.phase1_kp
        d_p1 = d.clone()
        d_p1[0, 2] = args_cli.kd_z if args_cli.kd_z is not None else 2.0 * args_cli.phase1_kp ** 0.5
        print(f"[INFO] Phase 1 ({args_cli.phase1_steps} steps): target {args_cli.phase1_target}, "
              f"Kp_z={args_cli.phase1_kp}, Kd_z={d_p1[0, 2].item():.1f}")

    success_step = None
    below_start = False
    for step in range(args_cli.steps):
        if phase1_active:
            if step == 0:
                env_unwrapped.fixed_target_z = args_cli.phase1_target
                env_unwrapped.base_gains = g_p1
                env_unwrapped.base_deriv = d_p1
            elif step == args_cli.phase1_steps:
                env_unwrapped.fixed_target_z = target_z
                env_unwrapped.base_gains = g
                env_unwrapped.base_deriv = d
                print(f"[INFO] Phase 2: target {target_z}, Kp_z={args_cli.kp_z}, "
                      f"Kd_z={kd_z:.1f}, max_force={args_cli.max_force}")
        obs, rew, terminated, truncated, info = env.step(action)

        ee = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        F = env_unwrapped.applied_wrench[0, :3].cpu().numpy()
        Fm = env_unwrapped._get_measured_force_mag()[0].item()
        cd = env_unwrapped._get_contact_degree()[0].item()

        # Gate-timing probe: does _sim_timestamp advance every substep?
        ts = float(env_unwrapped.robot._data._sim_timestamp)
        lts = float(env_unwrapped.last_update_timestamp)

        # Controller decomposition: recompute what _apply_action SHOULD have
        # commanded, from the SAME inputs it sees (obs-time values).
        # pos_error = target - ee ; F = Kp*err - Kd*v  (per oru_control).
        tgt = env_unwrapped.ground.data.root_pos_w[0].clone()
        tgt[2] = env_unwrapped.fixed_target_z
        pe = (tgt - env_unwrapped.ee_pos[0]).cpu().numpy()
        v = env_unwrapped.ee_linvel[0].cpu().numpy()
        kp = env_unwrapped.task_prop_gains[0].cpu().numpy()
        kd = env_unwrapped.task_deriv_gains[0].cpu().numpy()
        f_kp = kp[:3] * pe
        f_kd = -kd[:3] * v
        f_pred = f_kp + f_kd
        f_clamp = 20.0 if args_cli.max_force is None else args_cli.max_force
        f_pred_c = np.clip(f_pred, -f_clamp, f_clamp)

        if success_step is None and ee[2] < success_z:
            success_step = step
        if not below_start and ee[2] < ee_start[2] - 0.05:
            below_start = True  # actually moving down

        if step % 10 == 0 or step == args_cli.steps - 1:
            dxy = ((ee[0]-target_xy[0])**2 + (ee[1]-target_xy[1])**2) ** 0.5
            vel = env_unwrapped.ee_linvel_fd[0].cpu().numpy()
            vel_j = env_unwrapped.ee_linvel[0].cpu().numpy()  # Jacobian EE-origin velocity (what Kd now uses)
            oru_pos = env_unwrapped.oru.data.root_pos_w[0].cpu().numpy()
            gnd_pos = env_unwrapped.ground.data.root_pos_w[0].cpu().numpy()
            oru_rel = oru_pos - gnd_pos
            oru_xy_off = ((oru_rel[0])**2 + (oru_rel[1])**2) ** 0.5
            vel_r = env_unwrapped.robot.data.body_lin_vel_w[0, ee_idx].cpu().numpy()
            line = (
                f"Step {step:5d} | EE=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f}) "
                f"dXY={dxy:+.4f} dZ_to_success={ee[2]-success_z:+.4f} | "
                f"F_cmd=({F[0]:+6.2f},{F[1]:+6.2f},{F[2]:+6.2f})N "
                f"F_meas={Fm:6.2f}N contact={cd:4.2f}\n"
                f"    vel_fd=({vel[0]:+.3f},{vel[1]:+.3f},{vel[2]:+.3f}) "
                f"vel_jac=({vel_j[0]:+.3f},{vel_j[1]:+.3f},{vel_j[2]:+.3f}) "
                f"vel_physx=({vel_r[0]:+.3f},{vel_r[1]:+.3f},{vel_r[2]:+.3f}) | "
                f"err=({pe[0]:+.4f},{pe[1]:+.4f},{pe[2]:+.4f}) "
                f"F_kp=({f_kp[0]:+5.1f},{f_kp[1]:+5.1f},{f_kp[2]:+5.1f}) "
                f"F_kd=({f_kd[0]:+5.1f},{f_kd[1]:+5.1f},{f_kd[2]:+5.1f}) "
                f"F_pred=({f_pred_c[0]:+5.1f},{f_pred_c[1]:+5.1f},{f_pred_c[2]:+5.1f}) "
                f"ts={ts:.4f} lts={lts:.4f}\n"
                f"    ORU rel ground: ({oru_rel[0]:+.4f},{oru_rel[1]:+.4f},{oru_rel[2]:+.4f}) "
                f"xy_off={oru_xy_off:.4f}"
            )
            if args_cli.show_pose:
                oru_q = env_unwrapped.oru.data.root_quat_w[0].cpu().numpy()
                jp = env_unwrapped.robot.data.joint_pos[0].cpu().numpy()
                print(
                    f"{line}\n"
                    f"    ORU quat=({oru_q[0]:+.3f},{oru_q[1]:+.3f},{oru_q[2]:+.3f},{oru_q[3]:+.3f})\n"
                    f"    joint_pos(deg): "
                    + " ".join(f"{j*180/3.14159:+.1f}" for j in jp)
                )
            else:
                print(line)

    ee = env_unwrapped.robot.data.body_pos_w[0, ee_idx].cpu().numpy()
    print(f"\n[RESULT] Final EE Z: {ee[2]:.4f} vs success_z={success_z:.4f} "
          f"(gap {ee[2]-success_z:+.4f})")
    if success_step is not None:
        print(f"[RESULT] SUCCESS: EE below success_z at step {success_step} "
              f"({success_step/15.0:.1f}s) — insertion is physically possible.")
    else:
        print("[RESULT] FAILED: EE never reached success_z — insertion stalls. "
              "Try higher Kp_z / lower target_z, or the dead zone is eating the force.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
