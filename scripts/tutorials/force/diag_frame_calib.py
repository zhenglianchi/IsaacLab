"""Calibrate the PhysX jacobian frame — SINGLE-AXIS probing.

The controller maps the world-frame task wrench to joint torques: τ = Jᵀ·F.
Which frame is J expressed in? Previous tests (multi-axis PD) were
inconclusive: the arm drifted +X under EVERY mapping variant, and the
moment feedback diverged in all of them. Instead of guessing frames, this
script commands ONE axis at a time (pure +5N force, or +5Nm moment, all
other axes held at the current pose) and measures the ACTUAL motion
direction of the driven axis over ~15 steps.

If the mapping is correct, X force → +X motion, Y → +Y, Z → +Z,
Mx → rotation about +X, etc. Any mirror/rotation shows up as a sign flip
or axis swap in the very first steps.

Usage:
  python scripts/tutorials/force/diag_frame_calib.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-axis jacobian frame calibration.")
parser.add_argument("--task", type=str, default="Isaac-Oru-Direct-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=15, help="Sim steps per axis test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import axis_angle_from_quat, quat_mul
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.direct.oru import oru_control


def manual_step(env, wrench: torch.Tensor) -> torch.Tensor:
    """Apply a given task wrench through τ = Jᵀ·wrench for one physics step.

    The jacobian/ee_pos tensors are re-allocated on every scene.update(),
    so refresh AFTER the step for the next iteration and for the caller's
    post-step reads.
    """
    torque = (env.jacobian.transpose(1, 2) @ wrench.unsqueeze(-1)).squeeze(-1)
    torque = torque.clamp(min=-100.0, max=100.0)
    env.robot.set_joint_effort_target(torque, joint_ids=env._arm_joint_ids)
    env.scene.write_data_to_sim()  # flush effort targets to physics
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env._compute_intermediate_values(env.physics_dt)  # refresh aliases for next step / post-step reads
    return torque


def run_axis_test(u, kind: str, axis: int, steps: int, label: str):
    """Command a pure +5N force (kind='lin') or +5Nm moment (kind='ang') along
    one world axis; hold everything else at the current pose. Report the
    observed motion direction of the driven axis."""
    u._compute_intermediate_values(u.physics_dt)
    ee_idx = u._ee_frame_idx

    # Target: current pose, offset ONLY on the driven axis
    target_pos = u.ee_pos.clone()
    target_quat = u.ee_quat.clone()
    if kind == "lin":
        target_pos[:, axis] -= 0.05  # err +0.05 → +5N along +axis
    else:
        # +0.05 rad about world axis → +5Nm
        e = torch.zeros(3, device=u.device)
        e[axis] = 0.05
        q_off = torch.zeros(4, device=u.device)
        q_off[0] = torch.cos(e[axis] / 2)
        q_off[1 + axis] = torch.sin(e[axis] / 2)
        target_quat = quat_mul(q_off.unsqueeze(0), u.ee_quat)

    pos_err, rot_err = oru_control.get_pose_error(
        ee_pos=u.ee_pos, ee_quat=u.ee_quat,
        ctrl_target_ee_pos=target_pos, ctrl_target_ee_quat=target_quat,
        jacobian_type="geometric", rot_error_type="axis_angle",
    )
    delta = torch.cat([pos_err, rot_err], dim=-1)
    wrench = oru_control.task_space_pd(
        delta, u.ee_linvel_fd, u.ee_angvel_fd, u.base_gains, u.base_deriv
    )
    # Clamp (same as oru_control)
    max_f = u.cfg.task.max_task_force
    max_t = u.cfg.task.max_task_torque
    w_lim = torch.tensor([max_f] * 3 + [max_t] * 3, device=u.device)
    wrench = wrench.clamp(-w_lim, w_lim)

    w_cmd = wrench[0].cpu().numpy()
    print(f"\n=== {label} : command F/M = ({w_cmd[0]:+6.2f},{w_cmd[1]:+6.2f},{w_cmd[2]:+6.2f},"
          f"{w_cmd[3]:+6.2f},{w_cmd[4]:+6.2f},{w_cmd[5]:+6.2f}) ===")

    ee0 = u.ee_pos[0].cpu().numpy().copy()
    q0 = u.ee_quat[0].cpu().numpy().copy()
    jp0 = u.joint_pos[0].cpu().numpy().copy()
    tau0 = None
    for step in range(steps):
        tau = manual_step(u, wrench)
        if step == 0:
            tau0 = tau[0].cpu().numpy()
            print(f"  tau0 = ({tau0[0]:+6.2f},{tau0[1]:+6.2f},{tau0[2]:+6.2f},"
                  f"{tau0[3]:+6.2f},{tau0[4]:+6.2f},{tau0[5]:+6.2f}) Nm")
        if step < 5 or step == steps - 1:
            ee = u.ee_pos[0].cpu().numpy()
            d = ee - ee0
            jp = u.joint_pos[0].cpu().numpy()
            # actual rotation of the EE frame since test start
            q_now = u.ee_quat[0].cpu().numpy()
            dq = quat_mul(torch.tensor(q_now, device=u.device),
                          torch.tensor([q0[0], -q0[1], -q0[2], -q0[3]], device=u.device))
            rot = axis_angle_from_quat(dq.unsqueeze(0))[0].cpu().numpy()
            print(f"  step {step:2d}: dEE=({d[0]:+.5f},{d[1]:+.5f},{d[2]:+.5f}) "
                  f"dRot=({rot[0]:+.4f},{rot[1]:+.4f},{rot[2]:+.4f}) rad "
                  f"dJoint1={jp[1]-jp0[1]:+.5f} dJoint2={jp[2]-jp0[2]:+.5f}")

    ee = u.ee_pos[0].cpu().numpy()
    d = ee - ee0
    _, rot_err_f = oru_control.get_pose_error(
        ee_pos=u.ee_pos, ee_quat=u.ee_quat,
        ctrl_target_ee_pos=target_pos, ctrl_target_ee_quat=target_quat,
        jacobian_type="geometric", rot_error_type="axis_angle",
    )
    print(f"  => driven-axis displacement: {d[axis]:+.5f} m  (|rot_err| now {rot_err_f.norm(dim=-1)[0].item():.4f})")


def debug_physics(u):
    """Print scene physics params + direct joint-torque response."""
    print("\n=== DEBUG: physics params ===")
    print(f"  gravity cfg: {u.sim.cfg.gravity}")
    for attr in ["joint_stiffness", "joint_damping", "joint_max_effort"]:
        try:
            print(f"  {attr}: {getattr(u.robot.data, attr)[0].cpu().numpy()}")
        except Exception:
            print(f"  {attr}: N/A")
    try:
        masses = u.robot.root_physx_view.get_masses()
        print(f"  link masses:   {masses.cpu().numpy()}")
    except Exception as e:
        print(f"  get_masses failed: {e}")
    try:
        inertias = u.robot.root_physx_view.get_inertias()
        # (N, n_bodies, 9) or (N, n_bodies, 3) — print first env compactly
        print(f"  link inertias (shape {tuple(inertias.shape)}):")
        for i in range(inertias.shape[1]):
            print(f"    link {i}: {inertias[0, i].cpu().numpy()}")
    except Exception as e:
        print(f"  get_inertias failed: {e}")

    # Drive properties: maxForce / maxVelocity tell us if the USD clamps torque
    # Dump the runtime DRIVE MODEL properties — PhysX 5.x motor envelope:
    # [speedEffortGradient, maxActuatorVelocity, velocityDependentResistance].
    # This envelope constrains |effort| and |velocity| and is the likely
    # source of the ~75x effort attenuation seen in the torque tests.
    try:
        dm = u.robot.root_physx_view.get_dof_drive_model_properties()
        dm = dm[0].cpu().numpy()  # (max_dofs, 3)
        print(f"  drive model props (shape {dm.shape}) [grad, maxVel, vResist]:")
        for i, row in enumerate(dm[:7]):
            print(f"    dof {i}: grad={row[0]:+.6f} maxVel={row[1]:+.3f} vResist={row[2]:+.6f}")
    except Exception as e:
        print(f"  get_dof_drive_model_properties: {type(e).__name__}: {e}")

    # GRAVITY TORQUE PROBE: does omni.physics.tensors expose the generalized
    # gravity vector? (For the RL controller's gravity-compensation term.)
    have_g = False
    try:
        g = u.robot.root_physx_view.get_generalized_gravity()
        print(f"  get_generalized_gravity OK: {g[0].cpu().numpy()}")
        have_g = True
    except Exception as e:
        print(f"  get_generalized_gravity: {type(e).__name__}: {e}")
    if not have_g:
        # Fallback: tau_g = sum_l m_l * J_lin,l(q, com_l)^T * (0,0,-g)
        try:
            inertias = u.robot.root_physx_view.get_inertias()[0]   # (n_bodies, 10)
            n_bodies = inertias.shape[0]
            com = inertias[:, 1:4].clone()                          # (n_bodies, 3)
            jac_idx = torch.arange(n_bodies, device=u.device).repeat(u.num_envs, 1)
            local_poses = torch.zeros(u.num_envs, n_bodies, 7, device=u.device)
            local_poses[..., :3] = com.unsqueeze(0).expand(u.num_envs, -1, -1)
            local_poses[..., 3] = 1.0
            J_all = u.robot.root_physx_view.get_jacobians(jac_idx, local_poses)
            g_vec = torch.tensor([0.0, 0.0, -9.81], device=u.device)
            tau_g = torch.zeros(6, device=u.device)
            for l in range(n_bodies):
                Jl = J_all[0, l][:3, :6]                            # world-aligned linear rows
                tau_g += inertias[l, 0] * (Jl.T @ g_vec)
            print(f"  gravity torque (CoM jac sum): {tau_g.cpu().numpy()}")
        except Exception as e2:
            print(f"  gravity fallback failed: {type(e2).__name__}: {e2}")
    # NOTE: the ORU env arm spawns with disable_gravity=True (weightless), so
    # gravity compensation is NOT needed in the RL controller. get_gravity_
    # compensation_forces() assumes gravity ON — applying it to the zero-g arm
    # over-drives it upward (observed: +0.27m rise in the zero-action diag).
    try:
        from isaaclab.sim import SimulationContext
        from pxr import Usd
        stage = SimulationContext.instance().stage
        jnames = set(u.robot.joint_names)
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if prim.GetName() in jnames:
                lines = []
                for a in prim.GetAttributes():
                    nm = a.GetName()
                    if any(k in nm for k in ("rive", "orce", "elocity", "ax")):
                        try:
                            lines.append(f"{nm}={a.Get()}")
                        except Exception:
                            lines.append(f"{nm}=?")
                print(f"  joint {prim.GetName()} ({prim.GetTypeName()}): {', '.join(lines)}")
    except Exception as e:
        print(f"  joint prim dump failed: {type(e).__name__}: {e}")

    # Where does the ~435 kg*m2 effective inertia at the shoulder come from?
    # The mass matrix diagonal answers it: PhysX computes M(q) from the link
    # inertias AND the link offsets. A huge M[1,1] = a link is very far from
    # the shoulder axis (mis-scaled USD offsets), or a chain body is far away.
    u._compute_intermediate_values(u.physics_dt)
    M = u.mass_matrix[0].cpu().numpy()
    print(f"  mass matrix diag: {M[0,0]:.3f} {M[1,1]:.3f} {M[2,2]:.3f} {M[3,3]:.3f} {M[4,4]:.3f} {M[5,5]:.3f}")
    # body world positions + link index -> find the far-away bodies
    body_names = u.robot.body_names
    body_pos = u.robot.data.body_pos_w[0].cpu().numpy()
    root_pos = u.robot.data.root_pos_w[0].cpu().numpy()
    print(f"  root/base pos: {root_pos}")
    for i, (name, p) in enumerate(zip(body_names, body_pos)):
        d = p - root_pos
        print(f"    body {i:2d} {name:24s} pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) offset=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})")

    # Torque scaling test: does accel scale with commanded torque?
    #  - scales linearly  -> torque is applied, arm inertia is genuinely huge
    #  - constant ~0.05   -> drive clamps the effort (maxForce limit)
    for mag in (20.0, 200.0, 2000.0):
        u.reset()
        jp0 = u.robot.data.joint_pos[0, u._arm_joint_ids].cpu().numpy().copy()
        deltas = []
        for step in range(5):
            torque = torch.zeros(1, 6, device=u.device)
            torque[0, 1] = -mag
            u.robot.set_joint_effort_target(torque, joint_ids=u._arm_joint_ids)
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
            jp = u.robot.data.joint_pos[0, u._arm_joint_ids].cpu().numpy()
            deltas.append(jp[1] - jp0[1])
        vels = [deltas[i] - (deltas[i - 1] if i else 0.0) for i in range(len(deltas))]
        a_per_step = vels[-1] - vels[-2] if len(vels) > 1 else 0.0
        print(f"  tau=-{mag:5.0f}Nm: joint1 deltas {['%+.6f' % d for d in deltas]} "
              f"-> accel {a_per_step / (1 / 15.0) ** 2:8.4f} rad/s2")

    # BODY FORCE TEST: bypass the drive entirely — pull the ORU body down
    # with -50N. If the physics is healthy the whole assembly accelerates at
    # ~2.4 m/s2 (50N / 21kg). If the arm barely moves, the physics layer
    # itself is attenuated (sleep / mass scale / constraint).
    # COMPENSATION SWEEP: the tensors API attenuates the commanded drive
    # effort by ~1/8..1/64 (single-use force buffer vs 8 decimation substeps).
    # Sweep the multiplier: the correct one makes -20 Nm produce the full
    # Newton response: accel = 20/5.7 = 3.5 rad/s2, i.e. per-step deltas
    # 0.00012 x 8 = ~0.0010..0.0018 rad/step growing linearly.
    for mult in (8.0, 16.0, 32.0, 64.0):
        u.reset()
        jp0 = u.robot.data.joint_pos[0, u._arm_joint_ids].cpu().numpy().copy()
        torque = torch.zeros(1, 6, device=u.device)
        # wake pulse first (the arm sleeps at rest; a sleeping arm distorts the
        # early response), then the scaled command
        torque[0, 1] = -20.0
        u.robot.set_joint_effort_target(torque, joint_ids=u._arm_joint_ids)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        torque[0, 1] = -20.0 * mult
        deltas = []
        for step in range(6):
            u.robot.set_joint_effort_target(torque, joint_ids=u._arm_joint_ids)
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
            jp = u.robot.data.joint_pos[0, u._arm_joint_ids].cpu().numpy()
            deltas.append(jp[1] - jp0[1])
        print(f"  mult={mult:5.0f}: deltas {['%+.5f' % d for d in deltas]} "
              f"(full-Newton target ~0.008, 0.023, 0.039, 0.055, 0.070 rad/step)")
    u.reset()

    # Free fall: zero all torques, does the arm drop on its own?
    u.reset()
    jp0 = u.robot.data.joint_pos[0, u._arm_joint_ids].cpu().numpy().copy()
    ee0 = u.ee_pos[0].cpu().numpy().copy()
    torque = torch.zeros(1, 6, device=u.device)
    u.robot.set_joint_effort_target(torque, joint_ids=u._arm_joint_ids)
    u.scene.write_data_to_sim()
    print(f"\n=== DEBUG: free fall (zero effort), 10 steps ===")
    for step in range(10):
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        jp = u.robot.data.joint_pos[0, u._arm_joint_ids].cpu().numpy()
        ee = u.ee_pos[0].cpu().numpy()
        if step in (0, 1, 2, 4, 9):
            print(f"  step {step:2d}: dEE=({ee[0]-ee0[0]:+.5f},{ee[1]-ee0[1]:+.5f},{ee[2]-ee0[2]:+.5f}) "
                  f"dJoint1={jp[1]-jp0[1]:+.5f} dJoint2={jp[2]-jp0[2]:+.5f}")
    print("  (if the arm does NOT fall: something holds it - drive clamps or chain contact)")

    # Inertia-weighted test: pure Z force through the operational-space mass
    # matrix Lambda = (J M^-1 J^T)^-1. The pure force test above sweeps +X
    # because the arm's low-inertia direction at this pose is the shoulder
    # arc, not vertical. Lambda-weighting should make the EE behave like a
    # free point mass -> straight descent.
    u.reset()
    print("\n=== TEST: inertia-weighted -5N Z pull (Lambda-weighted), 15 steps ===")
    u._compute_intermediate_values(u.physics_dt)
    ee0 = u.ee_pos[0].cpu().numpy().copy()
    for step in range(15):
        u._compute_intermediate_values(u.physics_dt)
        raw = torch.zeros(1, 6, device=u.device)
        raw[0, 2] = -5.0
        M_inv = torch.inverse(u.mass_matrix)
        Lam = torch.inverse(u.jacobian @ M_inv @ u.jacobian.transpose(1, 2))
        wrench = (Lam @ raw.unsqueeze(-1)).squeeze(-1)
        torque = (u.jacobian.transpose(1, 2) @ wrench.unsqueeze(-1)).squeeze(-1).clamp(-100, 100)
        u.robot.set_joint_effort_target(torque, joint_ids=u._arm_joint_ids)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        if step in (0, 1, 2, 4, 9, 14):
            ee = u.ee_pos[0].cpu().numpy()
            print(f"  step {step:2d}: dEE=({ee[0]-ee0[0]:+.5f},{ee[1]-ee0[1]:+.5f},{ee[2]-ee0[2]:+.5f})")

    # Inertia-weighted FULL PD descent (the proposed controller change)
    u.reset()
    print("\n=== TEST: inertia-weighted full PD descent to target, 15 steps ===")
    u._compute_intermediate_values(u.physics_dt)
    ee0 = u.ee_pos[0].cpu().numpy().copy()
    for step in range(15):
        u._compute_intermediate_values(u.physics_dt)
        target_pos, _ = u._get_target_ref()
        pos_err, rot_err = oru_control.get_pose_error(
            ee_pos=u.ee_pos, ee_quat=u.ee_quat,
            ctrl_target_ee_pos=target_pos, ctrl_target_ee_quat=u.fixed_target_quat,
            jacobian_type="geometric", rot_error_type="axis_angle",
        )
        delta = torch.cat([pos_err, rot_err], dim=-1)
        raw_wrench = oru_control.task_space_pd(
            delta, u.ee_linvel_fd, u.ee_angvel_fd, u.base_gains, u.base_deriv
        )
        M_inv = torch.inverse(u.mass_matrix)
        Lam = torch.inverse(u.jacobian @ M_inv @ u.jacobian.transpose(1, 2))
        wrench = (Lam @ raw_wrench.unsqueeze(-1)).squeeze(-1)
        max_f = u.cfg.task.max_task_force
        max_t = u.cfg.task.max_task_torque
        w_lim = torch.tensor([max_f] * 3 + [max_t] * 3, device=u.device)
        wrench = wrench.clamp(-w_lim, w_lim)
        torque = (u.jacobian.transpose(1, 2) @ wrench.unsqueeze(-1)).squeeze(-1).clamp(-100, 100)
        u.robot.set_joint_effort_target(torque, joint_ids=u._arm_joint_ids)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        if step in (0, 1, 2, 4, 9, 14):
            ee = u.ee_pos[0].cpu().numpy()
            d = ee - ee0
            print(f"  step {step:2d}: dEE=({d[0]:+.5f},{d[1]:+.5f},{d[2]:+.5f})")

    # Full-PD reference: replicate the 'direct' phase (descend to target)
    u.reset()
    print(f"\n=== DEBUG: full PD reference (descend to target), 10 steps ===")
    for step in range(10):
        u._compute_intermediate_values(u.physics_dt)
        target_pos, _ = u._get_target_ref()
        pos_err, rot_err = oru_control.get_pose_error(
            ee_pos=u.ee_pos, ee_quat=u.ee_quat,
            ctrl_target_ee_pos=target_pos, ctrl_target_ee_quat=u.fixed_target_quat,
            jacobian_type="geometric", rot_error_type="axis_angle",
        )
        delta = torch.cat([pos_err, rot_err], dim=-1)
        wrench = oru_control.task_space_pd(
            delta, u.ee_linvel_fd, u.ee_angvel_fd, u.base_gains, u.base_deriv
        )
        max_f = u.cfg.task.max_task_force
        max_t = u.cfg.task.max_task_torque
        w_lim = torch.tensor([max_f] * 3 + [max_t] * 3, device=u.device)
        wrench = wrench.clamp(-w_lim, w_lim)
        torque = (u.jacobian.transpose(1, 2) @ wrench.unsqueeze(-1)).squeeze(-1).clamp(-100, 100)
        u.robot.set_joint_effort_target(torque, joint_ids=u._arm_joint_ids)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        ee = u.ee_pos[0].cpu().numpy()
        print(f"  step {step:2d}: EE=({ee[0]:+.5f},{ee[1]:+.5f},{ee[2]:+.5f})")


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)
    # Exact home pose — start straight above the target, no randomization
    env_cfg.task.fixed_ik_offset_pos = (0.0, 0.0, 0.0)
    env_cfg.task.fixed_ik_offset_rot = (0.0, 0.0, 0.0)
    # SOLVER EXPERIMENT: TGS with 1 velocity iteration is the prime suspect
    # for the ~75x force attenuation. Bump velocity iterations massively.
    env_cfg.sim.physx.max_velocity_iteration_count = 64

    env = gym.make(args_cli.task, cfg=env_cfg)
    u: DirectRLEnv = env.unwrapped

    debug_physics(u)
    env.close()
    print("\n[DIAG DONE] (axis/ lambda tests skipped - questions settled)")
    return

    AXES = ["X", "Y", "Z"]
    for kind, kind_name, name_suffix in [("lin", "force", "N"), ("ang", "moment", "Nm")]:
        for axis in range(3):
            env.reset()
            run_axis_test(u, kind, axis, args_cli.steps, f"{kind_name} +5{name_suffix} on +{AXES[axis]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
