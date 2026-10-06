"""Checkpoint-free fixed-impedance pilot and reset-isolation regression check."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--steps', type=int, default=225)
parser.add_argument('--nominal', action='store_true')
parser.add_argument('--bias', type=float, default=None,
                    help='override cfg.task.insertion_bias in m (virtual equilibrium below the seat, '
                         'contact-gated). Default None keeps the task config value.')
parser.add_argument('--insertion-bias-legacy-default', type=float, default=0.01, help=argparse.SUPPRESS)
parser.add_argument('--max-force', type=float, default=None,
                    help='override cfg.task.max_task_force (free-space XY/Z cap, default 8 N); lower = gentler approach')
parser.add_argument('--max-force-z', type=float, default=None,
                    help='override cfg.task.max_task_force_z (default 60 N) to push harder than the cap')
parser.add_argument('--start-z-offset', type=float, default=None,
                    help='constant vertical bias of the reset start height in m '
                         '(e.g. -0.15 lowers every episode start by 15 cm; the x/y '
                         'randomization stays active)')
parser.add_argument('--reference-mode', choices=['ramp', 'setpoint'], default=None,
                    help="'ramp' = rate-limited anchor (default); 'setpoint' = single target point "
                         "(classic fixed-impedance baseline, excites the outward arc)")
parser.add_argument('--reference-speed', type=float, default=None,
                    help='override cfg.task.reference_speed in m/s (default 0.02); the anchor ramps at this rate')
parser.add_argument('--friction', type=float, default=None,
                    help='override static/dynamic friction (defaults 1.0/1.0). mu=1.0 -> 45 deg friction cone: self-locking')
parser.add_argument('--kp-scale', type=float, default=1.0,
                    help='scale all task Kp (default 1.0). Stiffer -> more force AND more oscillation')
parser.add_argument('--kd-scale', type=float, default=1.0,
                    help='scale all task Kd (default 1.0). Over-damping is the documented anti-limit-cycle lever')
parser.add_argument('--ground-drop', type=float, default=0.0,
                    help='lower the docking surface by this many metres ON TOP of the configured value '
                         '(positive = deeper effective hole / more penetration). Success is TWO-SIDED: '
                         '|gap| < seat_z_tolerance, so overshooting fails too - aim for oru_z ~= oru_seat_z')
parser.add_argument('--oru-usd', type=str, default=None,
                    help='override cfg.scene.ORU.spawn.usd_path (e.g. assets/USD/oru_sdf0/ORU.usd) to A/B '
                         'collision approximations without editing the task config')
parser.add_argument('--max-torque', type=float, default=None,
                    help='override cfg.task.max_task_torque in Nm (default 6.0). The pin/hole fit here is '
                         'line-to-line (2.5 mm pin in 2.6 mm hole): a 1.5 deg tilt displaces the tip ~7x the '
                         'clearance, and with mu=1.0 that wedge self-locks - rotational authority is the lever')
parser.add_argument('--penetration', type=float, default=0.0,
                    help='collision penetration allowance in metres: sets rest_offset = -penetration on the ORU, '
                         'so PhysX lets the two collision surfaces overlap by that much before they rest. '
                         'This is a real contact allowance, not a frame translation. Caveat: it applies in every '
                         'direction, so once it exceeds the pin radius (2.5 mm) the pin/hole lateral contact is gone')
parser.add_argument('--hold-force', type=float, default=None,
                    help='override cfg.task.hold_force in N: the constant downward preload applied '
                         'once the ORU is geometrically seated and the feedback gains are frozen. Zeroing the '
                         'gains alone lets the ORU float up (nothing pulls the weightless chain down)')
parser.add_argument('--depen-vel', type=float, default=None,
                    help='override max_depenetration_velocity (m/s) on the ORU and the chain bodies; '
                         'PhysX resolves over-penetration up to this speed, so it bounds the seat ejection')
parser.add_argument('--rand-pos', type=float, default=None,
                    help='override ik_rand_pos_noise per axis in m (e.g. 0.03 = +/-3 cm). '
                         'The shipped config uses a +-1 cm pilot; the legacy design used +-12 cm.')
parser.add_argument('--rand-rot', type=float, default=None,
                    help='override ik_rand_rot_noise per axis in rad (e.g. 0.0524 = +/-3 deg)')
parser.add_argument('--seed', type=int, default=1234,
                    help='env seed for the reset randomization; vary it to build an evaluation set '
                         '(all envs share one seed, so use --num-envs to get many conditions per run)')
parser.add_argument('--contact-force-threshold', type=float, default=None,
                    help='override cfg.task.contact_force_threshold in N (default 0.5) - for the switch '
                         'threshold sweep required by the stage-switch verification protocol')
parser.add_argument('--fault-inject', choices=['none', 'zero', 'bias', 'delay'], default='none',
                    help='deliberately corrupt the contact signal to prove the switch actually matters: '
                         'zero (always 0 N), bias (add a constant N to every component), delay (replay the '
                         'value from --fault-delay steps ago)')
parser.add_argument('--fault-bias', type=float, default=2.0, help='bias in N for --fault-inject bias')
parser.add_argument('--fault-delay', type=int, default=5, help='steps for --fault-inject delay')
parser.add_argument('--output', default='.installation/oru_v2_diagnostic')
parser.add_argument('--head', action='store_true', help='open the Isaac Sim viewer instead of running headless')
parser.add_argument('--real-time', action='store_true', help='pace steps at the 15 Hz policy rate for viewing')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.chdir(root)
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=not args.head)
env = None
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply
    cfg = parse_env_cfg('Isaac-Oru-Direct-v0', device='cuda:0', num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.rand_pos is not None:
        cfg.task.ik_rand_pos_noise = (args.rand_pos,) * 3
        cfg.task.ik_rand_pos_bounds = None  # symmetric sweep overrides the per-axis bounds
        print(f'[INFO] ik_rand_pos_noise = +/-{args.rand_pos * 100:.1f} cm per axis', flush=True)
    if args.rand_rot is not None:
        cfg.task.ik_rand_rot_noise = (args.rand_rot,) * 3
        print(f'[INFO] ik_rand_rot_noise = +/-{args.rand_rot:.4f} rad per axis', flush=True)
    cfg.task.experiment_method = 'fixed'
    if args.ground_drop:
        gx, gy, gz = cfg.scene.Ground.init_state.pos
        cfg.scene.Ground.init_state.pos = (gx, gy, gz - args.ground_drop)
        print(f'[INFO] Ground dropped by {args.ground_drop * 1000:.1f} mm -> z={gz - args.ground_drop:.5f} '
              f'(configured {gz:.5f}); ORU seat reference stays {cfg.task.oru_seat_z:.5f}', flush=True)
    if args.oru_usd:
        cfg.scene.ORU.spawn.usd_path = args.oru_usd
        print(f'[INFO] ORU asset -> {args.oru_usd}', flush=True)
    if args.penetration:
        if cfg.scene.ORU.spawn.collision_props is None:
            from isaaclab.sim.schemas import schemas_cfg
            cfg.scene.ORU.spawn.collision_props = schemas_cfg.CollisionPropertiesCfg()
        cfg.scene.ORU.spawn.collision_props.rest_offset = -abs(args.penetration)
        # PhysX: PxShape::setContactOffset requires contactOffset > 0 AND > restOffset.
        if not cfg.scene.ORU.spawn.collision_props.contact_offset:
            cfg.scene.ORU.spawn.collision_props.contact_offset = 0.02
        print(f'[INFO] ORU penetration allowance = {abs(args.penetration) * 1000:.1f} mm '
              f'(rest_offset = {cfg.scene.ORU.spawn.collision_props.rest_offset}, '
              f'contact_offset = {cfg.scene.ORU.spawn.collision_props.contact_offset})', flush=True)
    if args.depen_vel is not None:
        cfg.scene.ORU.spawn.rigid_props.max_depenetration_velocity = args.depen_vel
        for name in ('Bridge', 'SixForce', 'Gripper'):
            asset = getattr(cfg.scene, name, None)
            if asset is not None and getattr(asset.spawn, 'rigid_props', None) is not None:
                asset.spawn.rigid_props.max_depenetration_velocity = args.depen_vel
        print(f'[INFO] max_depenetration_velocity = {args.depen_vel} m/s (ORU + chain)', flush=True)
    if args.bias is not None:
        cfg.task.insertion_bias = args.bias
        print(f'[INFO] insertion_bias = {cfg.task.insertion_bias} m', flush=True)
    if args.contact_force_threshold is not None:
        cfg.task.contact_force_threshold = args.contact_force_threshold
        print(f'[INFO] contact_force_threshold = {args.contact_force_threshold} N '
              f'(leave {cfg.task.contact_leave_threshold} N release)', flush=True)
    if args.max_force is not None:
        cfg.task.max_task_force = args.max_force
    if args.max_torque is not None:
        cfg.task.max_task_torque = args.max_torque
    if args.hold_force is not None:
        cfg.task.hold_force = args.hold_force
    if args.max_force_z is not None:
        cfg.task.max_task_force_z = args.max_force_z
    if args.start_z_offset is not None:
        cfg.task.start_z_offset = args.start_z_offset
        print(f'[INFO] start_z_offset = {cfg.task.start_z_offset * 100:+.1f} cm', flush=True)
    if args.reference_mode is not None:
        cfg.task.reference_mode = args.reference_mode
        print(f'[INFO] reference_mode = {cfg.task.reference_mode}', flush=True)
    if args.reference_speed is not None:
        cfg.task.reference_speed = args.reference_speed
    if args.friction is not None:
        cfg.sim.physics_material.static_friction = args.friction
        cfg.sim.physics_material.dynamic_friction = args.friction
    if args.kp_scale != 1.0:
        cfg.task.default_task_prop_gains = tuple(g * args.kp_scale for g in cfg.task.default_task_prop_gains)
    if args.kd_scale != 1.0:
        cfg.task.default_task_deriv_gains = tuple(g * args.kd_scale for g in cfg.task.default_task_deriv_gains)
    if args.nominal:
        cfg.task.fixed_ik_offset_pos = (0., 0., 0.)
    env = gym.make('Isaac-Oru-Direct-v0', cfg=cfg)
    obs, _ = env.reset()
    task = env.unwrapped
    assert obs['policy'].shape == (args.num_envs, 57)

    # ── Fault injection on the contact signal ──────────────────────────
    # Deliberately corrupting the detector is the only way to show that the stage
    # switch has an effect at all (otherwise "it works with or without the switch"
    # cannot be excluded). Patched on the vector provider, which the magnitude
    # helper and the stage logic both go through.
    if args.fault_inject != 'none':
        _orig_contact_vec = task._get_contact_force_vec
        _delay_buf = []

        def _faulty_contact_vec():
            v = _orig_contact_vec()
            if args.fault_inject == 'zero':
                return torch.zeros_like(v)
            if args.fault_inject == 'bias':
                return v + args.fault_bias
            _delay_buf.append(v.clone())
            while len(_delay_buf) <= args.fault_delay:
                _delay_buf.insert(0, v.clone())
            return _delay_buf.pop(0)

        task._get_contact_force_vec = _faulty_contact_vec
        print(f"[INFO] FAULT INJECTION on contact force: {args.fault_inject} "
              f"(bias={args.fault_bias} N, delay={args.fault_delay} steps)", flush=True)

    # Resetting one environment must not advance time or move any other body.
    if args.num_envs > 1:
        before = [a.data.root_state_w[1:].clone() for a in (task.robot, task.oru)]
        joint_before = task.robot.data.joint_pos[1:].clone()
        time_before = task.sim.current_time
        task._reset_idx(torch.tensor([0], device=task.device))
        assert task.sim.current_time == time_before, 'Partial reset advanced physics time'
        for asset, state in zip((task.robot, task.oru), before):
            assert torch.allclose(asset.data.root_state_w[1:], state, atol=1e-6), 'Partial reset moved another env'
        assert torch.allclose(task.robot.data.joint_pos[1:], joint_before, atol=1e-6)
    print('RESET ISOLATION PASS', flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    success = torch.zeros(args.num_envs, dtype=torch.bool, device=task.device)
    max_force = 0.
    min_oru_z = float("inf")
    min_oru_step = -1
    max_streak = 0
    obs_mismatch = 0
    # Per-env peak of the consecutive seat-criterion streak: >= 5 steps means stable
    # success under the env's own rule, so this gives a success rate across envs.
    streak_peak = torch.zeros(args.num_envs, device=task.device)
    # CORRECT success accounting. The env terminates AND resets on the very step the seat
    # criterion has held for success_hold_s, and _reset_buffers clears _success_count, so
    # reading the counter after env.step() always shows one step less than required (a
    # successful episode looks like 'streak 4 / stable_success False'). Instead, read the
    # pre-reset success fraction the env publishes in extras['log'] on that step.
    n_reset_events = 0
    n_success_events = 0.0
    success_rate_samples = []
    last_reset_step = torch.zeros(args.num_envs, dtype=torch.long, device=task.device)
    ep_lengths = []  # steps between consecutive resets of the same env
    # Completion time of the CURRENT episode per env, sampled every step because
    # _reset_buffers clears ep_success_times during the reset step.
    last_success_step = torch.zeros(args.num_envs, dtype=torch.long, device=task.device)
    finished_success_steps = []
    seat = float(cfg.task.oru_seat_z)
    ee_seat = float(cfg.task.success_z)
    print(f"[INFO] targets: oru_seat_z={seat:.5f}  ee success_z={ee_seat:.5f}  "
          f"tolerances: xy {cfg.task.xy_tolerance * 1000:.2f} mm, gap {cfg.task.seat_z_tolerance * 1000:.2f} mm, "
          f"tilt {cfg.task.seat_angle_tolerance:.4f} rad, spd {cfg.task.success_speed_tolerance}, "
          f"angspd {cfg.task.success_angular_speed_tolerance}", flush=True)
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step','env','ee_z','ee_x','ee_y','oru_z','oru_x','oru_y','xy_error','angle_rad','gap_to_seat_m',
                         'phase','alpha','contact_alpha','reference_z',
                         'ee_xy_error','ee_angle_rad','ee_gap_to_seat_m',
                         'kp_z','command_fz','wrist_fx_world','wrist_fy_world','wrist_fz_world','stable_success',
                         'oru_spd_m_s','oru_angspd_rad_s','cand','succ_count',
                         'oru_wx','oru_wy','oru_wz','oru_qw','oru_qx','oru_qy','oru_qz',
                         'contact_force_N','contact_flag','contact_fx','contact_fy','contact_fz',
                         'rew_stage1','rew_stage2'])
        for step in range(args.steps):
            loop_start = time.time()
            obs, reward, terminated, truncated, info = env.step(torch.zeros(env.action_space.shape, device=task.device))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            force = quat_apply(task.ee_quat, task.robot.data.body_incoming_joint_wrench_b[:, task._ee_frame_idx, :3])
            max_force = max(max_force, torch.linalg.vector_norm(force, dim=-1).max().item())
            success |= task.ep_succeeded
            xy = torch.linalg.vector_norm(task.oru.data.root_pos_w[:, :2] - task.ground.data.root_pos_w[:, :2], dim=-1)
            _xy_dbg, gap_dbg, angle_dbg = task._oru_pose_errors()
            cand_now = task._get_curr_successes(task.cfg.task.success_threshold)
            contact_vec = task._get_contact_force_vec()
            contact_vec_mag = torch.linalg.vector_norm(contact_vec, dim=-1)
            # obs <-> internal stage state consistency (guards against stale gating:
            # _update_stage_state runs only once per policy step)
            stage_obs = obs['policy'][0, :5].float()
            stage_expect = torch.stack((
                task._insertion_phase[0].float(), task._stage_alpha[0], task._contact_alpha[0],
                task._control_z[0] - task.fixed_target_z, task._best_insertion_gap[0],
            ))
            if not torch.allclose(stage_obs, stage_expect, atol=1e-5):
                obs_mismatch += 1
                if obs_mismatch <= 3:
                    print(f"[WARN] obs stage_state mismatch at step {step}: "
                          f"obs={[round(v, 5) for v in stage_obs.tolist()]} "
                          f"internal={[round(v, 5) for v in stage_expect.tolist()]}", flush=True)
            z_now = float(task.oru.data.root_pos_w[0, 2])
            # The seat criterion is measured on the EE frame; log it alongside the
            # ORU-based proxies so the report checks the criterion that actually runs.
            ee_xy, ee_gap, ee_angle = task._ee_pose_errors()
            if z_now < min_oru_z:
                min_oru_z, min_oru_step = z_now, step
            max_streak = max(max_streak, int(task._success_count[0].item()))
            streak_peak = torch.maximum(streak_peak, task._success_count.float())
            _st = getattr(task, "ep_success_times", None)
            if _st is not None:
                _hit = _st > 0
                last_success_step = torch.where(_hit, _st.long(), last_success_step)
            reset_now = task.episode_length_buf <= 0
            if torch.any(reset_now):
                _re = torch.nonzero(reset_now, as_tuple=False).flatten()
                for _e in _re.tolist():
                    if int(last_success_step[_e].item()) > 0:
                        finished_success_steps.append(int(last_success_step[_e].item()))
                    last_success_step[_e] = 0
                for _e in torch.nonzero(reset_now, as_tuple=False).flatten().tolist():
                    _iv = int(step) - int(last_reset_step[_e].item())
                    if _iv > 20:
                        ep_lengths.append(_iv)
                    last_reset_step[_e] = int(step)
                n_reset = int(reset_now.sum().item())
                n_reset_events += n_reset
                rate = float(task.extras.get('log', {}).get('episode_success_rate', float('nan')))
                if rate == rate:  # not NaN: the env logged this reset batch
                    n_success_events += rate * n_reset
                    success_rate_samples.append(rate)
            # live readout: watch the seated height while the viewer runs
            if step % 50 == 0 or step == args.steps - 1:
                print(f"[step {step:4d}] ee_z={float(task.ee_pos[0, 2]):.5f}  oru_z={z_now:.5f}  "
                      f"target: ee={float(ee_gap[0].item()) * 1000:+6.2f} / oru={float(gap_dbg[0].item()) * 1000:+6.2f} mm  "
                      f"tilt(oru)={float(task._oru_pose_errors()[2][0].item()):.4f} rad  xy(ee)={float(ee_xy[0].item()) * 1000:.2f} mm  "
                      f"cand={int(cand_now[0].item())}  streak={int(task._success_count[0].item())}"
                      f"  phase={int(task._insertion_phase[0].item())}  contact={float(task._contact_alpha[0].item()):.2f}"
                      f"  Fcontact={float(task._get_contact_force_mag()[0].item()):.3f} N",
                      flush=True)
            rows = torch.stack((task.ee_pos[:, 2], task.ee_pos[:, 0], task.ee_pos[:, 1],
                                task.oru.data.root_pos_w[:, 2],
                                task.oru.data.root_pos_w[:, 0], task.oru.data.root_pos_w[:, 1],
                                xy, angle_dbg, gap_dbg,
                                ee_xy, ee_angle, ee_gap,
                                task._insertion_phase.float(), task._stage_alpha, task._contact_alpha, task._control_z,
                                task.task_prop_gains[:, 2], task.applied_wrench[:, 2], force[:, 0], force[:, 1], force[:, 2],
                                task._stable_success.float(),
                                task._oru_speed,  # finite-difference (the channel the criterion uses)
                                task._oru_angspd,  # finite-difference (the channel the criterion uses)
                                cand_now.float(),
                                task._success_count.float(),
                                task.oru.data.root_ang_vel_w[:, 0], task.oru.data.root_ang_vel_w[:, 1],
                                task.oru.data.root_ang_vel_w[:, 2],
                                task.oru.data.root_quat_w[:, 0], task.oru.data.root_quat_w[:, 1],
                                task.oru.data.root_quat_w[:, 2], task.oru.data.root_quat_w[:, 3],
                                contact_vec_mag, task._was_in_contact.float(),
                                contact_vec[:, 0], contact_vec[:, 1], contact_vec[:, 2],
                                task.stage1_reward, task.stage2_reward),
                           dim=1).cpu().tolist()
            # Autoreset rows are a new initial state; omit rather than mislabel terminal data.
            reset_ids = (terminated | truncated).cpu().tolist()
            for env_id, row in enumerate(rows):
                if not reset_ids[env_id]:
                    writer.writerow([step, env_id, *row])
            if args.real_time:
                sleep_time = task.step_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    final_oru = float(task.oru.data.root_pos_w[0, 2])
    final_ee = float(task.ee_pos[0, 2])
    hold = math.ceil(cfg.task.success_hold_s / task.step_dt)
    print("\n================ SEAT CHECK ================", flush=True)
    print(f"  ORU target (oru_seat_z) : {seat:.5f}", flush=True)
    print(f"  ORU lowest reached      : {min_oru_z:.5f} at step {min_oru_step}  "
          f"gap {(min_oru_z - seat) * 1000:+6.2f} mm  -> "
          f"{'PASS' if abs(min_oru_z - seat) < cfg.task.seat_z_tolerance else 'FAIL (|gap| >= 2 mm)'}", flush=True)
    print(f"  ORU final               : {final_oru:.5f}  gap {(final_oru - seat) * 1000:+6.2f} mm", flush=True)
    print(f"  EE  target (success_z)  : {ee_seat:.5f}", flush=True)
    print(f"  EE  final               : {final_ee:.5f}  diff {(final_ee - ee_seat) * 1000:+6.2f} mm", flush=True)
    print(f"  stable_success          : {bool(success.any().item())}  "
          f"(longest cand streak {max_streak} steps, needs {hold})", flush=True)
    print(f"  obs stage_state mismatch: {obs_mismatch} steps  (0 expected)", flush=True)
    n_success = int((streak_peak >= 5).sum().item())
    ep_rate = (n_success_events / n_reset_events) if n_reset_events else float("nan")
    import statistics as _st
    mean_ep_len = (sum(ep_lengths) / len(ep_lengths)) if ep_lengths else float("nan")
    med_ep_len = _st.median(ep_lengths) if ep_lengths else float("nan")
    p95_ep_len = (sorted(ep_lengths)[int(0.95 * (len(ep_lengths) - 1))] if ep_lengths else float("nan"))
    print("  --- TRUE success accounting (episodes that ran to a decision) ---", flush=True)
    print(f"  episodes finished        : {n_reset_events}  (success {n_success_events:.0f})", flush=True)
    print(f"  SUCCESS RATE             : {100.0 * ep_rate:.1f}%   "
          f"episode length mean/median/P95 = {mean_ep_len:.0f}/{med_ep_len:.0f}/{p95_ep_len:.0f} steps "
          f"({mean_ep_len / 15.0:.1f}/{med_ep_len / 15.0:.1f}/{p95_ep_len / 15.0:.1f} s)", flush=True)
    if finished_success_steps:
        _fs = sorted(finished_success_steps)
        print(f"  success step  median/P95 = {_fs[len(_fs)//2]} / {_fs[int(0.95*(len(_fs)-1))]} steps "
              f"({_fs[len(_fs)//2]/15.0:.1f} / {_fs[int(0.95*(len(_fs)-1))]/15.0:.1f} s)  over {len(_fs)} envs",
              flush=True)
    print(f"  [obsolete] peak streak   : {int(streak_peak.max().item())} steps "
          f"(always one less than required: the reset clears it)", flush=True)
    print("===========================================", flush=True)

    summary = {'num_envs':args.num_envs, 'steps':args.steps, 'nominal':args.nominal, 'bias_m':float(cfg.task.insertion_bias),
               'penetration_m':args.penetration, 'oru_usd':args.oru_usd,
               'contact_force_threshold_N':float(cfg.task.contact_force_threshold),
               'fault_inject':args.fault_inject, 'obs_stage_mismatch_steps':obs_mismatch,
               'seed':args.seed, 'episodes_finished':n_reset_events,
               'episodes_succeeded':n_success_events,
               'success_rate_true':ep_rate, 'mean_episode_steps':mean_ep_len,
               'success_envs_obsolete_streak_metric':n_success,
               'success_rate':n_success / max(args.num_envs, 1),
               'streak_peak_per_env':[float(x) for x in streak_peak.cpu().tolist()],
               'oru_seat_target_m':seat, 'oru_lowest_m':min_oru_z, 'oru_final_m':final_oru,
               'gap_to_target_mm':(min_oru_z - seat) * 1000, 'ee_final_m':final_ee,
               'max_cand_streak_steps':max_streak,
               'max_task_force_z_N':float(cfg.task.max_task_force_z), 'max_task_force_N':float(cfg.task.max_task_force),
               'reference_speed_m_s':float(cfg.task.reference_speed),
               'friction':float(cfg.sim.physics_material.static_friction),
               'kp_z_N_per_m':float(cfg.task.default_task_prop_gains[2]),
               'kd_z_Ns_per_m':float(cfg.task.default_task_deriv_gains[2]),
               'ever_success_fraction_observed':success.float().mean().item(),
               'max_wrist_force_policy_samples_N':max_force,
               'note':'Pilot only: wrist reaction is not isolated contact force; policy-rate samples miss substep peaks.'}
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('DIAGNOSTIC PASS ' + json.dumps(summary), flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    from isaaclab.sim import SimulationContext
    sim = SimulationContext.instance()
    if sim is not None:
        sim.clear_all_callbacks()
        sim.clear_instance()
    if env is not None:
        env.close()
    launcher.app.close(wait_for_replicator=False)
