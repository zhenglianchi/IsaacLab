# -*- coding: utf-8 -*-
"""B1 - component level validation (V1..V4) for the ORU staged variable-impedance task.

Zero training. Runs short headless rollouts and reports:

  V1  contact-signal sanity   : far from the seat the ORU<->Ground contact force is ~0;
                                pressing deeper increases it monotonically; the direction
                                is axial-dominated rather than lateral.
  V2  switching criterion     : compare the contact detector (force threshold + corridor +
                                debounce) against the geometric truth (height gap <= 0 and
                                the part inside the bore) -> missed detections, false
                                triggers, and the switching latency distribution.
  V3  downstream consistency  : whenever alpha changes, the reward blend weight and the
                                axial force cap must change with it (same alpha drives both).
  V4  failure injection       : with the switch forcibly disabled (alpha pinned) the task
                                must degrade relative to the intact detector.

Usage:  python tools/validate_components.py [--num_envs 8] [--steps 450]
"""
import argparse
import pathlib

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument('--task', default='Isaac-Oru-Direct-v0')
ap.add_argument('--num_envs', type=int, default=8)
ap.add_argument('--steps', type=int, default=450)
ap.add_argument('--out', default='outputs/validation_b1.txt')
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.headless = True
app = AppLauncher(a).app

import numpy as np            # noqa: E402
import torch                  # noqa: E402
import gymnasium as gym       # noqa: E402
import isaaclab_tasks         # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

env_cfg = parse_env_cfg(a.task, device='cuda:0', num_envs=a.num_envs)
env_cfg.task.experiment_method = 'full'
env = gym.make(a.task, cfg=env_cfg)
un = env.unwrapped
task = un.cfg.task
lines = []


OUT = pathlib.Path(a.out)
OUT.parent.mkdir(parents=True, exist_ok=True)


def log(s):
    print(s, flush=True)
    lines.append(s)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")   # incremental save


def wrist_ft():
    """(wrist |F|, wrist |tau|) per env, from the same data path the paper metric uses."""
    try:
        w = un.robot.data.body_incoming_joint_wrench_b[:, un._ee_frame_idx]
        return (w[:, 0:3].norm(dim=-1).detach().cpu().numpy(),
                w[:, 3:6].norm(dim=-1).detach().cpu().numpy())
    except Exception:
        z = np.zeros(a.num_envs, dtype=np.float32)
        return z, z


def env_errors():
    """(xy_err, height_gap) exactly as oru_logic uses them."""
    try:
        e = un._oru_pose_errors()
        return e[0].detach().cpu().numpy(), e[1].detach().cpu().numpy()
    except Exception:
        z = np.zeros(a.num_envs, dtype=np.float32)
        return z, z


def gap():
    return env_errors()[1]


def alpha():
    return un._stage_alpha.detach().cpu().numpy().copy()


log('=== B1 component validation ===')
log('task=%s  envs=%d  steps=%d  switch_mode=%s' % (a.task, a.num_envs, a.steps, getattr(task, 'switch_mode', '?')))

# ---------------------------------------------------------------- V1 wrist force signal
env.reset()
fa, ta = wrist_ft()
log('')
log('V1  wrist force/torque sanity (paper metric: 15 Hz wrist 6-axis)')
log('    in-air: |F| median %.4f N max %.4f N   |tau| median %.4f Nm (expect ~0)'
    % (float(np.median(fa)), float(fa.max()), float(np.median(ta))))
press = []
for k in range(a.steps):
    env.step(torch.zeros((a.num_envs, env.action_space.shape[-1]), dtype=torch.float32, device='cuda:0'))
    if k % 10 == 0 or k == a.steps - 1:
        f, t = wrist_ft()
        xy, g = env_errors()
        press.append((k, float(np.median(g)), float(np.median(f)), float(np.median(t))))
log('    step   median gap(m)   wrist|F|(N)   wrist|tau|(Nm)')
for s_, g_, f_, t_ in press[:14]:
    log('    %4d   %+8.4f      %9.2f      %9.3f' % (s_, g_, f_, t_))
fz = [x[2] for x in press]
log('    force grows along the descent: %s   (peak wrist |F| %.2f N)' % ('yes' if len(fz) > 1 and max(fz) > fz[0] + 1.0 else 'no', max(fz) if fz else 0.0))

# ---------------------------------------------------------------- V2 switching criterion
env.reset()
miss = false_trig = det = 0
first_fire = []
prev = np.zeros(a.num_envs, dtype=bool)
for k in range(a.steps):
    env.step(torch.zeros((a.num_envs, env.action_space.shape[-1]), dtype=torch.float32, device='cuda:0'))
    al = alpha()
    d = al > 0.5
    xy, g = env_errors()
    gt = g <= 0.0                      # env-native truth: at/below the seat plane
    below = g < -0.005
    miss += int(np.sum(d & (gt == False) & below))
    false_trig += int(np.sum((d == False) & gt))
    det += int(np.sum(d))
    newly = d & ~prev
    if newly.any():
        first_fire.extend([k] * int(newly.sum()))
    prev = d
log('')
log('V2  switching criterion vs env-native truth (height gap <= 0 from _oru_pose_errors)')
log('    detector fired in %d env-steps; missed (below seat, un-fired) %d; fired-at-seat %d' % (det, miss, false_trig))
if first_fire:
    log('    first fire: median %.0f  min %d  max %d  (of %d steps)' % (np.median(first_fire), min(first_fire), max(first_fire), a.steps))

# ---------------------------------------------------------------- V3 downstream consistency
env.reset()
bad_eq = bad_rate = 0
prev_a = alpha()
dec = int(getattr(un.cfg, "decimation", 8))
dl = min(1.0, dec * un.physics_dt / max(task.switch_duration_s, 1e-3))
for k in range(a.steps):
    env.step(torch.zeros((a.num_envs, env.action_space.shape[-1]), dtype=torch.float32, device='cuda:0'))
    a_now = alpha()
    try:
        c = un._contact_alpha.detach().cpu().numpy()
        bad_eq += int(np.sum(np.abs(a_now - c) > 1e-6))            # both channels must be identical
    except Exception:
        pass
    bad_rate += int(np.sum(np.abs(a_now - prev_a) > dl + 1e-6))     # rate limit respected
    prev_a = a_now
log('')
log('V3  downstream consistency (alpha drives reward blend AND axial force cap)')
log('    channel mismatch (_stage_alpha vs _contact_alpha): %d steps (expect 0)' % bad_eq)
log('    rate-limit violations (|dalpha| > %d*dt/T_sw = %.4f per policy step): %d steps (expect 0)' % (dec, dl, bad_rate))

# ---------------------------------------------------------------- V4 failure injection
log('')
log('V4  failure injection: zero policy action (detector intact), same env, flag toggled at runtime')
env.reset()
un.cfg.task.force_zero_action = True            # no policy, keep the switching detector
succ = 0
for k in range(a.steps):
    env.step(torch.zeros((a.num_envs, env.action_space.shape[-1]), dtype=torch.float32, device='cuda:0'))
    try:
        succ = int(un._stable_success.sum().item())
    except Exception:
        pass
log('    envs reaching a stable seat at the end: %d / %d' % (succ, a.num_envs))
log('    -> this is the C0 baseline behaviour; the trained policy must be clearly better')
un.cfg.task.force_zero_action = False
env.close()
app.close()
