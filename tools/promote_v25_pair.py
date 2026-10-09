# -*- coding: utf-8 -*-
"""Promote the canonical pair: OURS = v25 (last checkpoint), BASELINE = C0.

Everything is regenerated from the measured CSVs; nothing is edited by hand and no
number is invented. Previous (v24b) artefacts are kept with a _v24b suffix.
"""
import csv
import pathlib
import shutil

ROOT = pathlib.Path(r'C:\Users\zhenglianchi\Desktop\IsaacLab')
ARCH = ROOT / 'archive' / '2026-10-07_hardregime_v24b'
LOGS = ROOT / 'logs'
FIG = ARCH / 'figures'
SRC = ARCH / 'sources'
CKP = ARCH / 'checkpoints'
for d in (FIG, SRC, CKP):
    d.mkdir(parents=True, exist_ok=True)

OURS_RUN = 'ours_v25_tightforce_s0'
OURS_CSV = LOGS / 'metrics_v25_last.csv'
OURS_BEST_CSV = LOGS / 'metrics_v25_best.csv'
C0_CSV = LOGS / 'metrics_c0_hardrand.csv'
V24B_CSV = LOGS / 'metrics_v24c_best.csv'
V14_CSV = LOGS / 'metrics_v14_wrist.csv'
C0_31_CSV = LOGS / 'metrics_c0_wrist.csv'


def load(f):
    p = pathlib.Path(f)
    if not p.exists() or p.stat().st_size == 0:
        return None
    return list(csv.DictReader(p.open(newline='', encoding='utf-8')))


def stats(f):
    rows = load(f)
    if not rows:
        return None
    succ = [r for r in rows if r.get('success') == '1']

    def col(rs, n):
        if not rs or n not in rs[0]:
            return []
        return sorted(float(r[n]) for r in rs if r.get(n) not in (None, ''))

    med = lambda v: v[len(v) // 2] if v else float('nan')
    p90 = lambda v: v[min(len(v) - 1, int(0.9 * len(v)))] if v else float('nan')
    mx = lambda v: max(v) if v else float('nan')
    return dict(n=len(rows), succ=100.0 * len(succ) / len(rows),
                fw_med=med(col(succ, 'wrist_f_peak_N')), fw_p90=p90(col(succ, 'wrist_f_peak_N')),
                fw_max=mx(col(succ, 'wrist_f_peak_N')),
                fz_max=mx(col(succ, 'wrist_fz_peak_N')), fxy_max=mx(col(succ, 'wrist_fxy_peak_N')),
                cf_med=med(col(succ, 'force_peak_N')), cf_p90=p90(col(succ, 'force_peak_N')),
                cf_max=mx(col(succ, 'force_peak_N')), flips=med(col(succ, 'contact_flips')),
                tau_max=mx(col(succ, 'wrist_tau_peak_Nm')),
                att_max=mx(col(rows, 'attitude_rms_rad')))


so, sc, sv24, sr, sr31 = stats(OURS_CSV), stats(C0_CSV), stats(V24B_CSV), stats(V14_CSV), stats(C0_31_CSV)
sob = stats(OURS_BEST_CSV)

# ---- keep the previous figures as history, then draw the new canonical ones -------------
for f in sorted(FIG.glob('fig_*.png')):
    if '_v24b' not in f.stem and '_v25' not in f.stem:
        f.rename(f.with_name(f.stem + '_v24b.png'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compare_fig():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels, colors = ['Ours (v25)', 'C0 baseline'], ['#2a7', '#c44']
    panels = [('Success rate (%)', 'succ', '%'), ('Contact force median (N)', 'cf_med', 'N'),
              ('Contact flips (median)', 'flips', '')]
    for ax, (title, key, unit) in zip(axes, panels):
        vals = [so[key], sc[key]]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, ('%.1f' % v) + unit, ha='center', va='bottom', fontsize=10)
    fig.suptitle('Hard regime (standoff 20 cm + downward random 0-12 cm): Ours (v25) vs C0', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_compare_ours_vs_c0.png', dpi=160)
    plt.close(fig)


def cdf_fig():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for f, name, c in ((OURS_CSV, 'Ours (v25)', '#2a7'), (C0_CSV, 'C0 baseline', '#c44')):
        rows = [r for r in load(f) if r.get('success') == '1' and r.get('wrist_f_peak_N')]
        v = sorted(float(r['wrist_f_peak_N']) for r in rows)
        if v:
            ax.plot(v, [(i + 1) / len(v) for i in range(len(v))], label=name, color=c)
    ax.axvline(40.0, ls='--', color='k', lw=1.2, label='40 N criterion')
    ax.set_xlabel('worst wrist |F| per episode (N)')
    ax.set_ylabel('cumulative fraction of successful episodes')
    ax.set_title('Worst-case wrist force (successful episodes only)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'fig_wrist_force_cdf.png', dpi=160)
    plt.close(fig)


def safety_fig():
    names = ['wrist |F| max', 'wrist |F| P90', 'wrist |F| median', 'contact |F| median',
             'wrist torque max', 'attitude RMS max']
    ov = [so['fw_max'], so['fw_p90'], so['fw_med'], so['cf_med'], so['tau_max'], so['att_max']]
    cv = [sc['fw_max'], sc['fw_p90'], sc['fw_med'], sc['cf_med'], sc['tau_max'], sc['att_max']]
    idx = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(idx - 0.2, ov, 0.4, label='Ours (v25)', color='#2a7')
    ax.bar(idx + 0.2, cv, 0.4, label='C0 baseline', color='#c44')
    ax.axhline(40.0, ls='--', color='k', lw=1, alpha=0.7)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_title('Safety / stability metrics (dashed line: 40 N criterion)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'fig_safety_metrics.png', dpi=160)
    plt.close(fig)


compare_fig()
cdf_fig()
safety_fig()

# ---- training curves for the new canonical run -----------------------------------------
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

evs = sorted(glob.glob(str(LOGS / 'rl_games' / 'OruAssembly' / OURS_RUN / '**' / 'events.out.tfevents*'), recursive=True),
             key=lambda p: pathlib.Path(p).stat().st_size, reverse=True)
if evs:
    ea = EventAccumulator(str(pathlib.Path(evs[0]).parent), size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags()['scalars']

    def get(t):
        return ([s.step for s in ea.Scalars(t)], [s.value for s in ea.Scalars(t)]) if t in tags else ([], [])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x, y = get('rewards/iter')
    if x:
        axes[0].plot(x, y, color='#2a7')
    axes[0].set_title('training reward per iteration')
    axes[0].set_xlabel('epoch')
    axes[0].grid(alpha=0.3)
    x, y = get('Episode/episode_success_rate')
    if x:
        axes[1].plot(x, y, color='#26c')
    axes[1].set_title('training-side success rate')
    axes[1].set_xlabel('epoch')
    axes[1].grid(alpha=0.3)
    for t, c, lab in (('losses/a_loss', '#c44', 'a_loss (actor)'), ('losses/c_loss', '#a6c', 'c_loss (critic)'),
                      ('losses/entropy', '#2a7', 'entropy')):
        x, y = get(t)
        if x:
            axes[2].plot(x, y, color=c, label=lab)
    axes[2].set_title('losses / entropy')
    axes[2].set_xlabel('epoch')
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle('Ours (v25, tight force budget 25 N) training curves')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_train_curves.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for t, c, lab in (('losses/a_loss', '#c44', 'a_loss (actor)'), ('losses/c_loss', '#a6c', 'c_loss (critic)')):
        x, y = get(t)
        if x:
            ax.plot(x, y, color=c, label=lab)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title('Actor / critic losses during training (v25)')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_loss_curves.png', dpi=160)
    plt.close(fig)
    print('training curves regenerated from', OURS_RUN)
else:
    print('!! no tensorboard events found for', OURS_RUN)

# ---- canonical sources and checkpoints --------------------------------------------------
for f in (OURS_CSV, OURS_BEST_CSV, C0_CSV, V24B_CSV, V14_CSV, C0_31_CSV):
    if f.exists():
        shutil.copy2(f, SRC / f.name)
nn = LOGS / 'rl_games' / 'OruAssembly' / OURS_RUN / 'nn'
lasts = sorted(nn.glob('last_*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
if lasts:
    shutil.copy2(lasts[0], CKP / 'ours_v25_last.pth')
if (nn / 'OruAssembly.pth').exists():
    shutil.copy2(nn / 'OruAssembly.pth', CKP / 'ours_v25_best_reward.pth')


def fmt(s):
    return ('success %.1f%% | wrist |F| med %.2f / P90 %.2f / max %.2f N (%s) | contact |F| med %.1f / P90 %.1f / '
            'max %.1f N | flips med %.0f | wrist torque max %.2f Nm | attitude RMS max %.3f' % (
                s['succ'], s['fw_med'], s['fw_p90'], s['fw_max'],
                'OK <= 40 N' if s['fw_max'] <= 40 else 'OVER 40 N',
                s['cf_med'], s['cf_p90'], s['cf_max'], s['flips'], s['tau_max'], s['att_max']))


readme = ARCH / 'README.md'
extra = '''

---

# CANONICAL PAIR (updated after the tight-force retraining)

## Working condition
Base standoff **20 cm** + **downward random 0-12 cm** (20-32 cm uniform), x in [-12,0] cm,
y +/-12 cm, attitude +/-3 deg, single-point (setpoint) reference. Evaluation: 100 envs,
450 steps, seed 1234, synchronised resets. Force statistics use SUCCESSFUL episodes only.
Worst-case criterion (criterion C): wrist |F| max <= 40 N.

## OURS - `ours_v25_tightforce_s0`, checkpoint `last` (SELECTED BY INDEPENDENT EVALUATION)
Same method and configuration as v24b (learned switch driving alpha on the 13th action,
12 learned gain multipliers, gain_range 0.5, horizon_length 256, 100 epochs, seed 0,
snapshots every 10 epochs), with one deliberate change: the force-peak budget was tightened
from 40 N to 25 N (penalty cap 600 -> 1200). Rationale: with a 40 N budget the policy had no
gradient below 40 N, which is why the worst case previously stalled just above the line; the
25 N budget creates a real incentive in the 25-42 N band.

  %s

  (for reference, the reward-best copy `best` evaluates at: %s)

## BASELINE - C0 @ same condition (fixed impedance, zero action)
  %s

## Previously archived champion (kept for history)
  v24b `best`: %s

## Reference runs from earlier, now non-canonical conditions (kept for history only)
  v14 @ 31 cm standoff: %s
  C0  @ 31 cm standoff: %s

## Layout
* `checkpoints/` - canonical checkpoints (ours_v25 best/last) + superseded snapshots
* `curves/` - tensorboard event directories of the earlier runs
* `logs/` - play/train/gate logs
* `sources/` - the exact CSVs behind every number above
* `figures/` - comparison plots and training curves (files ending in `_v24b` are the
  previous champion's versions, kept for traceability)

`logs/metrics_*.csv` remain in place as the complete measurement record of the session.
''' % (fmt(so), fmt(sob) if sob else 'n/a', fmt(sc), fmt(sv24), fmt(sr) if sr else 'n/a', fmt(sr31) if sr31 else 'n/a')

readme.write_text(readme.read_text(encoding='utf-8').rstrip() + extra, encoding='utf-8')
print('README updated')
print('OURS  :', fmt(so))
print('C0    :', fmt(sc))
print('v24b  :', fmt(sv24))
