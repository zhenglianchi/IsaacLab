"""Archive the pre-canonical artifacts and promote the canonical pair.

Canonical setting (user decision 2026-10-07):
  * working condition : new regime  = base standoff 20 cm + downward random 0-12 cm
                        (x/y +/-12 cm, attitude +/-3 deg, single-point reference)
  * OURS              : run `ours_v24b_gainrange_s0` (learned switch, 100 epochs, no collapse)
  * BASELINE          : C0 @ new regime (fixed impedance, same condition)

This script:
  1. creates archive/2026-10-07_hardregime_v24b/{checkpoints,logs,curves,figures,sources}
  2. moves every OTHER run's checkpoints / run dirs / play logs into the archive
     (metrics CSVs are never touched - they are the measurement record)
  3. copies the canonical checkpoints and CSVs next to the archive README
  4. draws the comparison figures for the OURS method
  5. writes the README with the canonical numbers
"""
import csv
import pathlib
import shutil

ROOT = pathlib.Path(r"C:\Users\zhenglianchi\Desktop\IsaacLab")
ARCH = ROOT / "archive" / "2026-10-07_hardregime_v24b"
OURS = "ours_v24b_gainrange_s0"
KEEP_RUNS = {OURS}
LOGS = ROOT / "logs"
SUB = LOGS / "rl_games" / "OruAssembly"

for d in ("checkpoints", "logs", "curves", "figures", "sources"):
    (ARCH / d).mkdir(parents=True, exist_ok=True)


def load(f):
    p = pathlib.Path(f)
    if not p.exists() or p.stat().st_size == 0:
        return None
    return list(csv.DictReader(p.open(newline="", encoding="utf-8")))


def stats(f):
    rows = load(f)
    if not rows:
        return None
    succ = [r for r in rows if r.get("success") == "1"]
    def col(rs, n):
        return sorted(float(r[n]) for r in rs if r.get(n) not in (None, "")) if rs and n in rs[0] else []
    med = lambda v: v[len(v) // 2] if v else float("nan")
    p90 = lambda v: v[min(len(v) - 1, int(0.9 * len(v)))] if v else float("nan")
    g = lambda v: max(v) if v else float("nan")
    return dict(
        n=len(rows), succ=100.0 * len(succ) / len(rows),
        fw_med=med(col(succ, "wrist_f_peak_N")), fw_p90=p90(col(succ, "wrist_f_peak_N")),
        fw_max=g(col(succ, "wrist_f_peak_N")),
        cf_med=med(col(succ, "force_peak_N")), cf_p90=p90(col(succ, "force_peak_N")),
        cf_max=g(col(succ, "force_peak_N")), flips=med(col(succ, "contact_flips")),
        tau_max=g(col(succ, "wrist_tau_peak_Nm")),
    )


OURS_CSV = LOGS / "metrics_v24c_best.csv"
C0_CSV = LOGS / "metrics_c0_hardrand.csv"
V14_CSV = LOGS / "metrics_v14_wrist.csv"
C0_31_CSV = LOGS / "metrics_c0_wrist.csv"

# ---------------------------------------------------------------- 1) move non-canonical runs
moved_runs, moved_ckpts = [], []
for d in sorted(SUB.glob("*")):
    if d.is_dir() and d.name not in KEEP_RUNS:
        dst = ARCH / "curves" / d.name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(d), str(dst))
        moved_runs.append(d.name)
nn = SUB / OURS / "nn"
if nn.is_dir():
    keep = {"OruAssembly.pth"}
    lasts = sorted(nn.glob("last_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    keep |= {p.name for p in lasts[:1]}                       # newest last_* is enough
    for f in sorted(nn.glob("*.pth")):
        if f.name in keep:
            continue
        shutil.move(str(f), str(ARCH / "checkpoints" / f"{OURS}__{f.name}"))
        moved_ckpts.append(f.name)
    # canonical copies
    shutil.copy2(nn / "OruAssembly.pth", ARCH / "checkpoints" / f"{OURS}__best_reward.pth")
    if lasts:
        shutil.copy2(lasts[0], ARCH / "checkpoints" / f"{OURS}__last.pth")
for f in sorted(LOGS.glob("play_*.log")) + sorted(LOGS.glob("play_*.err")):
    shutil.move(str(f), str(ARCH / "logs" / f.name))
for f in sorted(LOGS.glob("train_ours_v*.log")) + sorted(LOGS.glob("c0_*.log")) + sorted(LOGS.glob("gate_*.log")):
    if f.name != f"train_{OURS}.log":
        shutil.move(str(f), str(ARCH / "logs" / f.name))
print(f"moved {len(moved_runs)} run dirs, {len(moved_ckpts)} checkpoints to the archive")
# canonical source CSVs (copies; the originals stay in logs/)
for f in (OURS_CSV, C0_CSV, V14_CSV, C0_31_CSV):
    if f.exists():
        shutil.copy2(f, ARCH / "sources" / f.name)

# ---------------------------------------------------------------- 2) figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

so, sc = stats(OURS_CSV), stats(C0_CSV)
if so and sc:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = ["Ours (v24b)", "C0 baseline"]
    colors = ["#2a7", "#c44"]
    for ax, (title, key, unit) in zip(axes, [
            ("Success rate (%)", "succ", "%"),
            ("Contact force median (N)", "cf_med", "N"),
            ("Contact flips (median count)", "flips", "")]):
        vals = [so[key], sc[key]]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}{unit}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Hard regime (standoff 20 cm + downward random 0-12 cm): Ours vs C0", fontsize=12)
    fig.tight_layout()
    fig.savefig(ARCH / "figures" / "fig_compare_ours_vs_c0.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for f, name, c in ((OURS_CSV, "Ours (v24b)", "#2a7"), (C0_CSV, "C0 baseline", "#c44")):
        rows = [r for r in load(f) if r.get("success") == "1" and r.get("wrist_f_peak_N")]
        v = sorted(float(r["wrist_f_peak_N"]) for r in rows)
        if v:
            ax.plot(v, [(i + 1) / len(v) for i in range(len(v))], label=name, color=c)
    ax.axvline(40.0, ls="--", color="k", lw=1.2, label="40 N criterion")
    ax.set_xlabel("worst wrist |F| per episode (N)")
    ax.set_ylabel("cumulative fraction of successful episodes")
    ax.set_title("Worst-case wrist force (successful episodes only)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARCH / "figures" / "fig_wrist_force_cdf.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = ["wrist |F| max", "wrist |F| P90", "wrist |F| median", "contact |F| median",
         "wrist torque max", "attitude RMS max"]
    keys = ["fw_max", "fw_p90", "fw_med", "cf_med", "tau_max", None]
    ov = [so[k] for k in keys if k]
    cv = [sc[k] for k in keys if k]
    if keys[-1] is None:
        ar_o = load(OURS_CSV)
        ar_c = load(C0_CSV)
        ov.append(max(float(r["attitude_rms_rad"]) for r in ar_o))
        cv.append(max(float(r["attitude_rms_rad"]) for r in ar_c))
    import numpy as np
    idx = np.arange(len(x))
    ax.bar(idx - 0.2, ov, 0.4, label="Ours (v24b)", color="#2a7")
    ax.bar(idx + 0.2, cv, 0.4, label="C0 baseline", color="#c44")
    ax.set_xticks(idx)
    ax.set_xticklabels(x, rotation=20, ha="right", fontsize=9)
    ax.set_title("Safety / stability metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARCH / "figures" / "fig_safety_metrics.png", dpi=160)
    plt.close(fig)
    print("figures written")

# ---------------------------------------------------------------- 3) README
def fmt(s):
    if not s:
        return "no data"
    return (f"success {s['succ']:.1f}% | wrist |F| med {s['fw_med']:.2f} / P90 {s['fw_p90']:.2f} / max {s['fw_max']:.2f} N "
            f"({'OK<=40N' if s['fw_max'] <= 40 else 'OVER 40N'}) | contact |F| med {s['cf_med']:.1f} / P90 {s['cf_p90']:.1f} / "
            f"max {s['cf_max']:.1f} N | flips med {s['flips']:.0f} | wrist torque max {s['tau_max']:.2f} Nm")


readme = f"""# Canonical archive - hard regime (2026-10-07)

## Working condition (canonical)
Base standoff **20 cm** + **downward random 0-12 cm** (`start_z_offset = -0.109`,
`ik_rand_pos_bounds z in (-0.12, 0)`), x/y still +/-12 cm, attitude +/-3 deg,
single-point (setpoint) reference. Evaluation: 100 envs, 450 steps, seed 1234,
synchronised resets. Force statistics use SUCCESSFUL episodes only.
Worst-case criterion (criterion C): wrist |F| max <= 40 N.

## OURS - `{OURS}`
Learned switch (13th action drives alpha) + 12 learned gain multipliers, reward v2
(success 10000, timeout 3000, all-phase stall 2.0/step, hover penalty 0,
stage-1 terms as potential differences, force penalty 40 N / 20 per N / cap 600),
`gain_range 0.5`, `horizon_length 256`, 100 epochs, seed 0.
Training: peak 0.969, final 0.969, **no collapse** (previous 6 runs died at epochs 50-75).

  {fmt(so)}

## BASELINE - C0 @ same condition (fixed impedance, zero action)
  {fmt(sc)}

## Reference - regular regime (standoff 31 cm)
  v14 threshold+learned impedance: {fmt(stats(V14_CSV))}
  C0 @ 31 cm                    : {fmt(stats(C0_31_CSV))}

## Layout
* `checkpoints/` - the canonical OURS checkpoints (best reward / last) plus every
  superseded snapshot from the earlier runs
* `curves/` - tensorboard event directories of the earlier runs
* `logs/` - play/train/gate logs of the earlier runs
* `sources/` - the exact CSVs the numbers above were computed from
* `figures/` - comparison plots generated from those CSVs

`logs/metrics_*.csv` were intentionally left in place: they are the measurement
record for the whole session.
"""
(ARCH / "README.md").write_text(readme, encoding="utf-8")
print("README written")
print(readme)
