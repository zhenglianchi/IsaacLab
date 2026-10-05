"""Stage-switch verification report from diagnostic CSVs.

Implements the protocol in ORU_EXPERIMENT_PLAN.md section 3:
  L0  signal sanity   - free-space zero, load calibration, direction
  L1  detection       - misses / false alarms / latency / chatter vs a GEOMETRIC truth
  L2  downstream      - blend rise time, Z force cap, pre-contact stage-2 tax, seat freeze,
                        observation consistency

Ground truth: the geometric contact height z_touch. Pass it explicitly with --z-touch
(calibrate it once with rest_offset=0 at a very slow approach and then keep it FIXED) -
without it the script estimates z_touch from the contact signal itself, which is
circular and is flagged as such.

Usage:
    python tools/check_stage_switch.py .installation/diag_*.csv
    python tools/check_stage_switch.py run.csv --z-touch 0.04348 --json report.json
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np

DT = 1.0 / 15.0  # policy step at 15 Hz

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("csv", nargs="+", help="diagnostic CSV(s); shell globs are expanded by the shell")
parser.add_argument("--z-touch", type=float, default=None,
                    help="geometric contact height in m (independent truth). Omit to auto-estimate "
                         "(circular - flagged in the report)")
parser.add_argument("--margin", type=float, default=0.001, help="airborne margin in m for false alarms (default 1 mm)")
parser.add_argument("--latency-window-s", type=float, default=0.5, help="miss window after a truth rising edge")
parser.add_argument("--json", type=str, default=None, help="also write the report as JSON")
args = parser.parse_args()

paths = []
for pattern in args.csv:
    paths.extend(sorted(glob.glob(pattern)) or [pattern])
paths = [p for p in paths if os.path.exists(p)]
if not paths:
    raise SystemExit("no CSV found")


def col(rows, name):
    return np.array([float(r[name]) for r in rows])


def r_squared(y, y_hat):
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def edges(mask):
    """Indices where mask turns True."""
    m = np.asarray(mask, dtype=bool)
    return np.where(m & ~np.r_[False, m[:-1]])[0]


REQUIRED = ("step", "oru_z", "gap_to_seat_m", "contact_force_N", "contact_fx", "contact_fy", "contact_fz",
            "contact_flag", "phase", "alpha", "command_fz", "kp_z", "oru_spd_m_s", "oru_angspd_rad_s",
            "rew_stage2")


def report(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, {}
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return None, {"file": path, "error": "CSV predates the switch-verification columns",
                      "missing": missing,
                      "hint": "re-run tools/diagnose_oru_v2.py with the current code to regenerate it"}
    step = col(rows, "step").astype(int)
    oru_z = col(rows, "oru_z")
    gap_mm = col(rows, "gap_to_seat_m") * 1000.0
    fc = col(rows, "contact_force_N")
    fvec = np.stack([col(rows, "contact_fx"), col(rows, "contact_fy"), col(rows, "contact_fz")], axis=1)
    flag = col(rows, "contact_flag").astype(bool)
    phase = col(rows, "phase").astype(bool)
    alpha = col(rows, "alpha")
    cap_cmd = np.abs(col(rows, "command_fz"))
    kp_z = col(rows, "kp_z")
    spd = col(rows, "oru_spd_m_s")
    angspd = col(rows, "oru_angspd_rad_s")
    r2 = col(rows, "rew_stage2")
    n_steps = int(step.max()) + 1

    meta = {}
    meta_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(meta_path):
        try:
            meta = json.loads(open(meta_path, encoding="utf-8").read())
        except Exception:
            meta = {}

    # ---- known Z force cap (from the task config) -------------------------
    cap_free = 8.0
    cap_contact = 60.0

    out = {"file": path, "rows": len(rows), "sim_seconds": round(n_steps * DT, 2),
           "metadata": {k: meta.get(k) for k in
                        ("fault_inject", "contact_force_threshold_N", "bias_m", "penetration_m",
                         "friction", "obs_stage_mismatch_steps", "max_task_force_N", "max_task_force_z_N")
                        if k in meta}}

    # ================= L0: signal sanity =================
    fs = gap_mm > 50.0
    out["L0.1_free_space"] = {
        "samples": int(fs.sum()),
        "contact_force_max_N": round(float(fc[fs].max()), 6) if fs.any() else None,
        "contact_force_p95_N": round(float(np.percentile(fc[fs], 95)), 6) if fs.any() else None,
        "pass": bool(fs.any() and fc[fs].max() < 0.1),
    }
    settled = flag & (spd < 0.01)
    levels = np.unique(np.round(cap_cmd[settled], 6)) if settled.any() else np.array([])
    if settled.sum() > 5 and len(levels) >= 3:
        x = cap_cmd[settled]
        y = fc[settled]
        slope, intercept = np.polyfit(x, y, 1)
        r2v = r_squared(y, slope * x + intercept)
        out["L0.2_load_calibration"] = {
            "samples": int(settled.sum()), "levels": int(len(levels)),
            "slope": round(float(slope), 3), "intercept_N": round(float(intercept), 3),
            "r2": round(float(r2v), 4), "pass": bool(0.8 <= slope <= 1.2 and r2v > 0.95),
        }
    else:
        # A single operating point cannot validate a magnitude law: report N/A rather
        # than a meaningless slope (a flat fit gives slope 0.5 / R2 = nan).
        out["L0.2_load_calibration"] = {
            "samples": int(settled.sum()), "levels": int(len(levels)), "pass": None,
            "note": "insufficient excitation - run the preload sweep (e.g. bias 0.01/0.05/0.2/0.4) "
                    "and pass all files at once",
        }
    loaded = fc > 0.2
    if loaded.any():
        fxy = np.linalg.norm(fvec[loaded][:, :2], axis=1)
        fmag = np.linalg.norm(fvec[loaded], axis=1)
        ratio = fxy / np.maximum(fmag, 1e-9)
        out["L0.3_direction"] = {
            "samples": int(loaded.sum()),
            "mean_lateral_fraction": round(float(ratio.mean()), 4),
            "mean_fz_sign": round(float(np.sign(fvec[loaded][:, 2]).mean()), 3),
            "pass": bool(ratio.mean() < 0.10),
            "note": "sign convention: +Fz means the ground pushes the ORU up (verify once by hand)",
        }
    else:
        out["L0.3_direction"] = {"samples": 0, "pass": False, "note": "no loaded samples"}

    # ================= L1: detection vs geometric truth =================
    z_touch = args.z_touch
    circular = z_touch is None
    if circular:
        nz = np.where(fc > 0)[0]
        z_touch = float(oru_z[nz[0]]) if len(nz) else float(oru_z.min())
    truth = oru_z <= (z_touch + 1e-9)
    t_edges = edges(truth)
    det_edges = edges(flag)
    window = int(round(args.latency_window_s / DT))
    latencies, misses = [], 0
    for i in t_edges:
        j = np.where(flag[i:i + window])[0]
        if len(j):
            latencies.append(int(j[0]) * DT * 1000.0)
        else:
            misses += 1
    false_alarms = 0
    for i in det_edges:
        lo = max(0, i - window)
        if not truth[lo:i + 1].any() and oru_z[lo:i + 1].min() > z_touch + args.margin:
            false_alarms += 1
    transitions = int(np.sum(flag[1:] != flag[:-1]))
    tp = int(np.sum(flag & truth))
    out["L1_detection"] = {
        "z_touch_m": round(float(z_touch), 6),
        "z_touch_source": "estimated from the contact signal (CIRCULAR - calibrate and pass --z-touch)"
        if circular else "provided by --z-touch (independent)",
        "truth_contact_steps": int(truth.sum()), "detector_on_steps": int(flag.sum()),
        "recall": round(tp / max(int(truth.sum()), 1), 4),
        "precision": round(tp / max(int(flag.sum()), 1), 4),
        "misses": misses, "false_alarms": false_alarms, "transitions": transitions,
        "latency_ms_mean": round(float(np.mean(latencies)), 1) if latencies else None,
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        "latency_ms_max": round(float(np.max(latencies)), 1) if latencies else None,
        "pass": bool(misses == 0 and false_alarms == 0 and transitions <= 2 and latencies
                     and float(np.percentile(latencies, 95)) < 200.0),
    }

    # ================= L2: downstream identities =================
    # Blend rate is a property of the switch itself, so measure it from the
    # DETECTOR edge (from the truth edge it would also include the detection
    # latency). With dt = 1/15 s and switch_duration_s = 0.3 the discrete ramp
    # reaches 0.95 in 4 steps = 0.267 s, hence the +-0.05 s band.
    rises = []
    for i in det_edges:
        j = np.where(alpha[i:] >= 0.95)[0]
        if len(j):
            rises.append(int(j[0]) * DT)
    pre = ~flag
    frozen = kp_z <= 1e-9
    out["L2_downstream"] = {
        "blend_rise_time_s": round(float(np.mean(rises)), 3) if rises else None,
        "blend_rise_expected_s": 0.30,
        "pre_contact_max_cmd_fz_N": round(float(cap_cmd[pre].max()), 3) if pre.any() else None,
        "pre_contact_cap_N": cap_free,
        "pre_contact_cap_pass": bool((not pre.any()) or cap_cmd[pre].max() <= cap_free + 1e-6),
        "pre_contact_alpha_max": round(float(alpha[pre].max()), 6) if pre.any() else None,
        "pre_contact_stage2_tax_mean": round(float(r2[pre].mean()), 3) if pre.any() else None,
        "pre_contact_tax_note": "raw stage-2 reward that the blend does NOT pay before contact",
        "frozen_steps": int(frozen.sum()),
        "seat_gap_mm_mean": round(float(gap_mm[frozen].mean()), 2) if frozen.any() else None,
        "frozen_speed_max_m_s": round(float(spd[frozen].max()), 4) if frozen.any() else None,
        "frozen_angspd_max_rad_s": round(float(angspd[frozen].max()), 4) if frozen.any() else None,
        "obs_mismatch_steps": meta.get("obs_stage_mismatch_steps"),
    }
    out["L2_downstream"]["pass"] = bool(
        out["L2_downstream"]["pre_contact_cap_pass"]
        and (out["L2_downstream"]["pre_contact_alpha_max"] in (None, 0.0))
        and out["L2_downstream"]["blend_rise_time_s"] is not None
        and abs(out["L2_downstream"]["blend_rise_time_s"] - 0.30) <= 0.05
        and out["L2_downstream"].get("obs_mismatch_steps", 0) in (None, 0)
    )
    out["fault_inject"] = meta.get("fault_inject", "none")
    return rows, out


print("==================== STAGE SWITCH VERIFICATION ====================")
reports = []
for p in paths:
    rows, rep = report(p)
    if rep is None:
        continue
    if "error" in rep:
        print(f"\n### {p}\n    SKIPPED: {rep['error']}")
        print(f"    missing: {', '.join(rep['missing'])}")
        print(f"    hint   : {rep['hint']}")
        continue
    reports.append(rep)
    print(f"\n### {p}   rows={rep['rows']}  ({rep['sim_seconds']} s)")
    if rep["metadata"]:
        print("    metadata:", rep["metadata"])
    if rep["fault_inject"] not in (None, "none"):
        print(f"    !! FAULT-INJECTED RUN ({rep['fault_inject']}) - expected to degrade, not to pass")
    a = rep["L0.1_free_space"]
    print(f"  L0.1 free-space zero   : n={a['samples']:4d}  max={a['contact_force_max_N']} N  "
          f"P95={a['contact_force_p95_N']} N  -> {'PASS' if a['pass'] else 'FAIL'}")
    b = rep["L0.2_load_calibration"]
    b_verdict = "N/A" if b.get("pass") is None else ("PASS" if b["pass"] else "FAIL")
    print(f"  L0.2 load calibration  : n={b['samples']:4d} levels={b.get('levels')}  slope={b.get('slope')}  "
          f"R2={b.get('r2')}  -> {b_verdict}" + (f"  ({b['note']})" if b.get("note") else ""))
    c = rep["L0.3_direction"]
    print(f"  L0.3 contact direction : n={c['samples']:4d}  lateral/|F|={c.get('mean_lateral_fraction')}  "
          f"mean sign Fz={c.get('mean_fz_sign')}  -> {'PASS' if c['pass'] else 'FAIL'}")
    d = rep["L1_detection"]
    print(f"  L1 detection           : z_touch={d['z_touch_m']} m ({d['z_touch_source'][:38]}...)")
    print(f"                           truth={d['truth_contact_steps']} det={d['detector_on_steps']} "
          f"steps  recall={d['recall']} precision={d['precision']}")
    print(f"                           misses={d['misses']} false_alarms={d['false_alarms']} "
          f"transitions={d['transitions']}  latency mean/P95/max="
          f"{d['latency_ms_mean']}/{d['latency_ms_p95']}/{d['latency_ms_max']} ms  "
          f"-> {'PASS' if d['pass'] else 'FAIL'}")
    e = rep["L2_downstream"]
    print(f"  L2 downstream          : blend rise={e['blend_rise_time_s']} s (expect {e['blend_rise_expected_s']})")
    print(f"                           pre-contact cap: max|cmd_fz|={e['pre_contact_max_cmd_fz_N']} N <= "
          f"{e['pre_contact_cap_N']} N -> {'PASS' if e['pre_contact_cap_pass'] else 'FAIL'}")
    print(f"                           pre-contact alpha_max={e['pre_contact_alpha_max']} "
          f"(must be 0) ; stage-2 tax not paid = {e['pre_contact_stage2_tax_mean']}/step")
    print(f"                           frozen steps={e['frozen_steps']} seat gap={e['seat_gap_mm_mean']} mm "
          f"spd_max={e['frozen_speed_max_m_s']} angspd_max={e['frozen_angspd_max_rad_s']}")
    print(f"                           obs mismatch steps={e['obs_mismatch_steps']}  "
          f"-> overall L2 {'PASS' if e['pass'] else 'FAIL'}")

if len(reports) > 1:
    print("\n==================== SWEEP TABLE ====================")
    print(f"{'file':34s} {'variant':>10s} {'L0.1':>5s} {'L1':>5s} {'miss':>5s} {'FA':>4s} "
          f"{'lat95':>7s} {'chat':>5s} {'L2':>5s} {'seat_gap':>9s}")
    for r in reports:
        a, d, e = r["L0.1_free_space"], r["L1_detection"], r["L2_downstream"]
        print(f"{os.path.basename(r['file'])[:34]:34s} {str(r['fault_inject'])[:10]:>10s} "
              f"{'OK' if a['pass'] else 'FAIL':>5s} {'OK' if d['pass'] else 'FAIL':>5s} "
              f"{d['misses']:5d} {d['false_alarms']:4d} {str(d['latency_ms_p95']):>7s} "
              f"{d['transitions']:5d} {'OK' if e['pass'] else 'FAIL':>5s} {str(e['seat_gap_mm_mean']):>9s}")

if args.json:
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"\n[json] {args.json}")
print("\nLegend: L0.1 free-space max<0.1N | L1 misses=0, FA=0, chatter<=2, latency P95<200ms | "
      "L2 cap+zero blend before contact+rise 0.30+-0.05s+obs consistent")
