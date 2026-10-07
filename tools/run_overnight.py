"""Overnight pipeline (Python, replaces the .ps1 which hit PS 5.1 parser/policy issues).

Phase 0  housekeeping (conservative deletions, prints freed MB)
Phase 1  train the threshold-switch method at the 12 cm start -> evaluate best + last
Phase 2  switch to the learned-switch config (13-dim action + contact-force obs, 12 cm
         start kept) -> train -> evaluate best + last -> restore the code
Phase 3  print the comparison table (success-only criterion, wrist force)

Everything is appended to logs/overnight_report.txt; the table also goes to
logs/overnight_table.txt. Child processes write to log files (no pipes), and the
evaluations stop as soon as the metrics CSV holds >= 100 rows.
"""
import csv
import glob
import os
import pathlib
import re
import shutil
import subprocess
import time

REPO = r"C:\Users\zhenglianchi\Desktop\IsaacLab"
PY = r"D:\anaconda3\envs\env_isaaclab\python.exe"
PYH = r"D:\DSH\DSH Desktop\resources\office-runtime\primary-runtime\dependencies\python\python.exe"
REPORT = pathlib.Path("logs/overnight_report.txt")
TABLE = pathlib.Path("logs/overnight_table.txt")
TRAIN = "scripts/reinforcement_learning/rl_games/train.py"
PLAY = "scripts/reinforcement_learning/rl_games/play.py"
ORU = "source/isaaclab_tasks/isaaclab_tasks/direct/oru"

os.chdir(REPO)
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["PYTHONIOENCODING"] = "utf-8"


def log(msg):
    line = time.strftime("[%H:%M:%S] ") + str(msg)
    print(line, flush=True)
    with REPORT.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd, logfile, timeout=None):
    """Run a child process with output to a file (no pipes)."""
    with open(logfile, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def kill_sims():
    """Best-effort cleanup; no pipes (the sandbox may forbid them) and never raises."""
    try:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "oru_pids.txt")
        with open(tmp, "w") as fh:
            subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                           stdout=fh, stderr=subprocess.STDOUT, timeout=60)
        for line in pathlib.Path(tmp).read_text(errors="replace").splitlines():
            if "python.exe" in line:
                parts = [x.strip('"') for x in line.split(",")]
                pid = parts[1] if len(parts) > 1 else ""
                if pid.isdigit():
                    subprocess.run(["taskkill", "/PID", pid, "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(3)


def rows_in(path):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    with p.open(encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def evaluate(tag, ckpt, suffix):
    if not ckpt or not pathlib.Path(ckpt).exists():
        log(f"  skip {tag}/{suffix}: no checkpoint")
        return None
    log(f"  evaluate {tag} ({suffix}): {pathlib.Path(ckpt).name}")
    tmp = "logs/oru_episode_metrics.csv"
    if pathlib.Path(tmp).exists():
        pathlib.Path(tmp).unlink()
    cmd = [PY, PLAY, "--task", "Isaac-Oru-Direct-v0", "--num_envs", "100", "--headless",
           "--checkpoint", ckpt, "env.task.experiment_method=full",
           "agent.params.config.player.deterministic=True"]
    logf = f"logs/play_{tag}_{suffix}.log"
    with open(logf, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        n = 0
        while time.time() - t0 < 12 * 60:
            time.sleep(5)
            n = rows_in(tmp)
            if n >= 100:
                break
            if proc.poll() is not None and n == 0:
                break
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=60)
    el = round(time.time() - t0, 1)
    if n >= 100:
        out = f"logs/metrics_{tag}_{suffix}.csv"
        shutil.move(tmp, out)
        log(f"    -> {out} ({n} rows, {el}s) OK")
        return out
    log(f"    !! {tag}/{suffix}: only {n} rows after {el}s; tail of {logf}:")
    try:
        tail = pathlib.Path(logf).read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
        for t in tail:
            log("      " + t[:130])
    except Exception as e:
        log(f"      (cannot read log: {e})")
    return None


def train_and_eval(tag):
    log(f"=== train {tag} (60 epochs) ===")
    # Retry: the asset server needs the network (Isaac Sim streams USD assets); a
    # transient outage kills scene creation after ~2.5 min. Retry up to 12 times with a
    # 5-minute pause so the run starts by itself as soon as the network is back.
    rc = 1
    for attempt in range(1, 13):
        rc = run([PY, TRAIN, "--task", "Isaac-Oru-Direct-v0", "--num_envs", "64", "--headless",
                  "--seed", "0", "--max_iterations", "60", "env.task.experiment_method=full",
                  f"agent.params.config.full_experiment_name={tag}"], f"logs/train_{tag}.log")
        log(f"  training attempt {attempt}/12 exit code {rc}")
        if rc == 0:
            break
        log("  retrying in 5 minutes (network / asset server likely down)")
        time.sleep(300)
    nn = pathlib.Path(f"logs/rl_games/OruAssembly/{tag}/nn")
    best = nn / "OruAssembly.pth"
    lasts = sorted(nn.glob("last_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    evaluate(tag, str(best), "best")
    if lasts:
        evaluate(tag, str(lasts[0]), "last")
    else:
        log("  no last_*.pth (training may not have finished)")
    kill_sims()


def patch_py(path, pattern, repl, count=1):
    p = pathlib.Path(path)
    s = p.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, s, count=count)
    if n:
        p.write_bytes(new.encode("utf-8"))
    return n


# ------------------------------------------------------------------ phase 0
log("################ overnight pipeline start ################")
log("=== phase 0: housekeeping (conservative) ===")
before = sum(f.stat().st_size for f in pathlib.Path("logs").rglob("*") if f.is_file())
freed = 0
patterns = ["logs/play_*.log", "logs/play_*.err", "logs/sweep_*.log", "logs/c0_*.log",
            "logs/bisect*.log", "logs/v*_gate.log", "logs/wrist_gate.log",
            "logs/restore_check.log", "logs/train_v12.log", "logs/train_v13.log",
            "logs/train_v16.log", "logs/train_v16b.log", "logs/train_v16c.log",
            "logs/train_v17.log"]
for pat in patterns:
    for f in glob.glob(pat):
        try:
            freed += pathlib.Path(f).stat().st_size
            pathlib.Path(f).unlink()
            log(f"  removed log {pathlib.Path(f).name}")
        except Exception:
            pass
for d in ["ours_v12_s0", "ours_v13_s0", "ours_v16_learned_s0",
          "ours_v16b_learned_s0", "ours_v16c_learned_s0"]:
    p = pathlib.Path(f"logs/rl_games/OruAssembly/{d}")
    if p.exists():
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        shutil.rmtree(p, ignore_errors=True)
        freed += sz
        log(f"  removed empty run dir {d} ({round(sz/1e6,1)} MB)")
for f in glob.glob("logs/rl_games/OruAssembly/ours_v17_learned_s0/nn/*.pth"):
    try:
        freed += pathlib.Path(f).stat().st_size
        pathlib.Path(f).unlink()
        log(f"  removed closed-branch checkpoint {pathlib.Path(f).name}")
    except Exception:
        pass
after = sum(f.stat().st_size for f in pathlib.Path("logs").rglob("*") if f.is_file())
log(f"  freed {round(freed/1e6,1)} MB; logs {round(before/1e6,1)} MB -> {round(after/1e6,1)} MB")

# ------------------------------------------------------------------ phase 1
log("=== phase 1: threshold switch (current main, 12 cm start, action_space=12) ===")
cfg = pathlib.Path(f"{ORU}/oru_tasks_cfg.py").read_text(encoding="utf-8")
for line in cfg.splitlines():
    if "start_z_offset" in line:
        log("  config: " + line.strip())
train_and_eval("ours_v18_z12_s0")

# ------------------------------------------------------------------ phase 2
log("=== phase 2: learned switch (13-dim action + contact-force obs, 12 cm kept) ===")
if run(["git", "checkout", "learned-switch-13dim-v17", "--", f"{ORU}/oru_env.py"],
       "logs/phase2_checkout.log") != 0:
    log("  !! git checkout of the learned-switch env failed; skipping phase 2")
else:
    n1 = patch_py(f"{ORU}/oru_env_cfg.py", r"(action_space\s*(?::[^=\n]*)?=\s*)12\b", r"\g<1>13")
    n2 = patch_py(f"{ORU}/oru_tasks_cfg.py", r'switch_mode: str = "contact"', 'switch_mode: str = "learned"')
    log(f"  patched action_space(12->13)={n1}, switch_mode(contact->learned)={n2}")
    rc = run([PYH, "-m", "py_compile", f"{ORU}/oru_env.py", f"{ORU}/oru_env_cfg.py",
              f"{ORU}/oru_tasks_cfg.py"], "logs/phase2_compile.log")
    if rc != 0:
        log("  !! compile failed; rolling back and skipping phase 2")
        subprocess.run(["git", "checkout", "--", ORU], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        train_and_eval("ours_v19_learned_z12_s0")
        log("  restoring main code state")
        subprocess.run(["git", "checkout", "main", "--", ORU], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ------------------------------------------------------------------ phase 3
log("=== phase 3: comparison table (success-only, wrist-force criterion) ===")
GROUPS = [
    ("C0 @ 12cm (baseline)", "logs/metrics_c0_z12.csv"),
    ("v18 threshold best", "logs/metrics_ours_v18_z12_s0_best.csv"),
    ("v18 threshold last", "logs/metrics_ours_v18_z12_s0_last.csv"),
    ("v19 learned best", "logs/metrics_ours_v19_learned_z12_s0_best.csv"),
    ("v19 learned last", "logs/metrics_ours_v19_learned_z12_s0_last.csv"),
    ("ref: C0 @ 31cm", "logs/metrics_c0_wrist.csv"),
    ("ref: v14 @ 31cm", "logs/metrics_v14_wrist.csv"),
]
lines = []
hdr = (f"{'group':<22}{'eps':>5}{'succ':>8}{'wristF med':>12}{'P90':>8}{'max':>9}{'verdict':>9}"
       f"{'cF med':>9}{'cF P90':>8}{'cF max':>9}{'flips':>7}{'tauMax':>8}")
lines.append(hdr)
lines.append("-" * len(hdr))
for tag, f in GROUPS:
    p = pathlib.Path(f)
    if not p.exists() or p.stat().st_size == 0:
        lines.append(f"{tag:<22}{'--':>5}{'no data':>8}")
        continue
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    succ = [r for r in rows if r.get("success") == "1"]
    if not rows or not succ:
        lines.append(f"{tag:<22}{len(rows):>5}{'0%':>8}")
        continue
    num = lambda k, rs: [float(r[k]) for r in rs if r.get(k) not in (None, "")]
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    p90 = lambda v: sorted(v)[min(len(v) - 1, int(0.9 * len(v)))] if v else float("nan")
    fw = num("wrist_f_peak_N", succ) if "wrist_f_peak_N" in succ[0] else []
    fk = num("force_peak_N", succ)
    fl = num("contact_flips", succ)
    tw = num("wrist_tau_peak_Nm", succ) if "wrist_tau_peak_Nm" in succ[0] else []
    verdict = "OK<=40N" if (fw and max(fw) <= 40) else "OVER40N"
    lines.append(f"{tag:<22}{len(rows):>5}{100.0*len(succ)/len(rows):>7.1f}%"
                 f"{med(fw):>12.2f}{p90(fw):>8.2f}{max(fw):>9.2f}{verdict:>9}"
                 f"{med(fk):>9.1f}{p90(fk):>8.1f}{max(fk):>9.1f}{med(fl):>7.0f}{max(tw):>8.2f}")
notes = [
    "",
    "All force/torque statistics use SUCCESSFUL episodes only (user decision 2026-10-07).",
    "wristF/tau: 15 Hz, includes chain inertia - the real-robot-comparable criterion (wrist max <= 40 N).",
    "cF: contact sensor, 1/120 s substeps, includes solver discretisation (reported for reference).",
]
lines.extend(notes)
text = "\n".join(lines)
TABLE.write_text(text + "\n", encoding="utf-8")
for ln in lines:
    log("  " + ln)
log("################ overnight pipeline end ################")
