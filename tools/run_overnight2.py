"""Overnight pipeline, strictly serial (one process at a time).

A) C0 baseline of the NEW regime        (diagnostic, 100 envs / 450 steps / seed 1234)
B) learned-switch training, 120 epochs  -> evaluate best + last
C) threshold-switch training, 120 epochs-> evaluate best + last
D) comparison table (success-only criterion, wrist |F| <= 40 N)

New regime (user design, commit 002ef78): base standoff 20 cm + DOWNWARD random 0-12 cm
(start_z_offset = -0.109, ik_rand_pos_bounds z in (-0.12, 0)); x/y stay +/-12 cm and the
attitude stays +/-3 deg (no DR scaling, no curriculum).

Child processes write to log files (no pipes; the sandbox forbids them). Every step is
appended to logs/overnight_report.txt; the table also lands in logs/overnight_table.txt.
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
ORU = "source/isaaclab_tasks/isaaclab_tasks/direct/oru"
TRAIN = "scripts/reinforcement_learning/rl_games/train.py"
PLAY = "scripts/reinforcement_learning/rl_games/play.py"
DIAG = "tools/diagnose_oru_v2.py"
REPORT = pathlib.Path("logs/overnight_report.txt")
TABLE = pathlib.Path("logs/overnight_table.txt")
TAG = "learned-switch-13dim-v17"
SUB = "logs/rl_games/OruAssembly"

os.chdir(REPO)
for k, v in (("HTTP_PROXY", "http://127.0.0.1:7890"), ("HTTPS_PROXY", "http://127.0.0.1:7890"),
             ("NO_PROXY", "127.0.0.1,localhost"), ("PYTHONIOENCODING", "utf-8")):
    os.environ[k] = v


def log(msg):
    line = time.strftime("[%H:%M:%S] ") + str(msg)
    print(line, flush=True)
    with REPORT.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd, logfile, timeout=None):
    with open(logfile, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def git(args):
    return subprocess.run(["git"] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def rows_in(path):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    with p.open(encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def patch_task_cfg():
    """Idempotently force the new height regime (base 20 cm + downward 0-12 cm)."""
    p = pathlib.Path(ORU) / "oru_tasks_cfg.py"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"start_z_offset: float = [^\n]*", "start_z_offset: float = -0.109", s, count=1)
    s = re.sub(r"ik_rand_pos_bounds: tuple \| None = [^\n]*",
               "ik_rand_pos_bounds: tuple | None = ((-0.12, 0.0), (-0.12, 0.12), (-0.12, 0.0))", s, count=1)
    p.write_bytes(s.encode("utf-8"))


def patch_learned_env():
    """13-dim action + learned switch + wrist |F| columns + clipped tail penalty."""
    b = pathlib.Path(ORU)
    p = b / "oru_env_cfg.py"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"(action_space\s*(?::[^=\n]*)?=\s*)12\b", r"\g<1>13", s, count=1)
    p.write_bytes(s.encode("utf-8"))
    p = b / "oru_tasks_cfg.py"
    s = p.read_text(encoding="utf-8")
    s = s.replace('switch_mode: str = "contact"', 'switch_mode: str = "learned"', 1)
    p.write_bytes(s.encode("utf-8"))
    p = b / "oru_env.py"
    s = p.read_text(encoding="utf-8")
    I = " " * 8
    if "_ep_fw" not in s:
        a = I + "self._ep_tauz = torch.zeros(N, device=self.device)\n"
        s = s.replace(a, a + I + "self._ep_fw = torch.zeros(N, device=self.device)\n"
                           + I + "self._ep_fwz = torch.zeros(N, device=self.device)\n", 1)
        a2 = I + "self._ep_tauz = torch.maximum(self._ep_tauz, _tw[:, 2].abs())\n"
        s = s.replace(a2, a2 + I + "self._ep_fw = torch.maximum(self._ep_fw, torch.linalg.vector_norm(_fw, dim=-1))\n"
                               + I + "self._ep_fwz = torch.maximum(self._ep_fwz, _fw[:, 2].abs())\n", 1)
        s = s.replace("wrist_tauz_peak_Nm\\n'", "wrist_tauz_peak_Nm,wrist_f_peak_N,wrist_fz_peak_N\\n'", 1)
        w = I + "                round(float(self._ep_tauz[_e].item()), 4))) + '\\n')\n"
        s = s.replace(w, I + "                round(float(self._ep_tauz[_e].item()), 4),\n"
                           + I + "                round(float(self._ep_fw[_e].item()), 3),\n"
                           + I + "                round(float(self._ep_fwz[_e].item()), 3))) + '\\n')\n", 1)
        r = I + "    self._ep_tauz[_rb] = 0.0\n"
        s = s.replace(r, r + I + "    self._ep_fw[_rb] = 0.0\n" + I + "    self._ep_fwz[_rb] = 0.0\n", 1)
    old = ("                _pen[_rb] = task.force_peak_penalty * torch.clamp(\n"
           "                    self._ep_fmax[_rb] - task.force_peak_budget, min=0.0)\n")
    if old in s:
        s = s.replace(old, ("                _pen[_rb] = torch.clamp(\n"
                            "                    task.force_peak_penalty * torch.clamp(\n"
                            "                        self._ep_fmax[_rb] - task.force_peak_budget, min=0.0),\n"
                            "                    max=getattr(task, \"force_peak_penalty_cap\", 1200.0))\n"), 1)
    p.write_bytes(s.encode("utf-8"))


def set_config(mode):
    if mode == "learned":
        git(["checkout", TAG, "--", f"{ORU}/oru_env.py"])
        git(["checkout", TAG, "--", f"{ORU}/oru_env_cfg.py"])
        patch_learned_env()
        rc = run([PYH, "-m", "py_compile", f"{ORU}/oru_env.py", f"{ORU}/oru_env_cfg.py",
                  f"{ORU}/oru_tasks_cfg.py"], "logs/cfg_compile.log")
    else:
        git(["checkout", "main", "--", ORU])
        rc = run([PYH, "-m", "py_compile", f"{ORU}/oru_env.py", f"{ORU}/oru_env_cfg.py",
                  f"{ORU}/oru_tasks_cfg.py"], "logs/cfg_compile.log")
    patch_task_cfg()
    rc2 = run([PYH, "-m", "py_compile", f"{ORU}/oru_tasks_cfg.py"], "logs/cfg_compile2.log")
    log(f"  config={mode}: compile rc={rc},{rc2}")
    return rc == 0 and rc2 == 0


def evaluate(tag, ckpt, suffix):
    if not ckpt or not pathlib.Path(ckpt).exists():
        log(f"  skip {tag}/{suffix} (no checkpoint)")
        return None
    tmp = "logs/oru_episode_metrics.csv"
    if pathlib.Path(tmp).exists():
        pathlib.Path(tmp).unlink()
    log(f"  evaluate {tag}/{suffix}: {pathlib.Path(ckpt).name}")
    cmd = [PY, PLAY, "--task", "Isaac-Oru-Direct-v0", "--num_envs", "100", "--headless",
           "--checkpoint", ckpt, "env.task.experiment_method=full",
           "agent.params.config.player.deterministic=True"]
    logf = f"logs/play_{tag}_{suffix}.log"
    t0 = time.time()
    with open(logf, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        n = 0
        while time.time() - t0 < 15 * 60:
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
    log(f"    !! {tag}/{suffix}: only {n} rows in {el}s")
    return None


def train_and_eval(tag):
    log(f"=== train {tag} (120 epochs) ===")
    rc = 1
    for attempt in range(1, 4):
        rc = run([PY, TRAIN, "--task", "Isaac-Oru-Direct-v0", "--num_envs", "64", "--headless",
                  "--seed", "0", "--max_iterations", "120", "env.task.experiment_method=full",
                  f"agent.params.config.full_experiment_name={tag}"], f"logs/train_{tag}.log")
        log(f"  attempt {attempt}/3 exit code {rc}")
        if rc == 0:
            break
        time.sleep(60)
    nn = pathlib.Path(f"{SUB}/{tag}/nn")
    evaluate(tag, str(nn / "OruAssembly.pth"), "best")
    lasts = sorted(nn.glob("last_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if lasts:
        evaluate(tag, str(lasts[0]), "last")


log("################ serial overnight pipeline start ################")

# ---- A) C0 baseline of the new regime -------------------------------------------------
log("=== A) C0 baseline, new regime (base 20cm + downward 0-12cm) ===")
if pathlib.Path("logs/oru_episode_metrics.csv").exists():
    pathlib.Path("logs/oru_episode_metrics.csv").unlink()
rc = run([PY, DIAG, "--num-envs", "100", "--steps", "450", "--seed", "1234",
          "--output", ".installation/c0_hardrand2"], "logs/c0_hardrand2.log")
log(f"  diagnostic exit code {rc}")
if rows_in("logs/oru_episode_metrics.csv") >= 100:
    shutil.move("logs/oru_episode_metrics.csv", "logs/metrics_c0_hardrand.csv")
    log("  -> logs/metrics_c0_hardrand.csv OK")
else:
    log("  !! C0 baseline produced too few rows")

# ---- B) learned-switch -----------------------------------------------------------------
log("=== B) learned switch (13-dim, clipped penalty 1200) ===")
if set_config("learned"):
    train_and_eval("ours_v21_learned_hardrand_s0")
else:
    log("  !! learned config failed to compile; skipping B")

# ---- C) threshold switch ---------------------------------------------------------------
log("=== C) threshold switch ===")
if set_config("threshold"):
    train_and_eval("ours_v22_thr_hardrand_s0")
else:
    log("  !! threshold config failed to compile; skipping C")

log("  restore main")
git(["checkout", "main", "--", ORU])

# ---- D) table ---------------------------------------------------------------------------
log("=== D) comparison table (success-only, wrist |F| criterion) ===")
GROUPS = [
    ("C0 @ new regime", "logs/metrics_c0_hardrand.csv"),
    ("C0 @ 12cm (old, ref)", "logs/metrics_c0_z12.csv"),
    ("v14 @ 31cm (main result)", "logs/metrics_v14_wrist.csv"),
    ("C0 @ 31cm (ref)", "logs/metrics_c0_wrist.csv"),
    ("v21 learned @new best", "logs/metrics_ours_v21_learned_hardrand_s0_best.csv"),
    ("v21 learned @new last", "logs/metrics_ours_v21_learned_hardrand_s0_last.csv"),
    ("v22 threshold @new best", "logs/metrics_ours_v22_thr_hardrand_s0_best.csv"),
    ("v22 threshold @new last", "logs/metrics_ours_v22_thr_hardrand_s0_last.csv"),
]
lines = [f"{'group':<30}{'eps':>5}{'succ':>8}{'wF med':>9}{'wF P90':>9}{'wF max':>9}{'verdict':>10}"
         f"{'cF med':>9}{'cF P90':>8}{'cF max':>9}{'flips':>7}{'tauMax':>8}", "-" * 122]
for tag, f in GROUPS:
    p = pathlib.Path(f)
    if not p.exists() or p.stat().st_size == 0:
        lines.append(f"{tag:<30}{'--':>5}{'no data':>8}")
        continue
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    succ = [r for r in rows if r.get("success") == "1"]
    if not rows or not succ:
        lines.append(f"{tag:<30}{len(rows):>5}{'0%':>8}")
        continue
    def col(n):
        if n in succ[0]:
            v = sorted(float(r[n]) for r in succ if r.get(n) not in (None, ""))
            if v:
                return v
        return []
    med = lambda v: v[len(v) // 2] if v else float("nan")
    p90 = lambda v: v[min(len(v) - 1, int(0.9 * len(v)))] if v else float("nan")
    fw, fk = col("wrist_f_peak_N"), col("force_peak_N")
    fl, tw = col("contact_flips"), col("wrist_tau_peak_Nm")
    verdict = "OK<=40N" if (fw and max(fw) <= 40) else ("OVER40N" if fw else "no |F| col")
    lines.append(f"{tag:<30}{len(rows):>5}{100.0*len(succ)/len(rows):>7.1f}%{med(fw):>9.2f}{p90(fw):>9.2f}"
                 f"{(max(fw) if fw else float('nan')):>9.2f}{verdict:>10}{med(fk):>9.1f}{p90(fk):>8.1f}"
                 f"{(max(fk) if fk else float('nan')):>9.1f}{med(fl):>7.0f}{(max(tw) if tw else float('nan')):>8.2f}")
lines += ["", "All force/torque statistics use SUCCESSFUL episodes only (user decision 2026-10-07).",
          "wristF: 15 Hz, includes chain inertia - the real-robot-comparable criterion (max <= 40 N).",
          "cF: contact sensor, 1/120 s substeps, includes solver discretisation (reference only).",
          "New regime: base standoff 20 cm + downward random 0-12 cm; x/y +/-12 cm; attitude +/-3 deg."]
text = "\n".join(lines)
TABLE.write_text(text + "\n", encoding="utf-8")
for ln in lines:
    log("  " + ln)
log("################ serial overnight pipeline end ################")
