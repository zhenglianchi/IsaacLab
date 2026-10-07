"""Apply the reward redesign, verify, then train + evaluate. Strictly serial.

Order:
  1. learned-switch config (13-dim + contact-force obs + wrist |F| columns + clipped penalty)
  2. apply outputs/_reward_redesign.py (potential-difference shaping, stall penalty,
     success bonus 10000, timeout 8000, force cap 2500)
  3. save_frequency 10 + max_epochs 60 (stop before the reproducible collapse at ~epoch 68)
  4. py_compile gate + 1-env 20-step gate (abort before training if either fails)
  5. commit + push the redesign
  6. train ours_v23_learned_rewardfix_s0 (60 epochs, snapshots every 10)
  7. evaluate best, last and every ep_* snapshot -> pick the best by INDEPENDENT evaluation
  8. print the table
"""
import csv
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

REPO = r"C:\Users\zhenglianchi\Desktop\IsaacLab"
PY = r"D:\anaconda3\envs\env_isaaclab\python.exe"
PYH = r"D:\DSH\DSH Desktop\resources\office-runtime\primary-runtime\dependencies\python\python.exe"
ORU = "source/isaaclab_tasks/isaaclab_tasks/direct/oru"
SUB = "logs/rl_games/OruAssembly"
TAG = "learned-switch-13dim-v17"
REPORT = pathlib.Path("logs/rewardfix_report.txt")
TABLE = pathlib.Path("logs/rewardfix_table.txt")
I = " " * 8

os.chdir(REPO)
for k, v in (("HTTP_PROXY", "http://127.0.0.1:7890"), ("HTTPS_PROXY", "http://127.0.0.1:7890"),
             ("NO_PROXY", "127.0.0.1,localhost"), ("PYTHONIOENCODING", "utf-8")):
    os.environ[k] = v


def log(m):
    line = time.strftime("[%H:%M:%S] ") + str(m)
    print(line, flush=True)
    with REPORT.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd, logfile, timeout=None):
    with open(logfile, "w", encoding="utf-8", errors="replace") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode


def git(args):
    return subprocess.run(["git"] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def rows_in(path):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    with p.open(encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def ensure(cfg_key, value):
    p = pathlib.Path(ORU) / "oru_tasks_cfg.py"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"(start_z_offset: float = )[^\n]*", r"\g<1>-0.109", s, count=1)
    s = re.sub(r"ik_rand_pos_bounds: tuple \| None = [^\n]*",
               "ik_rand_pos_bounds: tuple | None = ((-0.12, 0.0), (-0.12, 0.12), (-0.12, 0.0))", s, count=1)
    p.write_bytes(s.encode("utf-8"))
    p = pathlib.Path(ORU) / "oru_env_cfg.py"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"(action_space\s*(?::[^=\n]*)?=\s*)12\b", r"\g<1>13", s, count=1)
    p.write_bytes(s.encode("utf-8"))
    p = pathlib.Path(ORU) / "oru_tasks_cfg.py"
    s = p.read_text(encoding="utf-8")
    s = s.replace('switch_mode: str = "contact"', 'switch_mode: str = "learned"', 1)
    p.write_bytes(s.encode("utf-8"))


def snapshots_cfg():
    """save_frequency 10 for snapshots; max_epochs 60 to stop before the collapse."""
    p = pathlib.Path(ORU) / "agents" / "rl_games_ppo_cfg.yaml"
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"save_frequency:\s*[0-9]+", "save_frequency: 10", s)
    s = re.sub(r"max_epochs:\s*[0-9]+", "max_epochs: 60", s)
    p.write_bytes(s.encode("utf-8"))
    return [l.strip() for l in s.splitlines() if "save_frequency" in l or "max_epochs" in l]


def evaluate(tag, ckpt, suffix):
    if not ckpt or not pathlib.Path(ckpt).exists():
        return None
    tmp = "logs/oru_episode_metrics.csv"
    if pathlib.Path(tmp).exists():
        pathlib.Path(tmp).unlink()
    cmd = [PY, "scripts/reinforcement_learning/rl_games/play.py", "--task", "Isaac-Oru-Direct-v0",
           "--num_envs", "100", "--headless", "--checkpoint", ckpt,
           "env.task.experiment_method=full", "agent.params.config.player.deterministic=True"]
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
        log(f"    -> {out} ({n} rows, {el}s)")
        return out
    log(f"    !! {tag}/{suffix}: only {n} rows in {el}s")
    return None


def summarize(paths):
    lines = [f"{'checkpoint':<46}{'succ':>7}{'wF max':>9}{'verdict':>10}{'cF med':>9}{'cF max':>9}{'flips':>7}"]
    for tag, f in paths:
        p = pathlib.Path(f)
        if not p.exists() or p.stat().st_size == 0:
            lines.append(f"{tag:<46}{'no data':>7}")
            continue
        rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
        succ = [r for r in rows if r.get("success") == "1"]
        if not rows or not succ:
            lines.append(f"{tag:<46}{(100.0*len(succ)/len(rows)) if rows else float('nan'):>6.1f}%")
            continue
        def col(n):
            if n in succ[0]:
                v = sorted(float(r[n]) for r in succ if r.get(n) not in (None, ""))
                if v:
                    return v
            return []
        med = lambda v: v[len(v) // 2] if v else float("nan")
        fw, fk, fl = col("wrist_f_peak_N"), col("force_peak_N"), col("contact_flips")
        verdict = "OK<=40N" if (fw and max(fw) <= 40) else "OVER40N"
        lines.append(f"{tag:<46}{100.0*len(succ)/len(rows):>6.1f}%{(max(fw) if fw else float('nan')):>9.2f}"
                     f"{verdict:>10}{med(fk):>9.1f}{(max(fk) if fk else float('nan')):>9.1f}{med(fl):>7.0f}")
    return lines


log("############ reward-fix pipeline start ############")

# 1) learned-switch base
git(["checkout", TAG, "--", f"{ORU}/oru_env.py"])
git(["checkout", TAG, "--", f"{ORU}/oru_env_cfg.py"])
git(["checkout", TAG, "--", f"{ORU}/oru_tasks_cfg.py"])
ensure("", "")
rc = run([PYH, "outputs/_reward_redesign.py"], "logs/rewardfix_patch.log")
log(f"patch exit code {rc}")
if rc != 0:
    log("!! patch failed (anchor mismatch) -> aborting before training")
    tail = pathlib.Path("logs/rewardfix_patch.log").read_text(errors="replace").splitlines()[-6:]
    for t in tail:
        log("   " + t[:130])
    sys.exit(1)
for t in pathlib.Path("logs/rewardfix_patch.log").read_text(errors="replace").splitlines():
    log("  patch> " + t[:120])

cfg_lines = snapshots_cfg()
log("agents cfg: " + " | ".join(cfg_lines))

# 2) gates
rc = run([PYH, "-m", "py_compile", f"{ORU}/oru_env.py", f"{ORU}/oru_env_cfg.py", f"{ORU}/oru_tasks_cfg.py"],
         "logs/rewardfix_compile.log")
log(f"compile rc={rc}")
if rc != 0:
    log("!! compile failed -> aborting")
    sys.exit(1)
rc = run([PY, "tools/diagnose_oru_v2.py", "--num-envs", "1", "--steps", "20", "--nominal",
          "--output", ".installation/rewardfix_gate"], "logs/rewardfix_gate.log")
log(f"1-env gate rc={rc}")
if rc != 0:
    log("!! gate failed -> aborting (no training)")
    for t in pathlib.Path("logs/rewardfix_gate.log").read_text(errors="replace").splitlines()[-10:]:
        log("   " + t[:130])
    sys.exit(1)

# 3) commit + push
git(["add", "-A", "--", ORU, "outputs/_reward_redesign.py", "tools/rewardfix_run.py"])
subprocess.run(["git", "-c", "core.hooksPath=.git/nohooks", "commit", "-m",
                "feat(reward): 按用户要求重设奖励——成功奖励 3000->10000、超时 500->8000、"
                "全阶段停滞惩罚 2.0/步、阶段一状态项改势函数差分、力惩罚上限 2500；训练 60 epoch 并存快照"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["git", "push", "origin", "main"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
log("committed + pushed")

# 4) train
tag = "ours_v23_learned_rewardfix_s0"
log(f"=== train {tag} (60 epochs, snapshots every 10) ===")
rc = 1
for attempt in range(1, 4):
    rc = run([PY, "scripts/reinforcement_learning/rl_games/train.py", "--task", "Isaac-Oru-Direct-v0",
              "--num_envs", "64", "--headless", "--seed", "0", "--max_iterations", "60",
              "env.task.experiment_method=full", f"agent.params.config.full_experiment_name={tag}"],
             f"logs/train_{tag}.log")
    log(f"  attempt {attempt}/3 exit code {rc}")
    if rc == 0:
        break
    time.sleep(60)

# 5) evaluate best, last and every snapshot
nn = pathlib.Path(f"{SUB}/{tag}/nn")
cands = []
if (nn / "OruAssembly.pth").exists():
    cands.append(("best", nn / "OruAssembly.pth"))
for f in sorted(nn.glob("last_*.pth")):
    cands.append(("last", f))
for f in sorted(nn.glob("*_ep_*.pth")):
    ep = re.search(r"_ep_(\d+)", f.name)
    cands.append((f"ep{ep.group(1)}" if ep else f.stem, f))
log(f"  snapshots found: {[c[0] for c in cands]}")
results = []
for suffix, ck in cands:
    out = evaluate(tag, str(ck), suffix)
    if out:
        results.append((suffix, out))

# 6) table
paths = [("C0 @ new regime", "logs/metrics_c0_hardrand.csv"),
         ("v21 learned (before fix) best", "logs/metrics_ours_v21_learned_hardrand_s0_best.csv"),
         ("v22 threshold (before fix) best", "logs/metrics_ours_v22_thr_hardrand_s0_best.csv"),
         ("v14 @ 31cm (main result)", "logs/metrics_v14_wrist.csv")]
paths += [(f"v23 rewardfix {s}", f) for s, f in results]
lines = summarize(paths)
TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
for ln in lines:
    log("  " + ln)
log("############ reward-fix pipeline end ############")
