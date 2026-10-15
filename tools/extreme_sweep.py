import csv, importlib.util, pathlib, shutil, subprocess, time

ROOT = pathlib.Path(r"C:\Users\zhenglianchi\Desktop\IsaacLab")
PY = r"D:\anaconda3\envs\env_isaaclab\python.exe"
LOGS = ROOT / "logs"
OUT = ROOT / "outputs"

spec = importlib.util.spec_from_file_location("rc", str(ROOT / "tools/run_comparisons.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def last_of(var):
    nn = LOGS / "rl_games/OruAssembly" / var / "nn"
    if not nn.exists():
        return None
    ls = sorted(nn.glob("last_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ls[0] if ls else None


OURS = last_of("ours_v25_tightforce_s0")
RUNS = [
    ("本文方法", 13, "ours", ["env.task.switch_mode=learned"], OURS, "extreme_ours"),
    ("C0 固定阻抗", 13, "ours", ["env.task.switch_mode=contact", "env.task.force_zero_action=True"], OURS, "extreme_c0"),
    ("B1 只学阻尼", 6, "b1_rl_varimp", [], last_of("b1_rl_varimp"), "extreme_b1"),
    ("B2 有界残差", 12, "b2_residual", [], last_of("b2_residual"), "extreme_b2"),
    ("B4 两阶段", 12, "b4_twostage", [], last_of("b4_twostage"), "extreme_b4"),
    ("A2 去阶段1", 13, "a2_no_stage1", [], last_of("a2_no_stage1"), "extreme_a2"),
    ("A3 去力约束", 13, "a3_no_forcecap", [], last_of("a3_no_forcecap"), "extreme_a3"),
]

rows_out = []
for label, adim, mv, extra, ck, out in RUNS:
    if ck is None:
        print("  !! %s 无 last 检查点，跳过" % label, flush=True); continue
    m.set_cfg(adim, mv, "full")
    tmp = LOGS / "oru_episode_metrics.csv"
    if tmp.exists():
        tmp.unlink()
    cmd = [PY, "scripts/reinforcement_learning/rl_games/play.py",
           "--task", "Isaac-Oru-Direct-v0", "--num_envs", "64", "--headless", "--seed", "1234",
           "--checkpoint", str(ck),
           "agent.params.config.full_experiment_name=%s" % out,
           "agent.params.config.player.deterministic=True",
           "env.task.extreme_sweep=True"] + extra
    with open(LOGS / ("%s_play.log" % out), "w", encoding="utf-8", errors="replace") as fh:
        pr = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        t0 = time.time(); n = 0
        while time.time() - t0 < 12 * 60:
            time.sleep(5)
            if tmp.exists():
                n = max(sum(1 for _ in open(tmp, encoding="utf-8", errors="replace")) - 1, 0)
                if n >= 64:
                    break
            if pr.poll() is not None and n == 0:
                break
        if pr.poll() is None:
            pr.kill()
    if n < 64:
        print("  !! %s 仅 %d 行" % (label, n), flush=True); continue
    dst = LOGS / ("metrics_%s.csv" % out)
    shutil.move(str(tmp), str(dst))
    rows = list(csv.DictReader(dst.open(newline="", encoding="utf-8-sig")))
    suc = [r for r in rows if r.get("success") == "1"]
    F = sorted(float(r["wrist_F_max_N"]) for r in suc if r.get("wrist_F_max_N") not in (None, "", "nan"))
    def mx(c):
        v = [float(r[c]) for r in suc if r.get(c) not in (None, "", "nan")]
        return max(v) if v else float("nan")
    q = lambda v, p: (v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan"))
    row = [label, "%.1f" % (100.0 * len(suc) / len(rows)), "%.2f" % q(F, 0.5), "%.2f" % q(F, 0.9), "%.2f" % (max(F) if F else float("nan")),
           "%.2f" % mx("wrist_Fx_max_N"), "%.2f" % mx("wrist_Fy_max_N"), "%.2f" % mx("wrist_Fz_max_N"),
           "%.2f" % mx("wrist_Mx_max_Nm"), "%.2f" % mx("wrist_My_max_Nm"), "%.2f" % mx("wrist_Mz_max_Nm")]
    rows_out.append(row)
    print("  %s -> 成功率 %s%%  |F|最大 %s N（%d/64 工况成功）" % (label, row[1], row[4], len(suc)), flush=True)

m.set_cfg(13, "ours", "full")
HEAD = m.HEAD
p = OUT / "extreme_regime_table.csv"
with p.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(HEAD)
    for r in rows_out:
        w.writerow(r)
print("=== 极限工况表已写入 %s ===" % p, flush=True)
print("=== 已恢复规范配置 ours/13/full ✓ ===", flush=True)
