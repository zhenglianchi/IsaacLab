import importlib.util, subprocess, sys, pathlib
PY = r"D:\anaconda3\envs\env_isaaclab\python.exe"
spec = importlib.util.spec_from_file_location("rc", "tools/run_comparisons.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
CASES = [
    ("ours", 13, "full"),
    ("b1_rl_varimp", 6, "full"),
    ("b2_residual", 12, "full"),
    ("b4_twostage", 12, "hard_switch"),
    ("a2_no_stage1", 13, "full"),
    ("a3_no_forcecap", 13, "full"),
]
print("=== 烟测（不训练，各跑 5 步）===", flush=True)
ok_all = True
for variant, adim, em in CASES:
    m.set_cfg(adim, variant, em)
    log = pathlib.Path("logs/smoke_%s.log" % variant)
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run([PY, "tools/_smoke_variant.py", "--steps", "5"],
                            stdout=fh, stderr=subprocess.STDOUT, cwd=".").returncode
    tail = " | ".join(log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-2:])
    status = "OK ✓" if (rc == 0 and "SMOKE_OK" in log.read_text(encoding="utf-8", errors="replace")) else "FAIL ✗"
    if status.startswith("FAIL"):
        ok_all = False
    print("  %-16s act=%-3d em=%-11s rc=%d  %s\n      %s" % (variant, adim, em, rc, status, tail[:220]), flush=True)
m.set_cfg(13, "ours", "full")
print("=== 已恢复规范配置 ours/13/full ✓ ===", flush=True)
print("=== 总结：%s ===" % ("全部通过 ✓✓" if ok_all else "存在失败 ✗（见上）"), flush=True)
