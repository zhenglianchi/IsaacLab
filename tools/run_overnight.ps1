# =====================================================================================
# Overnight pipeline: (1) train the threshold-switch method -> evaluate
#                    (2) train the learned-switch method   -> evaluate
#                    (3) print a comparison table (success-only criterion, wrist force)
# Everything is appended to logs\overnight_report.txt.
# Housekeeping (conservative): deletes diagnostics logs, empty run dirs and the v17
# checkpoints (that branch is closed); keeps every metrics CSV, every run's
# OruAssembly.pth and the last_* checkpoint of the runs we evaluate.
# =====================================================================================
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\zhenglianchi\Desktop\IsaacLab"
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:NO_PROXY = "127.0.0.1,localhost"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\anaconda3\envs\env_isaaclab\python.exe"
$pyH = "D:\DSH\DSH Desktop\resources\office-runtime\primary-runtime\dependencies\python\python.exe"
$REPORT = "logs\overnight_report.txt"

function Log($m) {
    $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $m
    Write-Output $line
    Add-Content -Path $REPORT -Value $line -Encoding UTF8
}

function Kill-Sims {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" -or $_.CommandLine -like "*play.py*" -or $_.CommandLine -like "*diagnose_oru*" } |
        ForEach-Object { Log "    kill PID $($_.ProcessId)"; taskkill /PID $_.ProcessId /F 2>&1 | Out-Null }
    Start-Sleep -Seconds 5
}

# Evaluate a checkpoint with the accelerated protocol: poll the metrics CSV until it has
# >= 100 rows, then stop. Returns the path of the saved CSV.
function Eval-Ckpt($tag, $ckpt, $suffix) {
    if (-not (Test-Path $ckpt)) { Log "  跳过 $tag/$suffix：检查点不存在"; return $null }
    Log "  评测 $tag ($suffix)：$([System.IO.Path]::GetFileName($ckpt))"
    Remove-Item "logs\oru_episode_metrics.csv" -ErrorAction SilentlyContinue
    $t0 = Get-Date
    $a = @("scripts\reinforcement_learning\rl_games\play.py", "--task", "Isaac-Oru-Direct-v0",
           "--num_envs", "100", "--headless", "--checkpoint", $ckpt,
           "env.task.experiment_method=full", "agent.params.config.player.deterministic=True")
    $p = Start-Process -FilePath $py -ArgumentList $a -PassThru -NoNewWindow `
         -RedirectStandardOutput ("logs\play_" + $tag + "_" + $suffix + ".log") -RedirectStandardError ("logs\play_" + $tag + "_" + $suffix + ".err")
    $n = 0
    for ($i = 0; $i -lt 150; $i++) {
        Start-Sleep -Seconds 5
        if (Test-Path "logs\oru_episode_metrics.csv") {
            $n = (Get-Content "logs\oru_episode_metrics.csv" -ReadCount 0).Count - 1
            if ($n -ge 100) { break }
        }
        if ($p.HasExited -and $n -eq 0) { break }
    }
    $el = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if (-not $p.HasExited) { $p | Stop-Process -Force }
    Start-Sleep -Seconds 3
    if ($n -ge 100) {
        $out = ("logs\metrics_" + $tag + "_" + $suffix + ".csv")
        Move-Item "logs\oru_episode_metrics.csv" $out -Force
        Log "    -> $out（$n 行，用时 $el 秒）✓"
        return $out
    }
    Log "    !! $tag/$suffix 未产出足够数据（$n 行，用时 $el 秒）；err 尾部："
    if (Test-Path ("logs\play_" + $tag + "_" + $suffix + ".err")) {
        Get-Content ("logs\play_" + $tag + "_" + $suffix + ".err") -Tail 4 | ForEach-Object { Log "      $($_.Substring(0,[Math]::Min(120,$_.Length)))" }
    }
    return $null
}

# Train 60 epochs, then evaluate both the best-reward and the final checkpoint.
function Train-And-Eval($tag) {
    Log "=== 训练 $tag（60 epoch）==="
    & $py "scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Oru-Direct-v0 `
        --num_envs 64 --headless --seed 0 --max_iterations 60 `
        env.task.experiment_method=full agent.params.config.full_experiment_name=$tag *> "logs\train_$tag.log"
    Log "  训练退出码 $LASTEXITCODE"
    $dir = "logs\rl_games\OruAssembly\$tag\nn"
    $best = Join-Path $dir "OruAssembly.pth"
    $last = (Get-ChildItem (Join-Path $dir "last_*.pth") -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    Eval-Ckpt $tag $best "best" | Out-Null
    if ($last) { Eval-Ckpt $tag $last.FullName "last" | Out-Null } else { Log "  无 last_* 检查点（可能训练未完成）" }
    Kill-Sims
}

Log "################ 夜间流水线开始 ################"

# ---------------- 0) Housekeeping ----------------
Log "=== 清理无用日志与 checkpoint（保守）==="
$before = (Get-ChildItem "logs" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
$delLogs = @("logs\play_*.log", "logs\play_*.err", "logs\sweep_*.log", "logs\c0_*.log",
             "logs\bisect*.log", "logs\v*_gate.log", "logs\wrist_gate.log", "logs\restore_check.log",
             "logs\train_v12.log", "logs\train_v13.log", "logs\train_v16.log", "logs\train_v16b.log",
             "logs\train_v16c.log", "logs\train_v17.log")
$freed = 0
foreach ($pat in $delLogs) {
    Get-ChildItem $pat -ErrorAction SilentlyContinue | ForEach-Object {
        $freed += $_.Length
        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
        Log "  删除日志 $($_.Name)"
    }
}
# empty / failed shells of the 13-dim attempts that never produced metrics
foreach ($d in @("ours_v12_s0", "ours_v13_s0", "ours_v16_learned_s0", "ours_v16b_learned_s0", "ours_v16c_learned_s0")) {
    $p = "logs\rl_games\OruAssembly\$d"
    if (Test-Path $p) {
        $sz = (Get-ChildItem $p -Recurse -File | Measure-Object Length -Sum).Sum
        $freed += $sz
        Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
        Log "  删除无效 run 目录 $d（$([math]::Round($sz/1MB,1)) MB）"
    }
}
# v17 (learned-switch) branch is closed: keep its metrics CSVs, drop its checkpoints
$p17 = "logs\rl_games\OruAssembly\ours_v17_learned_s0\nn"
if (Test-Path $p17) {
    Get-ChildItem "$p17\*.pth" -ErrorAction SilentlyContinue | ForEach-Object {
        $freed += $_.Length
        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
        Log "  删除已结题检查点 $($_.Name)（$([math]::Round($_.Length/1MB,1)) MB）"
    }
}
$after = (Get-ChildItem "logs" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
Log "  清理完成：释放约 $([math]::Round($freed/1MB,1)) MB；logs 目录由 $([math]::Round($before/1MB,1)) MB 变为 $([math]::Round($after/1MB,1)) MB"

# ---------------- 1) Threshold-switch method (current main: 12 cm start, 12-dim, contact switch) ----------------
Log "=== 阶段一：阈值切换（当前 main：近距起点 12 cm、action_space=12、switch_mode=contact）==="
Select-String -Path "source\isaaclab_tasks\isaaclab_tasks\direct\oru\oru_tasks_cfg.py" -Pattern "start_z_offset: float" | ForEach-Object { Log "  配置核对：$($_.Line.Trim())" }
Train-And-Eval "ours_v18_z12_s0"

# ---------------- 2) Learned-switch method (patch current tree: alpha action + contact-force obs) ----------------
Log "=== 阶段二：学习切换（13 维动作 + 观测用接触力）==="
git checkout learned-switch-13dim-v17 -- source/isaaclab_tasks/isaaclab_tasks/direct/oru/oru_env.py 2>&1 | Out-Null
$patch = @'
import pathlib, re
b = pathlib.Path("source/isaaclab_tasks/isaaclab_tasks/direct/oru")
p = b / "oru_env_cfg.py"; s = p.read_text(encoding="utf-8")
s = re.sub(r"(action_space\s*(?::[^=\n]*)?=\s*)12\b", r"\g<1>13", s, count=1)
p.write_bytes(s.encode("utf-8"))
p = b / "oru_tasks_cfg.py"; s = p.read_text(encoding="utf-8")
s = s.replace('switch_mode: str = "contact"', 'switch_mode: str = "learned"', 1)
p.write_bytes(s.encode("utf-8"))
print("patched: action_space=13, switch_mode=learned (start_z_offset kept)")
'@
$patch | & $pyH -
& $py -m py_compile "source\isaaclab_tasks\isaaclab_tasks\direct\oru\oru_env.py" "source\isaaclab_tasks\isaaclab_tasks\direct\oru\oru_env_cfg.py" "source\isaaclab_tasks\isaaclab_tasks\direct\oru\oru_tasks_cfg.py"
if ($LASTEXITCODE -ne 0) {
    Log "  !! 阶段二编译失败，回滚并跳过学习切换训练"
    git checkout -- source/isaaclab_tasks/isaaclab_tasks/direct/oru/
} else {
    Select-String -Path "source\isaaclab_tasks\isaaclab_tasks\direct\oru\oru_tasks_cfg.py" -Pattern "start_z_offset: float|switch_mode" | ForEach-Object { Log "  配置核对：$($_.Line.Trim())" }
    Train-And-Eval "ours_v19_learned_z12_s0"
    Log "  恢复 main 的代码状态"
    git checkout main -- source/isaaclab_tasks/isaaclab_tasks/direct/oru/
}

# ---------------- 3) Final table ----------------
Log "=== 汇总表（口径：力=接触传感器；最坏力判定=成功回合的腕部 |F| 最大 ≤40 N）==="
$analysis = @'
import csv, pathlib
import numpy as np
GROUPS = [
    ("C0 @ 12cm (基线)",      "logs/metrics_c0_z12.csv"),
    ("v18 threshold-best",   "logs/metrics_ours_v18_z12_s0_best.csv"),
    ("v18 threshold-last",   "logs/metrics_ours_v18_z12_s0_last.csv"),
    ("v19 learned-best",     "logs/metrics_ours_v19_learned_z12_s0_best.csv"),
    ("v19 learned-last",     "logs/metrics_ours_v19_learned_z12_s0_last.csv"),
    ("参考: C0 @ 31cm",       "logs/metrics_c0_wrist.csv"),
    ("参考: v14 @ 31cm",      "logs/metrics_v14_wrist.csv"),
]
def load(f):
    p = pathlib.Path(f)
    if not p.exists() or p.stat().st_size == 0: return None
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    return rows if rows else None
def key(rows, *names):
    for n in names:
        if n in rows[0]: return n
    return None
print(f"{'组':<22}{'回合':>5}{'成功率':>8}{'腕|F|中位':>10}{'腕|F|P90':>9}{'腕|F|最大':>10}{'判定':>9}{'接触力中位':>10}{'接触力P90':>9}{'接触力最大':>11}{'翻转':>6}{'腕力矩max':>10}")
print("-" * 128)
for tag, f in GROUPS:
    rows = load(f)
    if rows is None:
        print(f"{tag:<22}{'--':>5}{'无数据':>8}"); continue
    succ = [r for r in rows if r.get("success") == "1"]
    if not succ:
        print(f"{tag:<22}{len(rows):>5}{'0%':>8}"); continue
    fw_k = key(rows, "wrist_f_peak_N")
    g = lambda k, rs: np.array([float(r[k]) for r in rs if r.get(k) not in (None, "")])
    fw = g(fw_k, succ) if fw_k else np.array([np.nan])
    fk = g("force_peak_N", succ); fl = g("contact_flips", succ)
    tw_k = key(rows, "wrist_tau_peak_Nm"); tw = g(tw_k, succ) if tw_k else np.array([np.nan])
    verdict = "达标" if (len(fw) and fw.max() <= 40) else "超40N"
    print(f"{tag:<22}{len(rows):>5}{100.0*len(succ)/len(rows):>7.1f}%{np.median(fw):>10.2f}{np.percentile(fw,90):>9.2f}{fw.max():>10.2f}{verdict:>9}"
          f"{np.median(fk):>10.1f}{np.percentile(fk,90):>9.1f}{fk.max():>11.1f}{np.median(fl):>6.0f}{np.nanmax(tw):>10.2f}")
print()
print("说明：所有力/力矩统计**只取成功回合**（用户 2026-10-07 决定：失败回合不计入安全判据）。")
print("腕部 |F|/|tau| 为 15 Hz、含链条惯性，是真机六维力传感器的可比口径；接触力为 1/120 s 子步口径，含求解器离散成分。")
'@
$analysis | & $pyH - 2>&1 | Tee-Object -FilePath "logs\overnight_table.txt" | ForEach-Object { Log "  $_" }

Log "################ 夜间流水线结束 ################"
