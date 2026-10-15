$ErrorActionPreference = "Continue"
Set-Location "C:\Users\zhenglianchi\Desktop\IsaacLab"
$py = "D:\anaconda3\envs\env_isaaclab\python.exe"
$log = "logs\watchdog.log"
"=== watchdog start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Content $log -Encoding UTF8
$restarts = 0
while ($true) {
    $procs = Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue
    $pipe  = @($procs | Where-Object { $_.CommandLine -like "*run_comparisons*" }).Count
    $train = @($procs | Where-Object { $_.CommandLine -like "*train.py*" }).Count
    $play  = @($procs | Where-Object { $_.CommandLine -like "*play.py*" }).Count
    $last = ""
    if (Test-Path "logs\run_comparisons.log") {
        $last = (Get-Content "logs\run_comparisons.log" -Raw -Encoding UTF8 -ErrorAction SilentlyContinue) -split "`n" |
                Where-Object { $_ -match "写入|恢复|end|!!|rc=|====" } | Select-Object -Last 1
    }
    $stamp = Get-Date -Format "HH:mm:ss"
    "$stamp  pipe=$pipe train=$train play=$play | $($last.Trim())" | Add-Content $log -Encoding UTF8
    # 检测到"全部结束"里程碑则退出
    if ($last -match "end ##" -or $last -match "恢复规范配置") { "$stamp 检测到结束 ✓ 退出守护" | Add-Content $log -Encoding UTF8; break }
    # 流水线与训练/评估都不在运行 → 异常终止，尝试重启（最多 3 次）
    if ($pipe -eq 0 -and $train -eq 0 -and $play -eq 0) {
        if ($restarts -ge 3) { "$stamp 已重启 3 次，停止重试 ✗" | Add-Content $log -Encoding UTF8; break }
        $restarts++
        "$stamp 流水线不在运行且未见结束 → 第 $restarts 次重启" | Add-Content $log -Encoding UTF8
        & $py "tools\run_comparisons.py" *>> "logs\run_comparisons_restart$restarts.log"
    }
    Start-Sleep -Seconds 300
}
"=== watchdog end $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Content $log -Encoding UTF8
