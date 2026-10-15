# -*- coding: utf-8 -*-
"""用新的腕部六维指标（13 列）重评已有 5 行，并按新表头重建两张 CSV。

新表头（两张表统一，接触传感器力已剔除）：
  方法, 成功率%, |F|中位N, |F|P90N, |F|最大N, |Fx|最大N, |Fy|最大N, |Fz|最大N,
  |Mx|最大N·m, |My|最大N·m, |Mz|最大N·m  [, 参考论文]

|F| 的三个统计取自“成功回合的逐回合峰值”分布；6 个分力取“成功回合的峰值最大”。
"""
import csv
import pathlib
import re
import shutil
import subprocess
import time

ROOT = pathlib.Path(r'C:\Users\zhenglianchi\Desktop\IsaacLab')
PY = r'D:\anaconda3\envs\env_isaaclab\python.exe'
ORU = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/direct/oru'
LOGS = ROOT / 'logs'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
REPORT = LOGS / 'reeval_existing.log'

HEAD = ['方法', '成功率%', '|F|中位N', '|F|P90N', '|F|最大N',
        '|Fx|最大N', '|Fy|最大N', '|Fz|最大N',
        '|Mx|最大N·m', '|My|最大N·m', '|Mz|最大N·m']
CMP_CSV = OUT / 'comparison_table.csv'
ABL_CSV = OUT / 'ablation_table.csv'


def log(m):
    line = time.strftime('[%H:%M:%S] ') + str(m)
    print(line, flush=True)
    with REPORT.open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def set_cfg(action_space, method_variant, experiment_method='full'):
    p = ORU / 'oru_env_cfg.py'
    s = p.read_text(encoding='utf-8')
    s = re.sub(r'action_space: int = [0-9]+', 'action_space: int = %d' % action_space, s, count=1)
    p.write_text(s, encoding='utf-8')
    p = ORU / 'oru_tasks_cfg.py'
    s = p.read_text(encoding='utf-8')
    s = re.sub(r'method_variant: str = "[a-z0-9_]+"', 'method_variant: str = "%s"' % method_variant, s, count=1)
    s = re.sub(r'experiment_method: str = "[a-z_]+"', 'experiment_method: str = "%s"' % experiment_method, s, count=1)
    p.write_text(s, encoding='utf-8')


def run_eval(label, ckpt, extra_args, out_name):
    tmp = LOGS / 'oru_episode_metrics.csv'
    if tmp.exists():
        tmp.unlink()
    cmd = [PY, 'scripts/reinforcement_learning/rl_games/play.py',
           '--task', 'Isaac-Oru-Direct-v0', '--num_envs', '100', '--headless', '--seed', '1234',
           '--checkpoint', str(ckpt),
           'agent.params.config.full_experiment_name=%s' % out_name,
           'agent.params.config.player.deterministic=True'] + extra_args
    with open(LOGS / ('%s_play.log' % out_name), 'w', encoding='utf-8', errors='replace') as fh:
        pr = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        t0 = time.time(); n = 0
        while time.time() - t0 < 12 * 60:
            time.sleep(5)
            if tmp.exists():
                with open(tmp, encoding='utf-8', errors='replace') as f2:
                    n = max(sum(1 for _ in f2) - 1, 0)
                if n >= 100:
                    break
            if pr.poll() is not None and n == 0:
                break
        if pr.poll() is None:
            pr.kill()
    if n < 100:
        log('  !! %s 仅 %d 行，评估失败' % (label, n))
        return None
    dst = LOGS / ('metrics_%s.csv' % out_name)
    shutil.move(str(tmp), str(dst))
    log('  %s -> %s (%d 行)' % (label, dst.name, n))
    return dst


def stats_new(f):
    rows = list(csv.DictReader(open(f, newline='', encoding='utf-8-sig')))
    if not rows:
        return None
    suc = [r for r in rows if r.get('success', '') == '1']
    if not suc:
        return None
    F = sorted(float(r['wrist_F_max_N']) for r in suc if r.get('wrist_F_max_N') not in (None, '', 'nan'))
    def mx(col):
        v = [float(r[col]) for r in suc if r.get(col) not in (None, '', 'nan')]
        return max(v) if v else float('nan')
    def q(v, p):
        if not v:
            return float('nan')
        k = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
        return v[k]
    return (100.0 * len(suc) / len(rows), q(F, 0.5), q(F, 0.9), (max(F) if F else float('nan')),
            mx('wrist_Fx_max_N'), mx('wrist_Fy_max_N'), mx('wrist_Fz_max_N'),
            mx('wrist_Mx_max_Nm'), mx('wrist_My_max_Nm'), mx('wrist_Mz_max_Nm'))


def fmt(st):
    out = []
    for v in st:
        out.append(('%.2f' % v) if isinstance(v, float) and v == v else v)
    return out


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


OURS_BEST = LOGS / 'rl_games/OruAssembly/ours_v25_tightforce_s0/nn/OruAssembly.pth'
B1_LAST = LOGS / 'rl_games/OruAssembly/b1_rl_varimp/nn/last_OruAssembly_ep_100_rew__5277.8325_.pth'
OURS_CK = LOGS / 'rl_games/OruAssembly/ours_v25_tightforce_s0/nn/last_OruAssembly_ep_100_rew_6845.4463.pth'
A1_BEST = LOGS / 'rl_games/OruAssembly/ours_a1_threshold_s0/nn/OruAssembly.pth'
A1_LAST = LOGS / 'rl_games/OruAssembly/ours_a1_threshold_s0/nn/last_OruAssembly_ep_100_rew_7611.099.pth'
B1_BEST = LOGS / 'rl_games/OruAssembly/b1_rl_varimp/nn/OruAssembly.pth'

# (label, action_space, method_variant, extra_args, ckpt, out_name)
RUNS = [
    ('本文方法（13维学习α,best）', 13, 'ours', ['env.task.switch_mode=learned'], OURS_BEST, 'reeval_oursbest'),
    ('本文方法（13维学习α,last）', 13, 'ours', ['env.task.switch_mode=learned'], OURS_CK, 'reeval_ours'),
    ('C0 固定阻抗', 13, 'ours', ['env.task.switch_mode=contact', 'env.task.force_zero_action=True'], OURS_CK, 'reeval_c0'),
    ('B1 无分段RL变阻抗 best', 12, 'b1_rl_varimp', ['env.task.switch_mode=learned'], B1_BEST, 'reeval_b1best'),
    ('B1 无分段RL变阻抗 last', 12, 'b1_rl_varimp', ['env.task.switch_mode=learned'], B1_LAST, 'reeval_b1last'),
]

if __name__ == '__main__':
    log('########## 重评已有行（新 13 列指标）start ##########')
    rows = {}
    for label, adim, mv, extra, ck, out in RUNS:
        if not pathlib.Path(ck).exists():
            log('  !! 检查点缺失，跳过 %s: %s' % (label, ck)); continue
        set_cfg(adim, mv, 'full')
        f = run_eval(label, ck, extra, out)
        if f:
            st = stats_new(f)
            if st:
                rows[label] = fmt(st)
                log('  %s -> %s' % (label, rows[label]))
    set_cfg(13, 'ours', 'full')
    log('  已恢复规范配置 ours/13/full ✓')

    # 对比表：本文、C0、B1（B2/B4 待训练）
    cmp_rows = []
    for lab, paper in [('本文方法（13维学习α,best）', '—'), ('本文方法（13维学习α,last）', '—'), ('C0 固定阻抗', '—'), ('B1 无分段RL变阻抗 best', 'IEEE 11615839'), ('B1 无分段RL变阻抗 last', 'IEEE 11615839')]:
        if lab in rows:
            cmp_rows.append([lab] + rows[lab] + [paper])
    write_csv(CMP_CSV, HEAD + ['参考论文'], cmp_rows)
    # 消融表：本文、A1 best、A1 last（A2/A3 待训练）
    ab_rows = []
    for lab in ['本文方法（13维学习α,best）', '本文方法（13维学习α,last）', ]:
        if lab in rows:
            ab_rows.append([lab] + rows[lab])
    write_csv(ABL_CSV, HEAD, ab_rows)
    log('  两张 CSV 已按新表头重建 ✓')
    log('########## end ##########')
