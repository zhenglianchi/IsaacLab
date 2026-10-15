# -*- coding: utf-8 -*-
"""Sequential comparison + ablation pipeline.

Runs B1->B2->B4 (comparison) then A2->A3 (ablation), trains 100 epochs each,
evaluates best/last, and appends metrics to two CSVs:
  outputs/comparison_table.csv  (ours, C0, B1, B2, B4)
  outputs/ablation_table.csv    (ours, A1, A2, A3)

One process at a time; logs to logs/run_comparisons.log; restores canonical cfg.
"""
import csv
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(r'C:\Users\zhenglianchi\Desktop\IsaacLab')
PY = r'D:\anaconda3\envs\env_isaaclab\python.exe'
OUT = ROOT / 'outputs'
LOGS = ROOT / 'logs'
ORU = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/direct/oru'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = LOGS / 'run_comparisons.log'
SEEN_MD5 = {}
# B1 的配置切换是正确的，其检查点与评估均有效：只重评、不重训
SKIP_TRAIN = set()   # B1 已改为“只学阻尼(6维)”，必须重训

CMP_CSV = OUT / 'comparison_table.csv'
ABL_CSV = OUT / 'ablation_table.csv'
HEAD = ['方法', '成功率%', '|F|中位N', '|F|P90N', '|F|最大N',
        '|Fx|最大N', '|Fy|最大N', '|Fz|最大N',
        '|Mx|最大N·m', '|My|最大N·m', '|Mz|最大N·m']
CMP_HEADER = HEAD + ['参考论文']
ABL_HEADER = HEAD



def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prune_snapshots(nn_dir):
    """只保留最新一个 last_*.pth；best(OruAssembly.pth) 与更早的中间快照都删除。"""
    if not nn_dir.exists():
        return 0
    freed = 0
    for f in nn_dir.glob('OruAssembly.pth'):
        freed += f.stat().st_size
        f.unlink()
    lasts = sorted(nn_dir.glob('last_*.pth'), key=lambda x: x.stat().st_mtime, reverse=True)
    for f in lasts[1:]:
        freed += f.stat().st_size
        f.unlink()
    return freed


def log(m):
    line = time.strftime('[%H:%M:%S] ') + str(m)
    print(line, flush=True)
    with REPORT.open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def init_csvs():
    """按重评结果(只取 last)构建两表；B2/B4 与 A2/A3 由主循环追加。"""
    def row_from(fname, label, paper=None):
        f = LOGS / fname
        if not f.exists():
            return None
        st = stats(f)
        if st is None:
            return None
        vals = [label] + [('%.2f' % v) if isinstance(v, float) and v == v else v for v in st]
        if paper is not None:
            vals.append(paper)
        return vals

    cmp_rows = [r for r in [
        row_from('metrics_reeval_ours.csv', '本文方法（13维学习α,last）', '—'),
        row_from('metrics_reeval_c0.csv', 'C0 固定阻抗', '—'),
    ] if r]
    write_csv(CMP_CSV, CMP_HEADER, cmp_rows)
    ab_rows = [r for r in [row_from('metrics_reeval_ours.csv', '本文方法（13维学习α,last）')] if r]
    write_csv(ABL_CSV, ABL_HEADER, ab_rows)
    log('已按重评结果(last)重建两张表：对比 %d 行 / 消融 %d 行' % (len(cmp_rows), len(ab_rows)))


def set_cfg(action_space, method_variant, experiment_method='full'):
    """必须正则替换：字面串替换在第二个实验起会静默失效（历史 bug）。"""
    p1 = ORU / 'oru_env_cfg.py'
    t1 = p1.read_text(encoding='utf-8')
    t1 = re.sub(r'action_space: int = [0-9]+', 'action_space: int = %d' % action_space, t1, count=1)
    p1.write_text(t1, encoding='utf-8')
    p2 = ORU / 'oru_tasks_cfg.py'
    t2 = p2.read_text(encoding='utf-8')
    t2 = re.sub(r'method_variant: str = "[a-z0-9_]+"', 'method_variant: str = "%s"' % method_variant, t2, count=1)
    t2 = re.sub(r'experiment_method: str = "[a-z_]+"', 'experiment_method: str = "%s"' % experiment_method, t2, count=1)
    p2.write_text(t2, encoding='utf-8')
    _t = p2.read_text(encoding='utf-8'); _c = p1.read_text(encoding='utf-8')
    if ('method_variant: str = "%s"' % method_variant) not in _t or ('experiment_method: str = "%s"' % experiment_method) not in _t or re.search(r'action_space: int = %d\b' % action_space, _c) is None:
        raise RuntimeError('set_cfg 未生效: %s / %s / %d' % (method_variant, experiment_method, action_space))
    return True


def compile_ok():
    r = subprocess.run([PY, '-m', 'py_compile',
                        str(ORU / 'oru_env.py'), str(ORU / 'oru_tasks_cfg.py'), str(ORU / 'oru_env_cfg.py')],
                       cwd=str(ROOT))
    return r.returncode == 0


def train(name, extra_args):
    cmd = [PY, 'scripts/reinforcement_learning/rl_games/train.py',
           '--task', 'Isaac-Oru-Direct-v0', '--num_envs', '64', '--headless',
           '--seed', '0', '--max_iterations', '100',
           'agent.params.config.full_experiment_name=%s' % name] + extra_args
    with open(LOGS / ('%s_train.log' % name), 'w', encoding='utf-8', errors='replace') as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode


def eval_ckpt(name, tag, ckpt, extra_args):
    tmp = ROOT / 'logs/oru_episode_metrics.csv'
    if tmp.exists():
        tmp.unlink()
    cmd = [PY, 'scripts/reinforcement_learning/rl_games/play.py',
           '--task', 'Isaac-Oru-Direct-v0', '--num_envs', '100', '--headless', '--seed', '1234',
           '--checkpoint', str(ckpt),
           'agent.params.config.full_experiment_name=%s' % name,
           'agent.params.config.player.deterministic=True'] + extra_args
    with open(LOGS / ('%s_play_%s.log' % (name, tag)), 'w', encoding='utf-8', errors='replace') as fh:
        pr = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        t0 = time.time(); n = 0
        while time.time() - t0 < 15 * 60:
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
    dst = LOGS / ('metrics_%s_%s.csv' % (name, tag))
    if n >= 100:
        shutil.move(str(tmp), str(dst))
        log('  %s/%s -> %s (%d 行)' % (name, tag, dst.name, n))
        return dst
    log('  !! %s/%s 仅 %d 行' % (name, tag, n))
    return None


def stats(f):
    """读新 13 列格式：成功率 + |F| 中位/P90/最大（成功回合的逐回合峰值分布）+ 六维分力峰值最大。"""
    if f is None or not pathlib.Path(f).exists():
        return None
    rows = list(csv.DictReader(open(f, newline='', encoding='utf-8-sig')))
    if not rows:
        return None
    suc = [r for r in rows if r.get('success', '') == '1']
    if not suc:
        return (0.0, float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
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


def append_row(csv_path, header, row):
    with csv_path.open('a', newline='', encoding='utf-8-sig') as fh:
        csv.writer(fh).writerow(row)


# (method_variant, action_space, experiment_method, table, label, paper)
EXPERIMENTS = [
    ('b1_rl_varimp', 6, 'full', 'cmp', 'B1 只学阻尼变阻抗(自适应阻尼)', 'IEEE 11615839'),
    ('b2_residual', 12, 'full', 'cmp', 'B2 有界残差策略(标称固定)', 'IROS 2021 (9636547)'),
    ('b4_twostage', 12, 'hard_switch', 'cmp', 'B4 两阶段(粗调固定+精调RL)', '仪器仪表学报 J2413206'),
    ('a2_no_stage1', 13, 'full', 'abl', 'A2 去除阶段1', ''),
    ('a3_no_forcecap', 13, 'full', 'abl', 'A3 去除阶段2力约束', ''),
]


def run():
    log('########## 对比+消融流水线 start ##########')
    SEEN_MD5.clear()
    init_csvs()
    for variant, adim, em, tbl, label, paper in EXPERIMENTS:
      try:
        for variant, adim, em, tbl, label, paper in EXPERIMENTS:
            log('==== %s（%s，action=%d，em=%s）====' % (label, variant, adim, em))
            set_cfg(adim, variant, em)
            if not compile_ok():
                log('!! 配置切换后编译失败，跳过 %s' % label); continue
            extra = []   # set_cfg 已改源码；CLI 的 env.action_space 会触发类型错误，故不传 env.* 覆盖
            nn_dir = ROOT / 'logs/rl_games/OruAssembly' / variant
            if variant in SKIP_TRAIN and list((nn_dir / 'nn').glob('last_*.pth')):
                log('  跳过训练（复用已有有效检查点）: %s' % variant)
                rc = 0
            else:
                if nn_dir.exists():
                    shutil.rmtree(nn_dir, ignore_errors=True)
                    log('  已清空旧运行目录（防止误评旧检查点）: %s' % variant)
                rc = train(variant, extra)
            if rc != 0:
                log('  !! train %s 首次失败 rc=%d，重试一次' % (variant, rc))
                rc = train(variant, extra)
            log('  train %s rc=%d' % (variant, rc))
            nn = ROOT / 'logs/rl_games/OruAssembly' / variant / 'nn'
            _lasts_for_md5 = sorted(nn.glob('last_*.pth'), key=lambda x: x.stat().st_mtime, reverse=True) if nn.exists() else []
            if _lasts_for_md5:
                _h = md5_of(_lasts_for_md5[0])
                if _h in SEEN_MD5:
                    log('  !! 警告：%s 的 last 与 %s 字节相同(MD5 %s)，该行不可信' % (variant, SEEN_MD5[_h], _h[:12]))
                else:
                    SEEN_MD5[_h] = variant
            last_cks = sorted(nn.glob('last_*.pth'), key=lambda x: x.stat().st_mtime, reverse=True) if nn.exists() else []
            rows = []
            if last_cks:
                _lf = eval_ckpt(variant, 'last', last_cks[0], extra)
                rows.append(('last', stats(_lf)))
            else:
                log('  !! %s 无 last 检查点，跳过评估' % label)
            # 两行都保留：best 与 last 都写入，供后续挑选
            _written = False
            for tag, st in rows:
                if st is None:
                    continue
                vals = ['%s（%s）' % (label, tag)]
                for v in st:
                    vals.append(('%.2f' % v) if isinstance(v, float) and v == v else v)
                if tbl == 'cmp':
                    append_row(CMP_CSV, CMP_HEADER, vals + [paper])
                else:
                    append_row(ABL_CSV, ABL_HEADER, vals)
                _written = True
                log('  %s（%s）写入 %s：%s' % (label, tag, tbl, vals))
            if not _written:
                log('  !! %s 无有效评估，跳过' % label); continue
            _fr = prune_snapshots(nn)
            if _fr:
                log('  自动清理 %s 中间快照，释放 %.0f MB' % (variant, _fr / 1e6))

      except Exception as _e:
          log('  !! 实验 %s 异常终止：%r（继续下一个）' % (label, _e))
          try:
              set_cfg(13, 'ours', 'full')
          except Exception:
              pass
    # 恢复规范配置
    set_cfg(13, 'ours', 'full')
    compile_ok()
    log('已恢复规范配置 ours/13维/full')
    log('########## 对比+消融流水线 end ##########')


if __name__ == '__main__':
    run()
