# -*- coding: utf-8 -*-
"""Autonomous pipeline: B2 (mechanism visualisation) then A1 (switch ablation).

B2  : runs the proven play.py path twice (v25 policy, then C0 zero-action) with a
      flag-gated CSV tracer inside the env, then plots
        * alpha vs insertion depth       (what the policy learned to do)
        * distribution of the 12 learned gain multipliers
        * force-depth phase plot         (ours vs C0)
B1  : already produced by tools/validate_components.py (separate job).
A1  : retrains the SAME configuration but with a 12-dim action space and threshold
      switching (no learned alpha), evaluates best/last, and prints the comparison.

Everything is sequential (one process at a time) and logs to logs/pipeline_b2a1.log.
"""
import csv
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(r'C:\Users\zhenglianchi\Desktop\IsaacLab')
PY = r'D:\anaconda3\envs\env_isaaclab\python.exe'
ORU = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/direct/oru'
CK = ROOT / 'logs/rl_games/OruAssembly/ours_v25_tightforce_s0/nn/last_OruAssembly_ep_100_rew_6845.4463.pth'
OUT = ROOT / 'outputs'
FIGS = OUT / 'figs'
LOGS = ROOT / 'logs'
for d in (OUT, FIGS):
    d.mkdir(parents=True, exist_ok=True)
REPORT = LOGS / 'pipeline_b2a1.log'


def log(m):
    line = time.strftime('[%H:%M:%S] ') + str(m)
    print(line, flush=True)
    with REPORT.open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def py(code, logfile):
    with open(logfile, 'w', encoding='utf-8', errors='replace') as fh:
        return subprocess.run([PY, '-c', code], stdout=fh, stderr=subprocess.STDOUT).returncode


# ------------------------------------------------------------------ 0) CSV tracer in the env
TRACER = '''

        # ---- B2 tracer (flag gated, default off) -------------------------------------
        if getattr(self.cfg.task, "trace_csv", ""):
            try:
                import csv as _csv, os as _os
                _f = self._trace_fh if hasattr(self, "_trace_fh") else None
                if _f is None:
                    _new = not _os.path.exists(self.cfg.task.trace_csv)
                    _f = open(self.cfg.task.trace_csv, "a", newline="")
                    if _new:
                        _f.write("env,step,gap_m,xy_m,alpha,wrist_f_N,wrist_tau_Nm,force_cap_N" +
                                 "".join([",gain%d" % i for i in range(12)]) + "\\n")
                    self._trace_fh = _f
                if not hasattr(self, "_trace_step"):
                    self._trace_step = 0
                self._trace_step += 1
                _w = self.robot.data.body_incoming_joint_wrench_b[:, self._ee_frame_idx]
                _gap = float(self._oru_pose_errors()[1].mean().item())
                _xy = float(self._oru_pose_errors()[0].mean().item())
                _a = float(self._stage_alpha.mean().item())
                _wr = _w[:, 0:3].norm(dim=-1)
                _wt = _w[:, 3:6].norm(dim=-1)
                _g = self.actions[:, 0:12].mean(dim=0).detach().cpu().tolist()
                _f.write("%d,%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%s\\n" % (
                    0, self._trace_step, _gap, _xy, _a, float(_wr.mean()), float(_wt.mean()),
                    8.0 + 52.0 * _a, ",".join("%.5f" % v for v in _g)))
                _f.flush()
            except Exception:
                pass
'''


def patch_env_add_tracer():
    p = ORU / 'oru_env.py'
    s = p.read_text(encoding='utf-8')
    if 'B2 tracer' in s:
        print('  env tracer already present')
        return
    m = re.search(r'^([ \t]*)def _get_rewards\(self\)[^\n]*:\n', s, re.M)
    assert m, 'anchorage for tracer not found'
    ind = m.group(1) + '    '
    body = '\n'.join((ind + l[8:]) if l.startswith(' ' * 8) else l for l in TRACER.strip('\n').split('\n'))
    s = s[:m.end()] + body + '\n' + s[m.end():]
    p.write_bytes(s.encode('utf-8'))
    print('  env tracer inserted before _get_rewards')


def set_cfg(action_space, switch_mode):
    p = ORU / 'oru_env_cfg.py'
    s = p.read_text(encoding='utf-8')
    s = re.sub(r'(action_space\s*(?::[^=\n]*)?=\s*)\d+', r'\g<1>%d' % action_space, s, count=1)
    p.write_bytes(s.encode('utf-8'))
    p = ORU / 'oru_tasks_cfg.py'
    s = p.read_text(encoding='utf-8')
    s = re.sub(r'switch_mode: str = "[a-z]+"', 'switch_mode: str = "%s"' % switch_mode, s, count=1)
    p.write_bytes(s.encode('utf-8'))


def run_play(trace, zero, logfile, video=False):
    cmd = [PY, str(ROOT / 'scripts/reinforcement_learning/rl_games/play.py'),
           '--task', 'Isaac-Oru-Direct-v0', '--num_envs', '2', '--checkpoint', str(CK), '--headless',
           'env.task.experiment_method=full', 'agent.params.config.player.deterministic=True',
           'env.task.trace_csv=%s' % trace]
    if zero:
        cmd += ['env.task.force_zero_action=True', 'env.task.switch_mode=contact']
    # ALWAYS bound the run: without --video play.py runs forever
    cmd += ['--video', '--video_length', '450']
    with open(logfile, 'w', encoding='utf-8', errors='replace') as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


if __name__ == '__main__':
    log('########## pipeline B2 + A1 start ##########')
    patch_env_add_tracer()
    rc = py('import py_compile; py_compile.compile(r"%s")' % (ORU / 'oru_env.py'), str(LOGS / 'b2_compile.log'))
    log('compile rc=%d' % rc)
    if rc != 0:
        log('!! compile failed, aborting'); sys.exit(1)

    # ---------------- B1 (separate job) result check -------------------------------
    b1 = OUT / 'validation_b1.txt'
    if b1.exists():
        log('B1 report present: %d lines' % len(b1.read_text(encoding='utf-8').splitlines()))
    else:
        log('B1 report not present yet (separate job may still be running)')

    # ---------------- B2 -----------------------------------------------------------------
    for tag, zero in (('ours', False), ('c0', True)):
        trace = str(OUT / ('b2_%s_trace.csv' % tag))
        pathlib.Path(trace).unlink(missing_ok=True)
        set_cfg(13 if not zero else 13, 'learned' if not zero else 'learned')
        rc = run_play(trace, zero, str(LOGS / ('b2_play_%s.log' % tag)), video=True)
        n = 0
        if pathlib.Path(trace).exists():
            with open(trace, encoding='utf-8') as fh:
                n = max(sum(1 for _ in fh) - 1, 0)
        log('B2 %s: play rc=%d, trace rows=%d' % (tag, rc, n))

    rc = py('''
import csv, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = pathlib.Path(r"%s"); FIGS = pathlib.Path(r"%s")
def load(tag):
    p = OUT / ("b2_%%s_trace.csv" %% tag)
    if not p.exists(): return None
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    return rows
o, c = load("ours"), load("c0")
print("rows ours=%%s c0=%%s" %% (len(o) if o else 0, len(c) if c else 0))
if o:
    depth = np.array([-float(r["gap_m"]) for r in o]); al = np.array([float(r["alpha"]) for r in o])
    f = np.array([float(r["wrist_f_N"]) for r in o])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(depth, al, color="#2a7"); ax[0].set_xlabel("insertion depth (m)"); ax[0].set_ylabel("learned alpha")
    ax[0].set_title("alpha vs insertion depth (ours)"); ax[0].grid(alpha=.3)
    for i in range(12):
        g = np.array([float(r["gain%%d" %% i]) for r in o])
        ax[1].hist(g, bins=30, histtype="step", label="gain%%d" %% i)
    ax[1].set_xlabel("action value"); ax[1].set_title("learned gain multipliers (12-D)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "b2_alpha_and_gains.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(depth, f, color="#2a7", label="ours")
    if c:
        dc = np.array([-float(r["gap_m"]) for r in c]); fc = np.array([float(r["wrist_f_N"]) for r in c])
        ax.plot(dc, fc, color="#c44", label="C0")
    ax.axhline(40.0, ls="--", color="k", lw=1); ax.set_xlabel("insertion depth (m)"); ax.set_ylabel("wrist |F| (N)")
    ax.set_title("force-depth phase plot (ours vs C0, 40 N line)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "b2_force_depth.png", dpi=160); plt.close(fig)
    print("figures written:", sorted(p.name for p in FIGS.glob("b2_*.png")))
''' % (OUT, FIGS), str(LOGS / 'b2_plots.log'))
    log('B2 plots rc=%d' % rc)

    # ---------------- A1 -----------------------------------------------------------------
    set_cfg(12, 'contact')
    rc = py('import py_compile; py_compile.compile(r"%s")' % (ORU / 'oru_env.py'), str(LOGS / 'a1_compile.log'))
    log('A1 config set: action_space=12, switch_mode=contact; compile rc=%d' % rc)
    for attempt in range(1, 3):
        with open(LOGS / 'a1_train.log', 'w', encoding='utf-8', errors='replace') as fh:
            r = subprocess.run([PY, 'scripts/reinforcement_learning/rl_games/train.py',
                                '--task', 'Isaac-Oru-Direct-v0', '--num_envs', '64', '--headless',
                                '--seed', '0', '--max_iterations', '100',
                                'env.task.experiment_method=full',
                                'agent.params.config.full_experiment_name=ours_a1_threshold_s0'],
                               stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        log('A1 train attempt %d rc=%d' % (attempt, r.returncode))
        if r.returncode == 0:
            break
        time.sleep(60)

    nn = ROOT / 'logs/rl_games/OruAssembly/ours_a1_threshold_s0/nn'
    for tag, ck in (('best', nn / 'OruAssembly.pth'),
                    ('last', sorted(nn.glob('last_*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)[0]
                     if list(nn.glob('last_*.pth')) else None)):
        if not ck or not pathlib.Path(ck).exists():
            log('  skip A1/%s (no ckpt)' % tag); continue
        tmp = ROOT / 'logs/oru_episode_metrics.csv'
        if tmp.exists():
            tmp.unlink()
        with open(LOGS / ('a1_play_%s.log' % tag), 'w', encoding='utf-8', errors='replace') as fh:
            pr = subprocess.Popen([PY, 'scripts/reinforcement_learning/rl_games/play.py',
                                   '--task', 'Isaac-Oru-Direct-v0', '--num_envs', '100', '--headless',
                                   '--checkpoint', str(ck), 'env.task.experiment_method=full',
                                   'agent.params.config.player.deterministic=True'],
                                  stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
            t0 = time.time()
            n = 0
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
        if n >= 100:
            shutil.move(str(tmp), str(LOGS / ('metrics_a1_threshold_%s.csv' % tag)))
            log('  A1/%s -> metrics_a1_threshold_%s.csv (%d rows)' % (tag, tag, n))
        else:
            log('  !! A1/%s only %d rows' % (tag, n))

    # ---------------- summary table --------------------------------------------------------
    rc = py('''
import csv, pathlib
def stats(f):
    p = pathlib.Path(f)
    if not p.exists() or p.stat().st_size == 0: return None
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    suc = [r for r in rows if r.get("success") == "1"]
    def col(rs, n): return sorted(float(r[n]) for r in rs if r.get(n) not in (None, "")) if rs and n in rs[0] else []
    med = lambda v: v[len(v)//2] if v else float("nan")
    fw, fk, fl = col(suc, "wrist_f_peak_N"), col(suc, "force_peak_N"), col(suc, "contact_flips")
    return (100.0*len(suc)/max(len(rows),1), max(fw) if fw else float("nan"), med(fk) if fk else float("nan"), med(fl) if fl else float("nan"))
rows = [("Ours 13-D learned switch (v25)", "logs/metrics_v25_last.csv"),
        ("A1 12-D threshold switch best", "logs/metrics_a1_threshold_best.csv"),
        ("A1 12-D threshold switch last", "logs/metrics_a1_threshold_last.csv"),
        ("C0 fixed impedance", "logs/metrics_c0_hardrand.csv")]
print("%-34s %8s %10s %10s %8s" % ("group", "success", "wFr max", "cF med", "flips"))
for tag, f in rows:
    s = stats(f)
    print("%-34s %7.1f%% %10.2f %10.1f %8.0f" % ((tag,) + s) if s else "%-34s no data" % tag)
''', str(LOGS / 'a1_table.log'))
    log('A1 table rc=%d' % rc)
    if (LOGS / 'a1_table.log').exists():
        for l in (LOGS / 'a1_table.log').read_text(encoding='utf-8', errors='replace').splitlines():
            log('  ' + l)

    # ---------------- restore the canonical config (13-D learned) --------------------------
    set_cfg(13, 'learned')
    py('import py_compile; py_compile.compile(r"%s")' % (ORU / 'oru_env.py'), str(LOGS / 'restore_compile.log'))
    log('restored canonical config: action_space=13, switch_mode=learned')
    log('########## pipeline B2 + A1 end ##########')
