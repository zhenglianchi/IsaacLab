"""Throwaway self-test for tools/check_stage_switch.py.

Synthesises runs with a known ground truth and checks that the report's verdicts are
the ones we designed: 'good' must pass L1, 'latency5' must fail on latency and
'false_alarm' must fail on false alarms (negative controls).
"""
import csv
import json
import pathlib
import subprocess
import sys

DT = 1.0 / 15.0
SEAT = 0.03750
Z_TOUCH = 0.04348  # geometric contact height = the independent truth
COLS = ['step', 'env', 'ee_z', 'ee_x', 'ee_y', 'oru_z', 'oru_x', 'oru_y', 'xy_error', 'angle_rad',
        'gap_to_seat_m', 'phase', 'alpha', 'contact_alpha', 'reference_z', 'kp_z', 'command_fz',
        'wrist_fx_world', 'wrist_fy_world', 'wrist_fz_world', 'stable_success', 'oru_spd_m_s',
        'oru_angspd_rad_s', 'cand', 'succ_count', 'oru_wx', 'oru_wy', 'oru_wz', 'oru_qw', 'oru_qx',
        'oru_qy', 'oru_qz', 'contact_force_N', 'contact_flag', 'contact_fx', 'contact_fy', 'contact_fz',
        'rew_stage1', 'rew_stage2']


def oru_height(i, touch=200):
    if i >= touch:
        return SEAT + 0.0002
    if i >= 150:  # final approach: gap 102 mm -> 0
        f = (i - 150) / 50.0
        return 0.14 + f * (Z_TOUCH - 0.14)
    f = i / 150.0  # long descent: gap 302 mm -> 102 mm
    return 0.34 + f * (0.14 - 0.34)


def make(path, *, latency_steps=2, false_alarm=False, n=400, touch=200, preloads=(1.0, 3.0, 5.0, 10.0)):
    rows, alpha = [], 0.0
    for i in range(n):
        oru_z = oru_height(i, touch)
        in_contact = i >= touch
        preload = preloads[min(len(preloads) - 1, (i - touch) * len(preloads) // max(1, n - touch))] if in_contact else 0.0
        flag = i >= touch + latency_steps
        if false_alarm:
            flag = flag or (20 <= i < 40)  # detector ON while 2xx mm airborne
        target = 1.0 if flag else 0.0
        alpha += max(-DT / 0.3, min(DT / 0.3, target - alpha))
        rows.append({
            'step': i, 'env': 0, 'ee_z': oru_z + 0.39230, 'ee_x': 0.4, 'ee_y': 0.0,
            'oru_z': oru_z, 'oru_x': 0.39996, 'oru_y': 0.00001, 'xy_error': 0.00008,
            'angle_rad': 0.021, 'gap_to_seat_m': oru_z - SEAT, 'phase': int(i >= 120),
            'alpha': alpha, 'contact_alpha': alpha, 'reference_z': 0.4197,
            'kp_z': (0.0 if flag else 100.0), 'command_fz': -preload if in_contact else -0.5,
            'wrist_fx_world': 0.0, 'wrist_fy_world': 0.0, 'wrist_fz_world': -preload,
            'stable_success': int(flag), 'oru_spd_m_s': 0.002 if in_contact else 0.2,
            'oru_angspd_rad_s': 0.004, 'cand': int(flag),
            'succ_count': min(9, i - touch - latency_steps) if flag else 0,
            'oru_wx': 0, 'oru_wy': 0, 'oru_wz': 0, 'oru_qw': 0, 'oru_qx': 1, 'oru_qy': 0, 'oru_qz': 0,
            'contact_force_N': preload, 'contact_flag': int(flag),
            'contact_fx': 0.0, 'contact_fy': 0.0, 'contact_fz': preload,
            'rew_stage1': 1.2, 'rew_stage2': -10.4 if in_contact else -10.4,
        })
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    pathlib.Path(str(path).replace('.csv', '.json')).write_text(json.dumps({
        'fault_inject': 'none', 'contact_force_threshold_N': 0.5, 'bias_m': 0.01,
        'penetration_m': 0.0005, 'friction': 1.0, 'obs_stage_mismatch_steps': 0,
    }, indent=2), encoding='utf-8')


out = pathlib.Path('.installation/selftest')
out.mkdir(parents=True, exist_ok=True)
make(out / 'good.csv')
make(out / 'latency5.csv', latency_steps=5)
make(out / 'false_alarm.csv', false_alarm=True)

r = subprocess.run([sys.executable, 'tools/check_stage_switch.py',
                    str(out / 'good.csv'), str(out / 'latency5.csv'), str(out / 'false_alarm.csv'),
                    '--z-touch', str(Z_TOUCH), '--json', str(out / 'report.json')],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode:
    print("STDERR:", r.stderr[-1200:])

rep = {pathlib.Path(x['file']).name: x for x in json.loads((out / 'report.json').read_text(encoding='utf-8'))}
checks = [
    ("good L1 pass", rep['good.csv']['L1_detection']['pass'] is True),
    ("good L0.1 pass", rep['good.csv']['L0.1_free_space']['pass'] is True),
    ("good L0.2 pass", rep['good.csv']['L0.2_load_calibration']['pass'] is True),
    ("good L0.2 slope~1", abs(rep['good.csv']['L0.2_load_calibration']['slope'] - 1.0) < 0.05),
    ("good L2 pass", rep['good.csv']['L2_downstream']['pass'] is True),
    ("latency5 L1 fail", rep['latency5.csv']['L1_detection']['pass'] is False),
    ("latency5 lat95~333", abs(rep['latency5.csv']['L1_detection']['latency_ms_p95'] - 333.3) < 5),
    ("false_alarm L1 fail", rep['false_alarm.csv']['L1_detection']['pass'] is False),
    ("false_alarm FA>=1", rep['false_alarm.csv']['L1_detection']['false_alarms'] >= 1),
]
print("\n--- self-test assertions ---")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print("\nALL SELF-TEST CHECKS PASSED" if all(ok for _, ok in checks) else "\nSELF-TEST FAILURES PRESENT")
