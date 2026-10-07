# Canonical archive - hard regime (2026-10-07)

## Working condition (canonical)
Base standoff **20 cm** + **downward random 0-12 cm** (`start_z_offset = -0.109`,
`ik_rand_pos_bounds z in (-0.12, 0)`), x/y still +/-12 cm, attitude +/-3 deg,
single-point (setpoint) reference. Evaluation: 100 envs, 450 steps, seed 1234,
synchronised resets. Force statistics use SUCCESSFUL episodes only.
Worst-case criterion (criterion C): wrist |F| max <= 40 N.

## OURS - `ours_v24b_gainrange_s0`
Learned switch (13th action drives alpha) + 12 learned gain multipliers, reward v2
(success 10000, timeout 3000, all-phase stall 2.0/step, hover penalty 0,
stage-1 terms as potential differences, force penalty 40 N / 20 per N / cap 600),
`gain_range 0.5`, `horizon_length 256`, 100 epochs, seed 0.
Training: peak 0.969, final 0.969, **no collapse** (previous 6 runs died at epochs 50-75).

  success 99.0% | wrist |F| med 13.03 / P90 25.36 / max 42.18 N (OVER 40N) | contact |F| med 53.2 / P90 152.1 / max 697.8 N | flips med 39 | wrist torque max 8.00 Nm

## BASELINE - C0 @ same condition (fixed impedance, zero action)
  success 83.0% | wrist |F| med 15.97 / P90 28.37 / max 54.15 N (OVER 40N) | contact |F| med 104.4 / P90 172.2 / max 437.6 N | flips med 55 | wrist torque max 8.66 Nm

## Reference - regular regime (standoff 31 cm)
  v14 threshold+learned impedance: success 95.0% | wrist |F| med 14.99 / P90 26.37 / max 37.95 N (OK<=40N) | contact |F| med 47.0 / P90 194.1 / max 533.5 N | flips med 39 | wrist torque max 7.26 Nm
  C0 @ 31 cm                    : success 91.0% | wrist |F| med 12.24 / P90 27.25 / max 31.63 N (OK<=40N) | contact |F| med 86.9 / P90 182.9 / max 450.9 N | flips med 55 | wrist torque max 8.66 Nm

## Layout
* `checkpoints/` - the canonical OURS checkpoints (best reward / last) plus every
  superseded snapshot from the earlier runs
* `curves/` - tensorboard event directories of the earlier runs
* `logs/` - play/train/gate logs of the earlier runs
* `sources/` - the exact CSVs the numbers above were computed from
* `figures/` - comparison plots generated from those CSVs

`logs/metrics_*.csv` were intentionally left in place: they are the measurement
record for the whole session.
