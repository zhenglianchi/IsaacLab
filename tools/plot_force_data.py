#!/usr/bin/env python3
"""
Demo: plot the latest force/torque wrench data from force_data_logs/.

Reads the most recent CSV, plots forces (Fx, Fy, Fz) on the top subplot
and torques (Tx, Ty, Tz) on the bottom subplot.  Saves the figure as a
PNG next to the CSV and attempts to show an interactive window.

Usage:
    python tools/plot_force_data.py                    # latest file
    python tools/plot_force_data.py <path/to/file.csv> # specific file
"""

import os
import sys
import glob

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def find_latest_csv(log_dir: str) -> str | None:
    """Return the path of the most recently modified CSV in *log_dir*."""
    pattern = os.path.join(log_dir, "wrench_data_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def plot_wrench(csv_path: str, save: bool = True) -> None:
    """Read wrench CSV, print summary, and render a two-panel force/torque plot."""
    df = pd.read_csv(csv_path)

    # ── Console summary ──────────────────────────────────────────────
    print(f"File      : {csv_path}")
    print(f"Steps     : {len(df)}")
    print(f"Duration  : {len(df) - 1} steps  (assumes 1 row per simulation step)")
    print()
    for col in ["Fx(N)", "Fy(N)", "Fz(N)", "Tx(Nm)", "Ty(Nm)", "Tz(Nm)"]:
        s = df[col]
        print(
            f"  {col:<8s}  mean={s.mean():+10.4f}  "
            f"min={s.min():+10.4f}  max={s.max():+10.4f}"
        )
    print()

    # ── Figure ───────────────────────────────────────────────────────
    fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Force-Torque Wrench  —  {os.path.basename(csv_path)}", fontsize=14)

    steps = df["Step"]

    # Forces
    ax_f.plot(steps, df["Fx(N)"], label="Fx", linewidth=0.8)
    ax_f.plot(steps, df["Fy(N)"], label="Fy", linewidth=0.8)
    ax_f.plot(steps, df["Fz(N)"], label="Fz", linewidth=0.8)
    ax_f.set_ylabel("Force (N)")
    ax_f.legend(loc="upper right", ncol=3)
    ax_f.grid(True, alpha=0.3)
    ax_f.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    # Torques
    ax_t.plot(steps, df["Tx(Nm)"], label="Tx", linewidth=0.8)
    ax_t.plot(steps, df["Ty(Nm)"], label="Ty", linewidth=0.8)
    ax_t.plot(steps, df["Tz(Nm)"], label="Tz", linewidth=0.8)
    ax_t.set_xlabel("Step")
    ax_t.set_ylabel("Torque (Nm)")
    ax_t.legend(loc="upper right", ncol=3)
    ax_t.grid(True, alpha=0.3)
    ax_t.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    plt.tight_layout()

    # Save PNG next to the CSV
    if save:
        png_path = os.path.splitext(csv_path)[0] + ".png"
        fig.savefig(png_path, dpi=150)
        print(f"[SAVED] {png_path}")

    # Show interactive window if a display is available
    if matplotlib.get_backend() != "agg":
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    # Allow passing a specific CSV as an argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.isfile(csv_path):
            print(f"[ERROR] File not found: {csv_path}")
            sys.exit(1)
    else:
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "force_data_logs",
        )
        csv_path = find_latest_csv(log_dir)
        if csv_path is None:
            print(f"[ERROR] No wrench_data_*.csv files found in {log_dir}")
            sys.exit(1)

    plot_wrench(csv_path)


if __name__ == "__main__":
    main()
