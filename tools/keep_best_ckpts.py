"""Keep only three checkpoints per run (user request 2026-10-07).

  1. best reward      -> nn/OruAssembly.pth        (rl_games' own best-reward copy)
  2. best success     -> nn/best_success.pth       (picked by INDEPENDENT evaluation of the
                                                    snapshots: the metrics CSV with the highest
                                                    success rate)
  3. last epoch       -> nn/last_<...>.pth         (kept as-is)

Everything else in nn/ (the periodic snapshots) is deleted, together with the metrics CSVs
that were only used for the selection. Run it after training + evaluation:

    python tools/keep_best_ckpts.py <run_tag> [more_tags...]

It prints what it kept and what it removed, and never touches other runs.
"""
import csv
import pathlib
import shutil
import sys

SUB = pathlib.Path("logs/rl_games/OruAssembly")
LOGS = pathlib.Path("logs")


def rows(path):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def success(path):
    r = rows(path)
    if not r:
        return None
    ok = sum(1 for x in r if x.get("success") == "1")
    return 100.0 * ok / len(r), len(r)


def keep_best(tag):
    nn = SUB / tag / "nn"
    if not nn.is_dir():
        print(f"[{tag}] no nn dir - skipped")
        return
    # 1) best reward
    best_reward = nn / "OruAssembly.pth"
    # 3) last epoch
    lasts = sorted(nn.glob("last_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    last = lasts[0] if lasts else None
    # 2) best success: look at every evaluated metrics CSV belonging to this tag
    graded = []
    for f in sorted(LOGS.glob(f"metrics_{tag}_*.csv")):
        s = success(f)
        if s:
            graded.append((s[0], f, f.stem.replace(f"metrics_{tag}_", "")))
    graded.sort(reverse=True)
    best_success_ckpt = None
    kept_csv = []
    if graded:
        top = graded[0]
        src = nn / f"{top[2]}.pth"
        if not src.exists():
            for cand in nn.glob("*.pth"):
                if top[2] in cand.name:
                    src = cand
                    break
        if src.exists():
            dst = nn / "best_success.pth"
            shutil.copy2(src, dst)
            best_success_ckpt = dst
            print(f"[{tag}] best success {top[0]:.1f}% -> kept as best_success.pth (from {src.name})")
            kept_csv = [f for _, f, _ in graded[:1]]
        else:
            print(f"[{tag}] best-success snapshot not found on disk (looked for {top[2]}.pth)")
    # delete every other snapshot
    removed = 0
    for f in list(nn.glob("*.pth")):
        if f in (best_reward, last, best_success_ckpt):
            continue
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    # delete the metrics CSVs used for grading except the winning one
    for _, f, _ in graded:
        if f not in kept_csv:
            try:
                f.unlink()
            except Exception:
                pass
    keep = [p.name for p in (best_reward, best_success_ckpt, last) if p and p.exists()]
    print(f"[{tag}] kept: {keep}   removed {removed} snapshot(s)")


if __name__ == "__main__":
    tags = sys.argv[1:]
    if not tags:
        print("usage: python tools/keep_best_ckpts.py <run_tag> [more_tags...]")
        sys.exit(1)
    for t in tags:
        keep_best(t)
