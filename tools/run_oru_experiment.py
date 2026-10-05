"""Launch reproducible ORU v2 learning runs; default is a command preview."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['full', 'single', 'no_path', 'no_stage', 'hard_switch', 'all'], default='full')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.num_envs < 1 or (args.num_envs * 128) % 512:
        parser.error('epochs must be positive; num-envs*128 must be divisible by minibatch_size=512')
    root = Path(__file__).resolve().parents[1]
    methods = ['full', 'single', 'no_path', 'no_stage', 'hard_switch'] if args.method == 'all' else [args.method]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for method in methods:
        for seed in args.seeds:
            name = f'oru_v2_{method}_s{seed}_{stamp}'
            command = [sys.executable, str(root / 'scripts/reinforcement_learning/rl_games/train.py'),
                       '--task', 'Isaac-Oru-Direct-v0', '--headless', '--num_envs', str(args.num_envs),
                       '--seed', str(seed), '--max_iterations', str(args.epochs),
                       f'env.task.experiment_method={method}',
                       f'agent.params.config.full_experiment_name={name}']
            print(subprocess.list2cmdline(command), flush=True)
            if args.execute:
                subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
