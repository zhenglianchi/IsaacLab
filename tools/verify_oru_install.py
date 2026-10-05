"""Verify local ORU registration and a small CUDA rollout, without training."""

import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--registration-only", action="store_true")
parser.add_argument("--steps", type=int, default=5)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.chdir(root)  # ORU USD paths are relative to the checkout.

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
env = None
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab.sim import SimulationContext

    print(f"Task package: {isaaclab_tasks.__file__}", flush=True)
    spec = gym.spec("Isaac-Oru-Direct-v0")
    import isaaclab_tasks.direct.oru.oru_env as oru_env

    expected = root / "source/isaaclab_tasks/isaaclab_tasks/direct/oru/oru_env.py"
    assert Path(oru_env.__file__).resolve() == expected.resolve(), oru_env.__file__
    print(f"ORU registration PASS: {spec.id} -> {oru_env.__file__}", flush=True)

    if not args.registration_only:
        from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner
        from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

        agent_cfg = load_cfg_from_registry(spec.id, "rl_games_cfg_entry_point")
        assert agent_cfg["params"]["config"]["name"] == "OruAssembly"
        cfg = parse_env_cfg(spec.id, device="cuda:0", num_envs=1)
        env = gym.make(spec.id, cfg=cfg)
        obs, _ = env.reset()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(
                torch.zeros(env.action_space.shape, device="cuda")
            )
            assert torch.isfinite(reward).all(), reward
            assert torch.isfinite(obs["policy"]).all()
        print(f"ORU CUDA rollout PASS: {args.steps} steps; policy={tuple(obs['policy'].shape)}", flush=True)
except BaseException:
    # Kit shutdown may exit the process directly: emit failures before closing.
    import traceback
    traceback.print_exc()
    raise
finally:
    # Unsubscribe the published Lab STOP callback before stopping its timeline.
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    if sim is not None:
        sim.clear_all_callbacks()
        sim.clear_instance()
    if env is not None:
        env.close()
    launcher.app.close(wait_for_replicator=False)
