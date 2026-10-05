"""Validate pinned packages, CUDA math, and optionally a short Isaac Lab run."""

import argparse
import importlib.metadata as metadata
import json

parser = argparse.ArgumentParser()
parser.add_argument("--simulation", action="store_true")
args = parser.parse_args()

expected = {
    "isaacsim": "5.1.0.0",
    "isaaclab": "0.54.2",  # Core extension version in repository release 2.3.2.
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "torchaudio": "2.7.0+cu128",
    "h5py": "3.15.1",
    "pip": "23.0",
    "setuptools": "65.0.0",
    "flatdict": "4.0.0",
}
versions = {name: metadata.version(name) for name in expected}
print(json.dumps(versions, indent=2), flush=True)
assert versions == expected, (versions, expected)

import torch
import torchvision
import torchaudio
import h5py
import flatdict

assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"
x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
assert torch.allclose((x @ x.T).cpu(), x.cpu() @ x.cpu().T)
torch.cuda.synchronize()
print(f"CUDA math PASS: {torch.cuda.get_device_name(0)}", flush=True)

if args.simulation:
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext

        simulation = SimulationContext(SimulationCfg(device="cuda:0"))
        simulation.reset()
        for _ in range(10):
            simulation.step()
        print("Isaac Lab simulation PASS: 10 steps", flush=True)
        # The published Lab wheel's STOP callback keeps rendering until play.
        # Unsubscribe via the public cleanup API before stopping the timeline.
        simulation.clear_all_callbacks()
        simulation.clear_instance()
        simulation.stop()
        print("Isaac Lab physics cleanup PASS", flush=True)
    finally:
        # This smoke test has no Replicator writers or pending data to flush.
        launcher.app.close(wait_for_replicator=False)
        print("Isaac Lab shutdown PASS", flush=True)
