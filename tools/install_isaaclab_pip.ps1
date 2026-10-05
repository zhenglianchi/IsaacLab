param(
    [string]$Python = 'D:\anaconda3\envs\env_isaaclab\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$constraints = Join-Path $projectRoot 'constraints-isaaclab-cu128.txt'

function Invoke-Pip {
    & $Python -m pip @args --disable-pip-version-check --progress-bar off
    if ($LASTEXITCODE -ne 0) { throw "pip failed (exit $LASTEXITCODE)" }
}

# Python 3.11 must already exist; both Sim and Lab are installed with pip.
& $Python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required' }
Invoke-Pip install pip==23.0 setuptools==65.0.0 wheel==0.45.1
# pip 23 needs these from PyPI before using the PyTorch-only index.
Invoke-Pip install typing-extensions sympy==1.14.0 networkx jinja2 fsspec filelock 'numpy<2' pillow==11.3.0 h5py==3.15.1 flatdict==4.0.0 -c $constraints
Invoke-Pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 -c $constraints
Invoke-Pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com -c $constraints
# Keep Lab and the custom ORU task linked to this checkout.
Invoke-Pip install --no-build-isolation -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks -e source/isaaclab_rl -e source/isaaclab_mimic -e source/isaaclab_contrib -c $constraints
# NVIDIA's Python 3.11 compatible rl_games fork, pinned to the verified commit.
Invoke-Pip install 'rl-games @ git+https://github.com/isaac-sim/rl_games.git@6b3534f29568158e9e29ec8bf83cc88fce5f0cae' ray==2.45.0 wandb==0.19.11 -c $constraints
Invoke-Pip check
