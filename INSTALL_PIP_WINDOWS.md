# Windows：本地 Isaac Lab + pip Isaac Sim

Python 环境：D:\anaconda3\envs\env_isaaclab（Python 3.11）。
项目目录：C:\Users\zhenglianchi\Desktop\IsaacLab。

Isaac Sim 使用官方 pip 包。Isaac Lab 已通过 pip install -e 安装当前项目 source 下的六个子包。
ORU 随本地任务包加载，修改源码直接生效；移动项目目录后需重新安装。
不再使用之前的官方 Lab wheel、wheel 依赖补丁或临时 ORU 目录链接。

## 激活与验证

在 Anaconda Prompt 中：

```bat
conda activate env_isaaclab
cd /d C:\Users\zhenglianchi\Desktop\IsaacLab
python -m pip check
python tools\verify_isaaclab_install.py
python tools\verify_oru_install.py --steps 5
```

## 固定版本

| 组件 | 版本 |
| --- | --- |
| Isaac Sim | 5.1.0（pip 元数据：5.1.0.0） |
| Lab 仓库发行版 | 2.3.2（根目录 VERSION） |
| isaaclab 核心子包 | 0.54.2 |
| isaaclab_assets / isaaclab_tasks | 0.2.4 / 0.11.12 |
| isaaclab_rl / isaaclab_mimic / isaaclab_contrib | 0.4.7 / 1.0.16 / 0.0.2 |
| torch / torchvision / torchaudio | 2.7.0+cu128 / 0.22.0+cu128 / 2.7.0+cu128 |
| h5py / pip / setuptools / flatdict | 3.15.1 / 23.0 / 65.0.0 / 4.0.0 |
| rl_games | 1.6.1，NVIDIA Python 3.11 分支 |
| ray / wandb | 2.45.0 / 0.19.11 |

Lab 子包版本来自各自 config/extension.toml，与仓库发行版本采用不同编号。
rl_games 的 Git 提交为 6b3534f29568158e9e29ec8bf83cc88fce5f0cae。

## 重装

完整复用脚本：tools/install_isaaclab_pip.ps1。
仅重新安装本地 Lab：

```bat
python -m pip install --no-build-isolation -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks -e source/isaaclab_rl -e source/isaaclab_mimic -e source/isaaclab_contrib -c constraints-isaaclab-cu128.txt
```

后续 pip 安装继续传入 -c constraints-isaaclab-cu128.txt，保留固定依赖。
为匹配 Sim 5.1 的 FastAPI 0.115.7，本地 source/isaaclab/setup.py 中 Starlette
已从 0.49.1 调整为 0.45.3。Ray、wandb 使用兼容版本，避免升级 Sim 要求的
packaging==23.0 和 click==8.1.7。

## ORU

任务 ID：Isaac-Oru-Direct-v0。
源码：source/isaaclab_tasks/isaaclab_tasks/direct/oru。
资产使用相对路径，因此请在项目根目录运行。

```bat
python tools\verify_oru_install.py --registration-only
python tools\verify_oru_install.py --steps 5
```

第一条核对实际加载的 ORU 源码路径；第二条读取 RL 配置、创建单个 CUDA 环境并步进，不执行训练。
结果应包含 PASS 标记；Kit 关闭可能覆盖 Python 异常退出码，不能仅凭退出码判断成功。

训练入口（本次未执行训练）：

```bat
python -u scripts/reinforcement_learning/rl_games/train.py --task Isaac-Oru-Direct-v0 --headless --num_envs 4
```

本机 RTX 4060 8GB，先用较少环境数，不要直接套用交接文档的 128 环境配置。
安装验证不代表策略已训练或插入成功率已验证。

## 本机验证

2026-09-26：六个 Lab 子包均确认以 editable 方式指向当前项目；pip check 无依赖冲突。
ORU 已完成单环境 CUDA 仿真 5 步，策略观测维度为 (1, 52)。
日志：.installation/oru-local-smoke.log。

场景加载仍提示 ORU/Ground 的 instanceable geometry 未应用运行时碰撞属性覆盖；
这不妨碍本次加载和步进，但进行装配效果验证前应检查实际接触参数。
Windows 长路径支持已开启，驱动仍为 566.03，未更换显卡驱动。
安装日志与版本快照保存在 .installation（已加入 .gitignore）。
