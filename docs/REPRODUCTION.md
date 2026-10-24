# 复现指南

区分历史验收、本次发行烟测与未完成测试；本次状态见[TEST_RESULTS](../audit/TEST_RESULTS.md)。目标Ubuntu22.04/WSL2、Python3.10、NVIDIA GPU。以下均从独立检出根目录执行；Windows通过`wsl.exe -d Ubuntu-22.04 -- bash -lc '<Linux命令>'`调用，Python/仿真在WSL运行。不覆盖原环境或更改全局CUDA。

## 安装

确认`/usr/bin/python3 --version`为3.10，并有外部工具uv；工具不随仓库分发。

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r configs/requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
.venv/bin/python scripts/probe_python.py
```

核心版本：mjlab1.6.0、MuJoCo/MuJoCo Warp3.11.0、Warp1.14.0、PyTorch2.7.1、rsl-rl-lib5.4.2。lock是精确版本快照，**不是带下载哈希的供应链锁文件**。本次联网获取失败；离线独立重装不证明公网依赖可获取。离线测试用原缓存的独立复制件，新建空env，不读取原site-packages或启用system-site-packages。公开包不含9GB缓存，外部用户仍须获取上游依赖。

机器人由mjlab加载，仓库不带XML/mesh；安装上游不代表本项目已证明所有资产再分发权。待授权见审计。完整策略入口固定CUDA，CPU检查不能代替完整策略评价。

## 权重恢复

`release/weights-manifest.json`记录十个权重的角色、相对路径、大小、SHA。六部署角色为首跳、中间恢复、准备、位置、第二跳、终点恢复，还需standing/jump构造先验及V2比较准备策略。只保存六个主权重会缺依赖。

尚未上传Release、无下载URL。本地审核包`_release_assets/`中有拟发布ZIP。取得后运行：

```bash
.venv/bin/python scripts/prepare_weights.py --archive g1-dynamicparkour-v2.2-weights.zip
```

该命令先验ZIP整体SHA，只恢复清单成员，拒绝路径越界并逐权重校验。保留原`experiments/.../ppo/`相对目录，Git忽略这些文件。checkpoint含训练器序列化信息，只加载此清单校验过的可信本项目文件。视频不是运行依赖，仍在本地待授权区。

## 局部检查

```bash
.venv/bin/python scripts/test_event_detector.py
.venv/bin/python scripts/run_experiment.py --label command-continuity --kind audit --timeout 60 -- .venv/bin/python scripts/validate_command_continuity.py
.venv/bin/python scripts/run_experiment.py --label platform-cpu --kind audit --timeout 180 -- .venv/bin/python scripts/validate_platform_physics.py --device cpu-native
```

逻辑测试使用合成状态，不是机器人成功率；CPU平台测试不是G1完整策略评价。`run_experiment.py`每次保存唯一目录、commit（无提交则null）、工作树/源码、命令、配置、种子、stdout/stderr、退出码；无episode记不适用。

## 最小完整回放

```bash
.venv/bin/python scripts/reproduce_course_release.py \
  --python "$PWD/.venv/bin/python" \
  --config configs/v22_course_gap010_limited_middle064_jump015_v1.json \
  --episodes 1 --seed 98000 --video-episodes 0 --label public-smoke
```

部署配置不带formal100约束。入口强制本副本src、校验主权重，并对新路线独立物理审计。GPU非逐位确定，一次烟测不验证历史76%。`--video-episodes 1`可本地生成完整回放；模型权利问题解决前不公开发布新渲染。

## 正式评价

```bash
.venv/bin/python scripts/reproduce_course_release.py \
  --python "$PWD/.venv/bin/python" \
  --config configs/v22_course_formal100_gap010_limited064_v1.json \
  --episodes 100 --seed 98000 --video-episodes 0 --label public-formal100
```

正式配置有100回合和冻结种子约束，不随意修改后仍称原协议。检查新`course_evaluation.json`、独立`course_independent_audit.json`的qualified_successes、失败原因、AIR/接触/稳定判据，退出0仅表示执行完成。本次没有重跑正式100，历史76/100见results。V2.1正式种子须读取其配置，不能套用98000。

## 训练入口与范围

小规模PPO集成烟测（显式训练8次迭代，非成功验收）：

```bash
.venv/bin/python scripts/run_experiment.py --label standing-smoke --kind train --seed 0 --config configs/stage0_smoke.json --timeout 600 -- .venv/bin/python scripts/train_baseline.py --config configs/stage0_smoke.json
```

保留`train_jump.py`、`train_recovery_standing.py`、`train_prepared_jump.py`、`train_end_recovery.py`等真实PPO入口。其完整历史课程另依赖父checkpoint和真实接管bank；本发行只含必要部署闭包与一个landing bank，**不宣称整个训练链从零一键复现**。未纳入的训练素材仍在原科学归档；依赖图与待测项如实列明。

本次不启动大规模训练；未执行的小规模训练命令也不标通过。操作日志会含本机路径，默认本地保存并被Git忽略，后续公开须重新扫描。跨GPU/驱动/依赖源的可重复性与长期稳定性待验证。
