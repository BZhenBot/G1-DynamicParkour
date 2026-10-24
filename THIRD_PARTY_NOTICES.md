# 第三方版权与许可证声明 / Third-party notices

原创代码采用根目录 MIT。用户明确确认原创代码、训练权重和自制视频发布权利；署名使用 G1-DynamicParkour contributors。MIT 不覆盖第三方代码、模型、商标或依赖组件；第三方不为本项目背书。

| 组件 | 版本 | 上游声明及本发行处理 |
|---|---|---|
| mjlab | 1.6.0 | Apache-2.0；lab_api 含 Isaac Lab BSD-3-Clause 部分。仅依赖/API调用，不打包vendor或wheel；原许可证在LICENSES |
| rsl-rl-lib | 5.4.2 | BSD-3-Clause，ETH Zurich / NVIDIA。仅依赖安装及派生API，原许可证在LICENSES |
| MuJoCo / MuJoCo Warp | 3.11.0 | 上游Apache-2.0；仅外部依赖，不分发二进制 |
| NVIDIA Warp | 1.14.0 | 上游Apache-2.0及附带组件声明；不打包 |
| PyTorch / TensorDict / NumPy | 2.7.1 / 0.14.3 / 2.2.6 | 各分发自带许可和依赖notices；不打包，见audit/dependency-licenses.json |
| CUDA用户态库 | 精确版本见lock | 各NVIDIA条款；不打包，不能以MIT覆盖 |
| Unitree G1 XML/mesh/纹理 | mjlab wheel内部 | 精确来源与资产级授权未闭合；原文件及含其渲染的视频暂不公开 |
| 其余依赖 | requirements.lock.txt | 各自metadata声明，不视为法务认证；只公开版本/声明索引 |

上游证据（2026-10-24访问）：

- [mjlab v1.6.0 LICENSE](https://github.com/mujocolab/mjlab/blob/v1.6.0/LICENSE) 与 [第三方说明](https://github.com/mujocolab/mjlab/blob/v1.6.0/README.md#third-party-code)
- [rsl_rl v5.4.2 LICENSE](https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/LICENSE)
- [MuJoCo 3.11.0 LICENSE](https://github.com/google-deepmind/mujoco/blob/3.11.0/LICENSE)
- [Warp v1.14.0 LICENSE](https://github.com/NVIDIA/warp/blob/v1.14.0/LICENSE.md)
- [Menagerie G1 BSD声明](https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_g1/LICENSE)：仅比较证据，没有证明本地模型逐文件相同，不能据此闭合授权链。

公开核心与本地mjlab源码中至少9行函数的AST精确比较未发现完整相同函数。该检查不能证明不存在改写或片段复用；项目不主张上游API、PPO和机器人动力学的原创性。

十个checkpoint由用户授权发布，为本地训练参数，含optimizer/normalizer及嵌套先验。基础站立入口从随机初始化训练，后续复用本工程checkpoint。训练代码、权重角色及SHA可核查；本发行不提供外部动作捕捉或第三方预训练策略。完整训练祖先链没有全部打包，底层机器人资产授权风险仍需人工核查。

完整视频的原创录制已授权，但这不能证明模型渲染权利已经闭合，因此暂置本地隔离区。仓库展示只用自行编写的结构图与数值结果。许可证元数据和自动比对属于工程审计证据，不是无侵权保证。
