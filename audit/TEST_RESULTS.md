# 独立复现测试实测

结论：独立空环境通过离线安装、依赖一致性、项目editable安装、CUDA/局部CPU运行探测、十checkpoint加载及完整V2.2单路线回放。种子98000的1/1路线独立200Hz审计合格。此处1/1仅烟测，不是泛化成功率，也不替代历史76/100。

| 检查 | 实测结果 | 限制/证据 |
|---|---|---|
| 独立环境创建 | 通过 | Python3.10.12；未启用system-site-packages |
| PyPI联网获取原lock | 失败，exit2 | 访问filelock索引超时；DNS探测也失败，未绕过安全机制 |
| 首次离线安装 | 中断，exit null | 服务/WSL重启；保留stdout/stderr，不称通过 |
| 第二个空环境离线安装 | 通过，exit0 | 138个锁定依赖，从原缓存的独立复制件获取；公开包不带缓存 |
| pip check / 项目安装 | 通过 | --no-deps editable安装，后续显式当前src验证 |
| CUDA/CPU基础探测 | 通过 | RTX4060；CPU仅球体，不是G1 |
| ZIP及十权重SHA/恢复 | 通过 | 逐项大小/SHA，参数保留原文件 |
| 十checkpoint反序列化/有限性 | 首次失败，修正后通过 | 审核脚本名inspect.py遮蔽stdlib inspect；改名并用隔离导入，非策略/权重失败 |
| 事件检测逻辑 | 通过 | 合成状态，不是RL episode |
| 命令连续性 | 首次失败，包装器重试通过 | 原命令缺G1_RUN_DIR；修正指南，未改核心 |
| CPU载荷平台物理 | 首次失败，补资产后通过 | 初次漏选原创箱体XML；补齐后3候选均通过，不含G1模型 |
| 最小完整V2.2回放 | 通过，exit0；独立合格1/1 | 两次AIR及落地稳定核验；新完整日志本地保留 |
| 正式100路线入口 | 仅dry-run预检通过 | 真实100回合未执行；协议证据路径迁移不改变控制或物理判据 |
| 源码语法 | 通过 | AST解析，不启动训练 |

原始命令、起止时间、耗时、stdout/stderr、退出码及源码快照在本地_local_review；无episode的审计记不适用。机器路径不进入公开日志。新烟测摘要见results/public-smoke。公开副本的SHA/源码快照支持核查实际被测试内容；测试时尚无新Git提交，因此相关commit为null。

未完成：公网全量依赖下载、同新仓库正式100重评、从零全链训练、小规模PPO训练、跨设备/驱动、CFF官方schema验证及底层模型逐文件授权核对。训练命令保留但本次没有启动训练。最终Windows副本检查另见WINDOWS_VERIFICATION。

Windows最终副本补充：同seed98000、自身src和SHA相同权重，执行exit0且audit_pass=true，但qualified_successes=0（leg2:premature_target_contact），任务0/1。WSL先前1/1不与其合并，重复性差异未决。失败数据和原始日志保留，见WINDOWS_VERIFICATION。
