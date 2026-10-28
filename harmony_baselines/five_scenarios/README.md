# AuctionNet 十种Baseline与五场景实验

这是独立的后续实验，输出到 `outputs/baseline_five_scenarios_20261028/`。原 `harmony_baselines` 执行文件、`outputs/baseline_comparison_20261028/`、原模型与报告保持不变。新评测器只组织观察、策略和结果，所有竞价仍调用项目唯一的 `AuctionEngine.run_matrix_slot`。

## 两组实验

- **A：冻结泛化。**旧9类保持原参数和权重，直接复用旧3场景4,320局；在AuctionNetPID和Mixed2补2,880局。新AuctionNetPID先沿旧3场景、6验证角色及相对旧LocalPID选模口径调参，再测5场景480局。A合计7,680局，其中新增3,360局。
- **B：五场景联合调优。**所有10类按五场景验证平均累计价值V选择；R、预算利用、最差场景和角色P10是诊断。神经模型两个训练seed一起选共同配置/步数，并一并测试。B在period-24有7,680个完整配置单元；仅与A策略规格、模型/LP资产和比赛条件完全相同的单元可复用。
- **独立补充测试。**预先指定period-25；B模型、对手及执行代码冻结后才首次读取其市场矩阵。另测完整7,680局，单列结果。period-24是历史已展示的可比性测试，不声称为全新盲测。

场景名为Fixed、LocalPID、Mixed、AuctionNetPID、Mixed2。历史的PID名称只在新结果中规范为LocalPID，保留原场景名和来源。Mixed与Mixed2分别有23名Fixed和24名对应PID，两者在相同目标/日期/seed下使用同一组Fixed成员。目标始终只替换一名广告主；每个对手有独立实例和reset。

## 实现与计划

`auctionnet_pid.py` 保留上游的预算消耗阈值乘法控制，它不是标准P/I/D控制器；具体公式、48→72、零余额、reset与数值边界见 `PID_ADAPTATION.md`。默认初始α=15、低/高阈值0.7/1.1、上/下调倍率1.2/0.7作为验证候选保留。目标参数共享；新对手可以按训练角色校准初始值。旧Fixed/LocalPID对手参数逐字段保持一致。

训练数据复用原288条72片轨迹，沿同一组六种行为策略补192条新场景轨迹，共480条、34,560个转移，每场景96条。场景标签只在旁路元数据中，不能进入actor。18维自身历史、动作和奖励编码沿原适配器。

神经训练从头开始：6算法×2学习率×2seed=24个run，每run1,800步，保留600/1,800步共48个checkpoint。CQL沿旧选中模型的linear_scaled动作坐标，学习率1e-4/3e-4；其余log1p，学习率3e-4/1e-3。网络和核心loss不变，不因测试分数调整预算。训练时间、损失、参数量、实际配置和资产SHA均记录。

Online LP用训练period-7的全部48角色、五场景、四个固定α共960局重建成本响应表；仍复用原查表核。经典候选范围、对手校准和选择规则在测试前写入不可变 `protocol.json`。

## 命令与恢复

从AuctionNet目录运行，使用已有 `.venv-baselines`，不修改系统Python：

```powershell
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline prepare
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline opponents --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline tune-A --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline test-A --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline trajectories --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline lp --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline tune-B-classical --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline train
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline tune-B-neural --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline test-B --workers 4
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.pipeline independent --workers 4
```

`all`按以上顺序执行。初始计划为4个worker；若先运行仅训练市场的资源测试，采用的资源数及实际耗时必须另外记录，不能将资源调整解释成算法改进。各阶段状态保存实际worker数；资源benchmark局不进入训练数据、模型选择或正式排名。

本轮已完成4/8 worker资源对照：同一训练市场的48局分别耗时54.007秒与45.082秒，48对价值、成本和胜出数完全一致。`resource_application.json`记录后续正式阶段采用8个worker；初始`protocol.json`中的4保持原样。因此，复现本轮后续执行可使用`pipeline all --workers 8`；正式输出日志为`pipeline_run_001.log`。该单次、固定顺序对照包含进程启动、导入和退出，不能排除文件缓存及机器负载影响。

每局有独立的原子提交JSON、逐片诊断JSONL和任务身份。任务键覆盖策略/模型/LP资产、市场、场景、角色、seed、对手参数、规则及执行代码。CSV由已提交结果派生；重跑同一阶段会校验后跳过成功局。失败、超时和取消另列，验证缺任一角色或训练seed的候选不能进入选择，测试失败不能临时换模型。

本轮独立周期发生了一次系统待机中断：原阶段记录5次资源超时，并按同组失败规则跳过141局。用户授权重跑后，`recovery.py`将这146个缺失身份单独封存到`independent_test_recovery_001/`，使用原模型、数据、seed、180秒超时规则及2个worker恢复；它不重算或覆盖已有成功局。用户报告的断网与Windows待机事件分别记录于`qa/independent_interruption_20261028/`。恢复入口为：

```powershell
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.recovery plan
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.recovery run --workers 2
& ./.venv-baselines/Scripts/python.exe -X utf8 -m harmony_baselines.five_scenarios.recovery finalize
```

此入口针对已封存的本次中断，不能用于任意修改失败范围。`plan`固定任务映射、证据及恢复脚本SHA；之后不得修改脚本。`finalize`只在原pipeline已退出、恢复完整成功且两部分严格覆盖原7,680个身份时发布最终表，先保留原不完整派生表的字节快照。原失败、跳过、诊断和成功结果不变，包括待机期间仍成功提交的局及其长时延。最终`independent_test/recovery_manifest.json`记录来源映射和SHA；状态中的历史失败/跳过与当前未解决数分开统计。重新生成报告必须读取最终`completed`且`valid`的状态，不能把进程退出码当作完整性证明。

训练恢复以完整run为单位：已完成run的输入、配置、源代码、环境、日志和checkpoint逐项核验后复用；中断run保留原attempt，在新attempt目录从头重启该run。它不声称从中间权重精确恢复optimizer/RNG。不要修改冻结执行源或既有权重来绕过身份校验。

## 结果与追溯

| 文件或目录 | 内容 |
|---|---|
| `historical_assets_manifest.json` | 原实验全部文件及关键源码/报告的SHA、大小和修改时间 |
| `protocol.json` | 预先定义的数据范围、候选空间、训练预算与选择规则 |
| `opponents_frozen.json` | 原Fixed/LocalPID和新AuctionNetPID对手 |
| `validation_A/` | 新AuctionNetPID的原3场景验证及选择 |
| `trajectory_extension/`、`train_transitions.npz`、`trajectory_episodes.csv` | 192条新增轨迹、合并训练数据和来源/场景标签 |
| `lp_training/`、`online_lp_table.csv` | 五场景成本响应及冻结LP表 |
| `models/runs/`、`models/candidates.json` | 独立训练attempt、日志、权重与全局候选索引 |
| `validation_B_classical/`、`validation_B_neural/` | 五场景候选验证与平均V选择 |
| `experiment_A/`、`experiment_B/`、`independent_test/` | 各自冻结选择、完整逐局CSV、逐片CSV和完成验收 |
| `*_new/` | 真正新增执行的任务、逐任务结果与失败记录 |
| `reporting/` | 派生统计、12组图PNG/SVG及配对分析 |
| `report_auctionnet_baselines_comparison.md`、`report_auctionnet_baselines_detailed.md` | 新版简报和完整技术报告 |

A与B的内部R分别除以各自LocalPID均值；比较提升以相同角色/场景/评测seed下、先合并训练seed的绝对V差及相对变化为主。零参考单列，不能加epsilon。扩大场景、重训和更长步数同时改变时，只能报告联合效果，不能据此证明某一个因素的独立因果贡献。
