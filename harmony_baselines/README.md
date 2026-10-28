# AuctionNet Baseline 本轮适配、训练与评测

该目录是2026-10-28新实验入口。`evaluator.py` 只组织合法观察、对手生命周期、计时和结果；全部竞价、预算重拍、支付及累计价值都交由 `harmony_bidding.auction.AuctionEngine`。核心环境的原入口仍为 `harmony_bidding`。

## 范围与比较口径

- 算法：FixedAlpha、LocalPID、OnlineLP，以及直接复用 AuctionNet 源码训练步骤的 BC、IQL、BCQ、TD3+BC、CQL、DT。Abid 实际为当前均值归一化的二次报价，不能冒称 FixedAlpha；MBRL 只有 wrapper 且训练代码/权重不完整，未伪造替代成绩。详见 [源码审计](UPSTREAM_AUDIT.md)。
- 使用保留的72片转换数据；**本轮正式范围为2个完整训练周期（7、8）、1个验证周期（21）、1个测试周期（24）**。每周期保留全部请求、48广告主和参与掩码。其余源周期没有纳入本轮比较，不将“完整周期”写成“全部源数据”。
- Fixed、PID、Mixed 对手先用训练/验证数据调优并冻结。Fixed 对手每角色可有独立系数；Mixed 在每局开始按 seed 固定23个 Fixed、24个 LocalPID（目标之外47人）。PID 的后续轨迹允许因目标行为而变化。
- 目标策略为跨场景、跨角色共享策略；固定/ PID 目标以冻结Fixed对手参数的中位数作为共享初始alpha，搜索倍率 `0.5/1/2/4`，不使用逐角色参数表，神经模型不按测试角色重新训练。全广告主轮换是每次仅替换一名目标，覆盖48角色；固定角色0结果只是辅助子集。
- 两个训练 seed `20261028/20261029` 均保留。按两 seed 的验证均值选择统一超参数与训练步数，最终测试两个模型，**不按 seed 择优**。竞价/分配评测 seed 也各为这两个值。
- 每个神经算法默认两种学习率 `3e-4/1e-3`，每个 seed 各训练600步，保存200/600步候选。它是公开的**有限搜索预算，不是收敛证明**。选择依据是完整验证周期、6个代表角色、三场景的相对 LocalPID 竞胜价值。
- CQL 的一个原始log动作候选在完整验证局发生真实溢出，因此额外训练 `linear_scaled + learning_rate=1e-4`、双seed、200/600步四个checkpoint。线性坐标为 `u=alpha/train_scale`，执行 `alpha=train_scale*max(0,u)`，无动作上限，保留上游CQL训练步骤。额外候选与健康原候选一同由验证集选择；本轮选中线性600步。CQL搜索预算多于其余算法，不能声称完全相同预算。52个候选记录及额外预算保存在 `models/candidates.json` 和 `models/stability_tuning.json`。
- 主实验只使用原预算 `budget_scale=1`。若另做预算压力实验，必须新建独立配置和结果表，不能混入主排名。

## 环境

在 `AuctionNet` 目录使用独立 `.venv-baselines`（本轮 Python 3.12、CPU Torch），不修改系统 Python，也不要求基础 `harmony_bidding` 使用神经依赖。

`requirements-baselines-lock.txt` 是本轮虚拟环境 `pip freeze` 的完整实录；`requirements-baselines.txt` 只列本实验实际直接依赖。锁文件包含已安装但**本项目适配版不需要**的 `transformers/tokenizers/huggingface_hub` 等包。DT 是上游自写 PyTorch Transformer，不需要下载 HuggingFace 模型，也不需要为复现强行安装这些可选包。

```powershell
# 已有本轮环境可直接使用；这些命令不会修改系统 Python。
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m pip install -r requirements-baselines.txt
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m pytest tests/test_harmony_baselines_neural.py tests/test_harmony_baselines_protocol.py -q -p no:cacheprovider
```

## 串行阶段命令

以下阶段必须依序完成，不能在训练/选模完成前运行 `test`。`--workers` 只并行该阶段内独立完整比赛，不改变阶段依赖。不要用旧研究目录或历史模型替换任一阶段。

```powershell
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.experiment prepare
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.experiment calibrate --workers 6
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.experiment tune --workers 6
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.resources --workers 1 4 12
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.refine --workers 4
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.experiment trajectories --workers 4
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.experiment train --steps 600
& '.\.venv-baselines\Scripts\python.exe' -X utf8 scripts/train_cql_linear.py
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.supervised select --workers 4 --timeout 180
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.supervised test --workers 4 --timeout 180
& '.\.venv-baselines\Scripts\python.exe' -X utf8 -m harmony_baselines.report --root outputs/baseline_comparison_20261028
```

`prepare` 只建立训练/验证缓存；测试缓存由已冻结模型的 `test` 阶段首次请求。缓存是原转换数据的有指纹矩阵I/O视图，保留源CSV、不重新下载或重新转换原始数据。`trajectories` 运行288个完整新局，保存20,736条目标转移，按任务号、角色和片号排序，避免并行完成顺序影响训练随机数。行为噪声围绕**已选最佳**Fixed/PID倍率乘 `0.7/1/1.4`，不会丢掉已选基数。

CLI默认4 workers。`resources` 以同12个训练完整局测试1/4/12 workers，本轮墙钟约50.4/28.1/30.2秒，选择4用于后续；细项见 `resource_benchmark.json`，这些局不进入排名。`refine` 检查OnlineLP是否选中最低倍率，如是则只在原验证条件上测试更低两档，最多3轮；初始选择另存 `classical_selected_initial.json`，扩展成绩与 `classical_refinement.json` 均保留。所有扩展发生于冻结和测试之前，完整候选见 `classical_search.json.algorithm_multipliers`。

输出默认 `outputs/baseline_comparison_20261028`。训练和初次阶段要求新目录；`supervised` 对中断的当前运行只恢复经过任务清单、checkpoint、成绩身份与数据指纹核对的成功行，绝不把失败当成功。重跑时在所有 `experiment/resources/refine/supervised` 和 `train_cql_linear.py` 命令后添加同一 `--output outputs/my_fresh_run`，报告使用 `--root outputs/my_fresh_run`；`run_id` 自动采用新目录名。**不要修改已封存源码或删除本轮目录来重跑**，否则旧结果的源代码完整性检查会失败。已验证的只读数据缓存仍从默认缓存位置复用。当前 CLI 的数据划分、seed 和角色集合来自 `experiment.py` 常量；[可读配置模板](../configs/baseline_comparison_20261028.json) 记录这些值，**不是额外的 `--config` CLI 参数**。实际运行以输出 `protocol.json`、冻结配置和checkpoint元数据为准。

`supervised` 使用持久spawn worker，只复用只读市场矩阵和封存网络，每局仍创建新Agent并reset。每个完整72片任务有180秒资源期限，逐片记录目标alpha和自身反馈。完整局异常与资源超时分开记录，均不给0分；验证候选缺少任一seed或角色即排除。测试若出现失败，保持冻结选择，不换checkpoint、不回退；保留成功原始行，整个不完整算法退出主排名，其余算法继续48角色全覆盖。原882个验证任务及新增72个CQL任务的索引、成功/失败/未运行集合都可审计。失败与跳过记录见 `validation_failures.json`、`validation_skipped.json`、`evaluation_failures.json`、`evaluation_skipped.json`。

Windows运行中建议查看追加日志，不同时轮询正在原子替换的 `*.status.json`。本轮一次状态文件替换遭遇 `WinError 5`，已有1,659个完整测试局经指纹核验后按同一CLI安全续跑，没有修改模型或比赛代码。中断原日志、CSV快照和分段时间均保留。`runtime_accounting.json` 区分可恢复的执行段与未完整保存的早期验证计时；报告中的已记录阶段时间不能解释为项目端到端耗时。全部任务结束后可运行 `scripts/audit_completed_baseline_pipeline.py --output <run>` 核对冻结资产、完整任务键、每局72片诊断和CSV价值/费用一致性。

## 主要文件与结果追溯

| 文件 | 内容 |
|---|---|
| `data.py` | 源SHA校验、完整矩阵缓存与数据清单；市场数据只在评测器 |
| `classical.py` | 本地共享Fixed/PID，以及重建合法表后复用上游在线LP查表 |
| `neural.py` / `upstream.py` | 合法18维自身历史、原始算法训练步骤、动作尺度、独立checkpoint加载 |
| `experiment.py` | 对手校准、基线调参、新轨迹、训练、验证选模和全角色测试 |
| `supervised.py` | 完整任务资源期限、失败隔离、校验后续跑、验证选择封存与全角色测试 |
| `report.py` | 仅从已完成CSV计算统计、绘图和生成简洁主报告 |
| `protocol.json` / `opponents_frozen.json` / `selected_frozen.json` | 数据划分、规则、种子、对手参数、候选选择和冻结摘要 |
| `models/*/config_*_seed_*/training.json` / `step_*.pt` | 实际损失、训练时长、参数量、配置与候选模型 |
| `models/candidates.json` / `failures.json` | 成功候选与真实失败；缺seed覆盖的候选不能伪装完成 |
| `episode_results.csv` / `players_Fixed.csv` 等 | 完整逐周期、角色、场景、竞价seed、训练seed原始成绩 |
| `fixed_player_results.csv` | 同一正式测试中player0的辅助结果 |

最终全角色指标先在同角色内平均完整测试日、评测seed和训练seed，再计算同角色的 `V_algorithm/V_LocalPID`，最后等权平均角色。LocalPID分母为0时单独标记且报告有效角色数，不加epsilon。跨角色标准差、seed变化和跨场景变化是不同口径，不互相替代。LocalPID及归一化指标均为本地参考，不是官方PID或官方G。

## 源码复用范围

上游版本 `c20de631a40e24669d1d6a8e2acefd8f492f172d`、Apache-2.0，保留原源码、许可证和入口。本轮没有恢复旧研究代码或加载旧训练结果。BCQ/TD3使用明确标注的128宽紧凑网络，修复two-hot上界及TD3策略Q解码；CQL修复target初值；DT改为72片时钟及短训练warmup。原始训练机制保留，修改和限制均列于 `UPSTREAM_AUDIT.md`，不能称作“官方原样复现”。

独立加载规格示例：

```json
{"kind":"custom","class_path":"harmony_baselines.neural:NeuralAgent","kwargs":{"checkpoint":"绝对路径/本轮step_600.pt"}}
```

每局 `reset/bidding/update` 只接触自身配置和已完成片反馈；测试前校验冻结的checkpoint、OnlineLP表、对手配置、适配器、核心和真实上游源码摘要。最终测试的三个推理p99只计目标自身原始 `bidding/update` 调用，排除诊断JSONL写入、动作合法性检查和市场结算；每片相加后求paired p99。进程RSS为运行环境测量，包含只读缓存。这些是本地测量，不代表主办方执行器认证。

