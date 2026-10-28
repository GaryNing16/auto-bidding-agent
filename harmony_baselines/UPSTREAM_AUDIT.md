# AuctionNet 固定源码审计与本地适配边界

审计日期：2026-10-28（Asia/Shanghai）。来源是本工作区保留的 `strategy_train_env`、`simul_bidding_env` 和 `NOTICE-HARMONY.txt`，不是根据 README 算法名称推断。来源记录指向 [AuctionNet](https://github.com/alimama-tech/AuctionNet)，commit `c20de631a40e24669d1d6a8e2acefd8f492f172d`，Apache-2.0。运行时 `upstream.provenance()` 为实际复用的六个神经算法源码计算 SHA256；每个本轮 checkpoint 都带该清单。未修改上游文件，未加载上游或归档模型。

## 1. 实际存在的算法与完整程度

| 算法 | 实际源码与入口 | 完整程度与本轮使用方式 |
|---|---|---|
| Abid | `simul_bidding_env/strategy/abid_bidding_strategy.py` | 有策略，没有训练器。**它不是 Fixed Alpha**：`alpha = temporal_ratio[t] * cpa / mean(current_pvalues)`，最终 `bid = alpha * pvalue²`。既依赖当前均值，也不符合片内统一 `alpha*pvalue`，因此不把本地 FixedAlpha 冒称原始 Abid。保留固定基准的比较意图，明确标为本地替代。 |
| PID | `simul_bidding_env/strategy/pid_bidding_strategy.py` | 有无需训练的预算反馈启发式：上片消耗推算未来预算，小于阈值乘 1.2、超过阈值乘 0.7；不是标准比例、积分、微分控制。原始 48 片、末期除零和 reset 状态需要适配。本轮采用项目已有 **LocalPID**，不称 AuctionNet 原始 PID 或主办方官方 PID。 |
| Online LP | `baseline/onlineLp/onlineLp.py`、`run/run_onlinelp.py`、两套 strategy wrapper | 有成本/价值比排序、累积成本阈值表构建和在线查表。不是调用通用 LP solver。原训练器固定第一个 CSV、48 片、类别内随机采样 1/8、预算阈值 6000、间隔 10，原 wrapper 还有 `1.5*cpa` 上限。本轮直接复用其在线查表，重建符合本地核心的成本响应表；这属于**查表机制适配**，没有声称完整保留其原始离线训练器。 |
| BC | `baseline/bc/behavior_clone.py`、`run/run_bc.py` | 完整两层 128 ReLU actor、MSE 训练和推理。直接复用 `BC.step` 和 actor。 |
| IQL | `baseline/iql/iql.py`、`run/run_iql.py` | 完整双 Q、expectile V、优势加权 Gaussian actor、目标网络和训练入口。直接复用训练步骤。上游权重是 `exp(Q−V)*temperature`（并截为 100），不是常见的 `exp(temperature*(Q−V))`；本轮保留此版本的实际公式，不偷换论文实现。 |
| BCQ | `baseline/bcq/bcq.py`、`run/run_bcq.py` | 完整 VAE 行为分布、受限扰动 actor、双 Q、10 候选训练备份和 100 候选推理。该版本 Q 为 101 桶 symlog 分布式回归；不是标准论文的标量 MSE Q。直接复用 step/forward，并作下述紧凑层和数值修复。 |
| TD3+BC | `baseline/td3_bc/td3_bc.py`、`run/run_td3_bc.py` | 有双 Q、噪声平滑、延迟 actor、BC 正则与 101 桶 Q。原 actor objective 错把 Q1 logits 当 Q 数值，见修复记录。直接复用 step，保持其算法流程。 |
| Decision Transformer | `baseline/dt/dt.py`、`baseline/dt/utils.py`、`run/run_decision_transformer.py` | 完整自写因果 Transformer，无 HuggingFace 依赖：3 层、64 维、1 头，以 return/state/action token 预测动作，动作 MSE 训练。直接复用模型、step、get_action；72 片轨迹采样与合法在线历史由本地适配器提供。 |
| CQL | `baseline/cql/cql.py`、`run/run_cql.py` | 额外实际可运行算法。双 Q、Gaussian policy、熵系数学习、logsumexp 保守项，以及本源码的 expectile Bellman loss。直接复用 step，修复 target 初始化。此版本采样 `[-1000,1000]`、重复拼接 data Q 的具体保守项仍保留；不冒称标准 CQL 论文完全一致实现。 |
| MBRL-MOPO / MBRL-COMBOMICRO | `simul_bidding_env/strategy/mbrl_*_bidding_strategy.py` | 仅模型加载/出价 wrapper；指向 `strategy/official_agent/...` 的 TorchScript 与 normalization 文件不存在，保留目录中也没有相应 dynamics/policy 训练实现。本轮未恢复、伪造或用别的模型替代它们，**未列入完成训练算法**。 |

上表 `baseline/` 和 `run/` 均相对 `strategy_train_env/bidding_train_env/` 和 `strategy_train_env/`；原始代码保留原入口和许可证。

## 2. 原始 observation、action、reward

上游神经策略与 `train_data_generator/train_data_generator.py` 使用 16 维状态：剩余时间、剩余预算比例；历史全部/近三片报价、成交最低价、pvalue、转化和竞胜统计；**当前片 pvalue 均值、当前片真实机会数**；近三片/历史机会数。wrapper 能直接收到当前 `pValues` 向量。上述当前值、历史失败机会价值、市场最低成交价和转化指标不属于本赛题接口，不能只删除当前两维而沿用原轨迹。

原轨迹 action 为 `sum(bid)/sum(pvalue)`，wrapper 以 scalar alpha 乘当前 pvalue。Abid 是例外，其二次报价无法用一个合法片内固定 alpha 精确表示。

原轨迹同时保存曝光后的采样转化 `reward` 与曝光 pvalue 合计 `reward_continuous`。IQL/BCQ/TD3+BC/CQL 的入口通常选择连续值并标准化；DT 的 replay buffer 直接读取 `reward`，BC 只模仿动作、不使用奖励优化。上游环境仍有多广告位、曝光、转化与 CPA 相关语义，因此**即使某个脚本选择连续奖励，也不能把它的旧轨迹当作本赛题训练轨迹**。

本轮训练数据必须来自当前 `AuctionEngine` 完整 72 片新回放：只有实际竞胜 pvalue 累加为 reward，没有 CPA 惩罚、曝光或转化抽样。奖励仅除以训练轨迹总回报第 75 分位这一固定正数；这不改变单局最大化累计价值的目标。所有 offline RL 的 `gamma=1`，终止只在第 72 片，零余额仍保留后续片。

## 3. 合法共同接口和尺度

`neural.encode_observation()` 仅接受 `harmony_bidding.agent_api.Observation`，输出 18 维：时间比例、剩余预算比例、初始预算对数、预算进度偏差；上一片 alpha/费用/价值/竞胜；累计价值与竞胜；最近 3 片及 6 片的平均 alpha/费用/价值/竞胜。alpha、价值和次数使用 `log1p`，费用除以自身初始预算。没有机会数、当前 pvalue、当前 mask、对手状态、加载器或环境句柄。模型训练集上的逐维均值/标准差随 checkpoint 冻结。模型共享所有角色，不使用测试角色专属拟合。

默认训练动作坐标 `u=log1p(alpha/action_scale)`，其中 `action_scale` 为训练正 alpha 中位数。部署还原为 `alpha=action_scale*expm1(max(0,u))`。坐标下限保证合法非负输出；溢出或非有限输出会报错，不会静默替换成成功策略。没有 5、几十或其他全局 alpha 上限。

CQL 的完整周期验证暴露了对数动作坐标与指数解码的正反馈失稳风险：其无界策略输出及历史 alpha 特征可相互放大。本地适配器因此增加显式候选配置 `action_transform="linear_scaled"`：训练 `u=alpha/action_scale`，部署 `alpha=action_scale*max(0,u)`。这是线性尺度适配，没有输出上限，也没有修改上游 CQL 网络、保守项或 `step` 目标。该稳定性候选仅在原 CQL 没有完整有效的成对训练 seed 配置时由实验监督器另目录补训；实际是否采用，以候选与选模记录为准。原权重、失败和超时记录保留。

`SUPPORTED_ACTION_TRANSFORMS` 明确列出 `log1p/linear_scaled`。每个算法和超参数配置分别构造动作训练张量，所选变换同时写入 checkpoint 顶层、config 和 metadata；字段矛盾时拒绝加载。历史 checkpoint 缺少该字段时严格使用原 `log1p/expm1` 表达式，其输出不变。DT 和其他既有算法没有切换动作坐标；新增线性候选也不会改变后续对数候选的训练张量。

BCQ 原生 VAE/扰动网络具有自身有限 action support，保留该机制；其坐标 `max_action=max(1,1.25*max(train_u))` 从训练动作确定。它是**BCQ 的模型搜索支持范围**，不是比赛规则，其他算法不受此限制。Online LP 保留上游 `1.5*fallback` 限制，它同样是该策略自身范围；fallback 和 multiplier 由训练/验证选择，并在报告中记录。

DT 所需 return-to-go 在训练中从完整**训练**轨迹奖励反向累计。部署初始 desired return 只来自训练轨迹总回报的第 90 分位，随后仅减去已经得到的自身反馈；不会获取测试局真实未来总回报。DT 过去动作 token 存已执行动作坐标。BCQ 推理有 100 个 VAE 候选的随机性，采用 `(own_seed*73+time_slot) mod (2^63−1)` 独立 Torch RNG；调用顺序不消耗其他策略随机流。

## 4. 必要的模型适配与修复

| 变更 | 原因 | 保留范围 |
|---|---|---|
| 16 → 18 输入，正 alpha 的对数动作坐标 | 删除非法观测；适应转换概率造成的几十以上 alpha 尺度 | 六种模型训练目标、优化器步骤和原网络算子继续使用上游 |
| BC每批 actor 更新从默认3次设为1次 | 让所记录600个训练step对应600次actor更新，便于明确计算预算 | 原actor、MSE与Adam更新步骤不变 |
| BCQ 的 1400/1300/1750 宽度、TD3+BC 的 756 宽度缩为 128 | 降低本地 CPU 训练与百候选推理开销；属于明确的紧凑模型版本 | 层数、激活、VAE、分布式双 Q、扰动、target、soft update、采样候选数不变 |
| BCQ/TD3 `two_hot` 采用 `n_bins−1` 间距及上界安全的 scatter_add | 上游按 `range/n_bins` 会在 vmax 产生 index=101 越界，且与解码 linspace 网格不一致 | 仍是相邻两桶插值、symlog 分布式回归，没有改成普通 MSE |
| TD3 critic.Q1 返回 `two_hot_inv(logits)` | 原函数输出101桶 logits，而 actor 当数值 Q 求均值；与 critic.forward 的数值 Q 含义不一致 | 原 TD3+BC actor loss、BC 项、延迟更新和双 Q 逻辑不变 |
| CQL target 初始复制 online Q | 上游构造器建立独立随机 target，未初始化为 online Q | 原熵训练、保守项、expectile 和 soft update 不变 |
| CQL 可选线性动作尺度候选 | 解除对数坐标和指数解码的正反馈放大；仅适用于显式新配置，旧 checkpoint 原样加载 | 复用原 CQL 网络与 step；无动作上限，alpha 仍只要求有限非负 |
| DT 时间 embedding 73、上下文10、warmup默认50步 | 覆盖72片而非原始48/24/96混杂设置；短预算训练可实际学习 | 原3层因果 Transformer、token顺序、attention mask、动作MSE与梯度裁剪不变 |
| 保存 state_dict + JSON 可表示元数据 | 原保存/加载入口存在 `.pkl/.pt/JIT` 不一致；防止加载旧权重 | 独立实例严格加载本轮训练权重，不依赖原模型文件或 pickle 类对象 |

因此推荐报告名称为“**AuctionNet 源码复用的本地赛题适配版**”；BCQ/TD3 标明 compact/修复，不能表述为官方原样复现。`upstream.py` 所有变更都发生在本地构造器层，源文件不被覆盖。

## 5. 训练与选模责任边界

`train_candidates()` 输入仅为训练 NPZ（states/actions/rewards/next_states/dones/episode_id），拒绝缺片、重复非连续 episode ID、非法动作/奖励/终止值及非有限数组。所有 episode 必须连续72条，最后一条 done。该函数不接受测试市场、不会读取验证成绩、不会自行选最优模型。

默认每算法测试学习率 `3e-4/1e-3`，其余参数见每候选 config。关键可调项包括 IQL expectile/temperature、BCQ phi/support/宽度、TD3 BC权重/宽度、CQL保守项权重、DT上下文/desired return/warmup。配置、训练 seed、数据SHA、数据量、参数量、Torch版本、累计训练时间、50步间隔loss和每个checkpoint的SHA都会保存。实际正式梯度步数和种子由主实验配置决定，不能把默认值或小smoke当正式完成量。

需要补训时，CQL 线性尺度候选使用上游默认学习率 `1e-4`，仍采用相同训练轨迹、两个既定训练 seed、600 步和 200/600 步 checkpoint。它是额外的失败恢复候选，不能隐藏原两组学习率的失败或将其成本从训练资源统计中删除；仍只能用 validation 选择，测试集不参与。

本轮已实际完成该补训：仅使用 `period-7/period-8` 的 288 条完整轨迹、20,736 条转移，输入 NPZ SHA256 为 `63047f205f86f3c6252f31fbeae32338edfe73ae4bd20ebd6cf713a1993d5c92`；训练前后均核验数据摘要。两 seed `20261028/20261029` 各训练600步，2个 CPU线程，总耗时22.72秒，生成4个 checkpoint，保存于 `outputs/baseline_comparison_20261028/models_cql_linear/`。原 `models/candidates.json` 和原48个 checkpoint 的摘要保持不变，完整执行记录为 `models_cql_linear_run.json`。补训目录内 `config_index=0` 只是局部编号，统一实验监督器另行分配候选编号并开展独立验证；这些训练完成记录不表示该配置已经获选或优于旧配置。

统一实验 runner 在独立完整 validation 周期、固定三对手场景和统一角色上评估各候选后冻结 checkpoint，再由 `NeuralAgent(checkpoint=...)` 从磁盘独立加载最终测试。不能用训练loss代替验证竞胜价值，也不能根据最终test回头选择模型。

## 6. 已执行的适配验收

`tests/test_harmony_baselines_neural.py` 在2026-10-28最新实际通过10项测试：保留原7项，即六种算法从同一核心引擎生成的72片稀疏合成轨迹训练，保存并独立加载，然后每种算法两次完成同一核心72片竞价，合法输出且成绩可复现；另外覆盖18维自身历史特征、two-hot上界、TD3标量Q、CQL target初始化、缺片轨迹拒绝。新增3项覆盖 CQL 线性及对数候选各自接收正确动作张量、线性训练/加载/72片闭环、六算法无变换字段的旧 checkpoint 与显式对数 checkpoint 的逐片动作完全一致，以及相互矛盾的变换元数据拒绝加载。最新线性适配验收目录为 `outputs/baseline_comparison_20261028/linear_cql_smoke_20261028`。

该测试只证明训练/加载/核心评测闭环，不是实际 AuctionNet 数据的性能结论。小smoke产物放在 `outputs/baseline_neural_smoke_tmp`，不参与正式候选选模和排名。正式数据量、训练时长、角色覆盖、误差线和结果另以本轮实验目录及主报告为准。
