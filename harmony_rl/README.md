# 72片广告竞价 PPO + LSTM 实验系统

本包在现有唯一 `harmony_bidding.AuctionEngine` 上增加完整轨迹训练、LSTM回报重分配、残差PPO、诊断和冻结评测。既有规则、Baseline代码、训练结果及源数据保持不变。实际结果由评测文件生成，不能把本说明中的计划数量当成完成数量。

规则、十种基线、五场景以及可复用结果的审计见 [AUDIT.md](AUDIT.md)。本包是受限计算预算的研究系统，不预设PPO或LSTM优于已有方法。

## 一键命令

在 `AuctionNet` 目录打开 PowerShell，使用现成虚拟环境，不需要安装系统Python或GPU软件：

```powershell
$rlPython = '.\.venv-baselines\Scripts\python.exe'

# 环境对拍、数学、泄漏、重分配守恒、恢复与最小端到端测试
& $rlPython -X utf8 -m harmony_rl.cli verify

# 小型合成稀疏市场功能检查；不是历史市场实验成绩
& $rlPython -X utf8 -m harmony_rl.cli smoke --variant E1 --output outputs/ppo_smoke_raw
& $rlPython -X utf8 -m harmony_rl.cli smoke --variant E4 --output outputs/ppo_smoke_lstm_aug
& $rlPython -X utf8 -m harmony_rl.cli smoke --variant E5 --output outputs/ppo_smoke_residual

# 单独训练：分别为标准PPO、PPO+LSTM、Residual PPO
& $rlPython -X utf8 -m harmony_rl.cli train --variant E1 --episodes 96 --workers 8 --max-seconds 1800 --output outputs/ppo_raw
& $rlPython -X utf8 -m harmony_rl.cli train --variant E3 --episodes 96 --workers 8 --max-seconds 1800 --output outputs/ppo_lstm
& $rlPython -X utf8 -m harmony_rl.cli train --variant E5 --episodes 96 --workers 8 --max-seconds 1800 --output outputs/ppo_residual

# 同一run恢复：加载last.pt的optimizer、RNG、轨迹buffer和已提交进度
& $rlPython -X utf8 -m harmony_rl.cli train --variant E1 --episodes 192 --workers 8 --max-seconds 1800 --output outputs/ppo_raw --resume

# E1-E6同交互预算训练、验证选模、完整48角色×5场景×2评测seed、报告
& $rlPython -X utf8 -m harmony_rl.experiments --output outputs/ppo_lstm_20261028/pilot --episodes 96 --workers 8 --training-seeds 20261028 --max-seconds 7200 --training-max-seconds 1800 --evaluation-max-seconds 3600

# 独立冻结评测：自动复用经核验的十种Baseline B；不重训Baseline
& $rlPython -X utf8 -m harmony_rl.evaluate --checkpoints outputs/ppo_raw/best.pt outputs/ppo_lstm/best.pt outputs/ppo_residual/best.pt --output outputs/ppo_comparison --workers 8 --max-seconds 3600

# 评测未完成时原命令再次执行，成功的原子单局结果严格核验后跳过
# 若策略集、数据、对手或执行源码变化，必须使用新的输出目录

# 只派生报告，不训练、不选模
& $rlPython -X utf8 -m harmony_rl.report --evaluation outputs/ppo_lstm_20261028/pilot/evaluation --training-root outputs/ppo_lstm_20261028/pilot/training

# 训练市场增强分布审计；不读Test
& $rlPython -X utf8 -m harmony_rl.diagnostics --audit-markets --day period-7 --output outputs/ppo_market_audit

# 一条原市场轨迹、预算有限差分、相同未来市场的双分支反事实
& $rlPython -X utf8 -m harmony_rl.diagnostics --checkpoint outputs/ppo_raw/best.pt --day period-7 --player 0 --scenario Mixed2 --output outputs/ppo_raw_diagnostics
& $rlPython -X utf8 -m harmony_rl.diagnostics --checkpoint outputs/ppo_lstm/best.pt --day period-7 --player 0 --scenario Mixed2 --output outputs/ppo_lstm_diagnostics
```

`cli evaluate/report/diagnose` 也可代理相应模块命令。独立评测支持 `--roles 0 8`、`--scenarios Fixed Mixed2`、`--seeds 20261028`；这类子集在报告中明确标识，不能声称全角色完整实验。

Smoke模型只用于功能验证，不能进入正式历史市场评测或与十种Baseline比较。正式评测会检查Checkpoint来源、训练是否完成、验证选模、配置和模型SHA；不要把smoke输出改名为正式`best.pt`来跳过这些检查。

训练时间限制在完整采样批次之间及验证开始前检查，正在运行的完整轨迹会结束，因此墙钟可能超过阈值一个批次或验证批次。评测达到阈值会终止未完成工作，已提交局保留；缺失局可恢复重跑。`early_stopping_patience`控制验证连续无改进的停止次数。没有无限时长训练默认入口。

需要调整超参数时，创建JSON配置并用 `train --config path.json`。`TrainConfig`是唯一训练配置定义；`config.json`保存最终有效值。`smoke`、`train`和`experiments`均不在Test上选超参数。

## 本轮预先声明的有限预算实验

| 组 | 名称 | 市场数据 | 训练reward | 残差基准 |
|---|---|---|---|---|
| E0 | 十种冻结Baseline B | 已完成的原训练/验证协议 | 原协议 | 不适用 |
| E1 | PPO-Raw | Historical | 原始价值 | 无 |
| E2 | PPO-Aug | 60% Historical /30% Bootstrap /10% Stress | 原始价值 | 无 |
| E3 | PPO-LSTM | Historical | LSTM重分配 | 无 |
| E4 | PPO-LSTM-Aug | 与E2相同采样比例 | LSTM重分配 | 无 |
| E5 | Residual-PPO | Historical，与E1一致 | 原始价值 | 冻结LocalPID B |
| E6 | Residual-PPO-LSTM | Historical，与E5一致 | LSTM重分配 | 冻结LocalPID B |

每个PPO变体请求96条完整训练轨迹，即6,912个环境step、12个8轨迹PPO批次；角色依次轮换，96局覆盖48角色各两次。比例是每episode的采样概率，不强行保证96局恰好达到60/30/10配额。每个变体只有1个训练seed=20261028。所有变体有相同验证周期与选模口径；实际完成数、总请求量、更新次数和耗时以`training_status.json`、`training_history.json`为准。

Train使用period-7和period-8；Validation使用period-21；测试比较使用period-24。基线B中的神经方法有两个训练seed，本轮PPO只使用一个，因此均值可比较，但不能证明训练稳定性或收敛性能。评测seed=20261028/20261029不会替代训练seed重复。

每变体全测试为48角色×5场景×2评测seed=480局；6变体共2,880新测试局。十种Baseline原7,680局完全复用；每个神经Baseline的两个训练seed先合并。每次测试只替换一个广告主，其他47名依冻结场景动态更新。

`experiment_protocol.json`在运行前记录上述设定；若计算预算导致候选未完成训练或尚未验证，该候选不进入测试。若测试覆盖不足，该变体不进入主排名，主表仍保留并标“未完成”。period-24已是历史展示数据，不声明新的盲测；period-25尚无可复用完整结果，不预填成绩。

## 环境、数据与状态

`env.BiddingEnv.reset()`返回冻结`Observation`，`step(alpha)`执行一个时间片并返回`(next_observation, raw_value, terminated, False, own_feedback)`。所有竞价直接调用唯一`AuctionEngine.run_matrix_slot`，没有第二套支付或预算逻辑。每个对手及残差控制器每局独立reset，其update收到该局实际自身竞拍结果。余额耗尽依然执行72片。非法动作在结算前拒绝；异常不会部分提交环境账本和控制器状态。

观察契约复用`harmony_bidding.agent_api`，Actor不能获得环境句柄。输入包含现成18维自身历史编码，以及从训练广告主配置构造的角色和类别one-hot；未知角色/类别各有独立unknown位置。没有当前片请求数、当前/未来pvalue、真实机会数、对手动作、对手身份或场景标签。固定归一化参数、特征名称与排列保存于Checkpoint的`encoder`。它们不在同一rollout和PPO更新之间改变，也不从Test估计。

训练episode始终由真实市场机会和动态策略交互生成。静态CSV不被视为已经具有动作/奖励的离线RL数据。历史`MatrixDay/MatrixSlot`加载器和进程内缓存直接复用，避免每局重读CSV。

`markets.bootstrap_market`以完整请求组矩阵行为单位，联动保留全部参与mask和原联合pvalue；新request_id唯一重建。支持时间片内采样、连续请求块和连续时间块抽样，以及流量倍率/逐片权重。不能独立合成广告主pvalue，也不能把缺行补为有效pvalue=0。

Stress使用原请求组进行前置/后置流量、后半程高价值重排、强/弱竞争、流量加大、紧/松预算等改变；市场source记录生成参数。参与人数、各角色有效机会数、pvalue分位数、竞争共现矩阵、每片请求数、预算/流量比由`distribution_summary`输出。其统计描述不保证分布等价；应结合实际审计图解释增强合理性。

预算来自原始异质角色配置。`randomize_budgets`支持`original`、仅目标`controlled`、全体独立`independent`、共同倍率`common`；默认扰动范围[0.8,1.2]，本轮默认原始预算。流量q倍时区分`traffic_budget_mode=fixed`和`proportional`，后者预算也乘q。每个episode保存原始预算、实际预算和实现倍率。

训练对手从原冻结参数附近进行默认±10%的系数/本地PID增益扰动；正式评测不扰动。增强stress的竞争倍率额外记录。AuctionNetPID沿用现成适配器，不改写为近似标准PID。

## PPO与重分配的实际公式

Actor和Critic分别使用小型两层64维Tanh MLP，共享一套角色条件模型。Actor采样正态latent：

\[
z_t\sim\mathcal N(\mu_\theta(s_t),\sigma_\theta),\qquad \alpha_t=\exp(z_t).
\]

没有人为alpha上限。`decode_action`只对无法表示的浮点溢出/下溢报错，不把安全处理变成策略裁剪。评测采用正态均值的确定性动作`exp(mu)`，不采样。

残差方法改为`alpha_t = alpha_base_t * exp(z_t)`，Actor输出latent修正量，初始化均值为0。baseline输出0时最终动作仍0。保存和重算PPO概率均使用Normal latent坐标；同一状态的固定指数变换Jacobian在新旧比值中抵消。`alpha_log_prob`另行保存给诊断，不能与latent的旧概率混用。

原始reward只有每片竞胜价值`r_t=V_t`。`reward_scale`只允许固定正比例，不添加预算利用奖励、花费惩罚或ROS项。TD/GAE按每条72步轨迹独立计算，终止时bootstrap为0：

\[
\delta_t=r_t+\gamma(1-d_t)V_\psi(s_{t+1})-V_\psi(s_t),\quad
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}.
\]

本轮`gamma=1, lambda=0.95, clip_epsilon=0.2`。Critic回归`A_t+V_old(s_t)`；Actor最小化负的PPO clipped surrogate；优势在当前新rollout批次归一化。默认4个epoch、minibatch256、学习率3e-4、价值系数0.5、熵系数0.005、梯度norm上限0.5、KL目标0.03。KL超过1.5倍目标时提前停止该批次更新。

每条Trajectory保存states、actions、latents、log_probs、values、rewards、spends、next_states、dones以及role/market/opponents/budgets/seed/policy_version。PPO拒绝缺少版本、版本不符、已消费或标记LSTM-replay-only的轨迹；旧轨迹只能进入LSTM监督buffer。

`ReturnPredictor`是单向LSTM，不是双向LSTM。每个位置输入为当前state、latent、当前已实现reward/spend、片后剩余预算、时间比例，最后一步之前的输出不能读取后续张量。全部前缀用相同最终回报G作为监督，均匀权重MSE：

\[
L_\phi=\frac1{72}\sum_t(F_\phi(h_t)-G)^2.
\]

先算`r_tilde_t=F(h_t)-F(h_(t-1))`，初值预测为0；最后一片直接取`G-sum(r_tilde[:71])`，每条轨迹独立验证累计回报守恒。预测按固定reward_scale还原回原始单位后再重分配。

训练顺序是：收集同版本新rollout → 用冻结LSTM版本计算reward → PPO更新 → 将完整轨迹放入独立LSTM buffer → 训练LSTM并保存新版本。第一批没有已训练LSTM，明确用原始reward热身。不存在同一PPO批次中改变reward定义。默认buffer最多256轨迹、LSTM隐藏64、学习率1e-3、每批之后2个epoch。LSTM完全不进入Actor部署推理。

回报守恒只保证完整轨迹总值一致，不证明截断GAE信用分配更好；原始reward本来是密集价值，因此增加LSTM是否有收益必须看冻结测试。前缀MSE/bias、损失与版本都在日志里，不能把预测增量称为动作真实贡献。

## 预算机会成本与反事实

`diagnostics.critic_shadow`将原始预算和当前剩余预算同时增加Delta，保持已实现花费/价值/动作固定；历史remaining_budget也一致增加，随后完整重编码所有派生特征。这定义的是一致的额外资金干预，不是只改某个归一化特征。

\[
\hat\mu_t=\frac{\hat V(s_t;B+\Delta)-\hat V(s_t;B)}{\Delta},\qquad
\widehat{OC}_t=\hat\mu_t C_t.
\]

Raw-PPO可将该值作为原始未来价值敏感性估计；LSTM-PPO Critic学习的是重分配后的剩余return，报告明确标为`redistributed_return_sensitivity`，不能冒称原始未来价值影子价格。两者都不加入训练reward。

环境快照保留未来市场、按请求key决定的平局seed、动态对手内部状态和余额。反事实分别从原预算和追加预算状态继续竞拍，对手按各分支实际结果正常演化。默认片36做两个剩余日分支，不每次训练运行。单角色单市场单seed的结果只能说明一个模拟干预，不能证明真实因果关系或泛化解释能力。

## 模型、恢复和结果文件

| 文件 | 含义 |
|---|---|
| `config.json` / `provenance.json` | 完整有效配置、市场/对手/代码指纹、库和资源信息 |
| `last.pt` | 完整已提交训练边界：Actor/Critic、两optimizer、LSTM、replay、版本、RNG、指标和验证状态 |
| `best.pt` / `selection.json` | 只由period-21验证平均原始V选择的部署模型及SHA |
| `lstm_versions/version_*.pt` | 各次LSTM监督更新后的独立版本 |
| `trajectories/episode_*.npz` | 完整72步轨迹及来源、控制参数、实际成本 |
| `training_history.json` | 原始return、actor/critic loss、KL、clip比例、训练耗时、LSTM前缀误差、守恒误差 |
| `validation_history.json` | 每次完整验证的均值、局数、模型更新位置和耗时 |
| `training_status.json` | 实际完成数、是否验证、停止原因、请求总量、完整轨迹数和有效checkpoint |
| `evaluation/protocol.json` | 冻结策略、市场、原始预算、对手、全部case与源码身份 |
| `evaluation/results/*.json` | 原子提交的单局结果和72片数据；恢复的权威来源 |
| `evaluation/status.json` | 实际完成/缺失/失败、墙钟限制、CSV/协议SHA |
| `evaluation/baseline_reuse.json` | 十种Baseline复用核验链，原资产不修改 |
| `evaluation/report/` | 全十种Baseline与PPO表、角色分布、seed统计、配对提升、图及research_report.md |

恢复加载Python、NumPy和Torch RNG、optimizer、LSTM replay及版本；每个episode的随机seed由训练seed和episode_index确定，worker调度顺序不改变采样身份。中断时丢弃尚未完成提交的PPO更新，以`last.pt`为准；不是把半条轨迹接进当前on-policy批次。

恢复只允许修改总episode上限、此次时间限制和worker数。其他配置或输入/执行源码指纹变化都会报错，要求新run目录。评测同理：既有成功结果只有与完整冻结任务身份相同才会复用。

报告严格区分“实现已通过功能验证”和“实验已完整运行”。少量训练、单训练seed、已展示的比较测试、未完成的独立测试、尚未收敛都限制结论。最终哪种方法适合正式比赛，应由实际结果和可用训练时间决定。

消融表另外核验`last.pt`中的已提交完整episode数、environment step、训练seed、有效配置、日志计数和所评测模型是否确为`selection.json`选中的模型。不同变体的完整episode/环境step不同时，单元标为“训练预算不匹配”，不以成绩差冒称消融收益。请求总量仍单独记录：增强可改变每episode请求数，因此相同episode预算不代表相同请求处理成本。

冻结LSTM的额外前缀Validation检查可单独执行：

```powershell
& $rlPython -X utf8 -m harmony_rl.prefix_validation --checkpoint outputs/ppo_lstm_20261028/pilot/training/E3_seed20261028/best.pt --output outputs/ppo_lstm_20261028/diagnostics/E3 --workers 2 --max-seconds 300
```

此命令只评估冻结预测器，不更新Actor或LSTM。Actor曾由同一Validation选模，结果不能冒充完全独立泛化；报告分别列早期1、中期36、终期72片的MSE和bias，并核对完整30局覆盖。训练/评测完成后重新生成报告即可纳入实际诊断文件。
