# PPO 实现前的基线与规则审计

审计时间：2026-10-28。本文件对应新 `harmony_rl` 实验，不改写原报告、Checkpoint、源市场文件或冻结执行代码。

## 已核实的十种 Baseline

| 算法 | 类型 | 训练/调优 | 冻结资产 |
|---|---|---|---|
| FixedAlpha | 固定系数 | 验证集搜索 | 全角色共享 alpha 和 multiplier |
| LocalPID | 自身预算反馈控制 | 验证集搜索 | ScaledPIDAgent 初值、P/I/D、倍率 |
| OnlineLP | 经验成本响应表 | 训练集拟合、验证倍率 | online_lp_table.csv 和共享参数 |
| AuctionNetPID | 上游预算阈值乘法控制 | 训练/验证集调优 | AuctionNetPIDAgent 适配器及阈值、倍率 |
| BC | 监督模仿 | 已训练和验证选模 | 两训练 seed Checkpoint |
| IQL | 离线强化学习 | 已训练和验证选模 | 两训练 seed Checkpoint |
| BCQ | 离线强化学习 | 已训练和验证选模 | 两训练 seed Checkpoint |
| TD3BC | 离线强化学习 | 已训练和验证选模 | 两训练 seed Checkpoint |
| CQL | 离线强化学习 | 已训练和验证选模 | 两训练 seed Checkpoint |
| DT | 轨迹 Transformer | 已训练和验证选模 | 两训练 seed Checkpoint |

最新五场景实验目录为 `outputs/baseline_five_scenarios_20261028`。采用 B 组五场景联合验证选模作为主要基线。
`experiment_B/selected_frozen.json` 包含全部策略规格、模型/表 SHA256、原执行源码 SHA256、对手文件 SHA256、协议 SHA256 和 policy_ids。

`experiment_B/episode_results.status.json` 明确记录 `status=completed, valid=true`，7,680/7,680 局，其中7,200局新增、480局严格相同策略复用；没有缺失。
`episode_results.csv` 的 SHA256 为 `428b4cf6e1987a2496b35aaf03d71c49704562a7125e0370f87ad431822859a6`。
`slots.csv` 的 SHA256 为 `866798a00dc8cd17cc78ffa7cba23a6e3e13f83401c6c6d02942bc056dcde240`。

实际执行 `harmony_rl.evaluate.audit_baseline()` 通过：不仅检查上述封存散列，还检查全部16个策略/训练seed规格的资产身份、所有冻结执行源码、完整笛卡尔覆盖、原始逐角色预算、当前市场指纹、对手分配与规则。复用时重新执行相同审计。不得仅因为CSV存在就宣称可比。

旧顶层三场景报告只有9种算法；这是旧实验口径。新实验增加上游 AuctionNetPID 的本地比赛接口适配，不是另一套拍卖规则。

## 数据与对手协议

- Train：period-7、period-8；Validation：period-21；比较 Test：period-24。
- period-24 已在先前报告中展示，属于可比性测试，不能称为新的盲测。
- 原计划独立 period-25 目录尚无完整 `episode_results.status.json`，本任务不把它推断为完成，也不把未封存成绩加入排名。
- 全部48个广告主；原始预算按角色保留，为2,000–4,850，绝非统一预算。
- 五场景：Fixed、LocalPID、Mixed、AuctionNetPID、Mixed2。
- Fixed/LocalPID 对手参数保持原冻结版本；Mixed 为23 Fixed +24 LocalPID；Mixed2 同一分配种子下保留同23个 Fixed 成员，其余24个改为 AuctionNetPID。
- 评测 seed 为20261028、20261029。神经 Baseline 使用同两训练seed；经典策略训练seed记0。
- 混合分配取决于评测seed、day_id和目标角色，不能依赖待评测算法或其训练seed。

## 拍卖和观察边界

唯一执行实现为 `harmony_bidding.auction.AuctionEngine.run_matrix_slot`，规则 `harmony-A-B-local-v2`，本地假设配置 `local-unified-v2`。
每片先得到全部广告主一次标量alpha，再依原始请求顺序结算单广告位二价；不可支付实际二价的候选仅在当前请求剔除重选。稀疏参与mask决定是否参加，不能把缺行当有效零值。alpha非负有限，无比赛全局上限；溢出显式报错而不裁剪。

Agent 只接收 OwnConfig、当前时钟、已完成片自身反馈。请求矩阵、真实参与机会数、对手策略及市场场景标签不进入Actor。比赛无ROS约束，reward只取自身竞胜pvalue。资金耗尽仍执行完整72片。

金额6位精度、零余额不参加、原生48→72片中点映射仍是项目既有本地假设；本任务不能将其表述为已获独立官方确认。

## 性能与工作量

历史完整 B 组逐局记录显示平均单局5.155秒、最慢9.528秒，包含当时并发竞争，不能当作新机器的独立严格基准。
每PPO变体×2训练seed×48角色×5场景×2评测seed=960局；6变体共5,760局。
以历史逐局均值和8worker作粗估约62分钟纯评测，实际由新执行记录为准。若新PPO推理成本更低可更快；不以猜测代替记录。
新评测提供明确wall-clock预算、spawn进程池、内存缓存、每局原子提交及完整性校验；到期未完成单元保持缺失并可恢复，不按0分填补。

## 复用、汇总与输出

新评测调用原五场景 `run_episode`，只通过标准Agent规格挂接 `harmony_rl.policy:PPOAgent`。
成功局的源市场、预算、模型资产、执行代码、对手及任务身份全部进入冻结协议和单局记录。
`report.py` 将不完整PPO变体排除出排名，同时在主表明确列出全部十种基线及六个PPO变体；没有结果的单元显示“未完成”。
主指标为先平均角色内训练seed、评测seed和日期，再等权平均角色的V；R与旧B组一致，为每角色均值除以对应 LocalPID 角色均值。零分母不加epsilon。
配对提升先合并训练seed，然后按相同场景、日期、角色、评测seed配对；按角色百分比均值和整体均值提升分别说明，不互相冒充。
