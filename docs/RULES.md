# 赛题规则、来源与当前本地协议

当前执行规则为 `harmony-A-B-local-v2`，默认假设配置为 `local-unified-v2`。本文件说明本地开发环境，不宣称取得主办方 SDK、官方 PID 或隐藏测试验收。

## 来源与优先级

| 来源 | 内容与使用边界 |
|---|---|
| A：用户提供题面及指定的《广告智能出价Agent赛题整理.docx》 | 72片、标量动作、单广告位二价、预算与价值目标、生命周期、角色轮换和评分；整理稿原文含“上限由配置给定”。原DOCX保留不改。 |
| B：用户转述的主办方答疑 | 决策时当前片机会数/预估价值不可见，只看自身过去反馈；value为竞胜机会预估价值之和。 |
| B2：用户补充及《auto-bidding-agent-doc.docx》记录 | 同request是同一次拍卖；只有存在该request/player记录者参与。缺行是不参与，真实pvalue=0仍是存在的记录；不同广告主机会数可不同。 |
| C：2026-10-22用户提供六字段说明 | index、time_step、request_id、player_id、player_type、pvalue；类型固定、类别无序。日期、预算、片号起点、请求作用域和到达顺序由外部配置明确。 |
| D：2026-10-27用户指定的重构要求 | 活动本地环境无全局alpha上限；内部补零+Participation Mask；不能付二价则排除并重拍；同价等概率且可复现；官方策略不可用时明确失败；统一核心、轨迹、评测与性能入口。 |

D是本次明确采用的**本地实现约定**。其中无全局alpha上限覆盖旧实现的限幅选择，不可反写成“主办方已确认无上限”。预算细节、金额/零余额、官方随机算法、正式接口和官方PID仍待规范确认。上游AuctionNet默认机制不是本赛题规则来源。具体假设见 [ASSUMPTIONS.md](ASSUMPTIONS.md)。

## 当前执行规则

1. N个广告主各有独立预算；正式一天72片，每片20分钟。内部 `0..71` 对应题面 `1..72`，tiny可缩短。每局一支队伍只代理一个指定角色。
2. 每片所有Agent先基于片前自身状态输出一次alpha，再统一竞价，片末各自update。alpha是非负有限实数，可大于1，无人为全局上限，0表示不出有效报价。拒绝bool、字符串、数组、负值、NaN、Inf；环境不裁到5、10或其他值。极大alpha的报价乘积非有限时明确失败。策略自身可以限制搜索范围，但这不是比赛规则。
3. 平台计算 `bid = alpha * 该广告主的预估价值`。策略不返回逐请求数组、不挑机会、同片不改alpha。
4. 内部每个请求保存value向量及同列顺序的bool `participation_mask`。缺行对应value=0、mask=false；真实零估值对应value=0、mask=true。只有mask=true且有效报价为正、符合预算政策者进入候选。源坏值不能当作缺席。
5. 单广告位二价：最高有效报价者获胜，支付其余候选最高有效报价；败者不扣费；没有候选则无人胜；只剩一个有效候选支付0。多个最高同价者采用本地等概率伪随机选择。
6. 片内严格按original_order即时结算。候选赢家付不起**实际二价**时只从本请求排除，再选赢家、重算价格；排除不影响下一请求资格。原始报价超过余额不构成自动无效，不截断报价、不随机删请求、不事后修补超支。
7. 目标只累计竞胜机会的本广告主预估价值。reward=value，不减费用、不采样点击/转化/曝光、不加ROI/ROS/CPA/CPC约束。费用允许大于单次价值，只须满足预算。
8. 空片、零预算、预算耗尽仍保留完整72片调用和反馈。done只在日终为真。每局重新创建策略并reset，不沿用上一局反馈、在线参数、积分、历史或随机状态；封存模型不写回。
9. 引擎在工作账本上顺序处理整片，验证成功才提交权威账本。非法动作/请求/报价溢出不会留下部分预算或累计价值更新。此事务不负责回滚任意自定义Python代码的外部副作用，也不是安全沙箱。

## 本地Agent接口与信息边界

```python
reset(own_config: OwnConfig, public_config: PublicConfig) -> None
bidding(observation: Observation) -> float
update(own_feedback: OwnFeedback) -> None
```

这是本地接口，未来官方SDK只能通过薄适配接入，不声称以上是已知官方签名。

- OwnConfig：自身advertiser_id、advertiser_type、initial_budget、私有random_seed。
- PublicConfig：total_slots。无alpha_max。
- Observation：own_config、time_slot、total_slots、remaining_budget、已完成片的自身history。终态next_observation.time_slot等于total_slots，不再调用bidding。
- OwnFeedback：time_slot、自己的alpha、wins、cost、value、remaining_budget、cumulative_cost、cumulative_value。

Observation和普通actor轨迹不含当前/未来真实请求数、价值/分布、Mask、对手配置/动作/结果、市场价、最低竞胜价、数据加载器或环境句柄。过去片真实机会数也不是默认反馈，不能用来计算真实胜率。Mask列和产生的真实参与机会数只属于评测器诊断/独立离线资料。预测若将来加入策略，必须来自合法训练数据和自身历史并标明预测。

冻结对象、tuple和复制防止意外共享状态；独立进程用于生命周期隔离。它们不是操作系统级恶意代码安全沙箱。

## 官方策略与本地参考

`OfficialBiddingStrategy` 是预留本地适配契约，当前 `available=False`。`kind: official` 或官方reference请求明确报 `OfficialStrategyUnavailable`；不能静默替换成LocalPID。

ZeroAgent、ConstantAlphaAgent、LocalPIDAgent是活动基础策略。LocalPID只根据本局自身实际消耗与目标进度在片后调整alpha，不读取当前/未来流量。它不是主办方官方PID，也不是上游AuctionNet原PID。

本地评测先独立运行同日、同角色、同预算/规则/随机条件的全LocalPID参考；所有替换局的PID仍根据各自当局反馈重新运行。single每队代理d个角色；multi让d队右旋d局，d=3为ABC→CAB→BCA。

```text
ratio(q,m,k) = V(q,m,k) / V_local_pid_reference(m,k)
G_m(q) = mean_k ratio(q,m,k)
local_relative_score_single(q) = 40 * G_single(q) / max_Q_local G_single
local_relative_score_multi(q)  = 15 * G_multi(q)  / max_Q_local G_multi
```

公式沿用题面，但当前参考策略及结果性质明确为本地实验，**不是正式比赛G的复现**。先算逐局比再等权平均：120/100和900/1000得到G=1.05；不能用1020/1100。最高本地G=1.20时single相对效果分为35。

全体有效G为零时相对项为零。PID参考零时严格未定义；诊断模式用null与诊断，不加epsilon、不把0/0当1、不跳局。异常/缺失局不能删除后平均。未知完整官方Q与其余45分细则不补造，不输出完整百分制总分。

## 性能与可复现性

性能针对实际指定策略测量。分别报告bidding/update P99、同一Agent同片配对和的P99、两个P99之和，初始化单列；模拟竞价耗时不算Agent函数耗时。进程RSS标为process_total_rss，不能称单Agent内存。官方200ms/4GB统计口径仍未明确，样本不足的测量仅作诊断。

每次运行携带run_id、输入/生效配置与数据/代码/策略指纹、rules_version、assumption_profile及本地参考标记。业务复现排除时间戳/耗时。旧v1结果保留旧版本，不改标签冒充v2验收。运行和错误状态见 [EVALUATION.md](EVALUATION.md)。
