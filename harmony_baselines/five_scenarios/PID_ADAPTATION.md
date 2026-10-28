# AuctionNetPID 原始机制与72片适配审计

本文件记录新增控制器的机制与工程边界，不包含正式调参、训练或测试成绩。`AuctionNetPID` 指 AuctionNet 开源仓库自带的 `PidBiddingStrategy`，不代表赛事组织方尚未公布的 PID，也不是本项目 `LocalPIDAgent`。它无需梯度训练；后续实验只校准初始动作并搜索控制参数。

## 1. 固定上游来源与真实控制机制

已核对本地 Git HEAD 与 `git show`，版本均为 `c20de631a40e24669d1d6a8e2acefd8f492f172d`。上游文件未修改，原文件 SHA256 为 `4a007726b6771281bcf0058b479dad97fad90e59b39f824dee966f9c706cfb46`。源码：[PidBiddingStrategy](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/simul_bidding_env/strategy/pid_bidding_strategy.py:5)。

该策略是**基于上一片预算消耗与剩余预算的双阈值乘法控制器**，没有 P、I、D 三项，没有累计误差积分，也没有误差差分。名称不能作为“标准 PID”的证据。与之区分，本项目旧 `LocalPIDAgent` 的 `update` 确实实现累计预算曲线误差、积分和差分：[LocalPIDAgent.update](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_bidding/agents.py:59)。本轮保留旧 LocalPID 源码。

上游初始动作是常数 `base_action=15`，并非由预算、CPA 或行业类别计算；`cpa`、`category` 被保存，但没有进入本策略控制公式。默认期望权重为48个1。出处：[构造器](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/simul_bidding_env/strategy/pid_bidding_strategy.py:6)、[首片初始化](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/simul_bidding_env/strategy/pid_bidding_strategy.py:23)。

记当前片为 `t`（从0计数），周期片数为 `T`，片前剩余预算为 `B_t`，上一片花费为 `c_(t-1)=B_(t-1)-B_t`，固定期望权重为 `w_0,...,w_(T-1)`。对于正余额：

```text
t=0: alpha_0 = initial_alpha
t>0:
    r_up   = c_(t-1) * sum(w[t:]) / w[t-1] / B_t
    r_down = c_(t-1) * (T-t) / B_t
    if r_up < lower_threshold:
        alpha_t = alpha_(t-1) * increase_factor
    elif r_down > upper_threshold:
        alpha_t = alpha_(t-1) * decrease_factor
    else:
        alpha_t = alpha_(t-1)
```

上游默认 `T=48, lower_threshold=0.7, upper_threshold=1.1, increase_factor=1.2, decrease_factor=0.7`。比较是严格的 `<` 和 `>`；达到阈值本身不会触发对应分支。**上调分支使用期望权重，下调分支只用 `48-t`；这是原版不对称设计，本地没有改成共同权重或累计消耗目标。** 如果非均匀权重使两种判定同时满足，`if/elif` 优先上调。出处：[PidBiddingStrategy.bidding](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/simul_bidding_env/strategy/pid_bidding_strategy.py:26)。

对均匀权重，上下两种比率相同，可以理解为“按上一片速度花完余下片数，预计花费占现有余额的比例”。低于0.7就提高出价系数，高于1.1就降低。该解释不意味着策略准确预测了未来流量；它只按最近一片消耗作启发式外推。上游没有全局或内部 alpha 上界。

## 2. 信息与接口边界

上游 `bidding` 接收当前 `pValues`、不确定度与多种历史数组，但控制系数本身只用时钟、自己余额和固定权重。`pValues` 仅用于末尾 `bids=alpha*pValues`；其他传入数组未在这个方法中读取。当前赛题中，alpha 产生与逐请求乘价分离：[原乘价行](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/simul_bidding_env/strategy/pid_bidding_strategy.py:32)。

新 Agent 仅接收项目已有 [OwnConfig / PublicConfig / OwnFeedback / Observation](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_bidding/agent_api.py:14)。`bidding` 读取自身配置、时钟和自身余额；上一片花费由 `update` 收到的连续自身余额差得到。没有当前请求数、当前 pvalue、对手参数、市场矩阵、加载器或未来数据句柄。与成本反馈的浮点一致性检查不改变控制输入。[AuctionNetPIDAgent.bidding](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/auctionnet_pid.py:87)、[update](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/auctionnet_pid.py:119)。

`expected_ratios` 是在比赛前冻结的配置先验。公式访问其未来下标只是在访问这条固定配置，不能向其中填入本局当前或未来的真实机会量。默认72个1，不依赖市场。若未来从训练数据估计非均匀先验，必须在验证/测试前冻结并记录来源；本适配实现没有进行这种估计。

## 3. 保留、必要修改与数值边界

| 项目 | 上游原实现 | 本地适配 | 影响与边界 |
|---|---|---|---|
| 控制公式 | 两阈值、乘法增减、上调/下调不对称 | 全部保留，常数改成显式参数 | 不是重新实现为标准 P/I/D |
| 时间片 | 权重长度48，下调写死 `48-t` | `PublicConfig.total_slots` 决定长度，正式72 | 48片仅用于上游差分测试，不修改正式比赛72片规则 |
| 初始动作 | `base_action=15` | `initial_alpha=15.0`；可配置 | 默认保留15；调优值必须另存冻结参数 |
| 输出 | 当前 pValues 的逐请求报价向量 | 每片一个有限非负标量 alpha | 乘价、参与掩码、预算重拍均由唯一 AuctionEngine 处理 |
| reset | 只恢复 remaining_budget | 重置 alpha、上一片花费、余额、反馈计数、最近决策片 | 修复重复 episode 残留状态；上游首片会重置alpha，但未恢复 `last_remaining_budget` |
| 零余额 | 比率分母为0，无明确保护 | 输出0；继续接收余下全部片反馈 | 避免除零，符合默认零余额不参与；没有提前终止周期 |
| 重复调用 | 同片再次 bidding 可能重复乘系数 | 同一片重复读取返回同一alpha | 一片一次决策的幂等保护，不改变合法调用序列 |
| 配置合法性 | 无主动校验 | 初值非负有限；阈值有序；增幅≥1；降幅∈(0,1]；权重正有限且长度匹配 | 非法配置直接失败；允许初值0，无任意alpha上限 |
| 权重累计 | NumPy后缀求和 | `math.fsum` 预计算后缀和 | 减少累计舍入；普通默认/整数权重分支与上游一致，非整数近阈值处不承诺所有浮点运算逐位相同 |
| 比率中间溢出/下溢 | 普通float算式，可能产生Inf/0 | 普通范围沿原运算次序；极端范围用精确有理数比较同一公式 | 仅保持数学判定，不限制alpha，不引入新目标 |
| 动作乘法溢出 | 可能返回非有限动作 | 明确抛出ValueError，且不提交该片策略状态 | 不截断、不回退，不把失败伪装成合法0动作 |
| 角色参数 | 每对象自身 `base_action` | 可选 `player_alphas` 字典，构造时复制 | 供冻结对手校准；目标共享策略不传此表 |

`expected_ratios` 每个元素均须严格大于0，以避免上调分母为0。配置权重总和若不可表示为有限float直接拒绝。初始预算0在所有72片输出0。其它预算/反馈异常（余额增加、成本不一致、乱序、NaN、Inf）显式报错；不会进行静默修复。`alpha*pvalue` 的报价乘积溢出仍由原核心校验，适配器不会修改市场值或竞价规则。

实现未直接调用上游 `bidding`，因为其48常量、硬编码阈值和向量签名不满足新接口；这里只把已审计的控制公式翻译到标量接口，并用原类作为差分oracle。原源码、原模型和历史结果均不修改。[新构造器](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/auctionnet_pid.py:39)、[数值比率辅助函数](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/auctionnet_pid.py:14)、[reset](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/auctionnet_pid.py:65)。

## 4. 稳定 API 与接入方式

```python
AuctionNetPIDAgent(
    initial_alpha=15.0,
    lower_threshold=0.7,
    upper_threshold=1.1,
    increase_factor=1.2,
    decrease_factor=0.7,
    expected_ratios=None,  # 正式 reset 时生成72个1；也可显式传72个正有限值
    player_alphas=None,    # 可选 {"广告主ID": 初始alpha}；目标共享参数时不传
)
```

通过现有 `make_agent` 的 custom 分支加载，无需改动旧模块：

```json
{
  "kind": "custom",
  "class_path": "harmony_baselines.five_scenarios.auctionnet_pid:AuctionNetPIDAgent",
  "kwargs": {"initial_alpha": 15.0}
}
```

每局必须 `reset(OwnConfig(...), PublicConfig(72))`，每片按 `bidding(observation)` → AuctionEngine 结算 → `update(own_feedback)` 调用。每个广告主使用独立实例。返回标准Python float；不是每请求重新计算动作。接口参数和配置表不包含训练/验证/测试周期ID或对手场景标签。

## 5. 已执行验证与尚未完成的实验

本实现的针对性测试为 [tests/test_auctionnet_pid.py](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/tests/test_auctionnet_pid.py:1)，在项目根目录执行：

```powershell
.\.venv-baselines\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests/test_auctionnet_pid.py
```

测试覆盖：16组48片逐步上游差分（均匀/非均匀整数权重、8条合法预算路径）；72片上调/维持/下调及严格阈值；非均匀权重时的 `if/elif` 优先级；可配置倍率；零余额完整周期；重复reset与玩家参数切换；配置复制与独立状态；非法数值/乱序拒绝；极大有限动作无cap；动作溢出原子失败；比率中间溢出的数学判定；只读允许信息的受限观察对象；两次使用唯一AuctionEngine的72片集成，改变第36片后的未来市场，前37次决策保持完全一致（第36片决策先于其反馈）。

2026-10-28 实际执行结果为 **49 passed in 0.24s**。随后只读核对旧 `selected_frozen.json` 中35个冻结源码SHA，全部一致；这次新增适配没有改动任何旧冻结模块。

这些是实现验证，不是正式5场景成绩，也不证明任意阈值/倍率都能稳定或高效竞价。正式对手校准、3场景版/5场景版目标调优、冻结参数和两组测试结果由新实验流程独立记录；本实现任务没有运行正式参数搜索或读取测试市场调参。
