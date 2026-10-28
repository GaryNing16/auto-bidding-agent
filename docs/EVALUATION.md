# 统一核心的本地评测与运行记录

所有活动运行路径都通过 `runner.run_episode → AuctionEngine.run_slot → BudgetPolicy`。Agent只实现reset/bidding/update；单、多Agent、轨迹导出和benchmark不另写一套拍卖规则。原上游AuctionNet入口保留用于上游用途，不是本赛题另一个评测器。

## 策略与参考场景

`evaluation.roles`指定测试角色，`evaluation.teams`保存队名到封存构造规格的映射。支持zero、constant、pid、custom；`kind: official`当前明确抛出OfficialStrategyUnavailable，不回退到PID。

`evaluation.reference_strategy: local_pid`是当前可用参考。每个完整日期独立运行一次全LocalPID，再逐局运行替换；数据、角色预算、执行规则及请求平局条件匹配。PID依据**各自当前局**反馈重新决策，不复用参考局的历史报价/市场价。LocalPID不等于官方PID，也不等于上游AuctionNet原PID。

队伍kind=pid的未指定参数继承config.pid；pid块不能覆盖kind。普通run/export可使用：

```yaml
run:
  assignments: {advertiser_1: constant}
  teams:
    constant: {kind: constant, alpha: 1.2}
```

未分配角色使用LocalPID；没有assignments时为全LocalPID。`evaluation`与`run`分开配置，避免把普通演示的角色映射误当正式轮换。

single：每队每日分别代理d个角色，每队d局。multi：本组必须恰有d队，角色顺序固定、队伍右旋d局；三队为ABC→CAB→BCA。支持d=1及d=N，合法范围由真实角色集合检查。

## 评分与无效情况

每局分母是独立全LocalPID参考中**相同日期、相同角色**的价值，不是当前局某个对手价值。先计算每局ratio再等权平均：

```text
G = mean(value / pid_reference_value)
local_relative_score = weight * G / max(G over valid local teams)
weight = 40 for single; 15 for multi
```

120/100和900/1000的G为1.05，不是1020/1100。最高G=1.20时single相对分35。当前G、reference及relative_score均标为本地实验；官方PID未提供，不能称正式比赛最终G复现，不编造未知45分和百分制总分。

所有有效G为零时相对分为零。零参考严格报UndefinedReferenceError；`diagnostic_undefined: true`保留该局null与诊断，并使受影响G及整个比较集合相对分为空。异常、缺失、重复日期或不完整对局不删除后平均。空日期生成器也失败。

## 每局重置和模型封存

默认 `evaluation.isolation: process` 使用spawn新解释器，每局重读封存构造规格、新实例/reset，重置预算、历史、PID积分、在线参数和普通进程级缓存。`in_process`只供可信调试，无法清除所有模块级状态。

`sealed_model_files`明确列出权重/统计文件；自定义入口模块和声明文件纳入SHA256。评测开始与每局前后核对，不允许更新写回封存模型。未声明任意外部资源不保证发现；进程/哈希不是恶意代码沙箱。在线学习只能发生于当前局允许的自身反馈。

按日期生成器消费，只保留当前日期的大矩阵和必要摘要；固定模型可共享只读文件，不共享可变在线状态。

## 每次运行独立run_id

CLI输出根为 `output_dir/<command>/<run_id>/`；`--output`替换output_dir，仍追加command/run_id。stdout JSON给出实际run_id和report_directory。旧成功结果保留为旧run_id，不能充当本次结果。

每次status.json初始为running、valid=false；成功完成才为completed，且只有valid=true可消费为有效成绩。失败写failed、valid=false和ERROR.json。评测失败另写evaluation_error.json/无G的无效evaluation.json；已经完成的局日志可保留用于诊断，不构成可用总分。异常中断留下running而非成功。

配置解析失败时无法信任配置中的output_dir：错误记录写 `--output/<command>/<run_id>`，未给--output则写 `outputs/local/<command>/<run_id>`。正式聚合应先调用 `runner.read_completed_run(directory)`或执行等价检查，核对status、valid、目录名与run_id，而不是只看某个JSON存在。

局级目录含episode.json、slots.csv、timing.csv；结果保存当前执行规则/假设、数据与源格式指纹、配置/代码/策略及封存资产。配置可能补入导入溯源等生效字段；requested_config_path/sha记录请求文件，config_fingerprint记录同次实际生效配置。不得把请求文件摘要与生效配置摘要混为同一概念。

完整N角色报告、真实参与机会量和PRIVILEGED_DEBUG属于评测器。Agent与普通actor轨迹不因此获得对手/Mask/真实流量。

## 通用合法轨迹

`export-trajectories`复用run的策略配置。轨迹版本 `harmony-trajectory-v2`，一条对应一个广告主一个时间片，核心字段为：

```text
observation_t, alpha_t, reward_t, cost_t, next_observation, done,
episode_id, advertiser_id, day_id, run_id,
rules_version, assumption_profile, trajectory_version, money_decimals
```

reward是未减费用的本片价值，done只在日终。Reader严格校验所有字段/嵌套字段白名单、标量类型、身份、历史长度/时序、余额/累计账本、当前结果一致性和完整日终；拒绝额外current_opportunity_count、Mask、对手字段及字符串done。

```python
from harmony_bidding.trajectory import read_trajectories, validate_trajectory_file
count = validate_trajectory_file(path)  # 消费到EOF，验证所有流完整终止
for transition in read_trajectories(path):
    use(transition)
```

`read_trajectories`需完整消费到EOF才验证全流终止。导出以临时文件验证全部流后原子发布，失败不发布貌似完整轨迹。旧七维编码、统计拟合和ReplayBuffer桥已移除；未来算法基于合法Observation自行定义特征，不沿用旧统计/权重。

## 指定目标的benchmark

```yaml
benchmark:
  target_role: advertiser_1
  agent: {kind: constant, alpha: 40}
  warmup_episodes: 2
  episodes: 20
  min_samples: 1000
```

目标策略实际构造并运行；其它角色为动态LocalPID。仅目标角色的bidding/update进入aggregate/by_advertiser，其它角色只列opponents_diagnostic。上述72片20测量局提供1440对调用，预热不入调用分位数。换成custom时填相同class_path/kwargs及封存模型列表，不能拿LocalPID的旧数据代替。

使用perf_counter_ns分别计bidding/update，报告nearest-rank的各P99、同角色同片配对和P99、两个P99之和；构造/reset初始化另列。竞价、数据复制和序列化不计入函数时间。记录真实策略、硬件、Python/依赖、线程、样本量及范围。

RSS为整个测试进程的process_total_rss，2ms采样并可记录Windows进程生命周期peak_wset；包含对手、数据、引擎、预热和解释器，不能可靠分离单Agent内存，采样也可能漏短峰。样本量门槛只用于本地诊断，200ms/4GB是否满足官方口径仍待规范，不能把两个函数各自分配200ms。
