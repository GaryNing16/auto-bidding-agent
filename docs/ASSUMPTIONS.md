# 当前本地约定与待官方确认事项

执行规则：`harmony-A-B-local-v2`；默认 `assumption_profile: local-unified-v2`。这些标识随报告与轨迹保存。来源区分见 [RULES.md](RULES.md)。2026-10-27用户指定的本地规则已经可直接实施，不需要再次等待同一选择；这不等于已取得主办方可执行规范。

保留H编号以解释旧报告。v1旧内容在清理归档快照中，不能用本表覆盖其历史结论。

| ID | 当前本地约定/官方未确认点 | 影响与替换位置 |
|---|---|---|
| H-001 | 用户D指定按真实参与和正报价确定候选，付不起实际二价时仅本请求排除重拍；原报价超过余额允许，单人付款0。当前实现先按H-003量化再判定正报价，这是额外的本地精度约定，不是D给定细则。 | `rules.budget_policy=exclude_unaffordable_reauction`、BudgetPolicy。官方预算保护分支仍未确认；变更会改变竞争/价值。 |
| H-002 | 默认零余额不参与；可通过 `rules.zero_balance_can_bid=true`允许其争取免费胜出。 | 两种模式参考分母不能混用；不由“单人免费”推导零余额必能参与。 |
| H-003 | 默认 `money_decimals=6`，预算必须精确可表示；报价使用十进制ROUND_HALF_UP量化、整数比较和扣费；价值不按货币位数舍入。输入/输出数值要求有限，报价乘积非有限时报错。 | `auction.Money`及数据校验。很小正报价可能量化为0、近似报价可能形成平局；官方金额单位/精度未知。不是人为alpha上限。 |
| H-004 | 同价：将候选ID排序，SHA256摘要 `[tie_seed, day_id, time_slot, opportunity_id, round_index, candidates]` 派生独立请求/重拍随机源；`random.Random(seed).randrange(n)`均匀选索引，使用拒绝采样避免取模偏差。 | `auction.choose_request_tie`、`rules.tie_seed`。ID只建立稳定候选顺序，不直接决定赢家；流量/探索/全局RNG调用不改变此请求。官方随机协议未知；复现需匹配执行规则与Python环境，Python版本可核对环境/benchmark记录。 |
| H-005 | 流量、平局、策略探索随机源独立；策略种子由agent_seed/day/role稳定派生，每局重置，同角色条件匹配。 | `synthetic.seed`、`agent_seed`、`rules.tie_seed`。不依赖未重置的全局随机状态；输入/角色轮换顺序不改变随机匹配。 |
| H-006 | 用户D指定活动环境无全局alpha上限，覆盖旧v1示例alpha_max=5；N、d、预算、数据规模仍是配置示例。 | Rules/PublicConfig/Observation均无alpha_max，旧 `rules.alpha_max`明确报迁移错误。策略可自行限搜索范围，但不能改全局合法性。官方最终动作范围仍待SDK确认。 |
| H-007 | 六字段已知，日期/预算/真实顺序/起点/ID范围仍显式指定；内部三表是本地格式。按完整日期划分train/validation/local_holdout。 | `organizer_import`、数据manifest。不得随机拆同日、把local_holdout称官方隐藏集；未来训练统计只拟合train。 |
| H-008 | 正式SDK签名和全字段可见性未提供。本地采用reset/bidding/update冻结自身对象；官方适配类当前不可用。 | `agent_api.py`、`OfficialBiddingStrategy.available=False`。官方请求显式失败，不回退或伪造官方PID。 |
| H-009 | LocalPID默认kp=1.5、ki=0.1、kd=0.15、initial_alpha=1，目标曲线默认线性、可配置单调累计比例；update用本局实际消耗误差更新积分/差分，alpha最低为0，无上界裁剪。 | `pid`与队伍PID参数、`agents.LocalPIDAgent`。本地代理基线，不是官方PID/上游PID；目标曲线不能读取当日未来真值。 |
| H-010 | 本地参考价值为0时严格未定义；诊断模式null，不加epsilon、不跳局。全部有效G为0单独记相对项0。 | `evaluation.diagnostic_undefined`。官方零基准/异常计分仍待确认；本地错误不得伪装成绩。 |
| H-011 | 默认reference为LocalPID、本地队伍集合；公式所得G与40/15相对分均为本地实验。 | `evaluation.reference_strategy=local_pid`；official请求不可用。无完整官方Q，不编造其余45分或百分制总分。 |
| H-012 | 严格拒绝非法类型/负/非有限Action与源坏值，不修正；整片引擎在工作账本成功后提交，失败不部分扣款/累加价值。 | `validate_alpha`、`AuctionEngine.run_slot`、runner异常处理。事务不回滚恶意/自定义外部副作用；官方异常队伍处置未知。 |
| H-013 | P99采用nearest rank；分别/配对和/两P99之和都报告；初始化单列。benchmark执行配置的目标策略，仅目标调用作统计。 | `benchmark.target_role/agent/warmup_episodes/episodes/min_samples`。函数计时不含竞价；样本不足是诊断，不断言正式达标。 |
| H-014 | RSS是整个测试仿真进程采样；Windows可补peak_wset。包括数据、框架、对手和目标，不声称单Agent RSS。 | `benchmark.py`。采样可能漏短峰，系统历史峰可能含预热/初始化；官方内存范围未知。 |
| H-015 | 每局深拷贝构造规格、新建/reset；评测默认spawn进程。显式封存模型文件与自定义模块起始摘要，每局前后核对。 | `evaluation.isolation`、`sealed_model_files`、`provenance`。未声明外部资源不保证发现；新进程/哈希不是恶意代码沙箱。 |
| H-016 | 初始预算0的使用率定义为0.0；空片/耗尽不提前done；终态next_observation片号为total_slots。 | `runner`、通用轨迹。使用率是诊断，不是优化目标。 |
| H-017 | index只是非负整数行号。必须显式选择file_first_seen或explicit_csv请求顺序，不按index/request_id排序。 | `organizer_import.ordering`。file_first_seen是本地假设，不是官方真实时间；保留源顺序会影响预算耗尽。 |
| H-018 | request_id范围需显式day或time_step；同日多源文件共享该范围，重复request/player即使值相同仍报错。 | `organizer_import.request_id_scope`。time_step内部组合ID避免冲突，范围会影响平局条件。 |
| H-019 | **旧“缺行含义未确认/必须完整覆盖”已被B2/D取代。** 缺行是不参与；内部0+false，真实零值0+true；源非法值仍报错。 | `Opportunity.participation_mask`、六字段/三表/历史导入。Mask进入语义指纹，真实参与量只在评测器，不扩大反馈字段。 |
| H-020 | 六字段无日期/预算；sources显式path/day_id，players完整身份/type/预算，budgets_by_day可按日覆盖完整roster。 | 不因index/片号重置猜日，不从pvalue猜预算；同ID跨日type必须固定。 |
| H-021 | time_step_base必须为整数0或1，映射内部0..71；ID/type保持字符串与前导零，类别无序。 | `organizer_import`。空片保留，不能根据最小观测片号猜起点。 |

## 归档实验与保留数据

H-022至H-025属于旧Criteo代理估值实验：采样机会假设、三个代理角色、train-only平滑CTR及人工预算。该实验实现/派生数据已退出活动环境，旧说明与结果在工作区 `_archive/algorithm_restart_20261027_v1/workspace_snapshot.zip` 中保留，ZIP成员为 `snapshot/AuctionNet/docs/REAL_DATA.md` 等原路径。原始小样本、来源和CC BY-NC-SA 4.0许可保留，见 [数据说明](../data/real_criteo_sample/README.md)。它不再是活动竞价数据入口。

AuctionNet原生数据继续保留 `local-auctionnet-48-to-72-v1` **源转换profile**：原始概率pValue、均匀中点48→72映射、原物理顺序、完整周期划分等NA假设见 [AUCTIONNET_NATIVE_DATA.md](AUCTIONNET_NATIVE_DATA.md)。源manifest的旧规则/转换版本和哈希不改写；新执行规则单独记录为v2。完整48角色旧数据经验证后加载为全true Mask，不重转5亿行。

## 版本迁移与正式对接

金额、零余额、tie、接口、异常评分或官方策略变更时另升执行版本，重建受影响轨迹/统计并重新运行参考场景；不能只换标签复用旧G。旧v1完整内部格式由兼容加载器验证原指纹后加载，新语义指纹含Mask；旧转换事实与本次执行事实分开保存。

官方策略到来后实现预留适配接口、冻结实际代码/参数/模型，再重新建立参考分母。现有本地测试不构成官方验收，现有LocalPID G不能直接当官方G。
