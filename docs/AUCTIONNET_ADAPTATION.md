# AuctionNet 源码审计与实际适配

上游：<https://github.com/alimama-tech/AuctionNet>。本次读取的实际提交为 `c20de631a40e24669d1d6a8e2acefd8f492f172d`。审计依据本地检出的源码，不依据对仓库行为的猜测。

## 获取与保留

目标工作区起初已有用户的题面 Word 和一份研究 PDF，没有 AuctionNet 工作树。初次搭建时在子目录 `AuctionNet` 新建浅克隆，保留用户原文件。上游 `LICENSE.txt`（Apache 2.0）和原源码保留；本模块归属说明见本文件。未提交或推送任何更改。

采用 sparse checkout 排除完整预生成数据、模型、资源图等目录。首次 sparse 范围过宽，曾取回仓库附带的模型/样本（包括约 71MB 的 pth 和约 93MB 的 CSV），随后收紧范围并从工作树排除；它们未运行或加载。这是获取范围的实际偏差，不应描述为“从未下载任何模型文件”。初次环境搭建阶段没有下载 README 描述的完整约 80GB 预生成数据集，也未安装运行完整上游 GPU/RL 环境。后续用户自行下载的21周期原生数据已完成转换并保留，见 [原生数据说明](AUCTIONNET_NATIVE_DATA.md)；不能把初次获取范围误当当前数据状态。

## 源码发现及处理

| 上游文件/符号 | 实际行为、可参考内容 | 不兼容点与当前活动环境处理 |
|---|---|---|
| `README.md`、`requirements.txt` | 环境、训练数据生成、策略训练三个部分；固定版本 NumPy/Pandas/Torch/gin/SciPy 等 | 完整安装会引入非必需 PyTorch 等依赖。赛题模块使用独立虚拟环境及最小依赖；原入口保留。 |
| `config/test.gin` | 默认 48 广告主、48 片、500000 请求，保留价 `0.0001`，最低余额 `0.1` | 这些是 AuctionNet 的实例，不是鸿蒙赛题参数。新增独立 `configs/`，正式规格示例为 72 片，N/d/预算/规模明确为本地例值。 |
| `simul_bidding_env/Environment/BiddingEnv.py::BiddingEnv.__init__` | `NUM_SLOTS=3`、`slot_coefficients=[1,0.8,0.6]`、`NUM_ADVERTISERS=48` | 赛题为单广告位。新增 `harmony_bidding.auction.AuctionEngine` 逐请求参考实现，不调用旧引擎。 |
| `BiddingEnv._get_sorted_bids_and_market_prices`、`_handle_unsold_slots` | 排序取前三位置，市场价下限为保留价；费用等于保留价时将该位标为未售 | 赛题未规定保留价，单有效者需免费获胜。本地有效正报价按二价结算，禁用原保留价路径。 |
| `BiddingEnv._calculate_exposure`、`_generate_values_matrix`、`_calculate_conversion_action` | 对位置曝光作 Bernoulli，预估值加截断正态扰动，再 Bernoulli 转化；多个函数重新建固定种子 RNG | 答疑明确价值为竞胜预估值之和。本地直接累计赢家对应 value，不采样曝光/转化或扰动价值。 |
| `BiddingEnv.generate_trunc_values` | 使用 `hash((advertiser_index,time_step_index,episode))` 生成随机源种子 | v2平局按显式seed、day/slot/request、重拍轮次和排序候选集合派生SHA256私有随机源，使用randrange均匀选索引；不依赖Python hash、全局RNG或调用计数。 |
| `simul_bidding_env/Controller/Controller.py::Controller` | 组装多种 RL/LP/PID 对手，固定预算/CPA 列表；重置环境、Agent、流量生成器 | 直接导入会拖入模型依赖；CPA 不是本题约束。新增 runner 和 Agent factory，从封存构造规格每局新建所有策略，对手统一可配置 LocalPID。 |
| `simul_bidding_env/strategy/base_bidding_strategy.py::BaseBiddingStrategy` | 只依赖 `abc`；提供抽象策略及 reset/bidding 生命周期骨架 | **真实代码继承**：`harmony_bidding.agent_api.Agent` 继承它。明确不调用旧构造函数，不创建旧 CPA 状态；重写 reset/bidding 签名，新增 update 抽象方法。不是旧逐请求接口兼容适配器。 |
| `simul_bidding_env/strategy/pid_bidding_strategy.py::PidBiddingStrategy` | `base_action=15`，依据历史片消费/余额比例乘 1.2 或 0.7；48 硬编码；`reset()` 只重置余额 | 此实现并非公开的官方鸿蒙 PID，也没有可配置 Kp/Ki/Kd；`last_remaining_budget` 等状态的 reset 不完整。新增 LocalPID，按自身实际消费与目标预算进度误差计算 P/I/D，仅在 update 修改系数，每局重置所有状态。 |
| `run/run_test.py::run_test` | 把当前 `pv_values[:,i]` 和 `pvalue_sigmas` 及历史逐请求数据传给 bidding，返回逐请求 bid 数组；余额低于阈值跳过调用；负 bid 截为 0 | 当前请求数/价值、历史最低竞胜价不可见；只允许标量动作。新 runner 只给冻结的自身历史观察；所有 Agent 每片照常调用；非法动作严格报错，绝不静默截断。 |
| `run/run_test.py::adjust_over_cost`、超支 while 循环 | 先整片模拟，再按超支比例随机选已获胜请求置零，然后整片重算，直到不超支 | 破坏片内顺序和即时余额语义。本地 `BudgetPolicy` 在每个请求即刻检查可支付性并扣款，绝不随机删已结算请求/回滚。 |
| `run/run_test.py::run_test` 中 `done_list` | 余额低于最低阈值提前 done | 本地只日终 done；预算耗尽仍生成后续片零/正常反馈和训练轨迹。 |
| `simul_bidding_env/Tracker/BiddingTracker.py` | 按请求保存全 N 广告主预算、CPA、价值、报价、曝光、转化、最低竞胜价等 | 可参考分层日志组织，但不能直接当 actor 轨迹。新增自身可见片级轨迹；内部全广告主调试数据与训练轨迹明确分离。未继承或调用旧 Tracker。 |
| `simul_bidding_env/Tracker/PlayerAnalysis.py::_get_score_neurips`、`get_return_res` | CPA 超约束时平方惩罚 reward；最后 score 除以 20000；日志计算真实请求胜率 | 不符合价值目标及 G 公式。新增独立全 PID 分母、逐局比及模式均值评分；不把请求总数/真实胜率提供给 Agent。 |
| `strategy_train_env/bidding_train_env/strategy/base_bidding_strategy.py` | 类似旧模拟策略接口，含 CPA、当前 pValues 和历史逐请求数据 | 未直接复用这个重复基类；赛题使用同仓库模拟基类的抽象骨架，并建立自己的 API。 |
| `strategy_train_env/bidding_train_env/strategy/iql_bidding_strategy.py::IqlBiddingStrategy.bidding`（其他策略亦使用类似特征） | 16 维状态含当前 pValues 均值/请求数、历史最低竞胜价、历史机会量；时间除以 48；加载模型及归一化 pickle | 原策略/模型输入维度、时序和信息边界不兼容。未加载原模型；新轨迹只从实际 bidding 观察生成。 |
| `strategy_train_env/bidding_train_env/train_data_generator/train_data_generator.py::TrainDataGenerator._generate_train_data` | 从请求 CSV 聚合 state/action/reward/next_state/done；当前均值/请求数、历史最低竞胜价进入 state；固定 48；曝光转化 reward，另有连续 reward | **组织方式参考，非直接代码复用**：保留片级 transition 思路，重新导出合法 observation/alpha/value/cost/next_observation/done。不能直接用旧生成器处理本地 CSV。 |
| `strategy_train_env/bidding_train_env/baseline/` | BC/BCQ/CQL/IQL/TD3-BC/DT/OnlineLP 实现；IQL 的 `ReplayBuffer.push` 接收五元组，`IQL` 可配置 observation/action 维度 | 保留上游源码供未来参考；活动核心不导入这些模块、不加载旧模型。独立旧算法适配研究曾存在，现已按确认范围归档；v2不保留ReplayBuffer桥，也不声称重构验证了算法效果。 |
| `strategy_train_env/bidding_train_env/common/utils.py` | 状态/奖励归一化与统计序列化工具 | 原工具在传入数据上拟合统计，不能使用验证/留出数据。旧本地七维统计已归档；当前通用轨迹不固定特征、不拟合统计，不直接复用旧工具或训练模型。 |

## 实际继承、复用、重写的范围

本项目是在 AuctionNet 工作树内新增赛题模块，保留上游源码及训练目录。唯一运行时的上游代码依赖是轻量 `BaseBiddingStrategy` 抽象基类。新 Agent 改写所有与旧协议相关的行为，故既不接受旧 pValues 参数，也不依赖旧 CPA 目标。不要把它称作旧 Agent 接口的无缝兼容。

模块职责参考上游 Controller / Environment / Tracker / Strategy / TrainDataGenerator 分层；引擎、预算政策、可见接口、PID、数据验证、日志和评分均为适应题面而重写。原 `main_test.py` 和 `run/run_test.py` 未改成鸿蒙赛题实现；它们仍是旧 AuctionNet 入口，未在最小虚拟环境中宣称跑通。

## v2统一与已归档实验

当前比赛规则只有 `runner → AuctionEngine → BudgetPolicy` 一条权威链；评测、轨迹和benchmark均调用它。内部显式Participation Mask保留稀疏参与；alpha无全局上限；同价本地均匀选择；工作账本整片成功后提交。官方策略适配契约当前不可用，不能把上游PID作为主办方PID填入。

`prediction`、`ppo_baseline`、`official_baselines`、`alpha_robustness` 的独立研究代码、快速引擎和模型已按确认清单退出活动目录，恢复快照位于工作区 `_archive/algorithm_restart_20261027_v1/workspace_snapshot.zip`。这些是历史研究成果，不能描述为从未实现过或普通垃圾。上游仓库自身的训练算法源码继续保留，不属于此次研究目录清理范围。

旧本地 `trajectory.feature_vector` 七维示例、`fit_training_statistics`、`push_to_auctionnet_buffer` 已移除。活动轨迹仅保留实际合法Observation、标量alpha、价值reward、cost、下一观察、日终done及身份/版本元数据，严格读取/导出验证。未来算法自行定义合法特征，训练/部署共享时序、统计只拟合train；改观察后必须重新生成轨迹、统计和模型，不能直接加载旧16维或旧七维权重。

上游Controller/Environment/Tracker的组织思路是参考，轻量抽象基类继承是实际代码复用；竞价、数据、API、PID与评测是重写。保留这些区别，不能把独立新规则包装成上游原引擎已支持鸿蒙规则。
