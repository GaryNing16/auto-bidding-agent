# Harmony bidding 开发约定

保留AuctionNet原工作树/许可证/入口；本赛题活动入口只有harmony_bidding。不要为基础运行安装上游Torch环境、恢复旧研究目录或加载旧模型。

- 当前 `rules_version=harmony-A-B-local-v2`、默认 `assumption_profile=local-unified-v2`。72片，每片所有Agent先输出一次标量alpha，再统一竞价/各自update。alpha非负有限、无全局上限，拒绝bool/string/NaN/Inf及报价乘积溢出，不替策略裁剪。旧rules.alpha_max必须迁移，算法内部搜索范围不是比赛规则。
- 六字段稀疏参与是已确认含义。内部values+必需bool participation_mask：缺行0/false，真实零值0/true；源非法值不能当缺席。Mask进入数据指纹，真实机会数按列和；Mask/真实机会数只在评测器，不进actor。
- 唯一核心AuctionEngine负责按original_order结算单广告位二价、整数金额账本和价值累计。不能付实际二价的候选仅本请求排除重拍；不因原bid高于余额直接排除。单人免费、默认零余额不参与、金额精度和均匀seed平局均须记录本地假设。
- 整片工作账本验证成功后提交，非法动作/数据不部分更新权威状态。reward=竞胜预估价值，无CPA/ROI/ROS、无点击转化采样；空片/耗尽完整72片。
- Agent只接收冻结的自身配置、片前历史与片后自身反馈；不传当前/未来机会矩阵、对手、环境/加载器句柄。普通轨迹严格白名单，不恢复旧七维统计/ReplayBuffer桥。对象封装/新进程不是安全沙箱。
- 每局新实例/reset；评测默认spawn，封存模型不可写回。OfficialBiddingStrategy当前不可用，官方请求明确失败；Zero/Constant/LocalPID为本地基础策略，LocalPID及其G均非官方结果。
- 所有评测/轨迹/benchmark调用同一核心；未来Gym只能是适配层。G先逐局比再等权平均，零参考严格未定义。每次run_id独立，消费成绩须completed且valid，不能读取旧成功状态冒充当前结果。
- 六字段index只为行号；日期/预算/base/scope/order显式配置；同ID跨日type固定。v1完整矩阵验证后加载全true mask，不改原manifest；源转换版本和新执行版本分别记录。
- 保留data/auto-bidding与auctionnet_72原文件/溯源，不重下/重转5亿行；普通按日矩阵加载器只用于有界样本。Criteo只保留source样本/来源/CC BY-NC-SA4.0许可，代理估值实验已归档。
- 原生48→72仍是本地中点映射，pValue保持原评分，旧bid/CPA/成本/曝光/转化不回放。先前ZIP删除授权不扩大为删除CSV授权。
- 新验证与历史报告分开，不将旧454项通过当新验收。勿自动commit/push、修改系统Python、删用户文件或改权限；清理归档和恢复见工作区cleanup_review。

在本目录运行：`./.venv/Scripts/python.exe -m pytest -q tests`；`./.venv/Scripts/python.exe -m harmony_bidding.cli run --config configs/demo.yaml`。安装/完整命令见README，规则来源及假设见docs。
