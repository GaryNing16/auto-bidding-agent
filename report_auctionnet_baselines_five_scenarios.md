# AuctionNet 十种 Baseline、五种对手场景：本轮交付入口

本轮实验已完成。历史三场景实验及其报告保持原样；以下链接指向独立的五场景实验目录。

- [简版报告](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/report_auctionnet_baselines_comparison.md)
- [详细技术与实验报告](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/report_auctionnet_baselines_detailed.md)
- [汇总 CSV 与报告事实](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/reporting)
- [12组 PNG/SVG 图表](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/reporting/figures)

实验A、实验B和额外独立周期分别完整覆盖7,680局。A复用历史4,320局、补测3,360局；B复用配置完全一致的AuctionNetPID共480局、重新执行7,200局；独立period-25执行7,680个完整案例。正式测试新增成功局共18,240，三个结果表合计23,040行，不能将复用行重复计入执行量。

独立测试初次执行发生5次资源超时，连带跳过141局；146局已按原配置、原模型、原数据与seed全部恢复成功。最终7,534个原成功局加146个恢复成功局完整覆盖7,680个身份，当前未解决数为0。原超时/跳过、诊断、成功结果及待机重叠局时延均保留，未用恢复结果抹去运行历史。

A的五场景平均价值领先者为DT（13.2733）。B的主排名领先者为LocalPID（14.0268），其平均V相对A增加23.96%；DT为12.7995，相对A下降3.57%，不能因平均逐对百分比变化为正而称总体改善。独立period-25前两名仍为LocalPID（14.5615）和DT（13.3658）。这些是所测周期的描述性结果，不是统计显著或官方竞赛排名；A/B各使用自己的LocalPID作为R分母。

## 可复核的输入与结果

- [预先冻结协议](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/protocol.json)、[对手配置](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/opponents_frozen.json)
- [A参数与权重身份](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/experiment_A/selected_frozen.json)、[B最终参数与checkpoint身份](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/experiment_B/selected_frozen.json)
- [A逐局CSV](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/experiment_A/episode_results.csv)、[B逐局CSV](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/experiment_B/episode_results.csv)、[独立测试逐局CSV](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/independent_test/episode_results.csv)
- [训练轨迹清单](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/trajectory_episodes.csv)、[训练来源](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/trajectory_provenance.json)、[训练run及权重](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/models/runs)
- [恢复映射与封存清单](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/independent_test/recovery_manifest.json)、[运行阶段与实际执行数量](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/reporting/execution_stages.csv)

训练集复用288条旧轨迹、补充192条新轨迹，共480条72片轨迹、34,560个transition，各场景96条。六种神经算法按两个学习率和两个训练seed从头训练，共24个run，每run1,800步，保存600/1,800步共48个checkpoint。模型只由验证集选择，没有按测试结果再调参。

## 实现与审计

- [AuctionNetPID实现](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/auctionnet_pid.py)、[上游机制与72片适配说明](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/PID_ADAPTATION.md)
- [五场景执行流程](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/pipeline.py)、[数据与训练代码](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/data_training.py)、[统计与绘图源码](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/report.py)、[恢复入口与命令](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/harmony_baselines/five_scenarios/README.md)
- [独立测试全量审计](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/qa/independent_test_completed_audit.json)：7,680局、552,960片、1,658,880条诊断及冻结身份均通过。
- [恢复前缀审计](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/qa/independent_recovery_prefix_audit.json)：5个原超时案例的2,131个业务数值比较，最大差异0。
- [最终统计与产物审计](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/qa/final_delivery_audit.json)：682项检查通过，独立算术复核最大差异3.55e-15，82个报告产物SHA一致。
- [历史完整性核验](C:/Users/ningy/Desktop/广告智能出价agent/AuctionNet/outputs/baseline_five_scenarios_20261028/historical_integrity_verification.json)：7,786个历史保护文件的内容和修改时间均一致。
