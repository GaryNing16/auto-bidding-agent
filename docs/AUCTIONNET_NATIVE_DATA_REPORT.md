# AuctionNet 原生数据处理实测报告

> 历史报告：以下记录2026-10-23的转换、旧规则运行及测试事实，原数值和结论保留。2026-10-27统一环境使用v2规则；本页367项通过及G=1不作为重构后验收。部分旧运行产物/脚本已按确认清单归档，可从[恢复快照](../../_archive/algorithm_restart_20261027_v1/workspace_snapshot.zip)的 `snapshot/AuctionNet/...` 路径恢复；当前命令与新旧profile边界见[原生数据说明](AUCTIONNET_NATIVE_DATA.md)。

2026-10-23，Windows 本地工作树 `C:\Users\ningy\Desktop\广告智能出价agent\AuctionNet`，独立 `.venv`，Python 3.14.5。无新增依赖；PyYAML/psutil/pytest 版本见 `outputs/native_delivery_summary.json`。上游提交 `c20de631a40e24669d1d6a8e2acefd8f492f172d`。

## 全量结果

- 11 个 ZIP 全部成员通过 CRC、长度及磁盘重读 SHA256 验证后删除，保留21个源CSV，共 81,690,289,604 字节。
- 转换 **503,976,864 行 / 10,499,518 个请求**，每请求完整48个广告主，输出六字段CSV共 32,178,722,538 字节。
- 文件在 `data/auctionnet_72/period-<N>/six_fields.csv`，配套 `request_order.csv`、`metadata.json` 和每片前4请求样本。全部21期源哈希匹配解压清单，输出哈希经汇总程序复核。
- train/validation/local_holdout按完整周期划分为14/3/4，分别7..20、21..23、24..27，无真实日历日期推断。
- pValue范围 [0.0, 0.005491793509549988]，大于1计数 0；原数值文本保留。弃用历史列的空值计数：`{'CPAConstraint': 0, 'remainingBudget': 0, 'pValueSigma': 0, 'bid': 62153, 'xi': 0, 'adSlot': 0, 'cost': 0, 'isExposed': 0, 'conversionAction': 0, 'leastWinningCost': 0, 'isEnd': 0}`。
- 原始与转换大文件保持Git忽略；未提交、推送或修改系统Python。

## 实物发现与修复

第一次试转换发现pvIndex每个原时间片重置，改用 `(period, source_slot, pvIndex)` 唯一键，输出如 `period-7_slot_0_pv_0`；同源片重复仍拒绝。全量扫描另发现部分历史bid为空，改为18列宽度严格、仅7个必需源字段非空合法；弃用字段空值只统计，不补值、不删请求。两类错误均先显式失败，修复后恢复，原始错误日志保留在outputs。已成功产物恢复时核对源/输出哈希，不重写不同文件。

## 实际验证

完整回归 **367 项通过**，耗时 109.80 秒。原生转换的人工例、请求顺序、复合ID、缺失/重复/非法输入、断点污染及双进程入口均有测试。

周期7/21/24共864个完整请求、41,472行快速样本逐字对应原CSV，三个源文件完整哈希核对；独立审计 `outputs/native_sample_source_audit.json` 通过。样本保持全量时间映射、48角色和原预算。

`import-organizer`、72片 `run`、`evaluate-single`、`evaluate-multi`、`export-trajectories --split train` 全部实际成功。16个局报告的完整生命周期、预算与价值累计复核通过；训练轨迹3,456条、48个日终done，片前历史时序正确。重复运行17个业务CSV/评测JSON逐字节一致，不比较耗时。

留出周期24的全PID小样本总价值 0.458711871300374，总费用 0.678282。训练样本选择的角色为14、21、41；A/B/C都用同一PID，G_single={'A': 1.0, 'B': 1.0, 'C': 1.0}，G_multi={'A': 1.0, 'B': 1.0, 'C': 1.0}。这是匹配参考的一致性验证，不是策略提升或官方成绩。

## 耗时、内存与限制

各批次完整argv、退出状态、墙钟耗时及每秒采样的RSS在 `outputs/native_conversion*_measurement.json`；首次全量批次因空历史bid失败的周期已在后续批次补齐。不同批次可能时间重叠，不能直接相加当墙钟总耗时。RSS为转换父进程和工作进程之和，可能重复计算共享页；不是单Agent内存或官方性能指标。

| 转换周期 | workers | 批次状态 | 秒 | 该批次进程树RSS峰值/MiB |
|---|---:|---|---:|---:|
| 7-27 | 4 | failed | 1229.08 | 166.66 |
| 9,13 | 2 | success | 265.62 | 102.27 |
| 27 | 1 | success | 195.49 | 69.73 |

全量CSV已转换并校验；竞价引擎的实际集成运行使用有界样本，没有把5亿行加载到现有内存式通用导入器。48→72采用本地中点时间推定，原pValue为概率评分，预算和alpha尺度仍需显式实验配置。未训练大型模型，未声称通过官方验收。详细规则、字段映射、恢复和运行命令见 [AUCTIONNET_NATIVE_DATA.md](AUCTIONNET_NATIVE_DATA.md)。

直接运行：`.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/auctionnet_native_sample.yaml`。
