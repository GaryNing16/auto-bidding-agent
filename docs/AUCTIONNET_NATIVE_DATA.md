# AuctionNet 原生预生成数据的赛题格式适配

本文记录 AuctionNet 原生 CSV 的语义审计与转换契约。本路径保留已有逐请求 `pValue`，不重新训练CTR代理模型，也不生成缺失角色的估值。旧Criteo代理实验已归档，活动目录只保留其[源样本与许可](../data/real_criteo_sample/README.md)。

**2026-10-27状态说明**：以下2026-10-23的规模、错误修复与测试结果是历史转换证据，不是v2重构验收。原CSV、完整72片文件、manifest及旧转换profile不改写；新执行使用 `harmony-A-B-local-v2` / `local-unified-v2`。该批每请求完整48角色，兼容加载时验证后赋全true Participation Mask；不把原生全覆盖特性强加给主办方稀疏数据。历史已移除运行产物/脚本保存在[恢复快照](../../_archive/algorithm_restart_20261027_v1/workspace_snapshot.zip)的 `snapshot/AuctionNet/...` 路径，保留来源与审计JSON仍在outputs。

**来源是 AuctionNet 模拟竞价环境的预生成记录，不是华为或鸿蒙赛题主办方提供的数据，也不是实际商业平台完整投放日志。** 转换后的六列形状可用于本地开发；48→72 片时间映射、预算尺度及本地竞价机制仍必须与源数据语义分开记录。

2026-10-23已完成全量处理：11个归档均为 `verified_and_deleted`，21个源CSV保留；全部21周期转换为503,976,864行、10,499,518个完整请求，六字段CSV约32.18 GB。367项回归通过，72片运行、单/多Agent评测、轨迹导出及独立源行审计均实际通过。实测与证据见 [AUCTIONNET_NATIVE_DATA_REPORT.md](AUCTIONNET_NATIVE_DATA_REPORT.md)、各周期metadata和汇总manifest。

以下保留转换设计与严格失败后修复的记录：先核实pvIndex按原时间片重置，再处理弃用历史bid列的空值。最终所有周期均已恢复完成；全部62,153个空历史bid仅计入诊断，不改变预估价值或请求覆盖。

## 来源、版本与许可证据

- 上游项目：[alimama-tech/AuctionNet](https://github.com/alimama-tech/AuctionNet)。本工作树封存提交为 `c20de631a40e24669d1d6a8e2acefd8f492f172d`。
- 数据说明：[封存版本的 readme_dataset.md](https://github.com/alimama-tech/AuctionNet/blob/c20de631a40e24669d1d6a8e2acefd8f492f172d/pre_generated_dataset/readme_dataset.md)。本地稀疏检出未包含该文件时，仍可从已有 Git 对象读取，不需下载数据。
- 官方发布入口为数据说明中的 `https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/`，归档名为 `autoBidding_general_track_final_data_period_<周期或范围>.zip`。
- 用户提供的 11 个 ZIP 范围为 `7-8、9-10、11-12、13、14-15、16-17、18-19、20-21、22-23、24-25、26-27`，合计 21 个投放周期；现已在 `data/auto-bidding/` 解压保存源 CSV，并按授权在核验成功后删除这 11 个 ZIP。每个源 CSV 约数 GB；确切大小、成员、摘要和删除状态见 `data/auto-bidding/extraction_manifest.json`。

仓库根目录 [LICENSE.txt](../LICENSE.txt) 是 Apache License 2.0，上游 README 也声明该项目使用 Apache-2.0。已核对的数据说明没有为外部 OSS ZIP 另列独立的数据许可条款。应保留上游归属、源码许可、原始下载地址及归档内若存在的说明；不能仅凭代码许可推断外部数据取得了额外的商业再分发授权，也不能把 Criteo 的 CC BY-NC-SA 条款套在这批数据上。此处记录已查到的许可证据范围，不伪造未提供的数据授权。[上游许可](https://github.com/alimama-tech/AuctionNet/blob/c20de631a40e24669d1d6a8e2acefd8f492f172d/LICENSE.txt)

原始 ZIP（保留期间）、解压 CSV、完整转换 CSV 和大规模内部数据均不加入 Git；校验后的 ZIP 已按授权清理，源 CSV 继续保留。转换代码、本文和适当大小的配置/摘要清单可版本管理；清单保留 SHA256、行数、周期、来源路径与转换假设，不复制全部请求值。

## 原始字段与完整矩阵的证据

已确认的实际表头为：

```text
deliveryPeriodIndex,advertiserNumber,advertiserCategoryIndex,budget,CPAConstraint,timeStepIndex,remainingBudget,pvIndex,pValue,pValueSigma,bid,xi,adSlot,cost,isExposed,conversionAction,leastWinningCost,isEnd
```

上游说明将这批数据描述为 21 个投放周期，每周期约 50 万个机会、48 个决策片、48 个广告主，并支持查看全部广告主在各机会上的投放过程。其中示例一仅列出出价最高的五个角色；这是演示表的截取，不能推断下载的完整 CSV 只有五个角色。[数据说明](https://github.com/alimama-tech/AuctionNet/blob/c20de631a40e24669d1d6a8e2acefd8f492f172d/pre_generated_dataset/readme_dataset.md)

源码给出更直接的结构证据：[BiddingTracker.train_logging](../simul_bidding_env/Tracker/BiddingTracker.py) 根据 `pValues.shape = (num_pv, num_agent)` 展开全部价值，并对每个请求重复完整广告主列表；没有按 `xi`、`isExposed`、预算余额或 `isEnd` 过滤行。当前源文件前缀观察到同一请求的 48 个角色连续出现，与该矩阵布局相符。源码通过传入的 `totalPvNum` 构造请求编号，但不能据此认定外部 release 中的 `pvIndex` 跨片累计；下面的实物核查已经否定这一假设。

### 实物核查：pvIndex 每个源片重置

第一次全量转换以周期内唯一的 `pvIndex` 去重，实际在 period 7 严格停止并报出 `ValueError`，诊断指出 `duplicate/noncontiguous pvIndex`、编号为 `0`。检查源 CSV 发现：

| 源文件位置 | 前一行 | 当前行 |
|---|---|---|
| period 7，第 600770 行 | `timeStepIndex=0, pvIndex=12515` | `timeStepIndex=1, pvIndex=0` |
| period 7，第 881714 行 | `timeStepIndex=1, pvIndex=5852` | `timeStepIndex=2, pvIndex=0` |

这是下载实物的编号语义，不能将跨源片再次出现的 `pvIndex=0` 当作重复机会或文件损坏。修复后的源请求唯一键为 **`(deliveryPeriodIndex, timeStepIndex, pvIndex)`**，单周期内去重键为 `(source_slot, pvIndex)`。同源片中的重复/非连续请求仍须拒绝，不得为了接受重置而取消去重检查。

输出 ID 显式保留源片，如 `period-7_slot_0_pv_0` 与 `period-7_slot_1_pv_0` 是两个请求。48→72 映射可能将相邻源片的部分请求合并到同一个新片，因此 ID 不能只保留新片或周期与 pvIndex。元数据中的 `source_request_scope` 与布局要求随之更新。首轮失败未发布成功 metadata、未修改源 CSV；修复回归与后续重跑完成范围须按实际执行记录确认，不能把失败的首次尝试写成转换通过。

### 实物核查：被丢弃的历史 bid 可以为空

后续转换在两个周期发现历史 `bid` 的空单元格，旧 `_stage` 因错误地要求全部 18 列都非空而严格停止：

| 源文件位置 | 请求/角色 | 实际空字段 |
|---|---|---|
| period 9，第 23247251 行 | `timeStepIndex=46, pvIndex=0, advertiserNumber=33` | `bid` |
| period 13，第 23174739 行 | `timeStepIndex=47, pvIndex=0, advertiserNumber=1` | `bid` |

这两行的映射必需字段——`deliveryPeriodIndex`、`advertiserNumber`、`advertiserCategoryIndex`、`budget`、`timeStepIndex`、`pvIndex`、`pValue`——均有合法值。历史结果为空不等于 `pValue` 缺失，也不表示少了一名角色的价值数据。由于新局不回放历史 bid，不能因此删除该角色行或整个请求。

修正规则保持表头和每行 **18 列宽度严格匹配**，但仅要求上述 7 个必需字段非空，并继续执行其身份、类型、预算、片号、概率评分与关联合法性检查。其余被丢弃的历史字段允许空单元格，按列累计到 `ignored_missing_field_counts`。该计数用于来源诊断：不把空 bid 填成 0，不补历史转化或价格，不将历史列加入策略输入，也不据此降低必需字段的校验要求。少列、多列或必需字段缺失仍必须报错。

本批在修复前已经成功输出的周期，生成时通过了原先更严格的全部 18 列非空检查，因此可继续保留并按既有摘要验证；其忽略字段空值计数已由那次检查保证为 0，不需要丢弃成功产物。失败的 period 9、13 则在修复及回归后继续恢复/重跑，最终计数与完成状态由实际 metadata 记录，不能将一次失败写成成功转换。

这些证据支持“完整矩阵是预期格式”，但不等于已证明所有下载文件完整。转换必须逐请求核验：

1. 每个 `(deliveryPeriodIndex, timeStepIndex, pvIndex)` 恰好有全部 48 个已登记角色，各出现一次；同源片中的缺失、重复、未知身份均报错。不同源片允许复用相同 pvIndex，不将其合并为同一请求。
2. 同一角色的行业类别与该周期预算保持一致；跨周期类别一致性也需要检查，不能把类别变化静默当作同一固定类型。
3. 请求组应连续，源片号非递减且在 `0..47` 范围；同一源片中的请求不能先输出、随后又在另一位置出现。不能看到前 48 行正确就跳过后续检查。
4. 不因为 `bid=0`、`xi=0`、未曝光、未转化、余额不足或 `isEnd=1` 删除角色/请求。这些字段是旧机制结果，不能作为新环境可见性与生命周期的过滤器。
5. 列宽必须为18，7个转换必需字段必须非空且合法；被丢弃历史列的空单元格单独计数，不等同请求或价值缺失。

若全量核验发现缺行，不补零、不均值填充、不生成模型预测来伪装原始 `pValue`。应输出涉及周期/片号/请求/角色的诊断；确认源文件损坏、真实稀疏含义或范围选择后，再决定是否建立新的明确适配政策。

## pValue 的含义和报价尺度

上游把 `pValue` 定义为广告曝光后发生转化行为的概率，`pValueSigma` 表示预测不确定性。按这个语义通常应处于 `[0,1]`，没有货币单位；它不是人民币、每次转化收入或已经观测的转化结果。此次格式转换严格拒绝负值/非有限值，但不额外增加 `<=1` 的硬截断或过滤，以保留来源数值；应记录最小值、最大值及大于 1 的数量。若观察到大于 1 的来源值，它是需要核对的语义异常，不能再将该值解释为合法概率，也不能无声改成 1。六字段通用接口本身允许非负估值，不强制所有数据源都采用概率单位。上游 [BiddingEnv](../simul_bidding_env/Environment/BiddingEnv.py) 使用概率和不确定性生成曝光/转化事件，本赛题模块不沿用这些抽样过程。

本地映射保持 `pvalue = pValue`，不乘 `CPAConstraint`、不乘旧曝光概率、不按旧赢家筛选，也不增加随机噪声。新局 reward 为竞胜机会的 `pvalue` 之和，其单位是“原始概率评分之和”；它不是实现了多少次真实转化，也不是货币收入。相同请求仍允许不同角色具有不同价值。

报价继续按 `alpha * pvalue` 在本地金额精度下结算。alpha 是可大于 1 的出价系数，承担原始价值评分与本地报价数值之间的尺度选择；不能称为概率。上游预算与概率评分的数值范围可能使小型示例策略的报价尺度下预算约束很松，或使过小报价被金额精度量化成零。当前执行环境没有全局alpha_max；旧限幅配置仅属于历史实验。应先报告值域、预算使用率和报价尺度，再显式配置实验参数；不能默默修改 `pValue`、引入 CPA 约束或声称已复现原平台的金额单位。

源 `budget` 可作为该周期的初始预算候选，并在外部角色配置中保留其来源。`remainingBudget` 是旧策略的历史反馈，不能用于新局初始化或预算推进。若采用不同预算或系数上限，必须将覆盖值、理由与配置指纹写入转换/运行记录，重新运行全 PID 参考场景。

## 六列映射与原始顺序

| 输出字段 | 来源与处理 |
|---|---|
| `index` | 输出文件中的行号，仅便于定位；不承担请求唯一键或到达时间语义。 |
| `time_step` | 按下节明确的 48→72 映射得到；输出起点由导入配置显式声明，不能自动猜测。 |
| `request_id` | 将投放周期、原始 `timeStepIndex`、原始 `pvIndex` 组合，例如 `period-7_slot_0_pv_0`。pvIndex 仅在源片内唯一，必须保留源片避免重置与映射合片造成碰撞；三个源字段留在映射溯源中。 |
| `player_id` | 来自 `advertiserNumber`，允许源整数或仅含零小数部分的表示，如 `7.0` 规范化为字符串 `7`，不通过浮点数舍入 ID。当前原生入口明确要求 `0..47` 完整角色名单，不把角色从 1 开始重编号，也不把 ID 当连续数值特征。 |
| `player_type` | 来自 `advertiserCategoryIndex`，作为无序类别保留；不由历史 CPA/消耗/胜负重新分类。 |
| `pvalue` | 原始 `pValue`；仅做合法性检查和明确的表示转换，不拟合或填补估值。 |

请求顺序以源文件首次遇到该请求的顺序为准。对于同一个请求，角色行的排列不产生额外机会，也不影响请求的先后位置。转换器保存显式顺序或能无歧义恢复顺序的映射；不按 `pvIndex`、字符串 request_id 或 advertiser 排序来替换源序列。

`deliveryPeriodIndex` 是投放周期标识，本地可将每个周期作为一个独立 episode/day_id。它没有提供真实日历日期或时区；不得把 period 7..27 写成推测出的真实年月日。

## 48→72 片：明确的本地时间假设

源数据提供 48 个片号，没有逐请求真实时间戳。本地将每个源片内请求视为均匀占据该片，取每个请求位置的中点，映射到 72 片。设：

- `s` 为源片号，`0 <= s < 48`；当前数据明确按 0 起点处理。
- `c` 为该源片内唯一请求总数。
- `r` 为该请求在源片内首次出现的顺序，`0 <= r < c`。

内部新片号为：

```text
new_slot = floor((s + (r + 0.5) / c) * 72 / 48)
         = (72 * (2*s*c + 2*r + 1)) // (96*c)
```

实现应采用等价整数运算，避免浮点数在片边界附近误分配。先计完整源片的唯一请求数，再赋予中点位置；`c` 和 `r` 都按请求计数，不能用 48 倍的广告主行数。一个请求的所有角色必须映射到相同片。

人工核对：源第 0 片有两个请求时，`r=0,1` 分别映到新片 0、1；源第 1 片也有两个请求时，分别映到新片 1、2。源相邻片可以并入同一新片，且会保留原先后顺序。源片为空时没有请求要映射；不能除以零。请求很少时，72 片中的部分片也可能为空，仍保持完整生命周期。

这种映射不增加、删除或复制机会，不改变原 `pValue`，但会改变哪些请求共享一次 bidding 决策以及 PID 获得反馈的时点。因此它不能被称为恢复了真实 20 分钟时间戳，也不能认为与原 48 步环境的业务结果等价。全 PID、单 Agent、多 Agent 和训练轨迹必须使用同一映射版本。

如果另做有界小样本，需要声明 rank/count 来自完整源片还是抽样后片。推荐保留完整源片的 rank/count，再选择完整请求组，以便同一源请求在全量版与小样本版中拥有相同的映射片号。不能采样单个角色行而破坏完整矩阵。

本数据路径的附加假设使用 `NA-*` 编号，与 [ASSUMPTIONS.md](ASSUMPTIONS.md) 的通用 `H-*` 规则一并适用：

| ID | 本地选择 | 影响与替换位置 |
|---|---|---|
| NA-001 | 连续请求组的源首次出现次序作为请求顺序。 | 来源只给片号；若未来提供更精细到达顺序，应替换转换器与 order 文件并重评。 |
| NA-002 | 48 片内均匀中点映射为 72 片。 | 改变决策/反馈边界；实现位于 `auctionnet_data._emit`，由转换版本与 profile 标识。 |
| NA-003 | 原 pValue 不变，作为本地价值评分；超 1 值只做来源诊断，不自动裁剪。 | 不构造货币价值，不沿用曝光/转化抽样；实际范围与异常数量写入元数据，源语义确认后可调整校验政策。 |
| NA-004 | 保留源预算，示例 alpha 上限 5。 | 不保证与概率评分规模匹配；覆盖参数必须显式记录，不能修改源 pValue 来隐藏尺度变化。 |
| NA-005 | period 按7..20/21..23/24..27划分，快速样本取每映射片前4请求。 | 是本地划分及有偏的小样本，不是官方测试或代表性随机采样；参数改变需重建派生产物及可比参考。 |

源请求范围不是待猜测的日内唯一假设：本次 release 的实物已确认应包含源片号，按 `(period, source_slot, pvIndex)` 标识；若未来更换数据版本，仍要重新验证其范围。

## 周期划分与数据用途

本地预定划分按完整投放周期进行：

| 划分 | 周期 | 周期数 |
|---|---|---:|
| `train` | 7..20 | 14 |
| `validation` | 21..23 | 3 |
| `local_holdout` | 24..27 | 4 |

这是公开数据的本地留出划分，不是华为官方隐藏测试集，也不证明周期之间具有真实日历上的因果连续性。不得随机将同周期请求拆到多个集合。原始 `pValue` 直接使用，不需要重新拟合估值；后续策略训练、归一化、预算/参数选择仍只使用训练划分，验证与留出用于相应评估。

源 `bid`、`xi`、`adSlot`、`cost`、`isExposed`、`conversionAction`、`leastWinningCost`、`isEnd`、`CPAConstraint`、`remainingBudget` 和 `pValueSigma` 不进入在线 observation、普通片级训练轨迹或新局竞价。保留原 CSV 便于离线追溯，不代表这些列获准成为策略输入。每局预算与 PID 按新规则实际反馈重新运行，不能回放源历史价格或旧预算轨迹。

## 有界小样本与快速验收入口

完整转换数据保留在 `data/auctionnet_72/period-<period>/`。本次默认每周期额外保存每个**映射后**片的前四个完整请求，保留其全量映射片号、原相对顺序、全部 48 个角色、原始 `pValue` 与预算；不按小样本重新计算 48→72 时间位置。默认小样本每周期最多 72×4=288 个请求、13,824 行，实际数量以 metadata 为准；转换参数允许显式指定其他每片上限。

`scripts/prepare_auctionnet_sample.py` 使用已完成转换的 period 7、21、24，分别代表 train、validation、local_holdout。它核对各周期 metadata 的转换/规则/profile 版本、样本上限及引用的小样本与顺序文件 SHA256，汇总显式请求顺序并写入 `configs/auctionnet_native_sample.yaml`；各周期实际采样上限来自 metadata，不能靠固定文字猜测。当前运行配置的 `rules_version` 为 `harmony-A-B-local-v2`，`assumption_profile` 为 `local-unified-v2`；源转换metadata仍保留 `local-auctionnet-48-to-72-v1`。`source_kind` 为 `auctionnet_native_simulation_bounded_sample`，默认运行留出周期24；每次输出在 `outputs/auctionnet_native_sample/<command>/<run_id>/`，实际目录由CLI返回。

为选择三个快速验收角色，准备脚本只统计选定训练周期小样本中每请求原始 `pValue` 最大的角色次数，再选次数最多的三个角色。并列最大值按源角色行顺序确定，角色计数相同时按角色字符串顺序确定；这是本地测试角色选择，不是策略优化目标。验证/留出信息不参与选择，不能为消除留出失败而用留出结果重选角色。

示例 A/B/C 都使用相同 LocalPIDAgent，目的为检查独立全 PID 参考、单角色替换和轮换的一致性。在分母大于零时预期 G=1；若某局 PID 参考价值为零，仍按严格规则报告归一化未定义，不能为了验收加 epsilon。原预算完整保留，但机会大幅减少，小样本很可能预算宽松，因此它适合验证接口与数据链路，不适合评价预算优化算法优劣。

从 AuctionNet 项目根目录运行以下命令。准备步骤要求相应周期的全量转换与小样本 metadata 已经成功产生；它本身不解压 ZIP，也不以小样本代替全量转换。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_auctionnet_sample.py --root data/auctionnet_72 --config configs/auctionnet_native_sample.yaml --periods 7 21 24
.\.venv\Scripts\python.exe -m harmony_bidding.cli import-organizer --config configs/auctionnet_native_sample.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/auctionnet_native_sample.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-single --config configs/auctionnet_native_sample.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-multi --config configs/auctionnet_native_sample.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli export-trajectories --config configs/auctionnet_native_sample.yaml --split train
```

首次 `import-organizer` 创建 `data/auctionnet_72/sample_internal/`；已导入后直接运行或评测，不反复覆盖内部数据目录。准备脚本允许相同配置/顺序文件复用，遇到内容不同的既有文件则拒绝覆盖。完整21周期产物与这个三周期快速入口分开，报告必须标明使用了哪种数据规模。

## 大文件、完整性与归档清理

21 个源 CSV 的总规模远超 tiny/demo。`harmony_bidding.auctionnet_data` 采用两阶段流式处理：先扫描原 CSV、累计每源片请求数并核验结构，将必需列写入临时 staging 文件；再扫描 staging，按完整请求组写出映射。请求去重使用磁盘 SQLite，摘要计算使用文件流。此设计避免把数千万行字典或整周期完整矩阵留在 Python 内存中，同时需要额外 staging 和输出磁盘空间。通用按日加载器或导入器也不能在未测量内存前被当作全量数据入口。

历史转换版本为 `auctionnet-six-fields-v1`，完整数据 `source_kind` 为 `auctionnet_native_generated_game_data`，源metadata记录规则版本 `harmony-A-B-local-v1`、profile `local-auctionnet-48-to-72-v1`；这是不改写的转换来源，不是当前执行规则。每周期目录包含：

```text
data/auctionnet_72/period-7/
  six_fields.csv           # 完整转换，不只保存快速验收样本
  request_order.csv        # 完整映射后的显式顺序
  sample.csv               # 每映射片前四个完整请求
  sample_request_order.csv
  metadata.json            # 成功产物及摘要
  checkpoint.json          # 分阶段恢复状态
```

根目录 `manifest.json` 汇总成功周期并检查跨周期角色类别；失败期不能混入完整成功集合。可分别选择周期转换，但仅在核对 7..27 全部存在且有效后才称21周期转换完成。默认恢复会核对源文件、参数、阶段及输出摘要；输入或配置不匹配时拒绝复用，不覆盖不明已有产物。

以下是从已下载 ZIP 开始的全量入口，**涉及大文件解压与转换，并非默认快速 demo**。当前工作区已经完成解压与 ZIP 清理，继续工作只需转换命令；前两条保留供全新数据目录复现。`--delete-archives` 对应用户已经授权的逐包校验后清理；不加该参数则保留 ZIP。首次可先用 `--list-only` 查看成员，列目录本身不验证完整 CRC。

```powershell
.\.venv\Scripts\python.exe scripts/extract_auctionnet_archives.py --root data/auto-bidding --list-only
.\.venv\Scripts\python.exe scripts/extract_auctionnet_archives.py --root data/auto-bidding --delete-archives
.\.venv\Scripts\python.exe scripts/convert_auctionnet_data.py --source-dir data/auto-bidding --output-dir data/auctionnet_72 --periods 7-27 --workers 1 --sample-requests-per-slot 4
```

`--workers` 控制按周期并行，默认 1；增加进程会提高磁盘争用、同时占用的 staging 空间和内存，不能据单周期小测推断任意并发都可接受。转换脚本不下载网络数据、不解压、不删除源 CSV，也不运行策略训练。重复转换相同源与参数可按 checkpoint 恢复；具体完成位置由各周期状态决定。

每个周期独立记录源路径、字节数、SHA256、请求数/角色行数、源片计数、输出片计数、角色/类型/预算、输出摘要、转换版本、`rules_version` 和 `assumption_profile`；新增 `ignored_missing_field_counts` 区分被丢弃历史字段中的空值。行数与 `request_count * 48` 相等是必要条件，不能替代逐请求身份覆盖检查。转换中断或校验失败的部分文件不能拥有成功 manifest。

用户已授权在 **CRC 校验通过且对应 CSV 完整解压保存后** 删除相应 ZIP，保留解压 CSV。归档清理必须满足：

1. 逐归档核对成员和目标路径，禁止路径穿越或写到目标工作区之外；不覆盖已有不匹配文件。
2. 流式完整读取/解压并验证成员 CRC、解压大小及输出文件摘要；仅列出 ZIP 目录不能证明 CRC 正确。
3. 清单确认本归档所有预期 CSV 都已成功保存后，才删除该**已核实的归档绝对路径**。出错、缺成员、大小不符、输出未完成时保留 ZIP。
4. CRC 用于检测传输/存储损坏，不是来源真实性认证。清单应同时保留 ZIP/CSV 的 SHA256 和官方来源地址；解压与转换的成功状态分开记录。

本次 `extraction_manifest.json` 已记录 11 个归档均为 `verified_and_deleted`，21 个源 CSV 已保留。该结论来自执行清单，不能只由用户授权推导；它也不代表后续 CSV 语义校验或转换已完成。

## 尚需在产物中确认的内容

应以小型人工 fixture 验证整数片映射、顺序、跨周期/跨源片 ID、pvIndex 跨源片重置仍保留为独立机会、同源片重复仍拒绝、完整 48 角色、非法数值及缺失行，并区分“历史 bid 为空可保留并计数”与“pValue 等必需字段缺失必须拒绝”，再开展修复后的全量扫描；大于 1 的源值按前述诊断政策单独核对。全量耗时、内存、校验计数与压缩包清理结果必须记录真实测量值。

即使转换验证全部通过，仍然保留以下适配差距：模拟源数据、概率评分单位、推断的 72 片时点、本地预算政策、未知官方 Agent 签名和对手 PID。有效结论是“这批 AuctionNet 原生预估值在已记录的本地假设下可用”，不是“已重现华为官方环境”或“已通过官方验收”。
