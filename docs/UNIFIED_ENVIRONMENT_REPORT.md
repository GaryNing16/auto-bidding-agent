# 项目清理与统一环境交付报告

日期：2026-10-27。执行位置：用户本机 Windows 工作区 `C:\Users\ningy\Desktop\广告智能出价agent`；活动仓库为其下 `AuctionNet`。本报告记录本次清理后的实际结果，取代旧审计对当前状态的描述，但不改写历史证据。

## 结论

已按用户最终确认清单完成可访问文件的归档、校验和移除，并完成现有 `harmony_bidding` 的统一环境重构。**最终383项测试通过，0失败、0错误；18条验证命令全部退出0。** 72片演示、单/多Agent评测、指定策略benchmark、自定义Agent和轨迹读回均实际运行。另独立重读32局CSV/JSON，核对预算账本、价值、调用次数、评分和业务复现。

现在可以在同一环境上重新开发策略并作本地比较。官方SDK、官方PID及部分结算细则仍缺失；这不是官方环境验收，也不是正式比赛成绩。尤其“alpha无全局上限”来自本次用户指定的本地协议，与原整理稿“上限由配置给定”有区别。

## 1. 清理及恢复

最终清单：[FINAL_SCOPE_AND_ARCHIVE.md](../../cleanup_review/20261027/FINAL_SCOPE_AND_ARCHIVE.md)。归档位于工作区 `_archive/algorithm_restart_20261027_v1/`，保留本次改动前快照及逐文件恢复清单。

| 项目 | 实际结果 |
|---|---:|
| 主快照文件数 / 原文件字节数 | 64,272 / 16,198,405,075 |
| 补充快照（两个data README） | 2 / 3,243 |
| 归档合计 | **64,274文件 / 16,198,408,318源字节** |
| 主ZIP实际大小 | 16,234,229,812字节 |
| 获批且实际移除 | **64,043文件 / 16,031,475,048字节** |
| 获批清单剩余路径 / 临时排除文件 | 0 / 0 |
| 不可访问、保留未处理目录 | **101** |
| 物理恢复并逐项核对的样本 | 14文件 |

每个归档成员已读回核验SHA256和CRC；移除前再次核对当前文件与归档SHA256。物理恢复样本包括五个历史交付ZIP、两个预测器、核心源码、配置、用户文档及两个数据README。另恢复工具实际完成64,274成员的全量验证模式。历史算法没有重新执行；恢复后旧绝对路径可能需要重新绑定。

主ZIP SHA256：`2aa80b4f37fd7c81d66da3fc4cc312925eb6d0545d65a1a65761d3b6fd4fa56a`。

补充ZIP SHA256：`05a5e751da40bd64eab70431fc9d0884e7ef952379f9d5c775f1aa726baa4ec6`。

最终删除清单SHA256：`91dfbcb7bfb93ce6ef6ddda6d10d7235a256edb29c0a6f39518f3c091e608f49`。

实际退出活动开发的范围：prediction、ppo_baseline、official_baselines、alpha_robustness中获批的算法/模型/专用环境、配置、依赖、测试、日志和交付物，以及Criteo代理估值派生实验、旧七维训练特征/统计/ReplayBuffer桥、指定历史脚本与输出。原五个交付ZIP和模型保存在归档。只逐文件移除获批路径，然后移除空目录；没有用递归父目录删除绕过不可访问项，因此上述研究目录可能仍保留包含不可访问缓存的外壳。

证据：[移除回执](../../_archive/algorithm_restart_20261027_v1/removal_result.json)、[实际移除明细](../../_archive/algorithm_restart_20261027_v1/removed_files.csv)、[恢复清单](../../_archive/algorithm_restart_20261027_v1/recovery_manifest.csv)、[补充恢复清单](../../_archive/algorithm_restart_20261027_v1/supplemental_recovery_manifest.csv)、[不可访问目录清单](../../cleanup_review/20261027/blocked_directories.json)。早期归档验证文件的阶段状态不代表最终移除状态，以移除回执为准。

保留原始数据、既有72片转换数据及溯源、原Criteo样本/许可证、用户DOCX/PDF、上游源码/许可证/Git与核心虚拟环境。清理后核对206个受保护大数据文件，共114,349,033,819字节，大小和mtime均未变；**没有重新计算整套114GB数据的SHA256**。用户文档、上游保留源码和Criteo源样本另作SHA256核对。详细修改文件清单、保护检查和依赖扫描见 [post_cleanup_audit.json](../../cleanup_review/20261027/post_cleanup_audit.json)；未缺失应保留文件、活动Python无旧研究模块导入。

恢复命令在工作区根目录执行，目标必须为新的独立目录，不覆盖活动项目：

```powershell
Set-Location -LiteralPath 'C:\Users\ningy\Desktop\广告智能出价agent'
# 仅验证两个归档及所有成员
.\AuctionNet\.venv\Scripts\python.exe -X utf8 cleanup_review/20261027/restore_verified_archive.py --archive-dir _archive/algorithm_restart_20261027_v1
# 需要恢复时执行；约需原快照大小的额外空间
.\AuctionNet\.venv\Scripts\python.exe -X utf8 cleanup_review/20261027/restore_verified_archive.py --archive-dir _archive/algorithm_restart_20261027_v1 --destination _archive/algorithm_restart_20261027_v1/restored_snapshot
```

## 2. 保留结构及实际复用

```text
广告智能出价agent/
  官方及用户提供的DOCX/PDF                 保留
  report_environment_alignment.md           历史审计，附新报告入口
  cleanup_review/20261027/                  清单、归档/恢复工具、保留检查
  _archive/algorithm_restart_20261027_v1/    已验证历史快照
  AuctionNet/
    .git/、LICENSE.txt                     原上游Git和许可证
    simul_bidding_env/、strategy_train_env/、run/、config/
                                           上游原功能，非本赛题活动入口
    harmony_bidding/                        唯一本赛题规则、API、评测基础
    configs/                               tiny/demo/single/multi/benchmark等
    examples/constant_agent.py              自定义Agent接口示例
    fixtures/                              手写稀疏、tiny和六字段样例
    data/                                  原始/既有转换数据和小型开发数据
    tests/、scripts/、docs/                  验证、工具与规则说明
    .venv/、requirements-harmony-lock.txt    独立CPU基础环境
    outputs/                               按命令和run_id隔离的新运行证据
```

真实继承关系：`agent_api.Agent`继承上游轻量 `simul_bidding_env.strategy.base_bidding_strategy.BaseBiddingStrategy`，重写初始化与reset/bidding签名，增加update，不继承CPA状态。拍卖、预算、轨迹和评分均按本题重写/修复，不调用上游三广告位引擎、原模型或Tracker。详见 [AUCTIONNET_ADAPTATION.md](AUCTIONNET_ADAPTATION.md)。原上游README部分保持原文；原上游Torch训练未安装或执行，本次不能据此宣称其完整训练栈通过测试。

主要保留文件改动：schema/data/organizer_data处理稀疏关系与跨日类型；auction/budget_policy处理统一结算；agent_api/agents/runner管理可见状态和局生命周期；evaluation/trajectory/benchmark/cli/provenance管理评分、记录、轨迹及实测；7份原配置迁移v2；prepare_auctionnet_sample与verify_local更新；机制/数据/评测/API测试补充；README、AGENTS和规则文档更新。新增手写稀疏fixture、自定义接口示例及统一链路验收测试。完整逐文件清单见上述清理后审计JSON。

## 3. 唯一执行链

```text
六字段/内部三表/合成数据
          ↓ 校验、按原始顺序组织
DayData(values + participation_mask)
          ↓
CLI run/export ─┐
Evaluator ──────┼─→ runner.run_episode → AuctionEngine.run_slot → BudgetPolicy
benchmark ──────┘           ↕
                 Agent.reset / bidding / update
```

本赛题只有 `harmony_bidding/auction.py:75` 的AuctionEngine负责权威结算；旧研究的独立拍卖器已退出。未来训练/Gym包装必须调用同一链，不复制竞价逻辑。上游原环境仍保留供其原用途，未成为比赛入口。

| 规则/问题 | 来源与实现 | 代码证据、验证结果 |
|---|---|---|
| 72片，先全部bidding再竞价再各自update | 题面A；空片、耗尽仍调用 | `runner.py:40`；test_runner完整生命周期、同步时序通过 |
| 自身历史观察，标量动作 | A/B；冻结自身对象，不传对手/当前请求/Mask | `agent_api.py:14`、`runner.py:40`；观察等价与轨迹一致测试通过 |
| alpha无全局上限，严格数值检查 | **本次用户D本地约定**；不裁剪，乘积溢出报错 | `auction.py:47`、`schema.py:111`；0/1/5/40/100/更大合法值、bool/string/非有限及溢出测试通过 |
| 同request，仅存在记录者参与 | B2/C确认仅有记录者参与；补零+显式bool Mask为D指定的本地内部表示，真零保留true | `schema.py:34`、`organizer_data.py:132`、`data.py:285`；F-01修复、稀疏往返及手算通过 |
| 二价、单人免费、费用可高于价值 | A；正有效报价，赢家独付，不加ROI/ROS/CPA | `auction.py:89`；A出12/B8/C11，A付11得10通过 |
| 付不起排除重拍、按实际二价判断 | D本地约定，零余额默认不参与另属本地假设 | `budget_policy.py:19`；高bid可支付、连续排除、重算二价与顺序耗尽通过 |
| 同价等概率、确定性重放 | D要求；SHA256键控私有Random/randrange为本地协议 | `auction.py:59`；身份/调用顺序、全局RNG扰动、重拍轮次和固定种子分布诊断通过 |
| 非法输入不部分扣款 | D；整片工作账本全部成功后提交 | `auction.py:89`；末广告主坏动作、后续坏请求及溢出账本不变通过 |
| reward=竞胜pvalue之和 | B；无点击/转化/曝光抽样 | `auction.py:89`、`runner.py:40`；手算、CSV账本和轨迹通过 |
| 每局重置、模型封存 | A；新实例/reset，评测默认spawn | `evaluation.py:116`；模块可变状态、日期/局顺序置换、重复运行、封存资产检查通过 |
| 旧成绩残留F-04 | 独立run_id、原子状态记录、完整性读回 | `runner.py:234`、`:275`；失败、预检失败、缺结果、外来指纹报告均拒绝 |
| 跨日player_type F-06 | C；通用导入、写入、读取和直接DayData评测共享登记校验 | `data.py:57`、`evaluation.py:224`；跨日类型冲突测试通过 |
| 轨迹契约F-07 | 严格字段白名单、类型、时序、身份、账本、终止完整性 | `trajectory.py:50`；未知字段、bool/string动作、断流、假done、不可能零价值竞胜等拒绝 |
| 真实指定Agent性能F-08 | 目标策略单独统计，对手仅诊断 | `benchmark.py:35`；自定义策略、官方不可用、预热剔除及样本数测试通过 |
| 单/多角色轮换与G | A公式；当前分母明确为LocalPID | `evaluation.py:35`、`:47`、`:88`；ABC/CAB/BCA、d=1/d=N、G=1与G=1.05算例通过 |

表中代码路径均相对 `harmony_bidding/`。详细来源A/B/B2/C/D见 [RULES.md](RULES.md)，全部本地假设ID、替换点见 [ASSUMPTIONS.md](ASSUMPTIONS.md)。

F-02/F-03对应的旧研究环境已移除；统一引擎补上不部分更新与不泄露对手的保障。F-05对应研究快速缓存也已退出活动链，未把其trusted_static行为带入新入口。另修复day_id报告路径越界，以及缺少evaluation/benchmark实际结果却被读取为成功的情况。

普通训练轨迹包含实际片前observation、alpha、reward、cost、next_observation、done、episode/advertiser身份及版本信息；**不包含Mask或真实机会数**。Mask与按广告主计算的机会数保存在数据/评测器诊断中，支持离线分析，不能混入actor输入。去除了旧七维特征和ReplayBuffer桥，未来改观察后必须重建训练数据/统计。

## 4. 实际运行与测试

解释器：`AuctionNet\.venv\Scripts\python.exe`，CPython **3.14.5**；Windows 11 10.0.22631，AMD64 Family25 Model80，8物理/16逻辑CPU，RAM 33,720,774,656字节。核心依赖PyYAML **6.0.3**、psutil **7.2.2**、pytest **9.1.1**；完整锁定版本见 [requirements-harmony-lock.txt](../requirements-harmony-lock.txt)。已有venv继续使用，本轮未修改系统Python、未增加GPU/Torch依赖。沙箱启动该venv曾返回AccessDenied，按执行权限流程获准后正常运行，未绕过ACL。

上游 `https://github.com/alimama-tech/AuctionNet.git`，HEAD保持 **c20de631a40e24669d1d6a8e2acefd8f492f172d**。初始浅克隆/稀疏检出和既有未提交变更记录于 [git_evidence.json](../../cleanup_review/20261027/git_evidence.json)。本轮未reset/clean/commit/push；未重新下载或转换完整数据。

实际最终命令（工作目录AuctionNet）：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/verify_local.py --output outputs/unified_validation/20261027_final_v3
```

该脚本实际执行pip check、CLI help、tiny生成/运行、72片demo生成/运行、single、multi、轨迹、benchmark、手写六字段导入/运行/轨迹、自定义Agent运行/single、既有原生数据样本运行、demo重复及pytest。完整命令参数、退出码、耗时和结果目录见 [validation.json](../outputs/unified_validation/20261027_final_v3/validation.json)。

| 验证 | 结果 |
|---|---|
| 最终pytest全基础测试 | **383通过，0失败，0错误；JUnit 104.419秒** |
| 命令集 | 18/18退出0；15次CLI均有各自匹配的completed run_id |
| 72片生命周期 | demo全部4角色均72次bidding、72次update，日终done |
| 独立结果重读 | 32局，CSV成本/余额/价值/竞胜数与JSON一致 |
| demo轨迹 | 288条，导出后严格读回通过 |
| 手写稀疏72片场景 | A赢3次、成本13、价值17、余额87；B/C无扣款 |
| 可见性/状态隔离/非法输入 | 观察等价、冻结对象、重置、独立进程、无部分账本更新通过 |
| 评分/失败状态 | G手算、零参考未定义、缺失/异常局、报告完整性通过 |
| 业务复现 | 相同数据/配置/种子/源码结果相同，排除耗时与run_id |

JUnit证据：[pytest.xml](../outputs/unified_validation/20261027_final_v3/pytest.xml)。手写fixture并非由待测引擎生成：7行/4请求，A/B/C机会数为3/2/2，含C的真实零值；A先付11得10，再付2得3，最后独赢免费得4，后续71片空片。测试分别检查机会数只出现在内部诊断、没有进入Agent观察和反馈。

中间两次验证记录也保留：第一次新增测试误写advertiser_0（实际ID为advertiser_1）导致1项失败，已修正；第二次重复导入同一新fixture目录触发正确的禁止覆盖保护，随后验证脚本改为每次独立数据目录。最终v3全通过，没有删除失败规则测试或关闭覆盖保护。历史454项不是本次测试结论。

## 5. 72片业务结果及本地评分

demo日期2026-10-01，4个广告主；advertiser_1为ConstantAlphaAgent(alpha=1.2)，其余为LocalPID。

| 角色 | 初始预算 | 消耗 | 剩余 | 价值 | 竞胜数 |
|---|---:|---:|---:|---:|---:|
| advertiser_1 | 218.384778 | 218.323465 | 0.061313 | 273.979176 | 331 |
| advertiser_2 | 186.874325 | 186.798866 | 0.075459 | 468.374297 | 401 |
| advertiser_3 | 178.261899 | 178.239001 | 0.022898 | 480.736801 | 310 |
| advertiser_4 | 126.248804 | 126.222674 | 0.026130 | 264.254146 | 135 |

single/multi配置使用local_holdout，d=3；每队各3局。A=LocalPID、B=Constant(1.0)、C=Constant(1.4)。全LocalPID参考独立运行，对战中的PID每局依据当局反馈重新更新。

| 队伍 | G_single | G_multi | 本地single相对分/40 | 本地multi相对分/15 |
|---|---:|---:|---:|---:|
| A | 1.000000000 | 1.265420645 | 40.000000 | 15.000000 |
| B | 0.795809013 | 0.793701037 | 31.832361 | 9.408346 |
| C | 0.617997393 | 0.628036461 | 24.719896 | 7.444597 |

A在single中同PID替换G=1；multi的对手组合改变，G不要求仍为1。以上仅证明本地流程可用，不表示策略优化或官方排名。自定义示例alpha1.2也完成运行和single，G=0.686328628；它演示接口接入，不是训练算法。

## 6. 性能实测

实测目标为 **advertiser_1 / ConstantAlphaAgent(alpha=40)**，不是用LocalPID替代目标性能。2局预热、20局测量、每局72片，目标每函数1440样本；对手4320样本另外报告，不混入目标分位数。一个Agent执行线程，一个RSS采样线程；进程结束时共有7线程。

| 目标指标 | 实测 |
|---|---:|
| P99(bidding) | 0.0009 ms |
| P99(update) | 0.0010 ms |
| 同角色同片配对 P99(bidding+update) | 0.0019 ms |
| P99(bidding)+P99(update) | 0.0019 ms |
| 最大配对耗时 | 0.0032 ms |
| 初始化/reset P99（20样本，单列） | 0.0604 ms |
| process_total_rss 采样峰值 | 37,359,616字节 |
| Windows进程生命周期peak_wset | 37,543,936字节（约35.80 MiB） |

计时用perf_counter_ns，仅包围bidding/update函数，不含拷贝、拍卖和IO；P99使用nearest-rank ceil(0.99*n)。内存用psutil，2ms间隔，共171样本；包含Python、引擎、对手、数据、日志和预热，短暂峰值可能漏采，OS peak_wset覆盖进程生命周期。**single_agent_rss=null**，不能解释为单Agent内存。固定系数函数极简单，这组数据不能代表未来预测/RL模型或官方200ms/4GB验收；需对实际新Agent重跑benchmark。初始化只有20样本，是局部诊断。

## 7. 最少运行及接入命令

当前工作区不需重新安装。常用命令：

```powershell
Set-Location -LiteralPath 'C:\Users\ningy\Desktop\广告智能出价agent\AuctionNet'
.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/demo.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-single --config configs/single.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-multi --config configs/multi.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli export-trajectories --config configs/demo.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli benchmark --config configs/benchmark.yaml
# 示例自定义Agent采用相同接口及规则
.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/custom_example.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-single --config configs/custom_example.yaml
# 全流程验证每次用新的输出目录；不会复用旧成功报告
.\.venv\Scripts\python.exe -X utf8 scripts/verify_local.py --output outputs/unified_validation/my_new_run
```

stdout给出report_directory。该目录下status.json标明本次状态；每局episode.json/slots.csv/timing.csv记录预算、价值与耗时；评测总表evaluation.json保存逐局比和G。不要用上一次运行目录推断本次成功，程序化读取使用 `runner.read_completed_run`。

新策略继承Agent并实现reset/bidding/update，参考 [examples/constant_agent.py](../examples/constant_agent.py)。在配置中使用 `kind: custom`、`class_path: examples.constant_agent:ExampleConstantAgent`、`kwargs: {alpha: 1.2}`；换成自己的模块类即可接入相同Evaluator。只在reset初始化本局状态，只从Observation/OwnFeedback学习；模型文件应作为封存资产声明，评测不能写回。正式SDK到达后实现OfficialBiddingStrategy薄适配；当前`kind: official`明确失败。

## 8. 剩余边界

- 官方PID、SDK函数签名、完整队伍集合、官方错误处置和零参考计分尚未获得；当前LocalPID及G只供本地比较。
- 用户D指定无alpha全局上限、预算重拍、Mask和均匀平局；其中待官方确认的细节不能标成官方。原整理稿上限条款仍保留来源记录。
- 金额默认6位、HALF_UP、整数账本；报价先量化再作正报价判断，极小正报价可能量化为0。这属于本地金额假设，不是主办方已确认。预算精度错误拒绝；价值不按货币精度舍入。
- 默认零余额不参与（可配置）、具体同价随机协议、种子跨局匹配、数据日期/顺序/片号来源仍按ASSUMPTIONS配置记录。
- 原AuctionNet的48→72转换是既有本地时间映射，不能恢复真实20分钟到达时刻；已有源数据和转换结果本轮保持不变，v1完整数据验证旧指纹后只读适配全true Mask。主办方真实稀疏数据应走六字段入口。
- 冻结Python对象/新进程是信息接口与生命周期隔离，不是恶意代码安全沙箱；引擎账本事务不回滚自定义策略的任意外部副作用。
- 不可访问的101目录没有处理；原大数据仅完成大小/mtime保护核对，没有全量重新哈希。归档完整性与已恢复样本检查的范围如上，不扩大解释。
- 本轮没有恢复或训练旧模型，没有运行大规模原生数据压力测试，没有验证上游Torch训练功能，也没有生成正式比赛总分。

后续算法开发应从此v2协议和新轨迹开始，先实现策略接口，通过同一组机制测试和Evaluator，再讨论算法收益。
