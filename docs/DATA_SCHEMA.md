# 数据格式、稀疏参与与加载契约

六字段源格式由用户提供：`index,time_step,request_id,player_id,player_type,pvalue`。内部三表格式是本地中间格式，不冒充主办方正式文件规范。当前内部schema为 `harmony-local-csv-v2`；兼容已有完整矩阵v1，但不改写原文件。

## 六字段源数据

| 字段 | 含义与严格处理 |
|---|---|
| index | 非负整数行号，可重复、不连续；不用于请求身份或到达排序。 |
| time_step | 片号，必须显式配置time_step_base为0或1。 |
| request_id | 同一作用域中的同一次拍卖；范围day/time_step显式选择。 |
| player_id | 字符串身份，保留前导零，须属于配置roster。 |
| player_type | 固定无序类别字符串；同一ID跨日期/文件必须相同。 |
| pvalue | 本请求对该player的非负有限预估价值；不限定到[0,1]、不估均值、不采样；存在行但空/负/NaN/Inf等非法值拒绝。 |

六字段表头顺序可不同，也可有额外源列；必需六列必须齐备。每个request/player至多一条，跨文件同值重复也拒绝。只有存在记录的player参与；缺行不是错误，也不制造真实零估值行。不同请求不能混合竞价。

## 内部value与Participation Mask

```text
values = ((10, 8, 11), (3, 2, 0), (4, 0, 0))
mask   = ((True, True, True), (True, True, False), (True, False, False))
```

`Opportunity.values`与`participation_mask`均为tuple，长度等于当日广告主数；Mask成员必须是bool。false列必须填0；真实pvalue=0对应true。不得通过value!=0反推Mask，不得将源NaN或坏值转成false。列索引始终对应advertisers列表顺序。

该例真实参与量为3/2/1。数量是Mask列和，不是矩阵行数，也不是wins。竞价只对true列计算报价，再应用正报价和预算政策。Mask、真实机会量、逐请求矩阵仅供评测器/独立离线资料；不能进入Observation、OwnFeedback或普通actor轨迹。

源稀疏记录→补零矩阵+Mask→仅写回true列应保持请求/player键及真实零值，不改变赢家、价格、价值或预算。Mask包含在v2语义指纹中。

## 六字段导入配置

入口：`import_organizer_csv(config: dict) -> dict`；CLI为 `import-organizer`。示例为 `configs/organizer_example.yaml` 与手写 `fixtures/organizer/`，不是主办方真实数据。

| organizer_import字段 | 契约 |
|---|---|
| sources: [{path,day_id}] | 有序且必填；一个源文件属于显式单日，同一解析路径不能重复或跨日分配。同日可有多个分片。 |
| time_step_base: 0或1 | 必填整数，不按观测最小值推断。 |
| request_id_scope: day或time_step | 必填；day模式同ID跨片报错；time_step模式用片号与源ID组合内部ID。同日跨文件共享作用域。 |
| ordering: {mode:file_first_seen} | 按sources顺序和request首次出现次序给片内顺序；是明确的本地假设，不是官方到达时间。 |
| ordering: {mode:explicit_csv,path:...} | 独立顺序表表头day_id,time_step,request_id,original_order；片号起点同源，不能缺、重复、未知映射或同片重复order。 |
| players: {'001':{player_type:'7',budget:100},...} | 完整角色/固定类型/初始预算，不要求每请求全员出现。ID与类别在YAML加引号。 |
| budgets_by_day | 可覆盖部分声明日期；覆盖日须给全部角色预算。其余日采用players预算。 |
| split_by_day | 可选；提供时精确覆盖全部日期；否则按完整日期划分。 |

顶层data_dir为输出，expected_n若给定须与完整roster一致。相对路径以运行命令的项目根为基准。导入拒绝非空目标，已有内部数据直接运行；需要新导入则选择新的data_dir，不能覆盖原数据。

日期不从文件名、index重置或片号下降推断；预算不从价值或流量估算。请求player行可交错/跨文件，导入器按明确作用域聚合。真正的片内顺序只按original_order，不能按request_id排序。

manifest.import_metadata记录源文件路径/哈希/行数/请求数、配置指纹、顺序、scope/base、参与政策和本地规则来源。来源信息只进报告，不进策略。每源文件request_count之和可能超过同日去重请求总数。全部校验成功才发布manifest；失败的部分CSV不视为完整数据集。

## 本地三表格式

每个完整日期一个分区；manifest指定日期、目录、split、total_slots、advertiser_count及可选内容指纹。CSV表头严格如下，不能漏列/多列或调序。

| 文件 | 字段 | 语义 |
|---|---|---|
| advertisers.csv | day_id,advertiser_id,advertiser_type,budget | 每日角色配置；同ID跨日type固定。 |
| opportunities.csv | day_id,time_slot,opportunity_id,original_order | 每日request唯一；片号0..total_slots-1；同片order非负唯一，可不连续。 |
| values.csv | day_id,opportunity_id,advertiser_id,estimated_value | v2仅保存真实参与记录；缺行加载为0+false，真实0行保留true。重复/未知关联拒绝。 |

空片无需伪造请求。ID必须非空且无首尾空白，预算/价值非负有限。预算须在money_decimals下精确表示，不能向上舍入；CSV先检查原十进制文本，不能让负数或极小正数经float静默变0。

```python
generate_synthetic(config) -> list[DayData]
validate_day(day, rules, expected_n=None) -> None
write_dataset(days, path) -> None
load_days(path, rules, expected_n=None, split=None) -> Iterator[DayData]
import_history_csv(...)  # 通过显式字段映射导入历史三表
```

`write_dataset`只写Mask=true的values行，拒绝非空目标；`load_days`按日期生成器加载。六字段、历史导入、写入及manifest加载统一检查跨日类型。manifest不能指向数据集外目录或重复分区；重复日期/非法split/错误总片数不能静默忽略。

加载器先流式检查manifest全部日期的关联、类型与数值，随后逐个物化选择的日期；最多持有单日矩阵，但初始化仍需读取全部声明数据。**此通用入口不是5亿行全量native数据的内存安全直读入口。** 使用已准备的有界样本进行小规模运行；全量保留在原位置，不为本轮重构重新下载或转换。

## v1兼容与源/执行版本

v1 `harmony-local-csv-v1` 的既有契约要求每请求全员值。加载时仍验证其完整覆盖与原指纹，然后赋全true Mask；不能把v1缺行暗当合法稀疏数据。v2允许稀疏，保存含Mask的新语义指纹。

DayData保留source_schema_version/source_fingerprint，并单独计算当前fingerprint。源指纹使用对应版本规范核验，不能因新增Mask关闭旧校验或重写原manifest。源转换profile和执行rules_version是两件事：native48→72的历史版本原样保留，新运行使用v2规则；新报告同时记录两类来源。

## 训练边界与数据划分

按完整日期划分train、validation、local_holdout；local_holdout不是官方隐藏测试集。未来标准化、预测或价值先验只能从train拟合。通用加载器不拟合统计、不把流量真值给策略。

活动轨迹只导出实际bidding观察、alpha、本片价值reward、cost、next_observation、done及身份/版本元数据。旧七维特征、统计拟合和AuctionNet ReplayBuffer训练桥已退出活动接口。新增算法自行在合法Observation上定义特征，不能将privileged数据伪装为actor输入。

现有数据：

- [原生AuctionNet转换](AUCTIONNET_NATIVE_DATA.md)：21投放周期，原pValue和48→72本地映射保留；不是鸿蒙官方数据。
- [Criteo源样本](../data/real_criteo_sample/README.md)：仅保留真实源样本与许可，旧代理估值派生链已归档，不再作为运行示例。
- [六字段人工样例](../fixtures/organizer/README.md)：明确标记手写来源、原始顺序和测试用途。
