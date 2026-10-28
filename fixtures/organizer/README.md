# 六字段人工样例

这些 CSV 是按用户于 2026-10-22 告知的六字段**手工构造的测试数据**，不是主办方真实数据。
原始表头为 `index,time_step,request_id,player_id,player_type,pvalue`。

- 三个广告主 ID 为字符串 `001`、`002`、`010`，类型分别为字符串 `7`、`2`、`A`；不把类别编号当作大小关系。
- `index` 故意不按数值递增，且不同文件重复；它不用于请求身份或到达排序。
- 训练日期由 `train_part1.csv` 和 `train_part2.csv` 两个分片构成。
- 训练日第一个机会 `r20` 的价值为10/8/11，系数1.2/1/1时人工期望广告主001获胜、支付11、价值10。
- 同片另一请求 `r03` 的行与 `r20` 交错，验证每个请求只结算一次，不能把长表每一行都当作一次机会。
- 不同日期复用 `request_id`，按显式 `request_id_scope: day` 解释，不跨日合并。
- `time_step` 使用1..72，只有1、36、72有请求，其他片仍需完整生命周期。
- `request_order.csv` 是另一份**本地构造的顺序表**，不是根据 index 推断的官方到达顺序。

日期、预算100、完整广告主 roster、时间片起点、请求作用域、文件顺序及整日划分由
`configs/organizer_example.yaml` 显式给出，均不是新增官方参数。

## v2参与语义

此组既有CSV没有为重构改写；它们目前每个请求都有三个player行，加载后Mask均为true，只能验证全覆盖和原顺序路径。v2导入器另外允许缺request/player行：内部补0且Mask=false；真实pvalue=0行仍Mask=true。源空值/NaN/负值仍拒绝，不能当缺席。稀疏/真实零值区别由独立手算测试覆盖，不能据本组全覆盖样例声称已验证稀疏语义。

当前执行为 `harmony-A-B-local-v2` / `local-unified-v2`，alpha没有全局上限；样例1.2/1/1只是人工算例选择。Mask和每人真实机会数留在评测器，不加进策略Observation或普通actor轨迹。字段与规则见 [DATA_SCHEMA](../../docs/DATA_SCHEMA.md)、[RULES](../../docs/RULES.md)。
