# 供给预测起点与数据划分

`autobid.forecast` 提供一个能直接训练、保存、加载和留出评估的统计预测器。它只用**时间片和自身类别**，预测该角色本片的有效请求数和潜在 `pvalue` 总和。所有值都来自训练日的同片同类别均值，暂未接入 PPO。

这两个标签描述机会供给，**不等于竞胜数和实际获得价值**。预测误差下降也不能证明出价策略收益提高。第一版用于建立可检查的预测任务、避免信息泄漏，并给后续回归模型提供比较基准。

## 接口

```python
from autobid.forecast import fit_forecast, SupplyForecast, evaluate_forecast

# config 是 AuctionConfig；每个 CSV 代表一个独立完整逻辑投放日。
fit_forecast(["data/day01.csv", "data/day02.csv"], config, "runs/forecast.json")
model = SupplyForecast.load("runs/forecast.json")
print(model.predict(time_step=30, player_type="type0"))
report = evaluate_forecast("runs/forecast.json", ["data/day03.csv"], config)
```

模型用 JSON 存储，包含训练文件内容校验和和来源标识。预测入口不会读取当前片请求、当前 `pvalue` 或任何对手状态。未见过的类别使用训练集内所有角色的同时段平均值，并返回 `used_fallback=True`。

统计分母包含配置中的每个角色、每个时间片、每个训练日。某角色在某片没有有效记录，供给标签为零；它不会被从均值分母删除。这个“聚合标签为零”与把缺失 `(request_id, player_id)` 补成参拍记录是不同操作。

## 按完整投放日划分

```python
from autobid.forecast import split_by_episode, write_split_manifest

manifest = split_by_episode(day_paths, seed=42, validation_fraction=.2, test_fraction=.2)
write_split_manifest(manifest, "data/splits.json")
train_paths = [item["path"] for item in manifest["splits"]["train"]]
```

划分从完整文件开始，不随机拆分行或时间片。默认三部分都非空，因此至少需要三个独立来源日。比例按来源日数取整，小数据集的实际比例会与输入比例不同。目录名不承担隔离作用：内容相同的文件被拒绝，同一来源投放期的不同重分片/缩放/抽样版本会被分到同组。

适配器旁路文件中的 `source_delivery_period` 与 `upstream_repository` 用于识别来源日；如提供 `source_dataset_id` 则优先使用它。独立生成的模拟日应使用不同的数据集标识（例如含生成种子），同一真实来源日的各种变体必须沿用同一个标识。没有旁路来源信息时只能按内容校验和分组，使用者仍须保证不同 CSV 不是同一天的分片。

训练与评估均拒绝把同一来源日的多个变体重复计算为多个样本。若一个划分组里有多个变体，实验应选择其中一种转换配置。评估会检查路径、内容校验和和来源日三种重叠，并拒绝训练/评估重叠。

## 后续交给数据方向同学

在保持相同整日划分的前提下，增加线性回归或树模型；需要历史特征时只能使用已经发生的自身反馈。历史竞胜数不等于历史总机会数。预处理、类别编码和模型选择只在训练/验证集进行，最终留出集不参与调参。

如果接入预测增强 PPO，训练和推理都必须调用预测器；训练时不能直接输入未来真值。模型和特征版本应一起冻结，再比较有无预测的策略价值。
