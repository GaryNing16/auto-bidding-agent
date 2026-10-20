# AuctionNet 来源、复用和差异

本项目以 [alimama-tech/AuctionNet](https://github.com/alimama-tech/AuctionNet) 为数据与系统结构基础，锁定提交 `c20de631a40e24669d1d6a8e2acefd8f492f172d`。本仓库不是上游的完整镜像：保留可独立运行的请求生成器和接口参考，新增符合当前赛题的数据适配、拍卖、策略及训练评估流程。

## 实际复用路径

1. `third_party/auctionnet/simul_bidding_env/PvGenerator/NeurIPSPvGen.py` 是上游原文件，未改动。
2. `scripts/generate_auctionnet.py` 动态加载该生成器，先生成原生 48 片的多广告主请求价值，再写出上游字段 CSV。
3. `autobid.adapters.convert_auctionnet` 把上游数据转换为六字段赛题请求表，并记录转换参数、来源哈希和时间重分片假设。
4. 新环境按本赛题规则重新竞拍。上游历史的 `bid`、`cost`、`remainingBudget`、曝光与转化结果不会直接充当本地竞拍轨迹。

运行小规模数据路径：

```bash
python -m pip install -e '.[rl]'
python scripts/generate_auctionnet.py --output data/auctionnet-small.csv --seed 7 --requests 1000 --players 6 --categories 3
```

生成三份文件：`auctionnet-small.auctionnet.csv` 为上游字段中间表，`auctionnet-small.csv` 为赛题六字段表，`auctionnet-small.manifest.json` 记录两次转换的来源和假设。`--requests` 是生成器目标规模；上游按片取整后实际数量会略小。

默认以 `pvalue_scale=1000` 把上游转化概率转换为方便实验的代理价值；这只是准备期数值尺度，预算也必须匹配。默认 `participation_rate=0.7` 对参与记录做显式合成抽样，以练习稀疏参与；每个请求至少保留一个参与者。使用 `--participation-rate 1` 可保留生成器原有全体参与关系。该抽样不是对真实 AuctionNet 日志的事实陈述。

## 为什么不能只改一个配置

| 位置 / 事实依据 | 锁定版本的行为 | 本项目处理 |
| --- | --- | --- |
| `config/test.gin` | 48 个时间片、48 个广告主 | 本地默认 72 片，广告主来自独立配置 |
| `NeurIPSPvGen.load_traffic_num_ratio_base` | 流量基础曲线写死为 48 个值 | 保留原生 48 片生成，再显式重分为 72 片 |
| `NeurIPSPvGen.reset` | 重新调用构造函数时只传 episode，其他参数回默认值 | 桥接脚本每次明确构造新对象，不使用其 reset 改造为 72 片 |
| `BaseBiddingStrategy.bidding`、`run/run_test.py` | 决策时传入当前片自身 `pValues` 和不确定性数组，返回逐机会报价 | 策略只读取自身历史，返回本片一个系数；环境才读取当前请求 |
| `AbidBiddingStrategy.bidding` | 用当前 `pValues.mean()` 和 `pValues` 的平方形成报价 | 只作接口参考；本地 `FixedPolicy` 不调用这段策略 |
| `Environment/BiddingEnv.py` | 三槽 GSP，带保留价、曝光和转化采样 | 单槽二价，奖励为竞胜 `pvalue` 之和 |
| `run/run_test.py` | 按片批量竞拍；超预算时丢弃部分报价并重算 | 按请求顺序逐次更新余额；预算保护规则独立配置 |
| `pre_generated_dataset/readme_dataset.md` | `pValue` 原义为曝光后的转化概率，含 18 个原始字段 | 提取六字段请求价值，保留语义与尺度说明 |

源代码链接均以 [固定提交目录](https://github.com/alimama-tech/AuctionNet/tree/c20de631a40e24669d1d6a8e2acefd8f492f172d) 为准。数据说明位于 [预生成数据说明](https://github.com/alimama-tech/AuctionNet/blob/c20de631a40e24669d1d6a8e2acefd8f492f172d/pre_generated_dataset/readme_dataset.md)。数据有多个大分片，应先下载一个投放期并保留完整请求；数据的使用条件需随下载来源一起记录。

## 时间转换的含义

原始文件只有 48 片标签时，没有足够信息恢复真实 20 分钟到达时刻。本适配器在每个源时间片内按请求顺序分配均匀伪时间，再映射到 72 片；同一请求的所有参与者共同转换。它保留请求、参与者和价值，但引入了片内到达分布假设。不能将新时间片视为主办方真实流量。

正式六字段数据无需这次重分片。原数据的时间起点由调用者明确指定，不能根据部分样本自动猜测。

## 上游完整复现入口

若要研究上游原始算法，可在仓库根目录运行：

```bash
python scripts/fetch_auctionnet.py
conda create -n auctionnet-reference python=3.9.12 pip=23.0.1
conda activate auctionnet-reference
cd external/AuctionNet
python -m pip install -r requirements.txt
python main_test.py
```

脚本获取固定提交并以 detached HEAD 打开；目标目录已存在时会停止，避免覆盖已有实验。完整上游包含模型和大数据文件，首次检出与依赖安装可能较大，不是本地赛题环境的必需步骤。上游训练入口参见其 `strategy_train_env/README_strategy_train.md`。

**验证边界：** 本次已运行保留的原始生成器及其六字段适配路径：seed=7，目标 1,000 个请求，实际 977 个请求、4,110 条参与记录；输出可加载为 72 片。未运行上游完整 `main_test.py` 和所有预训练算法。源代码中 `from collections import Iterable` 与上游 `torch==1.12.0` 等旧依赖需要原文档的旧环境；不要把上游依赖直接合并到本项目 Python 3.10+ 环境。这里的复现命令是入口说明，不代表全部原始算法已复现。

## 许可与来源

保留文件的上游路径、Git blob SHA、提交号见 `third_party/auctionnet/PROVENANCE.json`，完整 Apache-2.0 文本见同目录 `LICENSE.txt`。本项目新增代码与保留的上游原文件分开维护，便于检查来源和未来升级。修改保留文件时必须记录变更，不得继续标记为未修改。
