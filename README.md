# auto-bidding-agent

面向 2026 Hackathon 广告智能出价赛题的三人协作项目。基于 [AuctionNet](https://github.com/alimama-tech/AuctionNet) 的流量生成器与数据格式，增加符合本题的独立拍卖、训练和评估流程。

**可以立即运行：六字段数据校验 → 72 片拍卖回放 → 固定系数/本地 pacing 对照 → PPO 训练 → 新进程加载评估 → 多策略角色轮换。** 还提供按完整来源日划分数据与供给预测起点。

本仓库保留 AuctionNet 固定版本的生成器与参考代码，不是原版完整 fork。原版拍卖器、可见当前 pValues 的策略、CPA/转化目标不直接用于本题。正式 SDK 到来前，这里是开发与训练系统；官方函数签名和未确认的预算细节仍需对接。[差异与来源](docs/auctionnet.md) · [规则及假设](docs/rules.md) · [三人分工](docs/team.md)

## 10 分钟跑通基础流程

Python 3.10–3.12；首次实测环境为 Python 3.12。基础回放只需要标准库。

```bash
git clone https://github.com/GaryNing16/auto-bidding-agent.git
cd auto-bidding-agent
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m autobid validate --data examples/hand_checked.csv
python -m autobid evaluate --data examples/synthetic_day.csv --config configs/demo.json --output runs/demo --audit
```

查看 `runs/demo/summary.json` 和 `summary.csv`：每个策略轮流代理三个角色，对手是每局重新运行的本地 pacing。逐片 CSV 包含系数、竞胜数、价值、消耗、余额和 `bidding + update` 耗时。`--audit` 额外输出逐请求参与者、报价、赢家和付款，仅供环境调试，策略拿不到这些字段。

基础安装会跳过可选 RL/上游生成器测试；安装以下依赖后完整测试不应跳过。所有 `data/`、`models/`、`runs/` 输出默认不提交 Git。

## 从 AuctionNet 生成准备数据

```bash
python -m pip install -e '.[upstream]'
python scripts/generate_auctionnet.py --output data/auctionnet.csv --seed 7 --requests 1000
python -m autobid evaluate --data data/auctionnet.csv --config configs/auctionnet_demo.json --output runs/auctionnet --strategies fixed:1 pacing
```

此路径实际执行保留的原始 `NeurIPSPvGen`，默认六个广告主。原始生成器产生 48 片，再按片内请求顺序建立均匀伪时间重分为 72 片；同一请求所有参与者共享时间。默认加稀疏参与掩码并将概率值乘 1000 作为代理价值，**这些都是准备期合成假设**，完整记录在旁路 `.manifest.json`，预算也只是可修改的示例配置。正式六字段数据直接进入校验/回放，不需要上述合成处理。

已有真实 AuctionNet 日志时：

```bash
python -m autobid convert-auctionnet --source data/source.csv --output data/period7.csv --period 7 --time-base 0 --max-requests 10000
python -m autobid validate --data data/period7.csv
```

必须先检查实际日志的时间编号起点，再选 `--time-base 0` 或 `1`。每次只转换一个投放期；`--max-requests` 保留完整请求的所有参与者并跨原日抽样，不压缩时间。转换不读取历史 bid/cost/预算轨迹。适配输出 `.manifest.json` 有角色类型、尺度和来源信息；预算、角色类型须与所用数据匹配。公开大数据的来源与使用条件见 [数据适配说明](docs/auctionnet.md)。

## PPO 训练、保存与独立评估

CPU 即可验证工程流程。先装 CPU PyTorch 可避免下载无用的 CUDA 包：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[rl,upstream]'
python -m unittest discover -s tests -v
python -m autobid generate --output data/train.csv --seed 7
python -m autobid generate --output data/validation.csv --seed 17
python -m autobid train --data data/train.csv --config configs/demo.json --player 0 --rotate-roles --timesteps 4096 --output models/ppo
python -m autobid evaluate --data data/validation.csv --config configs/demo.json --strategies fixed:1 pacing ppo --model models/ppo.zip --seeds 7 17 --output runs/validation
```

训练与评估是独立命令/进程。模型包含 `ppo.zip` 和 `ppo.metadata.json`，部署必须一起复制；`ppo.run.json` 保存数据哈希、配置、依赖版本、源代码哈希与 Git 状态。训练可用 `--data day1.csv day2.csv ...` 在完整训练局间采样；`--rotate-roles` 每局抽一个角色。短训练验证运行链路，不表示 PPO 已超过基线。

PPO 使用自身最近四片反馈及时间/预算摘要，共 26 个固定特征；缺失历史有 mask；奖励为本片竞胜 `pvalue` 总和，`gamma=1`。不读取当前片机会数、当前价值、其他广告主状态或未来数据。新增预测特征之前需要同步训练、加载与提交的特征契约。

## 预测、划分与同场竞争

```bash
# 至少三个独立完整来源日；先生成第三个例子
python -m autobid generate --output data/test.csv --seed 27
python -m autobid split --data data/train.csv data/validation.csv data/test.csv --output data/splits.json
# 下例手动指定了既定 train/validation；真实项目应按 splits.json 中的路径运行
python -m autobid fit-forecast --data data/train.csv --config configs/demo.json --output models/supply.json
python -m autobid evaluate-forecast --data data/validation.csv --config configs/demo.json --model models/supply.json --output runs/forecast.json
python -m autobid arena --data data/validation.csv --config configs/demo.json --lineup fixed:1 pacing fixed:2 --output runs/arena
```

供给预测先用训练日的“时间片 + 类别”均值，预测本角色的有效机会数与潜在价值，报告留出日 MAE；它不是逐请求 pvalue 模型，也不是实际竞胜价值模型。预测器尚未加入 PPO。[预测接口与扩展](docs/forecast.md)

`arena` 对配置中的全部角色轮换策略位置，每局重新初始化预算/对手。`evaluate` 是单策略替换对照。所有相对值均明确命名为 **相对本地 pacing**；主办方 PID 未提供。统计采用逐局价值比的平均，基准零值记录为 null、单独计数，不冒充官方计分。

## 数据与模块约定

| 文件/目录 | 用途 |
| --- | --- |
| `src/autobid/data.py`, `adapters.py` | 六字段输入、严格校验、合成数据、AuctionNet 转换 |
| `engine.py`, `types.py` | 按 request_id 独立结算、预算账本、不可变观测与反馈 |
| `policies.py` | 固定系数与自建 pacing 对照 |
| `rl.py` | Gymnasium、统一历史特征、PPO 训练/保存/加载 |
| `forecast.py` | 训练期供给统计、留出误差、来源日隔离 |
| `evaluate.py`, `cli.py` | 角色轮换、评估、日志和可复现命令 |
| `submission.py` | 本地 `Agent.bidding()/update()`；正式 SDK 接口适配位置 |
| `configs/`, `examples/`, `tests/` | 可运行配置、样例与规则验收 |
| `third_party/auctionnet/` | 固定版本的原始来源文件及许可证 |
| `.github/` | 自动测试、任务模板和 PR 模板 |

每份 CSV 表示一个完整投放局，仅六列：`index,time_step,request_id,player_id,player_type,pvalue`。时间片内部用数字 `index` 的请求首次位置确定顺序；若正式数据另有顺序语义，必须更新适配。缺失 `(request_id,player_id)` 表示不参与；存在但非法的 pvalue 报错；合法零值保持零值。预算、日期/投放期和来源位于配置或 manifest，不添加伪造的官方字段。


## 来源及许可

AuctionNet 上游版本：`c20de631a40e24669d1d6a8e2acefd8f492f172d`。保留的上游代码适用 Apache-2.0，见 `third_party/auctionnet/LICENSE.txt` 和 `PROVENANCE.json`。本仓库新增团队代码尚未指定对外开源许可；私有协作不自动授予第三方分发许可。数据的使用条件另行核对。
