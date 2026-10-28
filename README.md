# 面向鸿蒙生态的广告智能出价：统一本地环境

在 **alimama-tech/AuctionNet `c20de631a40e24669d1d6a8e2acefd8f492f172d`** 工作树中保留上游功能，新增 `harmony_bidding` 作为本赛题唯一活动规则核心。当前支持72片、稀疏参与、标量出价、顺序二价拍卖、预算重拍、动态本地PID、单/多Agent轮换、合法轨迹和指定策略性能测量。

**当前是本地协议，不是主办方评测器。** 用户于2026-10-27指定无全局alpha上限、补零+Participation Mask、不能付款则排除重拍和等概率seed平局；这些本地选择与题面/用户转述分别记录。官方SDK/PID未提供，官方策略入口明确不可用。本地G和相对分不能称官方结果。

## 安装与最小运行

在本项目目录的PowerShell执行。当前工作区已有独立 `.venv`；运行核心不需要GPU、Torch或四个旧研究目录，也不下载完整数据/预训练模型。

```powershell
Set-Location -LiteralPath 'C:\Users\ningy\Desktop\广告智能出价agent\AuctionNet'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m harmony_bidding.cli generate-demo --config configs/demo.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/demo.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-single --config configs/single.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-multi --config configs/multi.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli export-trajectories --config configs/demo.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli benchmark --config configs/benchmark.yaml
.\.venv\Scripts\python.exe -m pytest -q tests
```

新机器需要安装时运行 `./scripts/setup.ps1`（默认使用Python3.14），或 `./scripts/setup.ps1 -Python '已有Python解释器的完整路径'`。它只创建/使用项目 `.venv` 并安装 `requirements-harmony-lock.txt`；不改系统Python。当前基本依赖为PyYAML 6.0.3、psutil 7.2.2、pytest 9.1.1。初始环境快照见 [ENVIRONMENT.json](docs/ENVIRONMENT.json)；实际每次性能报告另记解释器/依赖版本。

已有相同synthetic数据可校验复用，内容不同会报错并要求新data_dir，不覆盖原数据。demo为72片，角色advertiser_1运行ConstantAlphaAgent(alpha=1.2)，其它角色运行LocalPID。N=4、d=3、预算/流量/日期均是本地例值；tiny只用于短机制测试。没有assignments时run为全LocalPID参考。

## 唯一核心与规则边界

```text
六字段/三表/合成数据 → DayData(values + participation_mask)
                                  ↓
Evaluator / run / export / benchmark → runner → AuctionEngine → BudgetPolicy
                                          ↕
                             Agent.reset / bidding / update
```

上游simul_bidding_env与strategy_train_env保留原用途，不是本赛题另一套评测入口。未来Gymnasium只能包装同一核心，不能复制拍卖逻辑。

- 内部0..71对应题面1..72。每片先同步收集所有alpha，再严格按original_order结算，最后各自update。空片和预算耗尽仍完整推进72片。
- alpha为非负有限实数，无全局cap；拒绝字符串、bool、数组、NaN/Inf。平台计算alpha×value，报价溢出明确报错，不截到5或其它上限；算法可自行选择搜索范围。
- 单广告位二价、零报价不参加、单有效者免费。仅Mask=true参与；真实pvalue=0与缺席不同。默认余额0不参与，候选付不起实际二价则仅本请求排除重拍；报价超过余额但能付二价仍合法。
- reward是赢家对应预估价值，无value-cost、CPA/ROI/ROS或转化采样。金额默认6位整数账本、HALF_UP，价值不按货币精度舍入。
- 同价按稳定请求/seed/重拍轮次及规范候选集合派生私有随机源，均匀选取；不按ID大小决胜，不依赖全局随机调用。整片工作账本成功后才提交，失败不部分扣款。
- Observation只有自身配置、片号/总片数、余额和已完成片自身反馈；没有alpha_max、当前/未来真值、Mask/真实机会数、对手或环境句柄。过去片真实机会数也不默认提供。

来源、未知官方细则及版本见 [RULES](docs/RULES.md)、[ASSUMPTIONS](docs/ASSUMPTIONS.md)。当前执行版本为 `harmony-A-B-local-v2` / `local-unified-v2`，v1结果保留历史身份，不能改标签作为新验收。

## 输出、评分与失败状态

每次结果在 `output_dir/<command>/<run_id>/`。CLI stdout返回实际report_directory/run_id；`--output outputs/my-experiment`可指定根目录。局内episode.json、slots.csv、timing.csv保存价值/消耗/余额/调用耗时；评测总表为evaluation.json。

只消费status.json中 `status=completed` 且 `valid=true`、run_id与目录匹配的结果；可用 `runner.read_completed_run(path)`。失败记录failed/ERROR，不沿用旧成功分数。配置无法解析时，错误写 `--output`指定根，未指定则写 `outputs/local/<command>/<run_id>`。报告里的其它角色和诊断数据不属于在线输入。

single逐队逐角色，multi右旋ABC→CAB→BCA；每局新实例/reset，默认新进程，封存模型不得写回。分母为独立全LocalPID同日同角色价值，`G=mean(逐局value/reference)`；零参考严格未定义，缺局/异常不跳过。只给本地40/15相对分，不编造官方队伍集合、未知45分或总分。详细接口、轨迹和性能见 [EVALUATION](docs/EVALUATION.md)。

## 数据与真实输入

六字段 `index,time_step,request_id,player_id,player_type,pvalue` 由用户提供。缺request/player行表示不参与，内部补0且mask=false；真实零值mask=true。源NaN、负数、空值等拒绝。日期、预算、片号起点、request范围和原顺序显式配置，同ID跨日type固定。内部三表v2仅写真实参与值行；已有v1完整数据验证后映射全true Mask，源文件/旧指纹不改。格式见 [DATA_SCHEMA](docs/DATA_SCHEMA.md)。

```powershell
# 首次导入手写六字段样例；已有data_dir时直接运行，重导需另选空目录。
.\.venv\Scripts\python.exe -m harmony_bidding.cli import-organizer --config configs/organizer_example.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/organizer_example.yaml
# 已有原生数据的有界72片样本，无需重下载/重转换。
.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/auctionnet_native_sample.yaml
```

[原生数据说明](docs/AUCTIONNET_NATIVE_DATA.md)：21投放周期原CSV与完整72片转换保留；源pValue不改，48→72是本地中点映射，不是真实20分钟时间戳。全量5亿行不直接交给普通按日内存加载器；当前quickstart使用周期7/21/24的有界48角色样本。源转换profile与本次执行规则分别记录。

Criteo仅保留[真实源样本与CC BY-NC-SA4.0许可](data/real_criteo_sample/README.md)；旧代理估值模型、派生六字段和实验入口已归档，不再作为本环境运行例子。原始数据和用户文档未因规则迁移而改写。

## 写一个Agent

实现 [agent_api.py](harmony_bidding/agent_api.py) 的本地 `Agent`：reset接受OwnConfig/PublicConfig；bidding只返回一个合法alpha；update只使用OwnFeedback。完整可运行例子见 [ExampleConstantAgent](examples/constant_agent.py) 和 [custom_example.yaml](configs/custom_example.yaml)：

```yaml
run:
  assignments: {advertiser_1: example}
  teams:
    example:
      kind: custom
      class_path: examples.constant_agent:ExampleConstantAgent
      kwargs: {alpha: 1.2}
```

```powershell
.\.venv\Scripts\python.exe -m harmony_bidding.cli run --config configs/custom_example.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-single --config configs/custom_example.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli evaluate-multi --config configs/custom_example.yaml
.\.venv\Scripts\python.exe -m harmony_bidding.cli benchmark --config configs/custom_example.yaml
```

`evaluation.teams`与`run.teams`分别配置；默认对手/参考LocalPID。`kind: official`明确不可用；未来官方SDK由OfficialBiddingStrategy适配，不替换核心。自定义权重/统计必须列入sealed_model_files，实际源码/模型封存指纹进报告；普通进程隔离不是恶意代码沙箱。

轨迹版本为harmony-trajectory-v2，观察与实际bidding输入一致，reward为本片价值，只日终done。`trajectory.validate_trajectory_file(path)`完整校验读回；普通read迭代器需消费到EOF。旧七维特征、拟合统计和ReplayBuffer桥已退出，未来新算法从合法Observation自行定义特征。

## 性能、归档与验证说明

benchmark显式配置target_role和agent，实际运行该策略并只统计目标调用；默认constant40、预热2局、测量20局，72片下为1440对调用。报告各P99、配对和P99、两P99之和及初始化；RSS标process_total_rss，包含引擎/对手/数据，不能冒充单Agent内存或官方200ms/4GB认证。

四个旧算法研究目录、历史交付/模型、Criteo派生实验和指定旧脚本/报告已按确认清单退出活动目录。恢复快照在工作区 `_archive/algorithm_restart_20261027_v1/workspace_snapshot.zip`，成员以 `snapshot/`开头；恢复清单/核验文件同目录。不可访问项未绕过权限移除。范围及恢复流程见 [清理确认清单](../cleanup_review/20261027/FINAL_SCOPE_AND_ARCHIVE.md)。归档不是普通运行依赖，不恢复旧算法进行本轮训练。

[上游审计与真实复用](docs/AUCTIONNET_ADAPTATION.md)说明轻量抽象基类继承及重写范围，Apache-2.0许可保留。[本次清理与统一环境交付报告](docs/UNIFIED_ENVIRONMENT_REPORT.md)记录归档恢复、383项测试、72片业务结果、实测性能和剩余假设。历史报告的测试数与G只代表当时版本，不用旧454项通过代替。以下保留完整上游README，不与本赛题入口混用。

---

# 原版 AuctionNet README（保留）

# AuctionNet: A Novel Benchmark for Decision-Making in Large-Scale Games
![Static Badge](https://img.shields.io/badge/license-Apache_License_2.0-red)   &nbsp;&nbsp;&nbsp;&nbsp;  ![Static Badge](https://img.shields.io/badge/version-1.0-green)&nbsp;&nbsp;&nbsp;&nbsp;  ![Static Badge](https://img.shields.io/badge/filed-Decision--Making-blue)&nbsp;&nbsp;&nbsp;&nbsp;  ![Static Badge](https://img.shields.io/badge/organization-Alimama-purple)


---


[//]: # (<p align="center">)

[//]: # (<a href="https://openreview.net/forum?id=OTjTKFk7gb#discussion">Paper</a>)

[//]: # (</p>)

Decision-making in large-scale games is an essential research area in artificial intelligence (AI) with significant real-world impact. **AuctionNet** is a benchmark for bid decision-making in large-scale ad auctions derived from a real-world online advertising platform. AuctionNet is composed of three parts: 

- 🌏️ **Ad Auction Environment**: The environment effectively replicates the integrity and complexity of real-world ad auctions with the interaction of several modules: the ad opportunity generation module, the bidding module, and the auction module. 

- 🔢 **Pre-Generated Dataset**: We pre-generated a substantial dataset based on the auction environment. The dataset contains trajectories with 48 diverse agents competing with each other, totaling over 500 million records and 80GB in size.

- ✴️ **Several Baseline Bid Decision-Making Algorithms**: We implemented a variety of baseline algorithms such as linear programming, reinforcement learning, and generative models.

We note that AuctionNet is applicable not only to research on bid decision-making algorithms in ad auctions but also to the general area of **decision-making** in large-scale games. It can also benefit researchers in a broader range of areas such as reinforcement learning, generative models, operational research, and mechanism design.

## 🔥 News

---

- [**2024-12-14**] 🔥 The **AuctionNet-1.0** code has been officially open-sourced. We welcome everyone to give it a thumbs up and share valuable feedback.
- [**2024-10-24**] 💫  [NeurIPS 2024 Competition: Auto-Bidding in Large-Scale Auctions](https://tianchi.aliyun.com/competition/entrance/532226) has officially ended. The competition attracted more than **1,500 teams** to participate. The **auction environment for evaluation, dataset, baseline algorithms** used in the competition are derived from this project.
- [**2024-09-26**] 🎁 Our paper [AuctionNet](https://arxiv.org/abs/2412.10798) has been accepted by **NeurIPS 2024 Datasets and Benchmark Track**!

## 🥁 Background
<p align="center">
    <br>
    <img src="assets/background.png" width="700"/>
    <br>
</p>

Bid decision-making in large-scale ad auctions is a concrete example of decision-making in large-scale games.  
Numbers 1 through 5 illustrate how an auto-bidding agent helps advertisers optimize performance.  
For each advertiser's unique objective (I), the auto-bidding agent makes bid decision-making (II) for continuously arriving ad opportunities and competes against each other in the ad auction (III).  
Then, each agent may win some impressions (IV), which may be exposed to users and potentially result in conversions. Finally, the agents' performance (V) will be reported to advertisers.


## 🏛︎ Project Structure

---

```
├── config                        # Configuration files for setting up the hyperparameters.
├── main_test.py                  # Main entry point for running evaluations.
├── run                           # Core logic for executing tests.

├── simul_bidding_env             # Ad Auction Environment

│   ├── Controller                # Module controlling the simulation flow and logic.
│   ├── Environment               # The auction module.
│   ├── PvGenerator               # The ad opportunity generation module.
│   ├── Tracker                   # Tracking components for monitoring and analysis.
│   │   ├── BiddingTracker.py     # Tracks the bidding process and generates raw data on ad opportunities granularity.
│   │   ├── PlayerAnalysis.py     # Implements metrics to evaluate the performance of user-defined strategies.
│   └── strategy                  # The bidding module (competitors’ strategies).


├── pre_generated_dataset         # Pre-generated dataset.


├── strategy_train_env            # Several baseline bid decision-making algorithms.

│   ├── README_strategy_train.md  # Documentation on how to train the bidding strategy.
│   ├── bidding_train_env         # Core components for training bidding strategies.
│   │   ├── baseline              # Implementation of baseline bid decision-making algorithms.
│   │   ├── common                # Common utilities used across modules.
│   │   ├── train_data_generator  # Reads raw data and constructs training datasets.
│   │   ├── offline_eval          # Components required for offline evaluation.
│   │   └── strategy              # Unified bidding strategy interface.
│   ├── data                      # Directory for storing training data.
│   ├── main                      # Main scripts for executing training processes.
│   ├── run                       # Core logic for executing training processes.
│   ├── saved_model               # Directory for saving trained models.

```




## 🧑‍💻 Quickstart

---

### Create and activate conda environment
```bash
$ conda create -n AuctionNet python=3.9.12 pip=23.0.1
$ conda activate AuctionNet
```
### Install requirements
```bash
$ pip install -r requirements.txt
```

### Train Strategy & Offline Evaluation
*For detailed usage, please refer to `strategy_train_env/README_strategy_train.md`.*

```
cd strategy_train_env  # Enter the strategy_train directory
```
#### Data Processing
Run this script to convert the raw data on ad opportunities granularity into trajectory data required for model training.
```
python  bidding_train_env/train_data_generator/train_data_generator.py
```

#### Strategy Training
Load the training data and train the xxx (for example, IQL) bidding strategy.
```
python main/main_iql.py 
```

Use the xxxBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
```

#### Offline Evaluation
Load the raw data on ad opportunities granularity to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python main/main_test.py
```

### Online Evaluation
Set up the hyperparameters for the online evaluation process.
```
config/test.gin
```

Run online evaluation.
```bash
# Return to the root directory
$ python main_test.py
```

## 📖 User Case
### Train your own bidding strategy 'awesome_xx'
Refer to the baseline algorithm implementation and complete the following files.
```
├── strategy_train_env
│   ├── bidding_train_env
│   │   ├── baseline
│   │   │   └── awesome_xx
│   │   │       └──awesome_xx.py                # Implement model-related components.
│   │   ├── train_data_generator
│   │   │   └── train_data_generator.py         # Custom-built training Data generation Pipeline.
│   │   └── strategy
│   │       └── awesome_xx_bidding_strategy.py  # Implement Unified bidding strategy interface.
│   ├── main
│   │   └── main_awesome_xx.py                  # Main scripts for executing training processes.
│   └── run
│       └── run_awesome_xx.py                   # Core logic for executing training processes.

```
### Evaluate your own bidding strategy 'awesome_xx'
Use the awesome_xxBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .awesome_xx_bidding_strategy import awesome_xxBiddingStrategy as PlayerBiddingStrategy
```
Run the evaluation process.
```
# Return to the root directory
$ python main_test.py
```


### Generate new dataset
Set the hyperparameters and run the evaluation process.
```
config/test.gin
GENERATE_LOG = True

python main_test.py
```
The newly generated data will be stored in the /data folder.


### Customize new auction environment
We adhere to the programming principles of high cohesion and low coupling to encapsulate each module, making it convenient for users to modify various modules in the auction environment according to their needs.
```
├── simul_bidding_env             # Ad Auction Environment

│   ├── Environment               # The auction module.
│   ├── PvGenerator               # The ad opportunity generation module.
│   ├── Tracker                   
│   │   ├── PlayerAnalysis.py     # Implements metrics to evaluate the performance.
│   └── strategy                  # The bidding module (competitors’ strategies).
```


## 🎡 Implemented Bid Decision-Making Algorithms

---

| Category                  | Strategy                                     | Status |
|---------------------------|----------------------------------------------|--------|
| Reinforcement Learning    | [IQL](https://arxiv.org/abs/2110.06169)      |   ✅     |
|                           | [BC](https://arxiv.org/abs/1805.01954)       |   ✅    |
|                           | [BCQ](https://arxiv.org/pdf/1812.02900)      | ✅      |
|                           | [TD3_BC](https://arxiv.org/pdf/2106.06860)   | ✅      |
| Online Linear Programming | [OnlineLp](https://arxiv.org/abs/2006.16312) | ✅      |
| Generative Model          | [Decision-Transformer](https://arxiv.org/abs/2106.01345)                     | ✅      |
| Generative Model          | [Diffbid](https://arxiv.org/pdf/2405.16141)  | To be implemented    |
| Other                     | Abid (fixed bid rate)                        | ✅      |
|                           | [PID](https://arxiv.org/abs/1603.01055)      | ✅      |


## ✌ Contributing

---

The field of decision intelligence is a fascinating area, and we welcome like-minded individuals to contribute their wisdom and creativity to optimize this project. If you have great ideas, feel free to fork the repo and create a pull request.

1. Fork the project.
2. Create your feature branch (`git checkout -b new-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin new-branch`).
5. Open a pull request.

## 🏷️ License

---

Distributed under the Apache License 2.0. See `LICENSE.txt` for more information.

## 💓 Reference

---

- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)
- [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)



## 🧑‍🤝‍🧑 Contributors

---

Shuai Dou • Yusen Huo • Zhilin Zhang • Yeshu Li • Zhengye Han • Kefan Su  
 • Zongqing Lu • Chuan Yu • Jian Xu • Bo Zheng


## ✉️ Contact

---

For any questions, please feel free to email `doushuai.ds@taobao.com`.



## 📝 Citation

---

If you find our work useful, please consider citing:
```
@inproceedings{
su2024a,
title={AuctionNet: A Novel Benchmark for Decision-Making in Large-Scale Games},
author={Kefan Su and Yusen Huo and Zhilin Zhang and Shuai Dou and Chuan Yu and Jian Xu and Zongqing Lu and Bo Zheng},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://arxiv.org/abs/2412.10798}
}
```
