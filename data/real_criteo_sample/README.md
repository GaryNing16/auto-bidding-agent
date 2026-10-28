# 保留的Criteo真实展示日志样本

真实来源：[Criteo Attribution Modeling for Bidding Dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset)，发布方Criteo。论文：Eustache Diemert、Julien Meynet、Pierre Galland、Damien Lefortier（2017），*Attribution Modeling Increases Efficiency of Bidding in Display Advertising*。

当前只保留 `source/`：sample.csv、provenance.json、UPSTREAM_README.md、LICENSE.txt。固定数据revision为 `904188a63cbad78bee43cd26ff5ee4ac77903986`；样本保存真实源行标识、相对时间、campaign及已展示广告click。原始投影/采样信息和哈希以provenance为准；不是全量拍卖日志，没有所有候选广告主的反事实pvalue。

2026-10-27已按确认清单将旧代理估值实现、训练统计、derived数据与实验入口归档退出活动目录。过去将展示行扩展为共同机会、人工预算/类别及CTR代理pvalue的说明，仅用于解释历史实验，不能作为当前真实竞价数据。正常CLI不再下载或生成该代理链。

旧说明、模型、派生数据和结果保存在工作区[恢复快照](../../../_archive/algorithm_restart_20261027_v1/workspace_snapshot.zip)，成员使用 `snapshot/AuctionNet/data/real_criteo_sample/derived/...`、`snapshot/AuctionNet/docs/REAL_DATA.md` 等原路径。不要通过只恢复旧derived目录使其成为新环境默认数据；未来需要另行确认语义和规则版本。

样本及历史修改版本的数据许可为 **CC BY-NC-SA 4.0**，与代码许可分开。保留[LICENSE.txt](source/LICENSE.txt)、[原始说明](source/UPSTREAM_README.md)、[来源记录](source/provenance.json)及署名/修改说明；不改为Apache许可，不宣称可任意商业再分发。
