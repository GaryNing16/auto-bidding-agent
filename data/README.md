# 保留的本地数据

- `auto-bidding/`：用户下载的AuctionNet原生预生成数据，21个源CSV保留。11个ZIP此前经授权/核验后删除；这不授权删除源CSV。不是鸿蒙官方数据或真实商业完整拍卖日志。
- `auctionnet_72/`：全部21周期六字段、请求顺序、预算/类别、转换metadata与有界样本保留。原pValue不改，48→72为本地中点映射。源转换profile和新执行规则分开，验证后完整48角色可加载为全true Mask。普通按日内存入口不用于直接读取全量5亿行。详见[原生数据说明](../docs/AUCTIONNET_NATIVE_DATA.md)。
- `real_criteo_sample/source/`：保留真实展示日志小样本、provenance、UPSTREAM_README与LICENSE。旧代理估值/派生六字段及实验代码已归档，不再作为竞价输入。见[Criteo源样本说明](real_criteo_sample/README.md)。
- `organizer_schema_example/`：既有手写格式样例的内部数据，不是主办方真实数据。
- `synthetic_demo/`、`synthetic_tiny/`：固定种子合成开发样例。

Criteo样本和历史修改版本适用 **CC BY-NC-SA 4.0**，与代码Apache许可独立。保留署名、许可、来源与修改说明；不把它当比赛官方数据或任意商业再分发授权。该许可不套用AuctionNet数据；其外部下载说明未单独列明数据许可，本项目只处理用户已有本地副本。

原始数据/已有完整72片数据不因环境重构而重下载、转换或改写。v1完整内部格式验证原指纹后兼容加载；新v2格式支持稀疏值行与显式Mask。数据/运行结果继续按.gitignore管理，忽略不等于可删除；没有自动提交或推送。
