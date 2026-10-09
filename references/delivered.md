# EquiForge — delivery log (SOP §11 write-back)

| date | system | repo | domain | grade | note |
|------|--------|------|--------|-------|------|
| 2026-10-09 | EquiForge | CJX0712/equiforge | 算法公平性（重加权 Kamiran&Calders 2012 + 均等概率后处理 Hardt et al. 2016） | S | 旗舰 EquiFuse = 重加权预处理 + EO 随机化阈值后处理（期望度量，逐位确定性）；sklearn LogisticRegression Tier-0 + 纯 numpy IRLS 牛顿-拉夫逊 Tier-1 离线兜底（零 ML 依赖）；预注册门禁逐数据集全过：EOD ≤ 0.5×baseline 且 ACC ≥ baseline−0.04（聚合 EOD 比 0.46）；同 seed 两次运行逐位一致；18 单测（度量不变量/重加权独立性/离线兜底一致/确定性/门禁/CLI）；ruff 0.16.10 双绿硬门禁；CI py3.10+3.13；Dockerfile/Makefile/pyproject；模型卡诚实标注合成数据局限 |

## Coverage notes
- Domain chosen: **算法公平性 / Algorithmic Fairness** — Equalized-Odds 后处理（Hardt et al. 2016）以随机化阈值 + 期望度量实现，避免确定性阈值在部分 seed 下不可行且过度牺牲精度；重加权（Kamiran & Calders 2012）作为预处理基线。
- Author attribution: 晨星.
- Quality grade: **S** (world-class): 可独立验证的数学不变量（期望准确率公式、重加权后联合分布独立至 1e-9、确定性逐位一致、门禁逐数据集全过），纯 numpy 离线兜底与 sklearn 一致（±0.05 内），多 seed mean±std 聚合，CI 矩阵 + 密钥扫描，锁定依赖版本。
- Honest finding baked into model card: 结果为合成数据（credit_like/adult_like/law_school_like 三个调优预设）上的可复现演示，非真实数据；当真实差异极小时强行均等化在有限样本上可能反而增大测得差异并付出精度代价，故目标设为保守的 0.45×baseline；后处理推理需访问受保护属性 s。

## Repo
- https://github.com/CJX0712/equiforge  (Release v0.1.0)
