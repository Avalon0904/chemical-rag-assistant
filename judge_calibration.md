# judge 标定工作表（人工裁决）

## 判定纪律（与 judge.py 的 JUDGE_SYSTEM 同款口径，必须严格执行）

- 只依据 chunks.json（全部 35 个 chunk），禁止用你自己的化学知识补缺；
- 语料明确支撑（含语义等价改写）→ **supported**；
- 语料与之矛盾，或断言内容超出语料（编造、无中生有）→ **unsupported**；
- 语料完全没提、无法支撑也无法否定 → **insufficient**；
- 数值必须精确一致才算 supported（单位换算除外），近似即 unsupported。

判定时打开 chunks.json 全文检索关键词逐条核对。
表中「模型建议」列由 kimi-k3 按同一口径预填，仅作参考；最终判定以人工裁决为准。

## A. closed-book 侧（分层抽样：含 unsupported 的 31 个题每题各 1 条，全题覆盖）

| 题号 | 断言 | 模型建议 | 人工判定 | 备注 |
|---|---|---|---|---|
| q001 | THF（四氢呋喃）的常压沸点更准确约为 65–67 °C。 | unsupported | unsupported |  |
| q002 | 二氯甲烷（DCM）的沸点约为 39.6 °C（常压，101.3 kPa）。 | unsupported | unsupported |  |
| q003 | 可用：分子筛（3A/4A）适合深度干燥丙酮。 | unsupported | unsupported |  |
| q005 | DMF 的文献常见沸点值范围约 152–154 °C。 | unsupported | unsupported |  |
| q006 | 使用新剪/新锉的镁屑可机械活化镁 | insufficient_info | insufficient_info |  |
| q007 | 格氏反应引发后滴加卤代烃时，体系温度通常控制在 20–40 ℃ | unsupported | unsupported |  |
| q009 | 建议不以固定次数为准 | insufficient_info | insufficient_info |  |
| q010 | 水相需中和至中性后按含盐废水处理 | unsupported | unsupported |  |
| q011 | 格氏试剂本身对卤代烃敏感 | unsupported | unsupported |  |
| q013 | 应使用干燥的氯化钠粉末扑灭金属钠火灾。 | unsupported | unsupported |  |
| q015 | 少量残留钠的安全处理步骤包括隔离，将残留钠置于干燥的沙盘或耐腐蚀容器中，远离水、酸、卤代烃等。 | unsupported | unsupported |  |
| q018 | 无法确定。原因：您未提供具体专利（公开号/申请号）及实施例1的原文内容，我无法检索或核对其中甲苯加氢的收率数据。 | unsupported | unsupported |  |
| q019 | 我无法回答这个问题。 | insufficient_info | insufficient_info |  |
| q020 | 目标产物或关键中间体在 180 ℃ 发生热分解或聚合。 | unsupported | unsupported |  |
| q021 | 问题中未提供对比例 1 的 Pd/C 用量。 | supported | supported |  |
| q022 | 不确定。原因：你没有提供专利号或专利全文，我无法定位"该专利"具体指哪一件，因此无法确认其使用的催化剂及镍负载量。请补充专利公开号/申请号或相关段落，我再据此回答。 | unsupported | unsupported |  |
| q023 | 问题中未给出实验条件。 | insufficient_info | insufficient_info |  |
| q024 | 无法给出2-甲基环己酮的选择性数值 | unsupported | unsupported |  |
| q025 | 高温下副反应（过度氧化、异构化、缩合/聚合）增多，目标产物收率降低。 | unsupported | unsupported |  |
| q026 | 请补充异常类型线索（如收率、纯度、温度、压力、粒径、粘度、颜色、杂质等）。 | unsupported | unsupported |  |
| q027 | 通常首选负载型贵金属催化剂，如 Pd、Pt 负载于 Al₂O₃、SiO₂ 或分子筛载体，低温活性高、芳烃饱和能力强，适用于深度加氢。 | unsupported | unsupported |  |
| q028 | 综述通常推荐负载型贵金属催化剂，尤其是Pd/C（钯碳） | unsupported | unsupported |  |
| q030 | 高温会加剧镍基催化剂的积碳。 | unsupported | unsupported |  |
| q031 | 蒸馏前需先检验并去除过氧化物 | insufficient_info | insufficient_info |  |
| q032 | 甲苯更严格无水时可用4A分子筛干燥 | supported | supported |  |
| q033 | 负载型Pt催化剂（如Pt/Al₂O₃）寿命长 | insufficient_info | insufficient_info |  |
| q034 | 不确定的原因是缺少两份文档的具体内容（实验数据表、专利实施例/权利要求、所述效果数据），无法判断各自理由。 | unsupported | unsupported |  |
| q035 | Ni/Al₂O₃ 催化加氢常用温度为 120–200 °C | unsupported | unsupported |  |
| q036 | 钠是一种碱金属元素 | insufficient_info | insufficient_info |  |
| q041 | 在该温度区间主反应路径占优，竞争副反应活化能更高、升温不利 | insufficient_info | insufficient_info |  |
| q042 | 格氏试剂过夜存放需在干燥的氩气保护下密封存放，隔绝氧气和水分。 | unsupported | unsupported |  |

## A+. q030 过采样（5 条，机理断言重灾区）

| 题号 | 断言 | 模型建议 | 人工判定 | 备注 |
|---|---|---|---|---|
| q030 | 卤素改变镍基催化剂的活性位。 | unsupported | unsupported |  |
| q030 | 原料中H₂S含硫化合物强吸附于Ni活性位。 | insufficient_info | insufficient_info |  |
| q030 | 碳质沉积覆盖镍基催化剂的活性位。 | unsupported | unsupported |  |
| q030 | 上述失活原因为通用归纳。 | insufficient_info | insufficient_info |  |
| q030 | 硫中毒后再生困难。 | unsupported | unsupported |  |

## B. RAG 侧（7 条全量，headline 2.92% 的分母）

| 题号 | 断言 | 模型建议 | 人工判定 | 备注 |
|---|---|---|---|---|
| q013 | 关于水基灭火器与金属钠的具体相容性，语料中没有关于水基灭火器扑救金属钠火灾的专门依据。 | unsupported | unsupported |  |
| q014 | 金属钠储存时须严格防潮防水[5] | supported | supported |  |
| q017 | 语料中没有关于专利实施例 1 中氢气压力的依据 | unsupported | unsupported |  |
| q028 | 综述推荐 Pd/C 催化剂 | unsupported | unsupported |  |
| q031 | 建议查阅该化学品 SDS 或联系 EHS 确认 | insufficient_info | insufficient_info |  |
| q034 | 报告以 170 ℃ 以上选择性下降为依据设定温度上限 | supported | supported |  |
| q034 | 专利以 180 ℃ 实验及催化剂烧结分析为依据设定温度上限 | supported | supported |  |

## 回收后

把人工判定结果告诉我，算两个指标：
1. judge 假阳性率 = 人工判 supported 但原 judge 判 unsupported 的比例；
2. headline 修正：2.92% × (1 − RAG 侧假阳性率)。


---

## 标定结论（2026-10-21，人工判定与 kimi-k3 建议 43/43 一致）

见 judge_calibration_result.json 与测评报告表 3d。
