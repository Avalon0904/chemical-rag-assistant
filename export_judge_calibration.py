# export_judge_calibration.py
# judge 标定抽样导出：人工裁决工作表生成器（judge 假阳性率标定的第一步）。
#
# 抽样设计：
#   A. closed-book 侧：156 条 unsupported 按题分层——每题各抽 1 条。实际含 unsupported 的
#      题共 31 个，故 A=31（全题覆盖）。放弃"固定 20 条"：按题分层可避免 q030 的 66 条
#      淹没其他题，且全题覆盖比随机 20 条更抗题目级聚类偏差，人工成本仅多 ~15 分钟。
#      q030 单独补抽 5 条（机理断言是 judge 尺度可疑重灾区，过采样）。
#   B. RAG 侧：7 条 unsupported 全量（headline 数字的直接分母，一条都不能漏）。
# 固定 random.seed(42)，可复现。
#
# 产物：judge_calibration.md（人工判定工作表）+ judge_calibration.json（机读版，回收判定后算指标）

import json
import random

SEED = 42
CB_PATH = "eval_closedbook_faithfulness.json"
RAG_PATH = "eval_faithfulness.json"
OUT_MD = "judge_calibration.md"
OUT_JSON = "judge_calibration.json"

rng = random.Random(SEED)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    cb = load(CB_PATH)
    rag = load(RAG_PATH)

    # A. closed-book：按题分组的 unsupported
    by_q = {}
    for r in cb["details"]:
        if r["status"] != "ok":
            continue
        uns = [c for c in r["claims"] if c["verdict"] == "unsupported"]
        if uns:
            by_q[r["id"]] = uns

    # 分层：每题各抽 1 条（含 unsupported 的题全覆盖，实测 31 题）
    sample_a = []
    for qid, uns in sorted(by_q.items()):
        sample_a.append({"题号": qid, "断言": rng.choice(uns)["claim"]})
    remaining = [
        {"题号": qid, "断言": c["claim"]}
        for qid, uns in sorted(by_q.items())
        for c in uns
        if {"题号": qid, "断言": c["claim"]} not in sample_a
    ]
    rng.shuffle(remaining)
    sample_a += remaining[: max(0, 20 - len(sample_a))]

    # q030 过采样 5 条（排除 A 中已抽的那条，避免重复裁决）
    a_q030 = next((x["断言"] for x in sample_a if x["题号"] == "q030"), None)
    pool = [
        {"题号": "q030", "断言": c["claim"]}
        for c in by_q.get("q030", [])
        if c["claim"] != a_q030
    ]
    q030_extra = rng.sample(pool, k=5)

    # B. RAG 侧 7 条全量
    sample_b = [
        {"题号": r["id"], "断言": c["claim"], "judge_evidence": c.get("evidence", "")}
        for r in rag["details"]
        if r["status"] == "ok"
        for c in r.get("claims", [])
        if c["verdict"] == "unsupported"
    ]

    def md_table(rows, cols):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for i, row in enumerate(rows, start=1):
            lines.append(
                "| "
                + " | ".join(str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols)
                + " |"
            )
        return "\n".join(lines)

    lines = [
        "# judge 标定工作表（人工裁决）",
        "",
        "## 判定纪律（与 judge.py 的 JUDGE_SYSTEM 同款口径，必须严格执行）",
        "",
        "- 只依据 `chunks.json`（全部 35 个 chunk），禁止用你自己的化学知识补缺；",
        "- 语料明确支撑（含语义等价改写）→ **supported**；",
        "- 语料与之矛盾，或断言内容超出语料（编造、无中生有）→ **unsupported**；",
        "- 语料完全没提、无法支撑也无法否定 → **insufficient**；",
        "- 数值必须精确一致才算 supported（单位换算除外），近似即 unsupported。",
        "",
        "判定时打开 `chunks.json` 全文检索关键词逐条核对。判定结果写在每行最后一列，",
        "格式：`supported` / `unsupported` / `insufficient`，可附一句话备注。",
        "",
        "## A. closed-book 侧（分层抽样：含 unsupported 的 31 个题每题各 1 条，全题覆盖）",
        "",
        md_table(sample_a, ["题号", "断言", "人工判定", "备注"]),
        "",
        "## A+. q030 过采样（5 条，机理断言重灾区）",
        "",
        md_table(q030_extra, ["题号", "断言", "人工判定", "备注"]),
        "",
        "## B. RAG 侧（7 条全量，headline 2.92% 的分母）",
        "",
        md_table(sample_b, ["题号", "断言", "judge依据", "人工判定", "备注"]),
        "",
        "## 回收后",
        "",
        "把判定结果告诉我（或直接回写本文件），我用机读版算两个指标：",
        "1. judge 假阳性率 = 人工判 supported 但 judge 判 unsupported 的比例；",
        "2. headline 修正：2.92% × (1 − RAG 侧假阳性率)。",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": SEED, "A_closedbook": sample_a, "A_plus_q030": q030_extra, "B_rag": sample_b},
            f, ensure_ascii=False, indent=2,
        )
    print(f"导出 {OUT_MD} / {OUT_JSON}：A={len(sample_a)} 条，A+={len(q030_extra)} 条，B={len(sample_b)} 条")


if __name__ == "__main__":
    main()
