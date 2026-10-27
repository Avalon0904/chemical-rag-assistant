# run_closedbook_eval.py
# closed-book 对照实验：无检索裸生成 × 同一 Kimi judge。
# 目的：给 RAG 忠实度（2.92% micro 幻觉）一个锚——
# 同模型同题不带上下文作答时，语料支撑率掉到多少，差值即 RAG 的幻觉压制价值。
#
# 判定基准（设计决策，勿改）：全部语料（chunks.json 全文 35 chunk）。
# 裸生成不知道哪篇文档相关，公平口径是"断言在知识库任何角落有无依据"，
# 若只对照金 chunk，会把"答对但角度不同"错杀成幻觉。
#
# 用法：python run_closedbook_eval.py --env <项目 .env 路径>
# （不需要 --lite：本实验无检索，不触碰 Milvus 和 embedding）

import os
import sys

ENV_PATH = None
if "--env" in sys.argv:
    i = sys.argv.index("--env")
    ENV_PATH = sys.argv[i + 1]

if ENV_PATH:
    from dotenv import dotenv_values

    for k, v in dotenv_values(ENV_PATH).items():
        if v is not None:
            os.environ[k] = v  # 后加载者胜（运行环境可能注入同名变量）

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eval_run.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from answer import generate  # noqa: E402  复用带重试的 DeepSeek 生成
from judge import extract_claims, judge_claims  # noqa: E402

CACHE_PATH = "closedbook_cache.json"
RESULTS_PATH = "eval_closedbook_faithfulness.json"
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "chunks.json")
EARLY_STOP_CONSECUTIVE_ERRORS = 3

CLOSED_BOOK_PROMPT = """你是化工研发领域的问答助手。请回答下面的问题。
要求：简洁、分点、使用简体中文。如果不确定，请说明不确定的原因。"""


def load_scored(path: str = "test_set.json"):
    with open(path, encoding="utf-8") as f:
        return [t for t in json.load(f) if not t.get("must_refuse")]


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_corpus_context() -> str:
    """判定基准：全语料。chunk_pk 保留，便于逐条 evidence 回溯。"""
    chunks = load_json(CHUNKS_FILE, [])
    return "\n\n".join(
        f"[{c['chunk_pk']}] 来源:{c['doc_name']}/{c['section']}\n{c['text']}" for c in chunks
    )


def main():
    items = load_scored()
    corpus = build_corpus_context()
    logger.info(f"判定基准：全语料 {len(load_json(CHUNKS_FILE, []))} 个 chunk")

    cache = load_json(CACHE_PATH, {})
    done = {r["id"]: r for r in load_json(RESULTS_PATH, {}).get("details", [])}
    records = []
    consecutive_errors = 0

    for i, item in enumerate(items, start=1):
        if item["id"] in done:
            logger.info(f"[跳过] {item['id']} 已完成")
            records.append(done[item["id"]])
            continue

        try:
            if item["question"] not in cache:
                prompt = f"{CLOSED_BOOK_PROMPT}\n\n【问题】\n{item['question']}"
                cache[item["question"]] = generate(prompt)
                save_json(CACHE_PATH, cache)

            answer_text = cache[item["question"]]
            claims = extract_claims(answer_text)
            verdicts = judge_claims(claims, corpus)

            record = {"id": item["id"], "question": item["question"],
                      "answer_text": answer_text, "claims": verdicts, "status": "ok"}
            consecutive_errors = 0
        except Exception as e:
            record = {"id": item["id"], "question": item["question"],
                      "status": "error", "error": f"{type(e).__name__}: {e}"}
            consecutive_errors += 1
            logger.error(f"[失败] {item['id']}: {record['error']}")
            if consecutive_errors >= EARLY_STOP_CONSECUTIVE_ERRORS:
                records.append(record)
                save_json(RESULTS_PATH, {"details": records})
                raise SystemExit(f"连续 {EARLY_STOP_CONSECUTIVE_ERRORS} 题失败，已保存进度")

        records.append(record)
        save_json(RESULTS_PATH, {"details": records})
        logger.info(f"[完成 {i}/{len(items)}] {item['id']}")

    # ===== 聚合 + 与 RAG 版对照 =====
    def aggregate(records):
        total = sup = uns = ins = 0
        hallucinated = answered = 0
        for r in records:
            if r["status"] != "ok":
                continue
            answered += 1
            counts = {"supported": 0, "unsupported": 0, "insufficient_info": 0}
            for c in r["claims"]:
                counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
            total += sum(counts.values())
            sup += counts["supported"]
            uns += counts["unsupported"]
            ins += counts["insufficient_info"]
            hallucinated += 1 if counts["unsupported"] > 0 else 0
        return {
            "answered": answered,
            "total_claims": total,
            "supported": sup,
            "unsupported": uns,
            "insufficient_info": ins,
            "micro_faithfulness": round(sup / total, 4) if total else None,
            "micro_hallucination_rate": round(uns / total, 4) if total else None,
            "macro_hallucination_rate": round(hallucinated / answered, 4) if answered else None,
        }

    summary = aggregate(records)
    rag = load_json("eval_faithfulness.json", {}).get("summary", {})
    comparison = {
        "closed_book": summary,
        "rag": rag,
        "delta_micro_hallucination": (
            round(summary["micro_hallucination_rate"] - rag.get("micro_hallucination_rate", 0), 4)
            if rag
            else None
        ),
        "delta_macro_hallucination": (
            round(summary["macro_hallucination_rate"] - rag.get("macro_hallucination_rate", 0), 4)
            if rag
            else None
        ),
        "note": "负 delta = RAG 相对裸生成的幻觉压制幅度（语料支撑口径）",
    }
    save_json(RESULTS_PATH, {"summary": summary, "comparison": comparison, "details": records})
    print("\n===== closed-book 对照 =====")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
