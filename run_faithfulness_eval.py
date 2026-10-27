# run_faithfulness_eval.py
# 忠实度评估跑批（阶段 4 第二片）：
#   38 计分题 → answer() 生成（DeepSeek）→ 原子断言切分（Kimi）→ 逐条判定（Kimi）→ 双口径聚合。
# 独立脚本原则：不改 answer.py / judge.py，只调用；答案缓存落盘，重跑不重复花生成钱。
#
# 用法：
#   python run_faithfulness_eval.py          # 正常（项目 .env 指向的 Milvus）
#   python run_faithfulness_eval.py --lite   # Lite 兜底（时序解法见文件内）
#
# 聚合口径（设计定论）：
#   micro 忠实度 = supported / (supported + unsupported + insufficient) —— 断言级
#   幻觉率（micro）= unsupported / 同上总数
#   macro 含幻觉率 = 含 ≥1 条 unsupported 的答案占比 —— 化工安全盯这个

import os
import sys

USE_LITE = "--lite" in sys.argv

# --env <path>：显式加载指定 .env（answer/judge 的 key 在 import 时就读 os.environ，
# 必须在 import 前完成；见下方说明——运行环境可能注入指向别处的同名变量）
ENV_PATH = None
if "--env" in sys.argv:
    i = sys.argv.index("--env")
    ENV_PATH = sys.argv[i + 1]

if USE_LITE:
    # HF_ENDPOINT 需在 sentence_transformers 首次网络请求前设置，import 前最稳
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

if ENV_PATH:
    from dotenv import dotenv_values

    for k, v in dotenv_values(ENV_PATH).items():
        if v is not None:
            os.environ[k] = v  # 后加载者胜；pymilvus 的 load_dotenv override=False 抢不走

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eval_run.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from answer import answer  # noqa: E402
from judge import extract_claims, judge_claims  # noqa: E402

if USE_LITE:
    # 拔 MILVUS_URI 必须发生在所有 import 之后：pymilvus（经 retrieval 引入）
    # 与各模块的 load_dotenv 都可能重新注入，retrieval 已改为
    # 首次连接时才读 URI（懒读取），所以这里拔掉即最终生效。
    os.environ.pop("MILVUS_URI", None)

ANSWERS_CACHE = "answers_cache.json"
RESULTS_PATH = "eval_faithfulness.json"
EARLY_STOP_CONSECUTIVE_ERRORS = 3


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


def build_context(citations) -> str:
    """判定基准 = 该题实际检索回的上下文全文（模型生成时所见即 judge 所审）。"""
    return "\n\n".join(
        f"[{c['index']}] 来源:{c['doc_name']}/{c['section']}\n{c['text']}" for c in citations
    )


def main():
    items = load_scored()
    answers_cache = load_json(ANSWERS_CACHE, {})
    done = {r["id"]: r for r in load_json(RESULTS_PATH, {}).get("details", [])}
    records = []
    consecutive_errors = 0

    for i, item in enumerate(items, start=1):
        if item["id"] in done:
            logger.info(f"[跳过] {item['id']} 已判定")
            records.append(done[item["id"]])
            continue

        try:
            # 1) 生成（走缓存）：答案 + 引用上下文一起缓存，判定基准可复现
            if item["question"] not in answers_cache:
                r = answer(item["question"])
                answers_cache[item["question"]] = {
                    "answer_text": r.answer_text,
                    "citations": [
                        {
                            "index": c.index,
                            "doc_name": c.doc_name,
                            "section": c.section,
                            "text": c.text,
                        }
                        for c in r.citations
                    ],
                }
                save_json(ANSWERS_CACHE, answers_cache)

            cached = answers_cache[item["question"]]
            context = build_context(cached["citations"])

            # 2) 切分 + 3) 判定
            claims = extract_claims(cached["answer_text"])
            verdicts = judge_claims(claims, context)

            record = {
                "id": item["id"],
                "question": item["question"],
                "answer_text": cached["answer_text"],
                "claims": verdicts,
                "status": "ok",
            }
            consecutive_errors = 0
        except Exception as e:
            record = {
                "id": item["id"],
                "question": item["question"],
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            consecutive_errors += 1
            logger.error(f"[失败] {item['id']}: {record['error']}")
            if consecutive_errors >= EARLY_STOP_CONSECUTIVE_ERRORS:
                records.append(record)
                save_json(RESULTS_PATH, {"details": records})
                raise SystemExit(f"连续 {EARLY_STOP_CONSECUTIVE_ERRORS} 题失败，已保存进度，修复后续跑")

        records.append(record)
        save_json(RESULTS_PATH, {"details": records})  # 逐题落盘
        n_sup = sum(1 for c in record.get("claims", []) if c["verdict"] == "supported")
        logger.info(f"[完成 {i}/{len(items)}] {item['id']} ok, 断言 {n_sup}/{len(record.get('claims', []))} supported")

    # ===== 聚合 =====
    total = sup = uns = ins = 0
    hallucinated_answers = answered = 0
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
        if counts["unsupported"] > 0:
            hallucinated_answers += 1

    summary = {
        "answered": answered,
        "total_claims": total,
        "supported": sup,
        "unsupported": uns,
        "insufficient_info": ins,
        "micro_faithfulness": round(sup / total, 4) if total else None,
        "micro_hallucination_rate": round(uns / total, 4) if total else None,
        "macro_hallucination_rate": round(hallucinated_answers / answered, 4) if answered else None,
    }
    save_json(RESULTS_PATH, {"summary": summary, "details": records})
    print("\n===== 忠实度评估 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
