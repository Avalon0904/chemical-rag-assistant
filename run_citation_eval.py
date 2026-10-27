# run_citation_eval.py
# 引用准确率评估（质检第三类，独立实验）：RAG 答案里每个 [n] 指针是否真指向支撑该句的 chunk。
#
# 背景：忠实度评估抓到 1 例引用指针错挂（q014——论断被语料支撑，但 [n] 指向的 chunk 不含此文），
# 发生率未知。本实验全量量化 38 题 RAG 答案的引用质量。
#
# 方法（零新造轮子）：
#   - 数据源 answers_cache.json（answer_text + citations 映射，与忠实度评估同一批答案）；
#   - 解析每个 [n]（含 [n][m] 并列组）所在的论断句；
#   - 复用 judge.judge_claims([论断句], 单 chunk 全文)——"该 chunk 是否支撑该论断"与忠实度判定
#     是同一任务，三分类天然映射：supported=引用正确 / unsupported=引用矛盾 / insufficient_info=错挂；
#   - judge 与被测断言同族（k2.6）的标定注意事项与表 3 相同，结论引用时连口径一起报。
#
# 指标：
#   - micro 引用准确率 = supported 指针 / 全部指针；
#   - 细分：矛盾条数 vs 错挂条数（性质不同：矛盾是严重事故，错挂是可修复的指向失误）；
#   - macro 含错率 = 含 ≥1 个错误指针的答案占比。
#
# 用法：python run_citation_eval.py --env <项目 .env 路径>

import os
import re
import sys
import json
import hashlib
import logging

ENV_PATH = None
if "--env" in sys.argv:
    i = sys.argv.index("--env")
    ENV_PATH = sys.argv[i + 1]

if ENV_PATH:
    from dotenv import dotenv_values

    for k, v in dotenv_values(ENV_PATH).items():
        if v is not None:
            os.environ[k] = v  # 后加载者胜

import argparse  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--env", default=ENV_PATH)
parser.add_argument("--cache", default="answers_cache.json")
args, _ = parser.parse_known_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eval_run.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from judge import judge_claims  # noqa: E402  环境变量就绪后 import

CACHE_PATH = "answers_cache.json"
VERDICT_CACHE = "citation_verdict_cache.json"
RESULTS_PATH = "eval_citation.json"

# [n] 或 [1][3] 并列组；论断句 = 行首或行内上一处标记组之后，到本标记组之前的文本
# （v2：首版按句末标点切，遇到"……根据资料 [3]"这类行尾引用会把论断切剩一个残片）
MARKER_GROUP = re.compile(r"((?:\[\d+\])+)\s*")


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 元话语前缀：模型习惯写"根据资料[n]，<论断>"——[n] 语义上支撑的是后面的论断，
# 若按"行首到标记"切会把"根据资料"这个残片送去判定（v2 已踩过）
FILLER_PREFIX = re.compile(r"^(根据)?资料[:：]?$|^参见[:：]?$|^补充说明[:：]?(根据)?资料[:：]?$")


def _next_marker_start(line: str, from_pos: int) -> int:
    m = MARKER_GROUP.search(line, from_pos)
    return m.start() if m else len(line)


def extract_pointers(answer_text: str) -> list:
    """返回 [{"statement": 论断句, "indices": [n, ...]}, ...]

    粒度约定：以"行"为引用单位——每个标记组的论断 = 该行中自上一个标记组（或行首）
    以来的全部文字。例外：残片是元话语前缀（"根据资料"等）时，论断取标记组之后
    到下一个标记组（或行尾）的文字——该 [n] 语义上支撑的是后文论断。
    """
    pointers = []
    for line in answer_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pos = 0
        for m in MARKER_GROUP.finditer(line):
            statement = line[pos : m.start()].strip(" •-·：:，")
            nums = [int(x) for x in re.findall(r"\[(\d+)\]", m.group(1))]
            if not statement or FILLER_PREFIX.match(statement):
                statement = line[m.end() : _next_marker_start(line, m.end())].strip(
                    " •-·：:，；。"
                )
            if statement:
                pointers.append({"statement": statement, "indices": nums})
            pos = m.end()
    return pointers


def main():
    answers = load_json(CACHE_PATH, {})
    # answers_cache.json 以问题文本为键，test_set.json 提供 问题文本 → 题号 映射
    test_set = load_json("test_set.json", [])
    q2id = {t["question"]: t["id"] for t in test_set}
    verdict_cache = load_json(VERDICT_CACHE, {})
    results = load_json(RESULTS_PATH, {}).get("details", [])
    done = {r["id"] for r in results}
    logger.info(f"答案缓存 {len(answers)} 题，已完成 {len(done)}")

    total = correct = contradiction = wrong_hang = error = 0
    answers_with_error = 0
    details = list(results)

    for question, item in answers.items():
        qid = q2id.get(question, question)
        if qid in done:
            continue
        chunks = {c["index"]: c for c in item["citations"]}
        records = []
        try:
            for p in extract_pointers(item["answer_text"]):
                for idx in p["indices"]:
                    key = hashlib.md5(
                        f"{qid}|{idx}|{p['statement']}".encode("utf-8")
                    ).hexdigest()
                    if key not in verdict_cache:
                        chunk = chunks.get(idx)
                        if chunk is None:
                            verdict_cache[key] = {"verdict": "dangling",
                                                  "evidence": f"[{idx}] 不在上下文编号范围内"}
                        else:
                            context = (f"[{idx}] 来源:{chunk['doc_name']}/{chunk['section']}\n"
                                       f"{chunk['text']}")
                            v = judge_claims([p["statement"]], context)[0]
                            verdict_cache[key] = {"verdict": v["verdict"],
                                                  "evidence": v.get("evidence", "")}
                            save_json(VERDICT_CACHE, verdict_cache)
                    v = verdict_cache[key]["verdict"]
                    records.append({"statement": p["statement"], "index": idx, "verdict": v,
                                    "evidence": verdict_cache[key]["evidence"]})
            details.append({"id": qid, "pointers": records, "status": "ok"})
            save_json(RESULTS_PATH, {"details": details})
            bad = sum(1 for r in records if r["verdict"] in ("unsupported", "insufficient_info", "dangling"))
            logger.info(f"[完成] {qid}：指针 {len(records)}，错挂/矛盾 {bad}")
        except Exception as e:
            details.append({"id": qid, "pointers": records, "status": "error",
                            "error": f"{type(e).__name__}: {e}"})
            save_json(RESULTS_PATH, {"details": details})
            logger.error(f"[失败] {qid}: {e}")

    # ===== 聚合 =====
    for d in details:
        if d["status"] != "ok":
            continue
        ps = d["pointers"]
        total += len(ps)
        correct += sum(1 for r in ps if r["verdict"] == "supported")
        contradiction += sum(1 for r in ps if r["verdict"] == "unsupported")
        wrong_hang += sum(1 for r in ps if r["verdict"] in ("insufficient_info", "dangling"))
        answers_with_error += 1 if any(r["verdict"] != "supported" for r in ps) else 0

    summary = {
        "questions": sum(1 for d in details if d["status"] == "ok"),
        "total_pointers": total,
        "correct": correct,
        "contradiction": contradiction,
        "wrong_hang": wrong_hang,
        "micro_citation_accuracy": round(correct / total, 4) if total else None,
        "macro_contain_error_rate": round(answers_with_error / max(1, sum(1 for d in details if d["status"] == "ok")), 4),
    }
    save_json(RESULTS_PATH, {"summary": summary, "details": details})
    print("\n===== 引用准确率 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
