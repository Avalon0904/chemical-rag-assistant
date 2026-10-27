# run_calibration_suggest.py
# judge 标定加速：让模型（默认 kimi-k3，--model 可换）对 judge_calibration.json 的
# 43 条断言出一版"建议判定"，供人工审稿（模型只提建议，人工是否决方）。
#
# 设计要点：
#   - 判定基准与 closed-book 实验一致 = 全部 35 个 chunk 语料，口径与 JUDGE_SYSTEM 同款；
#   - 复用 judge.judge_claims（带重试与 JSON 纠偏），逐条调用便于单条失败不炸全批；
#   - 先探针一条，失败即停——k3 的平台参数约束未验证过（已知有平台参数约束），
#     不把 43 条挨个撞一遍才发现配置错。
#
# 用法：python run_calibration_suggest.py --env <项目 .env> [--model kimi-k3]

import os
import sys
import json
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eval_run.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True)
parser.add_argument("--model", default="kimi-k3")
args = parser.parse_args()

from dotenv import dotenv_values

for k, v in dotenv_values(args.env).items():
    if v is not None:
        os.environ[k] = v  # 后加载者胜

os.environ["KIMI_MODEL"] = args.model  # 覆盖 .env 中的 k2.6

from judge import judge_claims  # noqa: E402  必须在环境变量就绪后 import

SAMPLE_PATH = "judge_calibration.json"
OUT_PATH = "judge_calibration_suggest.json"
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "chunks.json")


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def main():
    chunks = load_json(CHUNKS_FILE, [])
    corpus = "\n\n".join(
        f"[{c['chunk_pk']}] 来源:{c['doc_name']}/{c['section']}\n{c['text']}" for c in chunks
    )
    logger.info(f"判定基准：全语料 {len(chunks)} 个 chunk，judge 模型 = {args.model}")

    # 探针：先确认模型可用 + 参数约束兼容，避免 43 条逐个撞错
    try:
        probe = judge_claims(["探针：本助手基于给定语料回答。"], corpus)
        logger.info(f"探针通过：{probe[0]['verdict']}")
    except Exception as e:
        raise SystemExit(f"探针失败（模型或参数约束不兼容，见 eval_run.log）：{type(e).__name__}: {e}")

    sample = load_json(SAMPLE_PATH, {})
    results = {}
    total = sum(len(sample[g]) for g in ("A_closedbook", "A_plus_q030", "B_rag"))
    i = 0
    for group in ("A_closedbook", "A_plus_q030", "B_rag"):
        rows = []
        for item in sample[group]:
            i += 1
            try:
                verdicts = judge_claims([item["断言"]], corpus)
                v = verdicts[0]
                row = {"题号": item["题号"], "断言": item["断言"],
                       "suggest_verdict": v["verdict"], "suggest_evidence": v.get("evidence", "")}
            except Exception as e:
                row = {"题号": item["题号"], "断言": item["断言"],
                       "suggest_verdict": "error", "suggest_evidence": f"{type(e).__name__}: {e}"}
            rows.append(row)
            logger.info(f"[{i}/{total}] {item['题号']} -> {row['suggest_verdict']}")
        results[group] = rows

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "note": "模型建议判定，供人工审稿；人工是否决方",
                   "groups": results}, f, ensure_ascii=False, indent=2)
    errs = sum(1 for g in results.values() for r in g if r["suggest_verdict"] == "error")
    logger.info(f"完成：{OUT_PATH}，共 {total} 条，错误 {errs} 条")


if __name__ == "__main__":
    main()
