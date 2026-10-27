# run_rewrite_ablation.py
# 改写消融实验：rewrite off vs on，检索链固定为当前默认（混合 + RRF + 重排）。
# 独立脚本原则：不改动 evaluate.py / answer.py，只调用其已验收的函数；
# 结果落盘为独立文件（eval_rewrite_off/on_top{k}.json），与历史实验互不覆盖。
#
# 用法：
#   python run_rewrite_ablation.py            # 正常（项目 .env 指向的 Milvus）
#   python run_rewrite_ablation.py --lite     # Lite 兜底（import pymilvus 后拔 MILVUS_URI）
#
# 成本说明：38 计分题的改写只算一次并缓存，跨 4 个 top_k 复用（152 次检索只花 38 次改写钱）。

import sys

USE_LITE = "--lite" in sys.argv

if USE_LITE:
    import os

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.pop("MILVUS_URI", None)
    import pymilvus  # noqa: F401  触发其内部 load_dotenv（注入 MILVUS_URI）
    os.environ.pop("MILVUS_URI", None)  # 注入后再拔掉

import json
import logging
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("eval_run.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

import evaluate  # noqa: E402
from retrieval import hybrid_rerank_retrieve  # noqa: E402
from rewrite import rewrite_query  # noqa: E402

TOP_K_LIST = [1, 3, 5, 10]
CACHE_PATH = "rewrite_cache.json"  # 改写缓存：实验可重跑不重复花钱


def load_test_set(path: str = "test_set.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_rewrite_cache() -> Dict[str, str]:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def build_rewrites(test_set) -> Dict[str, str]:
    """对全部计分题预计算改写（每题一次 DeepSeek 调用，结果缓存落盘）。
    消融的可比性前提：on 组所有 top_k 共用同一份改写，排除随机性。"""
    cache = load_rewrite_cache()
    scored = [t for t in test_set if not t.get("must_refuse")]
    for i, item in enumerate(scored, start=1):
        q = item["question"]
        if q in cache:
            continue
        cache[q] = rewrite_query(q)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        logger.info(f"[改写 {i}/{len(scored)}] {q[:30]}...")
    return cache


def main():
    test_set = load_test_set()
    rewrites = build_rewrites(test_set)

    def retrieve_off(question, top_k=5):
        return hybrid_rerank_retrieve(question, top_k=top_k)

    def retrieve_on(question, top_k=5):
        return hybrid_rerank_retrieve(rewrites.get(question, question), top_k=top_k)

    for name, fn in [("rewrite_off", retrieve_off), ("rewrite_on", retrieve_on)]:
        for k in TOP_K_LIST:
            result = evaluate.evaluate(test_set, fn, top_k=k, name=name)
            evaluate.save_report(result, f"eval_{name}_top{k}.json")

    logger.info("===== 改写消融汇总 =====")
    for k in TOP_K_LIST:
        off = json.load(open(f"eval_rewrite_off_top{k}.json", encoding="utf-8"))
        on = json.load(open(f"eval_rewrite_on_top{k}.json", encoding="utf-8"))
        logger.info(
            f"top{k:2d} | HR off={off['hit_rate']:.4f} on={on['hit_rate']:.4f} "
            f"| MRR off={off['mrr']:.4f} on={on['mrr']:.4f} "
            f"| miss off={len(off['misses'])} on={len(on['misses'])}"
        )


if __name__ == "__main__":
    main()
