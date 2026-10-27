# evaluate.py
# 检索效果评估：Hit Rate + MRR，拒答题单独计数
# 用法：先 python build_test_set.py 生成 test_set.json，再运行本脚本

import json
import logging
from pathlib import Path
from typing import List, Dict

from retrieval import vector_retrieve, hybrid_retrieve, hybrid_rerank_retrieve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_set(path: str = "test_set.json") -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"测试集不存在: {path}，请先运行 build_test_set.py")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(
    test_set: List[Dict],
    retrieve_func,
    top_k: int = 5,
    name: str = "baseline",
) -> Dict:
    """
    评估检索效果（must_refuse 题不计入 Hit Rate / MRR，单独计数）。
    retrieve_func: fn(question, top_k) -> list[dict]（含 chunk_pk / doc_name / section / text）
    """
    scored = [t for t in test_set if not t["must_refuse"]]
    refused = [t for t in test_set if t["must_refuse"]]
    total = len(scored)
    hit_count, mrr_sum = 0, 0.0
    details = []

    for item in scored:
        question = item["question"]
        relevant = set(item["relevant_chunk_pks"])

        results = retrieve_func(question, top_k=top_k)
        retrieved_pks = [r["chunk_pk"] for r in results]

        # Hit Rate：任一 gold chunk 出现在 Top-K
        hit = any(pk in relevant for pk in retrieved_pks)
        if hit:
            hit_count += 1

        # MRR：第一个命中的 gold chunk 的排名倒数
        rank = next(
            (i for i, pk in enumerate(retrieved_pks, start=1) if pk in relevant), 0
        )
        if rank > 0:
            mrr_sum += 1 / rank

        # 失败题记录检回内容，用于失败分类（诊断信息必须够，否则无法调优）
        if not hit:
            details.append({
                "id": item["id"],
                "question": question,
                "gold_pks": list(relevant),
                "retrieved": [
                    {
                        "chunk_pk": r["chunk_pk"],
                        "doc": r["doc_name"],
                        "section": r["section"],
                        "snippet": r["text"][:80],
                    }
                    for r in results[:3]
                ],
            })

    hit_rate = hit_count / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0

    result = {
        "name": name,
        "top_k": top_k,
        "total": total,
        "refuse_counted_separately": len(refused),
        "hit_count": hit_count,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "misses": details,
    }

    logger.info(
        f"[{name}] top_k={top_k} | Hit Rate={hit_rate:.4f} ({hit_count}/{total}) "
        f"| MRR={mrr:.4f} | miss {len(details)} 题"
    )
    return result


def save_report(result: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存: {path}")


def main():
    test_set = load_test_set("test_set.json")
    logger.info(f"加载测试集：{len(test_set)} 题")

    # 对比实验：同一测试集、同一评估函数，只换检索函数
    retrievers = [
        ("baseline_vector", vector_retrieve),
        ("hybrid_bm25_rrf", hybrid_retrieve),
        ("hybrid_bm25_rrf_rerank", hybrid_rerank_retrieve),
    ]
    summary = []
    for name, fn in retrievers:
        for k in [1, 3, 5, 10]:
            result = evaluate(test_set, fn, top_k=k, name=name)
            save_report(result, f"eval_{name}_top{k}.json")
            if k == 10:
                summary.append((name, result["hit_rate"], result["mrr"], len(result["misses"])))

    logger.info("===== @10 对比汇总 =====")
    for name, hr, mrr, misses in summary:
        logger.info(f"{name:20s} | Hit Rate={hr:.4f} | MRR={mrr:.4f} | miss={misses}")


if __name__ == "__main__":
    main()
