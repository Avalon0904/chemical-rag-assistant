# build_test_set.py
# 把 golden_set.jsonl（anchor 锚定）映射为 test_set.json（chunk_pk 锚定）
# 每次重跑 ingest / 改切分策略后，先运行本脚本再生测试集，再跑 evaluate.py

import json
import re
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def norm(s: str) -> str:
    """空白归一化：PDF 折行位置是排版产物，不能作为匹配依据。"""
    return re.sub(r"\s+", "", s)


def main(
    golden_file: str = "eval/golden_set.jsonl",
    chunks_file: str = "chunks.json",
    out_file: str = "test_set.json",
):
    chunks = json.load(open(chunks_file, encoding="utf-8"))
    golden = [json.loads(l) for l in open(golden_file, encoding="utf-8")]

    # 预归一化 chunk，避免每题重复计算
    normed = [
        {
            "chunk_pk": c["chunk_pk"],
            "doc_name": c["doc_name"],
            "text_n": norm(c["text"]),
        }
        for c in chunks
    ]

    test_set, unmapped = [], []
    for item in golden:
        entry = {
            "id": item["id"],
            "question": item["question"],
            "must_refuse": item["must_refuse"],
            "relevant_chunk_pks": [],
        }
        for g in item["gold"]:
            anchor_n = norm(g["anchor"])
            doc_stem = g["doc"].rsplit(".", 1)[0]  # 去扩展名，匹配 doc_name
            pk = next(
                (c["chunk_pk"] for c in normed
                 if c["doc_name"] == doc_stem and anchor_n in c["text_n"]),
                None,
            )
            if pk:
                if pk not in entry["relevant_chunk_pks"]:
                    entry["relevant_chunk_pks"].append(pk)
            else:
                unmapped.append((item["id"], g["doc"], g["anchor"][:40]))

        if not item["must_refuse"] and not entry["relevant_chunk_pks"]:
            logger.warning(f"{item['id']} 没有映射到任何 chunk，仍计入测试集（预计必 miss）")
        test_set.append(entry)

    Path(out_file).write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    scored = [t for t in test_set if not t["must_refuse"]]
    logger.info(
        f"生成 {out_file}：共 {len(test_set)} 题（计分 {len(scored)}，拒答 {len(test_set) - len(scored)}）"
    )
    if unmapped:
        logger.error(f"有 {len(unmapped)} 个 anchor 未映射到 chunk：")
        for u in unmapped:
            logger.error(f"  {u}")
    else:
        logger.info("全部 anchor 映射成功")


if __name__ == "__main__":
    main()
