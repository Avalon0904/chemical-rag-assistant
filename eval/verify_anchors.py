# -*- coding: utf-8 -*-
"""入库后校验：golden_set 里每个非拒答题的 anchor 都能在当前 chunks 中命中。
匹配前对两侧做空白归一化（PDF 换行位置是排版产物，不能作为匹配依据）。"""
import json
import re
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)  # 去掉全部空白后比较，规避 PDF 折行差异


def main(chunks_file=BASE + r"\chunks.json"):
    chunks = json.load(open(chunks_file, encoding="utf-8"))
    golden = [json.loads(l) for l in open(BASE + r"\eval\golden_set.jsonl", encoding="utf-8")]

    misses, checked = [], 0
    for item in golden:
        if item["must_refuse"]:
            continue
        for g in item["gold"]:
            checked += 1
            anchor_n = norm(g["anchor"])
            hit = any(
                c["doc_name"] == g["doc"][:-3] and anchor_n in norm(c["text"])
                for c in chunks
            )
            if not hit:
                misses.append((item["id"], g["doc"][:20], g["anchor"][:40]))

    print(f"校验 anchor 数: {checked}，未命中: {len(misses)}")
    for m in misses:
        print("  MISS:", m)
    sys.exit(1 if misses else 0)


if __name__ == "__main__":
    main()
