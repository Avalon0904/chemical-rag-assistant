# eval_answer.py
# 阶段 4 评估 · 第一片：拒答准确率
# 对 test_set.json 中 must_refuse 题跑 answer()，统计拒答率。
# 口径（设计定论，不可悄悄改）：
#   - 分母 = 成功调用数；调用失败与拒答失败分列，互不冒充；
#   - 层一阈值未标定（None），本片测的是"层二天然拦截率"——
#     生成模型在 prompt 规则 3 下主动声明资料不足的比例；
#   - 逐题落盘，中断可续跑（按 id 跳过已完成题）。

import json
import logging
from pathlib import Path

from answer import answer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_SET_PATH = "test_set.json"
RESULTS_PATH = "eval_answer_refusal.json"
EARLY_STOP_CONSECUTIVE_ERRORS = 3  # 连续失败达到此数 = 疑似系统性故障，中止


def load_must_refuse(path: str = TEST_SET_PATH):
    with open(path, encoding="utf-8") as f:
        test_set = json.load(f)
    items = [t for t in test_set if t.get("must_refuse")]
    logger.info(f"加载测试集 {len(test_set)} 题，其中 must_refuse {len(items)} 题")
    return items


def load_done(path: str = RESULTS_PATH) -> dict:
    """断点续跑：已完成的题按 id 索引返回。"""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)["details"]}


def save(results: list, path: str = RESULTS_PATH):
    called = [r for r in results if r["status"] != "error"]
    refused = [r for r in called if r["status"] == "refused"]
    summary = {
        "total": len(results),
        "called": len(called),
        "errors": len(results) - len(called),
        "refused": len(refused),
        "refusal_rate": round(len(refused) / len(called), 4) if called else None,
        "note": "分母为成功调用数；refused 来自层一闸门，本片层一关闭，"
        "故主要观察 answered 题的 answer_text 是否声明资料不足（层二）",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)


def main():
    items = load_must_refuse()
    done = load_done()
    results = []
    consecutive_errors = 0

    for item in items:
        if item["id"] in done:
            logger.info(f"[跳过] {item['id']} 已完成")
            results.append(done[item["id"]])
            continue

        try:
            r = answer(item["question"])
            record = {
                "id": item["id"],
                "question": item["question"],
                "status": "refused" if r.refused else "answered",
                "refuse_reason": r.refuse_reason,
                "answer_text": r.answer_text,
            }
            consecutive_errors = 0
        except Exception as e:  # 调用失败：记账续跑，不算拒答失败
            record = {
                "id": item["id"],
                "question": item["question"],
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            consecutive_errors += 1
            logger.error(f"[调用失败] {item['id']}: {record['error']}")
            if consecutive_errors >= EARLY_STOP_CONSECUTIVE_ERRORS:
                results.append(record)
                save(results)
                raise SystemExit(
                    f"连续 {EARLY_STOP_CONSECUTIVE_ERRORS} 题调用失败，疑似系统性故障，"
                    f"已保存进度至 {RESULTS_PATH}，修复后重跑可续跑"
                )

        results.append(record)
        save(results)  # 逐题落盘
        logger.info(f"[完成] {item['id']} -> {record['status']}")

    called = [r for r in results if r["status"] != "error"]
    refused = [r for r in called if r["status"] == "refused"]
    print("\n===== 拒答评估结果 =====")
    print(f"题目数: {len(results)} | 成功调用: {len(called)} | 调用失败: {len(results) - len(called)}")
    print(f"层一拒答: {len(refused)} | 层二声明不足（需人工看 answer_text）: {len(called) - len(refused)}")
    for r in results:
        print(f"\n--- {r['id']} [{r['status']}] {r['question']}")
        if r["status"] == "answered":
            print(r["answer_text"][:300])
        elif r["status"] == "refused":
            print(f"拒答原因: {r['refuse_reason']}")
        else:
            print(f"错误: {r['error']}")


if __name__ == "__main__":
    main()
