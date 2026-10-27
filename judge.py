# judge.py
# 忠实度裁判（阶段 4 第二片）：Kimi 对 DeepSeek 生成的答案做原子断言级审计。
# 设计定论：
#   - judge 与生成必须不同源（DeepSeek 生成 / Kimi 裁判），错误模式不相关；
#   - 判定基准 = 该题实际检索回的上下文全文（"只据资料回答"的契约审计）；
#   - 三分类：supported / unsupported（=幻觉）/ insufficient_info（上下文未提及，单列不计入分母）；
#   - temperature=0，JSON 结构化输出，解析失败即重试。

import os
import re
import json
import time
import logging

from dotenv import load_dotenv

# 注意：若 KIMI_BASE_URL 已在系统级设置，
# dotenv 默认 override=False 会让系统变量覆盖项目配置，导致调用 404。
# 项目约定"配置以 .env 为准"，故此处显式 override=True。
load_dotenv(override=True)
logger = logging.getLogger(__name__)

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.6")

MAX_RETRIES = 2
VERDICTS = ("supported", "unsupported", "insufficient_info")

# 采样温度平台约束：k2.6/k3 这类思考模型由平台固定 temperature=1，
# 传 0 会被 400 拒绝（"only 1 is allowed for this model"）。

# k2.6 默认开思考模式：推理 token 可能吃光 max_tokens 导致正文为空（实测 2048 即触发）。
# 切分/判定是格式关键任务，不需要长推理——禁用思考，输出更短更稳。
KIMI_DISABLE_THINKING = os.getenv("KIMI_DISABLE_THINKING", "1") == "1"

# 平台参数耦合（平台约束）：k2.6 思考开 → temperature 仅允许 1；
# 思考关 → 仅允许 0.6。默认值随模式切换，可用 KIMI_TEMPERATURE 显式覆盖。
KIMI_TEMPERATURE = float(
    os.getenv("KIMI_TEMPERATURE", "0.6" if KIMI_DISABLE_THINKING else "1.0")
)

_client = None


def get_client():
    global _client
    if _client is None:
        if not KIMI_API_KEY:
            raise RuntimeError("KIMI_API_KEY 未配置：请在项目根目录 .env 中填入")
        from openai import OpenAI

        _client = OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)
    return _client


def _call(messages, max_tokens=2048) -> str:
    """Kimi 调用，故障三分类重试（与 answer.generate 同一套约定）。"""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
        AuthenticationError,
        RateLimitError,
    )

    client = get_client()
    kwargs = {"max_tokens": max_tokens}
    if KIMI_DISABLE_THINKING:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=KIMI_MODEL,
                messages=messages,
                temperature=KIMI_TEMPERATURE,
                **kwargs,
            )
            content = resp.choices[0].message.content
            # 思考模式吃光 token 时正文为空——按可重试故障处理
            if content is None or not content.strip():
                if attempt < MAX_RETRIES:
                    logger.warning(f"空输出（finish_reason={resp.choices[0].finish_reason}），重试")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ValueError("模型连续返回空输出（思考耗尽了 max_tokens）")
            return content
        except AuthenticationError:
            raise
        except RateLimitError:
            if attempt < MAX_RETRIES:
                time.sleep(delay * 4)
                delay *= 2
                continue
            raise
        except (APIConnectionError, APITimeoutError, APIStatusError):
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def _extract_json(text: str):
    """从模型输出里稳健抽取 JSON（容 markdown 围栏和前后杂音）。"""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 截取第一个 [ 或 { 到末尾匹配的括号
    starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i != -1]
    if not starts:
        raise ValueError(f"输出中找不到 JSON: {text[:100]}...")
    start = min(starts)
    opener = cleaned[start]
    closer = "]" if opener == "[" else "}"
    end = cleaned.rfind(closer)
    if end <= start:
        raise ValueError(f"JSON 括号不配对: {text[:100]}...")
    return json.loads(cleaned[start : end + 1])


CLAIM_SYSTEM = """你是忠实度评估的断言切分器。把给定的答案切分为原子断言列表。

规则：
1. 原子断言 = 最小可验证单元，必须含具体实体、数值、条件或可核对的陈述；
2. 一句话含多个可验证点时必须拆开（如"温度 25–30 ℃，时间 1–2 h"是两条）；
3. 丢弃纯过渡、客套、重复修辞的句子；
4. 保留论断原有的引用编号 [n]；
5. 只输出 JSON 字符串数组，不要任何解释。"""

JUDGE_SYSTEM = """你是忠实度裁判。给定【上下文】与【断言列表】，逐条判定每条断言与上下文的关系：

- supported：上下文明确支撑该断言（含语义改写后的等价表述）；
- unsupported：上下文与之矛盾，或断言内容超出上下文（编造、无中生有）；
- insufficient_info：上下文完全没有提及，既无法支撑也无法否定。

判定纪律：
1. 只依据【上下文】，禁止使用你自己的知识补缺；
2. 数值必须精确一致才算 supported（单位换算除外），近似即 unsupported；
3. 只输出 JSON 数组，每项形如 {"claim": "...", "verdict": "...", "evidence": "上下文中的依据原文片段，unsupported 时写矛盾点"}，不要任何解释。"""


def _call_json(messages, max_tokens=2048):
    """调用并解析 JSON；解析失败追加一条纠偏消息重试一次。
    思考模型在 temperature=1 下偶发格式漂移，重试一次的成本远小于整题记 error。"""
    raw = _call(messages, max_tokens=max_tokens)
    try:
        return _extract_json(raw)
    except ValueError:
        logger.warning("JSON 解析失败，追加纠偏重试")
        raw = _call(
            messages
            + [
                {
                    "role": "user",
                    "content": "你上一次的输出无法被 JSON 解析。请重新输出，"
                    "严格只输出 JSON 本身，不要任何解释、注释或 markdown 围栏。",
                }
            ],
            max_tokens=max_tokens,
        )
        return _extract_json(raw)


def extract_claims(answer_text: str) -> list:
    """把答案切成原子断言列表。"""
    claims = _call_json(
        [
            {"role": "system", "content": CLAIM_SYSTEM},
            {"role": "user", "content": f"【答案】\n{answer_text}"},
        ]
    )
    if not isinstance(claims, list):
        raise ValueError("断言切分输出不是数组")
    return [str(c).strip() for c in claims if str(c).strip()]


def judge_claims(claims: list, context: str) -> list:
    """逐条判定断言对上下文的关系。返回 [{claim, verdict, evidence}, ...]。"""
    verdicts = _call_json(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"【上下文】\n{context}\n\n【断言列表】\n{json.dumps(claims, ensure_ascii=False)}",
            },
        ],
        max_tokens=4096,
    )
    if not isinstance(verdicts, list):
        raise ValueError("判定输出不是数组")
    results = []
    for item in verdicts:
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            verdict = "insufficient_info"  # 解析兜底：不猜，单列
        results.append(
            {
                "claim": str(item.get("claim", "")),
                "verdict": verdict,
                "evidence": str(item.get("evidence", "")),
            }
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = "格氏反应引发后滴加卤代烃时，体系温度应控制在 25–30 ℃ [5]。若放热剧烈，应立即用冰浴降温并减缓滴加 [5]。"
    claims = extract_claims(demo)
    print(json.dumps(claims, ensure_ascii=False, indent=1))
