# answer.py
# 阶段 4：生成与拒答防线
# 数据流：问题 → (可选改写) → 混合检索 top_k → 拒答闸门 → 拼上下文 → DeepSeek 生成 → 逐句引用 [n]

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from dotenv import load_dotenv

from retrieval import hybrid_rerank_retrieve

load_dotenv()
logger = logging.getLogger(__name__)

# ========== 配置（全部走 .env，密钥不落代码库）==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GENERATION_TEMPERATURE = 0.0  # RAG 生成要确定性，创造性是幻觉的近亲

TOP_K = 5  # 进上下文的 chunk 数（评估分层看 @3/@5 对忠实度的影响）

# 拒答闸门开关：None = 关闭。
# 阈值必须在 golden_set 的 must_refuse 题上标定后才能填数——
# fused_score 无绝对语义，拍脑袋填阈值等于在 5 道题上赌博。
REFUSE_FUSED_SCORE_THRESHOLD: Optional[float] = None


# ========== 数据结构 ==========
@dataclass
class Citation:
    """一条引用 = 正文里的编号 [n] + 可回溯的 chunk 元数据。
    text 保留全文：语义层检查（论断数字 vs 原文比对）要能离线做，
    snippet 仅供日志与人审速览。"""

    index: int  # 与拼进上下文的 [n] 严格同号，模型输出里的 [n] 靠它对回原文
    chunk_pk: str
    doc_name: str
    section: str
    text: str  # chunk 全文，供忠实度检查做原文比对
    snippet: str


@dataclass
class AnswerResult:
    """answer() 的唯一返回类型。拒答也是一种合法结果，不是一个异常——
    评估脚本要靠 refused/refuse_reason 统计拒答准确率，异常会打断批跑。"""

    question: str
    answer_text: str
    citations: List[Citation] = field(default_factory=list)
    refused: bool = False
    refuse_reason: Optional[str] = None
    rewritten_question: Optional[str] = None  # 改写模块启用时回填，用于消融对比
    prompt: Optional[str] = None  # dry_run 时返回组装好的 prompt，供审查与调试


# ========== 改写模块（实现已移至 rewrite.py，此处仅导入保持签名稳定）==========
from rewrite import rewrite_query


# ========== 拒答闸门（层一：检索级）==========
def should_refuse(retrieved: List[Dict], threshold: Optional[float]) -> Optional[str]:
    """检索级拒答闸门：最高分低于阈值即拒答，生成根本不发起。

    为什么放检索后、生成前：拒答要省的是 LLM 调用费和幻觉机会，
    放在生成之后就失去了这两个意义。
    为什么只信 top1 的 fused_score：RRF 融合后头部区分度最大，
    且 fused_score 无绝对语义——threshold 未标定前（None）闸门必须关闭。

    返回拒答原因文案（会展示给用户），不拒答返回 None。
    """
    if not retrieved:
        return "未检索到任何相关文档，语料中可能没有可靠依据"
    if threshold is None:
        return None
    if retrieved[0]["fused_score"] < threshold:
        return "检索结果与问题的相关度不足，语料中可能没有可靠依据"
    return None


# ========== Prompt 组装 ==========
def build_prompt(question: str, retrieved: List[Dict]) -> str:
    """检索结果编号拼上下文，要求逐句标注 [n]。

    独立成函数的两个理由：
    1. 它是全流程最该被人眼审查的工件——dry_run 直接把它交出来审；
    2. eval_answer.py 做忠实度判定时要复用同一份上下文做对照。
    """
    context_lines = []
    for i, r in enumerate(retrieved, start=1):
        context_lines.append(
            f"[{i}] 来源：{r['doc_name']} / {r['section']}\n{r['text']}"
        )
    context = "\n\n".join(context_lines)

    # 系统规则四条（阶段 4 设计定稿）：
    # 只据资料回答 / 逐句标注来源 / 不足即拒答 / 安全内容指引查 SDS
    return f"""你是化工研发领域的问答助手。严格按以下规则回答：
1. 只能依据下方编号资料回答，禁止利用模型内置知识补充。
2. 每个关键论断句末用 [n] 标注来源编号，多个来源用 [1][3]。
3. 资料不足以完整回答时，明确说明"语料中没有关于XX的依据"，并建议查阅该化学品 SDS 或联系 EHS，不要编造。
4. 使用简体中文，简洁、分点。

【资料】
{context}

【问题】
{question}
"""


# ========== 生成 ==========
_client = None  # OpenAI 兼容客户端单例：批跑评估时复用连接，不每题重建

MAX_RETRIES = 2  # 瞬时故障重试次数（不含首次尝试）
BASE_DELAY = 1.0  # 指数退避起点（秒）


def get_client():
    """懒加载 DeepSeek 客户端（OpenAI 兼容协议）。"""
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未配置：请在项目根目录 .env 中填入（.gitignore 已挡）"
            )
        from openai import OpenAI

        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _client


def generate(prompt: str) -> str:
    """调用 DeepSeek 生成答案（带重试与故障分类）。

    故障三分类重试策略：
    - 鉴权错（401）：立即抛出，无可救药，批跑层应整批中止；
    - 限流（429）：重试但退避拉长——限流的含义是"你打太快了"；
    - 瞬时故障（超时/断连/5xx）：重试 MAX_RETRIES 次，指数退避。
    重试耗尽后抛出，由批跑脚本记为"调用失败"并续跑下一题。
    """
    import time

    from openai import (
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
        AuthenticationError,
        RateLimitError,
    )

    client = get_client()
    delay = BASE_DELAY
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GENERATION_TEMPERATURE,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except AuthenticationError:
            raise  # 系统性故障：不浪费重试
        except RateLimitError:
            if attempt < MAX_RETRIES:
                logger.warning(f"限流，{delay * 4:.0f}s 后重试（第 {attempt + 1} 次）")
                time.sleep(delay * 4)
                delay *= 2
                continue
            raise
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{type(e).__name__}，{delay:.0f}s 后重试（第 {attempt + 1} 次）")
                time.sleep(delay)
                delay *= 2
                continue
            raise


# ========== 主流程 ==========
def answer(
    question: str,
    top_k: int = TOP_K,
    rewrite: bool = False,
    dry_run: bool = False,
) -> AnswerResult:
    """端到端问答接口。rewrite/top_k/dry_run 三个开关支撑后续所有消融实验：
    - 优化 1（改写）   : answer(q, rewrite=True)  vs  rewrite=False
    - top_k 忠实度实验 : answer(q, top_k=3/5)
    评估脚本只认这一个入口，内部组件怎么换都不影响它。
    """
    q = rewrite_query(question) if rewrite else question

    # 默认链路 = 混合召回 + 交叉编码器重排（2026-10-21 收口时接入，对齐已定的默认链路决策）。
    # 注意：忠实度/引用/closed-book 三项生成层评估的 38 题答案生成于接入前的 hybrid 链路
    # （answers_cache.json 为证），数字口径以 SUMMARY.md 的标注为准。
    retrieved = hybrid_rerank_retrieve(q, top_k=top_k)

    # 拒答闸门：命中即短路返回，不发起生成调用
    refuse_reason = should_refuse(retrieved, REFUSE_FUSED_SCORE_THRESHOLD)
    if refuse_reason:
        logger.info(f"[拒答] {question[:30]}... | 原因: {refuse_reason}")
        return AnswerResult(
            question=question,
            answer_text="",
            citations=[],
            refused=True,
            refuse_reason=refuse_reason,
            rewritten_question=q if rewrite else None,
        )

    prompt = build_prompt(q, retrieved)
    citations = [
        Citation(
            index=i,
            chunk_pk=r["chunk_pk"],
            doc_name=r["doc_name"],
            section=r["section"],
            text=r["text"],
            snippet=r["text"][:100],
        )
        for i, r in enumerate(retrieved, start=1)
    ]

    if dry_run:
        # 离线跑通"检索→拼prompt"前半链路，prompt 交人审，不花一分钱 API
        return AnswerResult(
            question=question,
            answer_text="",
            citations=citations,
            refused=False,
            prompt=prompt,
            rewritten_question=q if rewrite else None,
        )

    answer_text = generate(prompt)
    return AnswerResult(
        question=question,
        answer_text=answer_text,
        citations=citations,
        refused=False,
        rewritten_question=q if rewrite else None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = answer(
        "格氏反应引发后滴加卤代烃时，体系温度应控制在什么范围？",
        dry_run=True,
    )
    print("=== 检索命中 ===")
    for c in result.citations:
        print(f"[{c.index}] {c.doc_name} / {c.section} / {c.snippet}...")
    print("\n=== 组装后的 Prompt ===")
    print(result.prompt)
