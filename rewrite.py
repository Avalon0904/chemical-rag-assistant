# rewrite.py
# 查询改写模块（阶段 3+ 消融 · 优化 1/2 的实验件）
# 职责单一：口语问题 → 语料风格书面问题，配后校验，不过即回退原问题。
# 设计取舍：
#   - prompt 用 6 条编号约束 + few-shot，temperature=0（要确定性，不要创造性）；
#   - "劝"（prompt）与"拦"（代码校验）缺一不可：LLM 输出是概率性的；
#   - 保守兜底原则：校验拿不准一律保留原问题——改写模块的天花板是"无害"。
# 注意：本模块不得 import answer（answer 依赖本模块，方向单向）。

import os
import re
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("REWRITE_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

REWRITE_SYSTEM_PROMPT = """# 角色
你是化工研发文档的查询改写器。

# 任务
将用户口语化的提问改写为与化工 SOP/工艺规程语料风格一致的书面提问，
用于检索文档。只改表述，不改语义。

# 约束（必须逐条遵守）
1. 实体守恒：原问题中的化学品名称、反应名称、设备名、操作名必须原样保留，
   不得新增、不得替换为泛指。
2. 数值与单位逐字保留：所有数字、单位、范围照抄，
   禁止改写为"较高""适量""一段时间"等模糊表述。
3. 疑问意图不变：保持原疑问类型（问条件/问步骤/问原因/是非判断），不得转换。
4. 不合并、不拆分、不扩写：一个问题进，一个问题出，
   禁止补充原问题没有的背景、前提或上下文。
5. 书面语域对齐：使用 SOP 常用书面术语（如"滴加""引发""淬灭""保温""后处理"）
   替换口语表达。
6. 裁决规则：术语规范化属于允许的改写（"格氏"→"格氏反应"视为同一实体，
   可替换）；"格氏反应"→"金属有机反应"属于改变所指，禁止。
   拿不准时一律保留原写法。

# 输出格式
只输出改写后的问题本身。一行。不要解释、不要引号、不要"改写后："等前后缀。

# 示例
原问题：格氏反应做完之后怎么淬灭？
改写后：格氏反应结束后应如何进行淬灭操作？

原问题：滴加的时候温度是不是不能超过40度？
改写后：格氏反应滴加过程中体系温度应控制在什么范围？

原问题：引发老是不成功，能直接再丢点碘进去吗？
改写后：格氏反应引发失败时是否可以通过补加碘促进引发？
"""

# 后校验用的口语虚词：实体匹配时剔除，避免"吗/什么"这类词卡住校验
STOPWORDS = {
    "什么", "怎么", "怎样", "时候", "可以", "应该", "需要", "是否", "大概",
    "多少", "我们", "你们", "一下", "的话", "如果", "为什么", "哪里",
}

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def validate_rewrite(original: str, rewritten: str) -> bool:
    """后校验：数字逐字保留 + 实体子串匹配。任一项不过即 False（调用方回退）。

    设计取向是偏严：宁可回退不用改写，也不放一个有实体丢失风险的
    改写进检索——改写模块的定位是增益项，不是必经路。
    """
    if not rewritten or not rewritten.strip():
        return False
    # 长度 sanity：太短可能在概括，太长可能在扩写
    if not (0.4 <= len(rewritten) / max(len(original), 1) <= 3.0):
        return False
    # 数字守恒：原文每个数字串都必须出现在改写结果里
    for num in NUMBER_RE.findall(original):
        if num not in rewritten:
            return False
    # 实体守恒：jieba 切出的实义词（含中文、长度≥2、非虚词）须原样出现
    import jieba

    for token in jieba.cut(original):
        token = token.strip()
        if len(token) >= 2 and re.search(r"[\u4e00-\u9fff]", token):
            if token in STOPWORDS:
                continue
            if token not in rewritten:
                return False
    return True


_client = None


def _get_client():
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置：请在项目根目录 .env 中填入")
        from openai import OpenAI

        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


def rewrite_query(question: str) -> str:
    """术语对齐式改写。校验不过或调用失败一律回退原问题，并记日志。

    返回：改写后的问题，或原问题（回退）。
    """
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        rewritten = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[改写] 调用失败，回退原问题: {type(e).__name__}: {e}")
        return question

    if validate_rewrite(question, rewritten):
        if rewritten != question:
            logger.info(f"[改写] {question[:30]}... -> {rewritten[:30]}...")
        return rewritten

    logger.info(f"[改写] 后校验未通过，回退原问题: {question[:30]}...")
    return question


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(rewrite_query("格氏反应做完之后怎么淬灭？"))
