# retrieval.py
# 混合检索：稠密向量 + BM25 稀疏检索，RRF 倒数排名融合
# evaluate.py 通过 retrieve_func(question, top_k) 接口调用，向量/混合可互换

import os
import json
import logging
from typing import List, Dict

import jieba
from rank_bm25 import BM25Okapi
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 注意：MILVUS_URI 不在此处读取——模块导入期各库的 load_dotenv 注入时序
# 不可控（环境变量覆盖），URI 推迟到 get_client() 首次调用时读取，
# 保证"最后一次环境修改才作数"。
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chemical_docs")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "chunks.json")
RRF_K = int(os.getenv("RRF_K", "60"))          # RRF 平滑常数，越大排名差异被压缩得越平
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "20"))  # 每路召回的候选数

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 全局单例
_model: SentenceTransformer | None = None
_client: MilvusClient | None = None
_bm25: BM25Okapi | None = None
_bm25_chunks: List[Dict] | None = None


def get_model() -> SentenceTransformer:
    """懒加载 Embedding 模型"""
    global _model
    if _model is None:
        logger.info(f"加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_client() -> MilvusClient:
    """懒加载 Milvus 客户端。
    Milvus Lite 的 load 状态不跨进程持久化，每次启动先加载集合（幂等）。
    URI 在首次使用时读取，确保运行期对 os.environ 的最后修改生效。"""
    global _client
    if _client is None:
        milvus_uri = os.getenv("MILVUS_URI", "./milvus_chemical.db")
        logger.info(f"连接 Milvus: {milvus_uri}")
        _client = MilvusClient(uri=milvus_uri)
        _client.load_collection(collection_name=COLLECTION_NAME)
    return _client


def vector_retrieve(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
) -> List[Dict]:
    """
    纯向量检索（基线）。
    Args:
        query: 用户问题
        top_k: 返回条数
        doc_type: 可选，按文档类型过滤（如 "SOP"）
    Returns:
        [{"chunk_pk": ..., "text": ..., "doc_name": ..., "section": ..., "score": ...}, ...]
    """
    model = get_model()
    client = get_client()
    # 幂等加载：Lite 服务器状态可能漂移（崩溃重启后集合 released），
    # 搜索前确保集合可用；已加载时此调用近乎零成本
    client.load_collection(collection_name=COLLECTION_NAME)

    # 查询编码（加前缀，归一化）
    query_vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    # 过滤条件
    filter_expr = f'doc_type == "{doc_type}"' if doc_type else ""

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=[
            "text",
            "doc_name",
            "doc_type",
            "section",
            "chunk_index",
            "chunk_pk",
        ],
    )

    docs = []
    for hits in results:
        for hit in hits:
            entity = hit["entity"]
            docs.append(
                {
                    "chunk_pk": entity["chunk_pk"],
                    "text": entity["text"],
                    "doc_name": entity["doc_name"],
                    "doc_type": entity["doc_type"],
                    "section": entity["section"],
                    "chunk_index": entity["chunk_index"],
                    "score": hit["distance"],
                }
            )
    return docs


# ========== BM25 稀疏检索腿 ==========
def get_bm25() -> BM25Okapi:
    """懒加载 BM25 索引。语料来自 chunks.json（与灌库同一份，保证 chunk_pk 一致）。
    中文必须分词后喂 BM25，整句当词只有命中全句才得分。"""
    global _bm25, _bm25_chunks
    if _bm25 is None:
        logger.info(f"构建 BM25 索引 from {CHUNKS_FILE}")
        _bm25_chunks = json.load(open(CHUNKS_FILE, encoding="utf-8"))
        tokenized = [list(jieba.cut(c["text"])) for c in _bm25_chunks]
        _bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 索引就绪：{len(_bm25_chunks)} 个 chunk")
    return _bm25


def bm25_retrieve(query: str, top_n: int = HYBRID_CANDIDATES) -> List[Dict]:
    """BM25 检索，返回与 vector_retrieve 相同的字段结构。"""
    bm25 = get_bm25()
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)

    # 取 top_n：argsort 降序
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = []
    for i in top_idx:
        c = _bm25_chunks[i]
        results.append(
            {
                "chunk_pk": c["chunk_pk"],
                "text": c["text"],
                "doc_name": c["doc_name"],
                "doc_type": c["doc_type"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
                "score": float(scores[i]),
            }
        )
    return results


# ========== 混合检索：RRF 融合 ==========
def rrf_fuse(rankings: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """倒数排名融合：score(d) = Σ 1/(k + rank_i(d))。
    只用排名、不用原始分数——稠密余弦（0~1）和 BM25（0~几十）没有可比尺度，
    分数融合需要归一化且对分布敏感，排名融合开箱即用、抗尺度差异。"""
    fused: Dict[str, float] = {}
    best_row: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            pk = row["chunk_pk"]
            fused[pk] = fused.get(pk, 0.0) + 1.0 / (k + rank)
            best_row.setdefault(pk, row)  # 元数据任取一路的即可
    merged = [dict(best_row[pk], fused_score=s) for pk, s in fused.items()]
    merged.sort(key=lambda r: r["fused_score"], reverse=True)
    return merged


def hybrid_candidates(query: str) -> List[Dict]:
    """混合检索的完整候选池（融合后不截断）。
    重排需要尽量大的候选池；截断是 hybrid_retrieve 的最后一步。"""
    dense = vector_retrieve(query, top_k=HYBRID_CANDIDATES)
    sparse = bm25_retrieve(query, top_n=HYBRID_CANDIDATES)
    return rrf_fuse([dense, sparse])


def hybrid_retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """混合检索接口，与 vector_retrieve 同签名，evaluate.py 可直接互换。"""
    return hybrid_candidates(query)[:top_k]


# ========== 重排（交叉编码器）=========
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")

_reranker = None


def get_reranker():
    """懒加载重排模型。CPU 推理：双编码器负责 recall，交叉编码器只在
    小候选池上精排，这是 retrieve-and-rerank 的标准分工。"""
    global _reranker
    if _reranker is None:
        logger.info(f"加载重排模型: {RERANK_MODEL_NAME}")
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL_NAME, device="cpu")
    return _reranker


def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    """对候选池做交叉编码器重排。批量 predict 一次算完全部候选。
    rerank_score 取代 fused_score 决定最终顺序（RRF 分数只用于粗筛进池）。"""
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    scored = [dict(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
    scored.sort(key=lambda r: r["rerank_score"], reverse=True)
    return scored


def hybrid_rerank_retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """第三路检索：混合召回 → 交叉编码器重排 → top_k。
    与 vector_retrieve 同签名，evaluate.py 消融直接互换。"""
    return rerank(query, hybrid_candidates(query))[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = "格氏反应引发后滴加卤代烃时，体系温度应控制在什么范围？"
    for name, fn in [("vector", vector_retrieve), ("hybrid", hybrid_retrieve)]:
        print(f"\n=== {name}: {q} ===")
        for r in fn(q, top_k=3):
            score = r.get("fused_score", r["score"])
            print(f"[{score:.4f}] {r['doc_name']} / {r['section']} / {r['text'][:50]}...")
