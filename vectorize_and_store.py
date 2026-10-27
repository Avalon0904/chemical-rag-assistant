# vectorize_and_store.py
# 阶段三：向量化与存储（Milvus Lite 版）
# 流程：chunks.json → BGE-small 编码 → Milvus Lite 建集合 → 插入 → 建索引 → 加载 → 验证检索

import os
import json
import logging
from pathlib import Path
from typing import List, Dict

from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ========== 配置加载 ==========
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Milvus Lite 配置（本地文件，嵌入式，无需 Docker）
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_chemical.db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chemical_docs")

# Embedding 配置（bge-small-zh-v1.5 输出 512 维）
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))

# 数据配置
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "chunks.json")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# BGE 查询前缀（仅查询时使用，文档不加）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


# ========== 加载切片数据 ==========
def load_chunks(chunks_file: str) -> List[Dict]:
    """从 chunks.json 加载切片数据"""
    path = Path(chunks_file)
    if not path.exists():
        raise FileNotFoundError(f"切片文件不存在: {chunks_file}")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"加载 {len(chunks)} 个切片 from {chunks_file}")

    # 校验字段完整性
    required = {"chunk_pk", "text", "doc_name", "doc_type", "section", "chunk_index"}
    if chunks:
        missing = required - set(chunks[0].keys())
        if missing:
            raise ValueError(f"chunks.json 缺少字段: {missing}")

    return chunks


# ========== 初始化 Milvus 集合 ==========
def init_collection(
    client: MilvusClient,
    collection_name: str,
    dim: int,
    drop_existing: bool = True,
):
    """
    创建 Milvus 集合。
    drop_existing=True 时先删除同名集合（开发阶段重建用）。
    Milvus Lite 同样支持 has_collection / drop_collection。
    """
    if client.has_collection(collection_name):
        if drop_existing:
            logger.warning(f"集合 {collection_name} 已存在，删除重建")
            client.drop_collection(collection_name)
        else:
            logger.info(f"集合 {collection_name} 已存在，复用")
            return

    # 定义 schema：chunk_pk 作为主键（VARCHAR）
    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="chunk_pk", dtype=DataType.VARCHAR, max_length=100, is_primary=True
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
        ]
    )

    client.create_collection(collection_name=collection_name, schema=schema)
    logger.info(f"集合 {collection_name} 创建成功，维度 {dim}")


# ========== 编码并插入 ==========
def encode_and_insert(
    client: MilvusClient,
    collection_name: str,
    chunks: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 64,
):
    """
    分批编码切片并插入 Milvus。
    文档编码不加查询前缀，归一化。
    """
    total = len(chunks)
    inserted = 0

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]

        # 编码（文档不加前缀，归一化）
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # 组装插入数据
        data = []
        for chunk, vector in zip(batch, vectors):
            data.append(
                {
                    "chunk_pk": chunk["chunk_pk"],
                    "vector": vector,
                    "text": chunk["text"],
                    "doc_name": chunk["doc_name"],
                    "doc_type": chunk["doc_type"],
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

        client.insert(collection_name=collection_name, data=data)
        inserted += len(data)
        logger.info(f"已插入 {inserted}/{total}")

    logger.info(f"全部插入完成，共 {inserted} 条")


# ========== 创建索引并加载 ==========
def create_index_and_load(client: MilvusClient, collection_name: str):
    """创建向量索引并加载集合到内存"""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    logger.info("索引创建完成")

    client.load_collection(collection_name=collection_name)
    logger.info("集合已加载到内存")


# ========== 验证检索 ==========
def verify_search(
    client: MilvusClient,
    collection_name: str,
    model: SentenceTransformer,
    query: str,
    top_k: int = 3,
):
    """用一条查询验证检索是否正常"""
    query_vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "doc_name", "doc_type", "section", "chunk_index"],
    )

    logger.info(f"检索验证：query='{query}'")
    for hits in results:
        for hit in hits:
            entity = hit["entity"]
            logger.info(
                f"  distance={hit['distance']:.4f} "
                f"doc={entity['doc_name']} "
                f"section={entity['section']} "
                f"chunk_index={entity['chunk_index']} "
                f"text={entity['text'][:50]}..."
            )
    return results


# ========== 主流程 ==========
def main():
    # 1. 加载切片
    chunks = load_chunks(CHUNKS_FILE)
    if not chunks:
        logger.warning("切片为空，退出")
        return

    # 2. 加载 Embedding 模型
    logger.info(f"加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"模型加载完成，向量维度: {model.get_sentence_embedding_dimension()}")

    # 3. 连接 Milvus Lite（本地文件，无需 Docker）
    logger.info(f"连接 Milvus Lite: {MILVUS_URI}")
    client = MilvusClient(uri=MILVUS_URI)

    # 4. 创建集合（开发阶段重建）
    init_collection(client, COLLECTION_NAME, VECTOR_DIM, drop_existing=True)

    # 5. 编码并插入
    encode_and_insert(client, COLLECTION_NAME, chunks, model, BATCH_SIZE)

    # 6. 创建索引并加载
    create_index_and_load(client, COLLECTION_NAME)

    # 7. 验证检索
    verify_search(client, COLLECTION_NAME, model, "催化剂选择性下降怎么办", top_k=3)

    # 8. 统计
    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    logger.info(f"集合统计: {stats}")


if __name__ == "__main__":
    main()
