# document_pipeline.py
# 文档预处理管道：PDF/TXT → 提取（表格区域排除 + 表格转 Markdown）→ 章节感知切片 → chunks.json

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
OVERLAP = int(os.getenv("OVERLAP", "50"))
INPUT_DIR = os.getenv("INPUT_DIR", "corpus_final")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "chunks.json")

# 文件名前缀 → 文档类型（真实项目应从元数据/配置读取）
DOC_TYPE_MAP = {
    "SOP": "SOP", "SDS": "SDS", "PATENT": "专利", "REPORT": "实验报告",
    "REVIEW": "综述", "TABLE": "数据表",
}


# ========== 文本清洗 ==========
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ========== 表格转 Markdown ==========
def tables_to_markdown(tables: list) -> str:
    """把 pdfplumber 提取的表格转成 Markdown。
    注意：单元格内换行直接删除而非替换为空格——
    PDF 单元格折行不是语义断点（如 CAS 号 109-99-9 被折成两行）。"""
    if not tables:
        return ""
    md_parts = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        header = [str(c).replace("\n", "").strip() if c else "" for c in table[0]]
        md_parts.append("| " + " | ".join(header) + " |")
        md_parts.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in table[1:]:
            cells = [str(c).replace("\n", "").strip() if c else "" for c in row]
            md_parts.append("| " + " | ".join(cells) + " |")
        md_parts.append("")
    return "\n".join(md_parts)


# ========== TXT 提取 ==========
def extract_txt(file_path: Path) -> List[Tuple[str, str]]:
    """返回 [(章节标题, 正文)]。TXT 无布局信息，用启发式识别标题：
    短行（<30 字）+ 无句尾标点 + 前后有空行。启发式是脆弱的，
    真实项目应约定 TXT 保留标题标记（如 # 前缀）。"""
    try:
        try:
            text = open(file_path, "r", encoding="utf-8").read()
        except UnicodeDecodeError:
            text = open(file_path, "r", encoding="gbk").read()
    except Exception as e:
        logger.error(f"TXT 读取失败: {file_path.name}, {e}")
        return []

    lines = text.splitlines()
    sections, title, buf = [], "（文首）", []
    for i, line in enumerate(lines):
        s = line.strip()
        is_heading = (
            0 < len(s) < 30
            and not s.startswith(("-", "*", "|"))          # 列表/表格行不当标题
            and not re.match(r"^\d+\.", s)                  # 编号行不当标题
            and not s.endswith(("。", "；", "，", "：", ".", ";", ",", "）"))
            and (i + 1 >= len(lines) or not lines[i + 1].strip() or lines[i + 1].strip().startswith(("|", "-")))
        )
        if is_heading and buf:
            sections.append((title, "\n".join(buf)))
            title, buf = s, []
        else:
            buf.append(s)
    if buf:
        sections.append((title, "\n".join(buf)))
    return [(t, clean_text(b)) for t, b in sections if clean_text(b)]


# ========== PDF 提取（表格区域排除 + 字号识别标题） ==========
def _lines_with_size(page) -> List[Tuple[str, float]]:
    """把页内字符聚成行，返回 [(行文本, 行最大字号)]，用于识别标题层级。"""
    chars = sorted(page.chars, key=lambda c: (round(c["top"], 1), c["x0"]))
    lines, cur, cur_top = [], [], None
    for c in chars:
        if cur_top is None or abs(c["top"] - cur_top) > 2:
            if cur:
                lines.append(("".join(x["text"] for x in sorted(cur, key=lambda x: x["x0"])),
                              max(x["size"] for x in cur)))
            cur, cur_top = [c], c["top"]
        else:
            cur.append(c)
    if cur:
        lines.append(("".join(x["text"] for x in sorted(cur, key=lambda x: x["x0"])),
                      max(x["size"] for x in cur)))
    return lines


def extract_pdf(file_path: Path) -> List[Tuple[str, str]]:
    """返回 [(章节标题, 正文)]。表格只以 Markdown 形式出现一次：
    正文抽取时通过 outside_bbox 把表格区域排除，避免脏文本双份入库。"""
    try:
        with pdfplumber.open(file_path) as pdf:
            sections, title, buf = [], "（文首）", []

            def flush():
                nonlocal buf
                body = clean_text("\n".join(buf))
                if body:
                    sections.append((title, body))
                buf = []

            for page_num, page in enumerate(pdf.pages, start=1):
                # 1. 表格区域从正文中排除，单独转 Markdown
                tables = page.find_tables()
                table_md = tables_to_markdown([t.extract() for t in tables])
                for t in tables:
                    page = page.outside_bbox(t.bbox)  # 逐步裁掉表格区域

                # 2. 非表格文本：按字号区分标题/正文
                for text, size in _lines_with_size(page):
                    text = text.strip()
                    if not text:
                        continue
                    if size >= 12.5:  # 标题（13pt）或文档名（16pt）
                        flush()
                        title = text
                    else:
                        buf.append(text)

                # 3. 本页表格挂在当前章节下
                if table_md:
                    buf.append(f"【本页表格】\n{table_md}")
                page.flush_cache()
            flush()

            if not sections:
                logger.warning(f"可能是扫描件，需要 OCR: {file_path.name}")
            return sections
    except PDFSyntaxError:
        logger.error(f"PDF 文件损坏: {file_path.name}")
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
    except Exception as e:
        logger.error(f"PDF 解析未知错误: {file_path.name}, {e}")
    return []


# ========== 切片（递归分隔符 + 真实 overlap） ==========
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " "]
FALLBACK_SEP = ""


def _sliding_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    """最后兜底：按字符滑窗，保证绝不超 chunk_size。"""
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step) if text[i:i + chunk_size].strip()]


def _split_recursive(text: str, chunk_size: int, separators: List[str], level: int = 0) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if level >= len(separators):  # 分隔符用尽，滑窗兜底
        return _sliding_window(text, chunk_size, overlap=max(0, chunk_size // 10))
    sep = separators[level]
    if sep not in text:  # 当前层级无法切分，换更细的分隔符
        return _split_recursive(text, chunk_size, separators, level + 1)

    chunks, current = [], ""
    for part in text.split(sep):
        candidate = part if not current else current + sep + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            current = part
    if current.strip():
        chunks.append(current)

    out = []
    for c in chunks:
        if len(c) > chunk_size:  # 超大块递归用更细的分隔符
            out.extend(_split_recursive(c, chunk_size, separators, level + 1))
        else:
            out.append(c)
    return out


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """给相邻 chunk 之间加重叠：每个 chunk 开头带上前一片的尾部。"""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append(cur if cur.startswith(tail) else tail + cur)
    return out


# ========== 处理单个文件 ==========
def process_file(file_path: Path, chunk_size: int, overlap: int) -> List[Dict]:
    stem = file_path.stem
    doc_type = next((v for k, v in DOC_TYPE_MAP.items() if k in stem.upper()), "未知")

    if file_path.suffix == ".txt":
        sections = extract_txt(file_path)
    elif file_path.suffix == ".pdf":
        sections = extract_pdf(file_path)
    else:
        return []

    chunks_out = []
    idx = 1
    for section, body in sections:
        pieces = _split_recursive(body, chunk_size, SEPARATORS)
        pieces = _apply_overlap(pieces, overlap)
        for c in pieces:
            chunks_out.append({
                "chunk_pk": f"{stem}-c{idx:03d}",   # 全局唯一主键，进 Milvus 用
                "doc_name": stem,                   # 不带扩展名，与评估集匹配
                "doc_type": doc_type,
                "section": section,                 # 章节，引用和过滤都要用
                "chunk_index": idx,
                "text": c,
            })
            idx += 1
    return chunks_out


# ========== 主流程 ==========
def main(input_dir: str = INPUT_DIR, output_file: str = OUTPUT_FILE):
    p = Path(input_dir)
    if not p.exists():
        logger.error(f"目录不存在: {input_dir}")
        return

    all_chunks, processed, failed = [], 0, 0
    for file_path in sorted(p.rglob("*")):
        if not file_path.is_file() or file_path.suffix not in (".txt", ".pdf"):
            continue
        logger.info(f"正在处理: {file_path.name}")
        chunks = process_file(file_path, CHUNK_SIZE, OVERLAP)
        if chunks:
            all_chunks.extend(chunks)
            processed += 1
            logger.info(f"  → {len(chunks)} 个切片")
        else:
            failed += 1
            logger.warning("  → 无有效内容，跳过")

    if not all_chunks:
        logger.warning("没有找到任何可处理的文件")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"完成：成功 {processed} 个文件，失败 {failed} 个，共 {len(all_chunks)} 个切片 → {output_file}")


if __name__ == "__main__":
    main()
