"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0],
                      metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    from numpy import dot
    from numpy.linalg import norm

    model = _get_semantic_encoder()
    embeddings = model.encode(sentences, show_progress_bar=False)

    def cosine_sim(a, b) -> float:
        denom = norm(a) * norm(b)
        return float(dot(a, b) / denom) if denom else 0.0

    chunks: list[Chunk] = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current_group = []
        current_group.append(sentences[i])
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))
    return chunks


# Cache encoder across calls — loading is the bottleneck (~2s).
_SEMANTIC_ENCODER = None


def _get_semantic_encoder():
    global _SEMANTIC_ENCODER
    if _SEMANTIC_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_ENCODER


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], []

    parents: list[Chunk] = []
    children: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > parent_size and current:
            _flush_parent(current, parents, children, metadata, child_size)
            current = ""
        current += (para + "\n\n")
    if current.strip():
        _flush_parent(current, parents, children, metadata, child_size)

    return parents, children


def _flush_parent(text: str, parents: list, children: list, metadata: dict, child_size: int) -> None:
    """Build one parent + its sliding-window children. Used by chunk_hierarchical."""
    pid = f"parent_{len(parents)}"
    parent_text = text.strip()
    parents.append(Chunk(
        text=parent_text,
        metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
    ))
    # Slide non-overlapping window child_size chars across parent text
    for start in range(0, len(parent_text), child_size):
        chunk_text = parent_text[start:start + child_size].strip()
        if chunk_text:
            children.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_type": "child"},
                parent_id=pid,
            ))


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def _flush():
        if current_content.strip() or current_header:
            chunks.append(Chunk(
                text=f"{current_header}\n{current_content}".strip(),
                metadata={**metadata, "section": current_header, "strategy": "structure"},
            ))

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            _flush()
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
    _flush()

    return [c for c in chunks if c.text]


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lens = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lens) // len(lens),
            "min_length": min(lens),
            "max_length": max(lens),
        }

    all_basic, all_semantic, all_struct, all_parents, all_children = [], [], [], [], []
    for doc in documents:
        text, meta = doc["text"], doc.get("metadata", {})
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        all_struct.extend(chunk_structure_aware(text, metadata=meta))
        p, c = chunk_hierarchical(text, metadata=meta)
        all_parents.extend(p)
        all_children.extend(c)

    results = {
        "basic": stats(all_basic),
        "semantic": stats(all_semantic),
        # Hierarchical reports children stats (children are what get indexed) plus parents count.
        "hierarchical": {**stats(all_children), "num_parents": len(all_parents)},
        "structure": stats(all_struct),
    }

    print(f"\n{'Strategy':<14} | {'Chunks':>7} | {'Avg':>5} | {'Min':>4} | {'Max':>4}")
    print("-" * 50)
    for name in ["basic", "semantic", "hierarchical", "structure"]:
        s = results[name]
        chunks_label = f"{s['num_chunks']}c/{s['num_parents']}p" if name == "hierarchical" else str(s["num_chunks"])
        print(f"{name:<14} | {chunks_label:>7} | {s['avg_length']:>5} | {s['min_length']:>4} | {s['max_length']:>4}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
