"""Deduplication and Relevance Scoring for scraped content.

Pure in-memory processing - optimized for AWS Lambda deployment.
"""

from typing import List, Dict, Any
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Model cache for Lambda warm starts
_models = {}


def get_model(model_type: str):
    """Load and cache models for reuse across Lambda invocations."""
    if model_type not in _models:
        if model_type == "embedding":
            _models[model_type] = SentenceTransformer(settings.EMBEDDING_MODEL)
        elif model_type == "reranker":
            _models[model_type] = CrossEncoder(settings.RERANKER_MODEL)
        logger.info(f"Loaded {model_type} model")
    return _models[model_type]


def chunk_text(text: str) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE * 5,  # words to chars
        chunk_overlap=settings.CHUNK_OVERLAP * 5,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    for i, chunk_text in enumerate(splitter.split_text(text)):
        word_count = len(chunk_text.split())
        if word_count >= settings.MIN_CHUNK_WORDS:
            chunks.append({
                "chunk_id": f"chunk_{i:04d}",
                "text": chunk_text,
                "word_count": word_count
            })

    return chunks


def deduplicate(chunks: List[Dict], embeddings: np.ndarray) -> tuple:
    """Remove near-duplicate chunks using FAISS similarity search."""
    # Build index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    # Find duplicates
    keep = np.ones(len(chunks), dtype=bool)
    for i in range(len(chunks)):
        if not keep[i]:
            continue

        # Search similar chunks
        sims, idxs = index.search(embeddings[i:i+1].astype('float32'), min(20, len(chunks)))

        for sim, idx in zip(sims[0], idxs[0]):
            if idx != i and keep[idx] and sim >= settings.SIMILARITY_THRESHOLD:
                # Keep longer chunk
                keep[idx] = chunks[idx]["word_count"] > chunks[i]["word_count"]

    return [c for i, c in enumerate(chunks) if keep[i]], embeddings[keep]


def rerank(chunks: List[Dict], topic: str) -> List[Dict]:
    """Rank chunks by relevance to topic using cross-encoder."""
    if not chunks:
        return []

    pairs = [[topic, c["text"]] for c in chunks]
    scores = get_model("reranker").predict(pairs, show_progress_bar=False)

    for chunk, score in zip(chunks, scores):
        chunk["relevance_score"] = float(score)

    return sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)


def process(merged_text: str, topic: str, scraped_pages: List[Dict]) -> Dict[str, Any]:
    """Execute dedup + ranking pipeline (pure in-memory)."""
    logger.info(f"Processing {len(merged_text)} chars from {len(scraped_pages)} pages")

    # 1. Chunk
    chunks = chunk_text(merged_text)
    logger.info(f"Created {len(chunks)} chunks")

    if not chunks:
        return {"ranked_chunks": [], "stats": {"total_chunks": 0}}

    # 2. Embed
    texts = [c["text"] for c in chunks]
    embeddings = get_model("embedding").encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    logger.info(f"Generated embeddings: {embeddings.shape}")

    # 3. Deduplicate
    unique_chunks, unique_embs = deduplicate(chunks, embeddings)
    logger.info(f"Deduped: {len(chunks)} → {len(unique_chunks)}")

    # 4. Rerank
    ranked = rerank(unique_chunks, topic)
    top_k = ranked[:settings.TOP_K_CHUNKS]
    logger.info(f"Selected top {len(top_k)} chunks")

    return {
        "ranked_chunks": top_k,
        "stats": {
            "total_chunks": len(chunks),
            "duplicates_removed": len(chunks) - len(unique_chunks),
            "top_k_selected": len(top_k)
        }
    }
