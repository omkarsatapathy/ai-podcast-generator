"""Phase 2 Agent 1: Chapter Planner.

Transforms ranked source chunks from Phase 1 into structured chapter outlines.
Pipeline: Analyze Chunks → Cluster (FAISS) → 3-Act Sequence (LLM) → Chapter Outlines (LLM)
"""

from typing import List, Dict, Any
import asyncio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import faiss

from config.settings import settings
from src.agents.phase1.dedup_relevance_scorer import get_model
from src.api_factory.llm import get_llm
from src.llm.prompts import (
    CHUNK_ANALYSIS_PROMPT,
    NARRATIVE_SEQUENCE_PROMPT,
    CHAPTER_OUTLINE_PROMPT,
)
from src.models.chapter import (
    BatchChunkAnalysis,
    NarrativeSequence,
    BatchChapterOutlines,
    ChapterOutline,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_llm():
    """Get configured LLM for chapter planner."""
    return get_llm(
        tier=settings.CHAPTER_PLANNER_MODEL,
        temperature=settings.CHAPTER_PLANNER_TEMPERATURE,
    )


# ==================== STEP 1: ANALYZE CHUNKS ====================

async def _process_single_batch(
    llm, batch: List[Dict], topic: str, batch_num: int, total_batches: int
) -> List[Dict]:
    """Process a single batch of chunks asynchronously."""
    # Build chunk text for prompt
    chunks_text = ""
    for chunk in batch:
        chunks_text += (
            f"\n--- chunk_id: {chunk['chunk_id']} ---\n"
            f"{chunk['text'][:1500]}\n"  # cap per chunk to manage tokens
        )

    prompt = CHUNK_ANALYSIS_PROMPT.format(topic=topic, chunks_text=chunks_text)

    logger.info(f"Analyzing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
    result: BatchChunkAnalysis = await llm.ainvoke(prompt)

    # Merge analysis metadata into original chunks
    analysis_map = {a.chunk_id: a for a in result.analyses}
    batch_analyzed = []
    for chunk in batch:
        analysis = analysis_map.get(chunk["chunk_id"])
        if analysis:
            chunk["analysis_topic"] = analysis.topic
            chunk["analysis_subtopics"] = analysis.subtopics
            chunk["analysis_summary"] = analysis.summary
            chunk["analysis_tone"] = analysis.tone
        else:
            # Fallback if LLM missed a chunk
            chunk["analysis_topic"] = "unknown"
            chunk["analysis_subtopics"] = []
            chunk["analysis_summary"] = "No summary available"
            chunk["analysis_tone"] = "factual"
        batch_analyzed.append(chunk)

    return batch_analyzed


async def _analyze_chunks_async(ranked_chunks: List[Dict], topic: str) -> List[Dict]:
    """Extract metadata from chunks using parallel LLM calls.

    Processes 10 batches (20 chunks) in parallel for efficiency.
    """
    llm = _get_llm().with_structured_output(BatchChunkAnalysis, method="json_schema")
    batch_size = settings.CHAPTER_PLANNER_BATCH_SIZE
    parallel_batches = 10  # Process 10 batches simultaneously
    analyzed = []

    total_batches = (len(ranked_chunks) + batch_size - 1) // batch_size

    # Process in groups of 10 parallel batches (20 chunks at a time)
    for i in range(0, len(ranked_chunks), batch_size * parallel_batches):
        tasks = []

        for j in range(parallel_batches):
            start_idx = i + j * batch_size
            if start_idx >= len(ranked_chunks):
                break

            batch = ranked_chunks[start_idx : start_idx + batch_size]
            batch_num = start_idx // batch_size + 1
            tasks.append(_process_single_batch(llm, batch, topic, batch_num, total_batches))

        # Run all batches in parallel
        results = await asyncio.gather(*tasks)
        for result in results:
            analyzed.extend(result)

    logger.info(f"Analyzed {len(analyzed)} chunks with parallel processing")
    return analyzed


def analyze_chunks(ranked_chunks: List[Dict], topic: str) -> List[Dict]:
    """Extract metadata from chunks using LLM in batches of 2 with parallel processing.

    Adds topic, subtopics, summary, tone to each chunk.
    Original chunk data is preserved.
    Processes up to 20 chunks (10 batches) in parallel.
    """
    return asyncio.run(_analyze_chunks_async(ranked_chunks, topic))


# ==================== STEP 2: CLUSTER CHUNKS ====================

def cluster_chunks(analyzed_chunks: List[Dict]) -> List[List[Dict]]:
    """Cluster chunks by summary embeddings using FAISS + AgglomerativeClustering.

    Returns list of clusters, each cluster is a list of chunk dicts.
    """
    # Embed summaries (lightweight, focused text)
    summaries = [c["analysis_summary"] for c in analyzed_chunks]
    embedder = get_model("embedding")
    embeddings = embedder.encode(summaries, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")

    # Store in FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors")

    # Compute cosine distance matrix for clustering
    # cosine_distance = 1 - cosine_similarity (since embeddings are normalized, IP = cosine)
    similarity_matrix = np.dot(embeddings, embeddings.T)
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)  # zero self-distance

    # Determine cluster count within bounds
    n_clusters = min(
        settings.MAX_CHAPTERS,
        max(settings.MIN_CHAPTERS, len(analyzed_chunks) // 8)
    )

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    # Group chunks by cluster label
    clusters: List[List[Dict]] = [[] for _ in range(n_clusters)]
    for chunk, label in zip(analyzed_chunks, labels):
        clusters[label].append(chunk)

    # Log cluster sizes
    for i, cluster in enumerate(clusters):
        topics = [c["analysis_topic"] for c in cluster]
        logger.info(f"Cluster {i}: {len(cluster)} chunks - topics: {topics[:3]}")

    return clusters


# ==================== STEP 3: NARRATIVE SEQUENCE ====================

def generate_narrative_sequence(
    clusters: List[List[Dict]], topic: str
) -> NarrativeSequence:
    """Use LLM to organize clusters into 3-act podcast structure."""
    llm = _get_llm().with_structured_output(NarrativeSequence, method="json_schema")

    # Build cluster summary for prompt
    clusters_summary = ""
    for i, cluster in enumerate(clusters):
        topics = list({c["analysis_topic"] for c in cluster})
        tones = list({c["analysis_tone"] for c in cluster})
        clusters_summary += (
            f"Cluster {i}: {len(cluster)} chunks\n"
            f"  Topics: {', '.join(topics)}\n"
            f"  Tones: {', '.join(tones)}\n\n"
        )

    prompt = NARRATIVE_SEQUENCE_PROMPT.format(
        topic=topic,
        num_clusters=len(clusters),
        clusters_summary=clusters_summary,
    )

    logger.info("Generating narrative sequence via LLM")
    sequence: NarrativeSequence = llm.invoke(prompt)

    # Validate total duration
    total = sum(ch.estimated_duration_minutes for ch in sequence.chapters)
    logger.info(f"Narrative sequence: {len(sequence.chapters)} chapters, {total:.1f} min total")

    return sequence


# ==================== STEP 4: CHAPTER OUTLINES ====================

def generate_chapter_outlines(
    sequence: NarrativeSequence,
    clusters: List[List[Dict]],
    topic: str,
) -> List[ChapterOutline]:
    """Use LLM to generate titles, key_points, hooks for each chapter."""
    llm = _get_llm().with_structured_output(BatchChapterOutlines, method="json_schema")

    # Build detail for each chapter
    chapters_detail = ""
    for ch in sequence.chapters:
        # Collect chunk metadata from assigned clusters
        chapter_topics = []
        chapter_summaries = []
        for cid in ch.cluster_ids:
            if cid < len(clusters):
                for chunk in clusters[cid]:
                    chapter_topics.append(chunk["analysis_topic"])
                    chapter_summaries.append(chunk["analysis_summary"])

        chapters_detail += (
            f"Chapter {ch.chapter_number} (Act: {ch.act}, Energy: {ch.energy_level}):\n"
            f"  Topics: {', '.join(set(chapter_topics))}\n"
            f"  Content summaries: {'; '.join(set(chapter_summaries[:8]))}\n\n"
        )

    prompt = CHAPTER_OUTLINE_PROMPT.format(
        topic=topic,
        chapters_detail=chapters_detail,
        num_chapters=len(sequence.chapters),
    )

    logger.info("Generating chapter outlines via LLM")
    result: BatchChapterOutlines = llm.invoke(prompt)

    # Merge LLM creative output with deterministic data
    outline_map = {o.chapter_number: o for o in result.outlines}
    outlines = []

    for ch in sequence.chapters:
        generated = outline_map.get(ch.chapter_number)
        # Collect source chunk IDs from assigned clusters
        source_ids = []
        for cid in ch.cluster_ids:
            if cid < len(clusters):
                source_ids.extend(c["chunk_id"] for c in clusters[cid])

        outlines.append(ChapterOutline(
            chapter_number=ch.chapter_number,
            title=generated.title if generated else f"Chapter {ch.chapter_number}",
            act=ch.act,
            energy_level=ch.energy_level,
            key_points=generated.key_points if generated else [],
            source_chunk_ids=source_ids,
            transition_hook=generated.transition_hook if generated else "",
            estimated_duration_minutes=ch.estimated_duration_minutes,
        ))

    logger.info(f"Generated {len(outlines)} chapter outlines")
    return outlines


# ==================== ORCHESTRATOR ====================

def process(ranked_chunks: List[Dict], topic: str) -> Dict[str, Any]:
    """Execute the full chapter planner pipeline.

    Steps:
        1. Analyze chunks (LLM, batches of 5)
        2. Cluster by summary embeddings (FAISS + sklearn)
        3. Generate 3-act narrative sequence (LLM)
        4. Generate chapter outlines (LLM)

    Returns:
        dict with chapter_outlines and analysis_stats
    """
    logger.info(f"Starting Chapter Planner: {len(ranked_chunks)} chunks, topic='{topic}'")

    # Step 1: Analyze
    analyzed = analyze_chunks(ranked_chunks, topic)

    # Step 2: Cluster
    clusters = cluster_chunks(analyzed)

    # Step 3: Narrative sequence
    sequence = generate_narrative_sequence(clusters, topic)

    # Step 4: Chapter outlines
    outlines = generate_chapter_outlines(sequence, clusters, topic)

    total_duration = sum(o.estimated_duration_minutes for o in outlines)
    logger.info(f"Chapter Planner complete: {len(outlines)} chapters, {total_duration:.1f} min")

    return {
        "chapter_outlines": [o.model_dump() for o in outlines],
        "analyzed_chunks": analyzed,
        "stats": {
            "total_chunks_analyzed": len(analyzed),
            "num_clusters": len(clusters),
            "num_chapters": len(outlines),
            "total_duration_minutes": total_duration,
        },
    }
