"""Phase 3 Agent: Fact Checker.

Verifies factual claims in dialogue against source chunks from Phase 1.
"""

import re
from typing import List, Dict, Tuple

from langchain_openai import ChatOpenAI

from config.settings import settings
from src.llm.prompts import BATCH_FACT_CHECK_PROMPT
from src.models.dialogue import BatchFactCheckResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_claims(utterances: List[Dict]) -> List[Dict]:
    """Identify utterances containing factual claims to verify."""
    claims = []
    for utt in utterances:
        if not utt["grounding_chunk_ids"]:
            continue
        text = utt["text_clean"]
        has_factual = bool(
            re.search(r'\d+\.?\d*\s*%|million|billion', text, re.I)
            or re.search(r'\b(19|20)\d{2}\b', text)
            or re.search(r'caused|led to|resulted in|more than|faster|better', text, re.I)
        )
        if has_factual or utt["intent"] in ("answer", "challenge"):
            claims.append({
                "claim_index": len(claims),
                "utterance_id": utt["utterance_id"],
                "text": text,
                "grounding_chunk_ids": utt["grounding_chunk_ids"],
            })
    return claims


def check_facts(
    utterances: List[Dict], ranked_chunks: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Verify factual claims. Returns (utterances, issues_list)."""
    claims = _extract_claims(utterances)
    if not claims:
        logger.info("No factual claims to verify")
        return utterances, []

    chunk_map = {str(c["chunk_id"]): c for c in ranked_chunks}

    # Build prompt inputs
    claims_text = "\n".join(f"Claim {c['claim_index']}: \"{c['text']}\"" for c in claims)
    all_ids = {str(cid) for c in claims for cid in c["grounding_chunk_ids"]}
    source_text = "\n---\n".join(
        f"[Chunk {cid}]:\n{chunk_map[cid]['text'][:500]}"
        for cid in all_ids if cid in chunk_map
    ) or "No sources available."

    llm = ChatOpenAI(
        model=settings.FACT_CHECKER_MODEL,
        temperature=settings.FACT_CHECKER_TEMPERATURE,
    ).with_structured_output(BatchFactCheckResult)

    issues = []
    try:
        result: BatchFactCheckResult = llm.invoke(
            BATCH_FACT_CHECK_PROMPT.format(
                claims_text=claims_text, source_chunks_text=source_text,
            )
        )
        for v in result.results:
            if v.claim_index >= len(claims):
                continue
            claim = claims[v.claim_index]
            if v.verdict == "unsupported" or (
                v.verdict == "partially_supported" and v.confidence < 0.7
            ):
                issues.append({
                    "utterance_id": claim["utterance_id"],
                    "claim": claim["text"][:100],
                    "verdict": v.verdict,
                    "reasoning": v.reasoning,
                    "correction": v.correction,
                })
    except Exception as e:
        logger.warning(f"Fact check failed: {e}")

    logger.info(f"Fact check: {len(claims)} claims verified, {len(issues)} issues")
    return utterances, issues
