"""Phase 3 Agent: SSML Annotator.

Converts naturalness markers to TTS-ready output.
Google TTS: SSML markup | Eleven Labs: clean text with metadata.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _convert_to_ssml(text: str) -> Tuple[str, Dict]:
    """Convert naturalness markers to Google TTS SSML tags."""
    meta = {}

    # Extract audio mixer metadata (handled by Phase 5)
    m = re.search(r'\[INTERRUPT:([\d.]+s)\]', text)
    if m:
        meta["interrupt_duration"] = m.group(1)
        text = text.replace(m.group(0), "")

    m = re.search(r'\[BACKCHANNEL:(\w+)\]', text)
    if m:
        meta["backchannel_speaker"] = m.group(1)
        text = text.replace(m.group(0), "")

    # Pauses
    text = text.replace("[PAUSE:short]", '<break time="400ms"/>')
    text = text.replace("[PAUSE:long]", '<break time="800ms"/>')

    # Fillers
    text = text.replace("[FILLER:thinking]", '<break time="300ms"/>um,')
    text = text.replace("[FILLER:agreement]", "yeah,")

    # Emphasis: [EMPHASIS:word] → <emphasis level="strong">word</emphasis>
    text = re.sub(
        r'\[EMPHASIS:([^\]]+)\]',
        r'<emphasis level="strong">\1</emphasis>',
        text,
    )

    # Pacing — wrap to end of sentence (simplified approach)
    text = re.sub(r'\[PACE:fast\]\s*', '<prosody rate="fast">', text)
    text = re.sub(r'\[PACE:slow\]\s*', '<prosody rate="slow">', text)
    # Close any open prosody tags
    open_tags = text.count("<prosody ") - text.count("</prosody>")
    text += "</prosody>" * max(0, open_tags)

    # Laughs
    text = text.replace("[LAUGH:light]", '<break time="200ms"/>heh')
    text = text.replace("[LAUGH:medium]", '<break time="300ms"/>')

    # False starts
    text = text.replace("[FALSE_START]", "\u2014<break time=\"200ms\"/> well,")

    # Strip any leftover markers
    text = re.sub(r'\[[A-Z_]+(?::[^\]]+)?\]', '', text)

    return f"<speak>{text.strip()}</speak>", meta


def _convert_to_plaintext(text: str) -> Tuple[str, Dict]:
    """Strip markers for Eleven Labs (prosody controlled via API params)."""
    meta = {}

    m = re.search(r'\[INTERRUPT:([\d.]+s)\]', text)
    if m:
        meta["interrupt_duration"] = m.group(1)

    m = re.search(r'\[BACKCHANNEL:(\w+)\]', text)
    if m:
        meta["backchannel_speaker"] = m.group(1)

    # Convert fillers to spoken text
    text = text.replace("[FILLER:thinking]", "um,")
    text = text.replace("[FILLER:agreement]", "yeah,")
    text = text.replace("[LAUGH:light]", "heh,")
    text = text.replace("[FALSE_START]", "\u2014 well,")
    text = text.replace("[PAUSE:short]", "...").replace("[PAUSE:long]", "... ...")

    # Keep emphasized words, just remove the marker brackets
    text = re.sub(r'\[EMPHASIS:([^\]]+)\]', r'\1', text)

    # Strip remaining markers
    text = re.sub(r'\[[A-Z_]+(?::[^\]]+)?\]', '', text)
    return re.sub(r'\s+', ' ', text).strip(), meta


def _validate_ssml(ssml: str) -> bool:
    """Check SSML is well-formed XML."""
    try:
        ET.fromstring(ssml)
        return True
    except ET.ParseError:
        return False


def annotate_chapter(utterances: List[Dict], personas: List[Dict]) -> List[Dict]:
    """Convert naturalness markers to TTS output for all utterances."""
    provider = settings.TTS_PROVIDER
    convert = _convert_to_ssml if provider == "google" else _convert_to_plaintext

    for utt in utterances:
        source = utt.get("text_with_naturalness") or utt["text_clean"]
        result, meta = convert(source)

        # Validate SSML; fallback to plain text in <speak> if broken
        if provider == "google" and not _validate_ssml(result):
            result = f"<speak>{utt['text_clean']}</speak>"
            logger.warning(f"Invalid SSML for {utt['utterance_id']}, using fallback")

        utt["text_ssml"] = result
        utt["audio_metadata"].update(meta)

    logger.info(f"SSML annotation ({provider}): {len(utterances)} utterances done")
    return utterances
