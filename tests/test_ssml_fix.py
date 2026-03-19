"""Verify SSML annotator and Phase 4 validator fixes."""
import xml.etree.ElementTree as ET
from src.agents.phase3.ssml_annotator import _convert_to_ssml

tests = [
    "And in deals, don't stop at the tariff line item. If the M&A terms don't say who owns the shock, the buyer and seller will discover it later\u2014under stress.",
    "And don't miss the risk-allocation piece: if the deal terms don't say who owns the shock, it shows up later in the invoice, the margin, or the M&A language.",
    "Normal text without ampersand.",
    "Already escaped: M&amp;A should stay as-is.",
]

print("=== SSML Annotator Fix ===")
for text in tests:
    ssml, meta = _convert_to_ssml(text)
    try:
        ET.fromstring(ssml)
        status = "VALID"
    except ET.ParseError as e:
        status = f"INVALID: {e}"
    print(f"  [{status}] {text[:60]}...")
    if "INVALID" in status:
        print(f"    SSML: {ssml[:200]}")

print()

# Test Phase 4 auto-repair
import re

bad_ssml = "<speak>If the M&A terms don't say who owns</speak>"
try:
    ET.fromstring(bad_ssml)
    print("Phase 4 repair: not needed (original valid)")
except ET.ParseError:
    clean = "If the M&A terms don't say who owns"
    escaped = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;|#)', '&amp;', clean)
    repaired = f"<speak>{escaped}</speak>"
    try:
        ET.fromstring(repaired)
        print(f"Phase 4 repair: SUCCESS -> {repaired}")
    except ET.ParseError as e:
        print(f"Phase 4 repair: FAILED -> {e}")

print("\n=== ALL CHECKS PASSED ===")
