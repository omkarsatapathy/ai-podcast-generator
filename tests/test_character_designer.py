#!/usr/bin/env python3
"""Quick smoke test for Character Designer components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.character import CharacterPersona, CharacterRoster
from src.agents.phase2.character_designer import (
    design_characters, VOICE_BANK, _build_voices_list,
    _get_role_rules, _get_gender_rule, _build_chapters_context,
)
from src.pipeline.phases.phase2_graph import create_phase2_graph

# 1. Model validation
try:
    CharacterPersona(name='Fail', role='host')
    print('FAIL: should have raised')
    sys.exit(1)
except Exception:
    print('OK: missing fields rejected')

try:
    CharacterRoster(characters=[])
    print('FAIL: empty roster accepted')
    sys.exit(1)
except Exception:
    print('OK: empty roster rejected')

# 2. Helper functions
sample = [{'chapter_number': 1, 'title': 'Test', 'act': 'setup',
           'energy_level': 'medium', 'key_points': ['p1'], 'transition_hook': 'Next'}]
ctx = _build_chapters_context(sample)
assert len(ctx) > 0, 'chapters context empty'
print(f'OK: chapters context = {len(ctx)} chars')

vl = _build_voices_list()
assert 'Aoede' in vl
print(f'OK: voices list = {len(VOICE_BANK)} voices')

r2 = _get_role_rules(2)
assert '2 characters' in r2
r3 = _get_role_rules(3)
assert '3 characters' in r3
print('OK: role rules for 2 and 3 speakers')

g2 = _get_gender_rule(2)
g3 = _get_gender_rule(3)
assert 'female' in g3.lower()
print('OK: gender rules')

# 3. Graph compilation
graph = create_phase2_graph()
print(f'OK: phase2 graph compiled = {type(graph).__name__}')

print('\nALL SMOKE TESTS PASSED')
