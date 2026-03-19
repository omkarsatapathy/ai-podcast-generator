"""Generate a single combined PNG diagram for all phases (1-5) using real LangGraph nodes.

This script uses LangGraph's built-in mermaid diagram generation to extract
the actual node-by-node graph structure from all 5 phases and combines them
into a single diagram rendered as one PNG file.

Usage:
    python tests/generate_graph_diagram.py

Outputs:
    - Single PNG diagram: docs/full_pipeline_graph.png (all phases combined)
"""
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables (needed for settings)
load_dotenv()

OUTPUT_PNG = project_root / "docs" / "full_pipeline_graph.png"

PHASE_NAMES = {
    1: "Phase 1: Research & Ingestion",
    2: "Phase 2: Content Planning",
    3: "Phase 3: Dialogue Generation",
    4: "Phase 4: Voice Synthesis",
    5: "Phase 5: Audio Post-Processing",
}


def _parse_mermaid_body(mermaid_text: str, phase: int) -> tuple[list[str], list[str]]:
    """Extract node definitions and edges from a LangGraph mermaid diagram.

    Prefixes every node id with ``p{phase}_`` so that nodes from different
    phases don't collide when merged into one diagram.

    Returns (node_lines, edge_lines) with the prefixed ids.
    """
    prefix = f"p{phase}_"
    nodes: list[str] = []
    edges: list[str] = []

    for line in mermaid_text.splitlines():
        stripped = line.strip()
        # Skip the init directive, graph declaration, classDef, class lines
        if (
            not stripped
            or stripped.startswith("%%{")
            or stripped.startswith("graph ")
            or stripped.startswith("classDef ")
            or stripped.startswith("class ")
        ):
            continue

        # Edge lines contain --> or -.->
        if "-->" in stripped or "-.->" in stripped:
            # Replace node ids with prefixed versions
            # Edges look like: nodeA --> nodeB; or nodeA -.-> nodeB;
            prefixed = _prefix_edge(stripped, prefix)
            edges.append(prefixed)
        elif stripped.endswith(")") or stripped.endswith(";"):
            # Node definition line
            prefixed = _prefix_node(stripped, prefix)
            nodes.append(prefixed)

    return nodes, edges


def _prefix_node(line: str, prefix: str) -> str:
    """Prefix the node id in a node definition line."""
    # Match patterns like: __start__([<p>__start__</p>]):::first
    #                  or: initialize(initialize)
    match = re.match(r'^(\s*)(\S+?)(\(.*)', line)
    if match:
        indent, node_id, rest = match.groups()
        return f"{indent}{prefix}{node_id}{rest}"
    return f"{prefix}{line}"


def _prefix_edge(line: str, prefix: str) -> str:
    """Prefix node ids in an edge line."""
    # Replace patterns like: nodeA --> nodeB; or nodeA -.-> nodeB;
    # Handle both --> and -.->
    parts = re.split(r'(\s+-->\s+|\s+-\.->\s+)', line)
    result = []
    for part in parts:
        part_stripped = part.strip()
        if part_stripped in ["-->", "-.->"] or re.match(r'^-->\s*$', part_stripped) or re.match(r'^-\.->\s*$', part_stripped):
            result.append(part)
        else:
            # This is a node id (possibly with trailing ;)
            clean = part.strip().rstrip(";").strip()
            if clean:
                trailing = ";" if part.strip().endswith(";") else ""
                result.append(f"{prefix}{clean}{trailing}")
            else:
                result.append(part)
    return "".join(result)


def _build_combined_mermaid(phase_diagrams: dict[int, str]) -> str:
    """Build a single mermaid diagram combining all phases with subgraphs."""
    lines = [
        "%%{init: {'flowchart': {'curve': 'linear'}}}%%",
        "graph TD;",
    ]

    prev_end_node = None

    for phase in sorted(phase_diagrams.keys()):
        mermaid_text = phase_diagrams[phase]
        prefix = f"p{phase}_"
        nodes, edges = _parse_mermaid_body(mermaid_text, phase)

        # Add subgraph header
        lines.append(f'    subgraph sub_phase{phase}["{PHASE_NAMES[phase]}"]')

        # Add nodes
        for node in nodes:
            lines.append(f"        {node}")

        # Add edges
        for edge in edges:
            lines.append(f"        {edge}")

        lines.append("    end")
        lines.append("")

        # Connect previous phase's __end__ to this phase's __start__
        if prev_end_node:
            start_node = f"{prefix}__start__"
            lines.append(f"    {prev_end_node} --> {start_node};")
            lines.append("")

        prev_end_node = f"{prefix}__end__"

    # Styling
    lines.append("    classDef default fill:#f2f0ff,line-height:1.2")
    lines.append("    classDef first fill-opacity:0")
    lines.append("    classDef last fill:#bfb6fc")

    return "\n".join(lines)


def generate_combined_diagram():
    """Generate a single combined PNG diagram showing all phases 1-5."""
    print("=" * 70)
    print("🎨 FULL PIPELINE GRAPH (PHASES 1-5) — LANGGRAPH NODES")
    print("=" * 70)
    print()

    try:
        print("📦 Importing all phase graphs...")
        from src.pipeline.phases.phase1_graph import create_phase1_graph
        from src.pipeline.phases.phase2_graph import create_phase2_graph
        from src.pipeline.phases.phase3_graph import create_phase3_graph
        from src.pipeline.phases.phase4_graph import create_phase4_graph
        from src.pipeline.phases.phase5_graph import create_phase5_graph
        print("✅ All imports successful\n")

        print("🔧 Creating all phase graphs...")
        creators = {
            1: create_phase1_graph,
            2: create_phase2_graph,
            3: create_phase3_graph,
            4: create_phase4_graph,
            5: create_phase5_graph,
        }

        phase_diagrams = {}
        for phase_num, creator in creators.items():
            graph = creator()
            mermaid = graph.get_graph().draw_mermaid()
            phase_diagrams[phase_num] = mermaid
            node_count = sum(1 for l in mermaid.splitlines()
                             if l.strip() and not l.strip().startswith(("%%", "graph", "classDef", "class"))
                             and "-->" not in l and "-.->" not in l)
            print(f"  Phase {phase_num}: {node_count} nodes")

        print("✅ All graphs created\n")

        print("🎨 Combining into single mermaid diagram...")
        combined = _build_combined_mermaid(phase_diagrams)
        print("✅ Combined diagram generated\n")

        print("=" * 70)
        print("📊 COMBINED MERMAID DIAGRAM")
        print("=" * 70)
        print()
        print(combined)
        print()

        # Render to PNG using mmdc
        print("🖼️  Rendering to PNG...")
        OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(combined)
            temp_mmd = f.name

        try:
            result = subprocess.run(
                ['mmdc', '-i', temp_mmd, '-o', str(OUTPUT_PNG), '-s', '3', '-w', '2400'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print(f"✅ Saved PNG diagram to: {OUTPUT_PNG}")
            else:
                print(f"❌ mmdc failed (exit {result.returncode}):")
                print(f"   stderr: {result.stderr}")
                return False
        finally:
            os.unlink(temp_mmd)

        print()
        print("=" * 70)
        print("✅ DONE!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ DIAGRAM GENERATION FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_combined_diagram()
    exit(0 if success else 1)
