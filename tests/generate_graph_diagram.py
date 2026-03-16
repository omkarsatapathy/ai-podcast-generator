"""Generate mermaid diagram for Phase 1 graph visualization.

This script uses LangGraph's built-in mermaid diagram generation to visualize
the Phase 1 workflow graph structure.

Usage:
    python tests/generate_graph_diagram.py

Outputs:
    - Prints mermaid diagram to console
    - Saves mermaid diagram to: docs/phase1_graph.mmd
    - Saves PNG diagram to: docs/phase1_graph.png (if graphviz installed)
    - Saves JPG diagram to: docs/phase1_graph.jpg (converted from PNG)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables (needed for settings)
load_dotenv()

OUTPUT_MERMAID = project_root / "docs" / "phase1_graph.mmd"
OUTPUT_PNG = project_root / "docs" / "phase1_graph.png"
OUTPUT_JPG = project_root / "docs" / "phase1_graph.jpg"


def generate_mermaid_diagram():
    """Generate and save mermaid diagram for Phase 1 graph."""
    print("=" * 70)
    print("🎨 PHASE 1 GRAPH VISUALIZATION GENERATOR")
    print("=" * 70)
    print()

    try:
        print("📦 Importing Phase 1 graph...")
        from src.pipeline.phases.phase1_graph import create_phase1_graph
        print("✅ Import successful\n")

        print("🔧 Creating graph...")
        graph = create_phase1_graph()
        print("✅ Graph created\n")

        print("🎨 Generating mermaid diagram...")
        # Get the graph representation and draw mermaid
        mermaid_diagram = graph.get_graph().draw_mermaid()
        print("✅ Mermaid diagram generated\n")

        # Print to console
        print("=" * 70)
        print("📊 MERMAID DIAGRAM")
        print("=" * 70)
        print()
        print(mermaid_diagram)
        print()

        # Save to file
        OUTPUT_MERMAID.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_MERMAID.write_text(mermaid_diagram, encoding="utf-8")
        print(f"💾 Saved mermaid diagram to: {OUTPUT_MERMAID}")
        print()

        # Try to generate PNG (requires graphviz)
        png_generated = False
        try:
            print("🖼️  Attempting to generate PNG diagram (requires graphviz)...")
            png_data = graph.get_graph().draw_mermaid_png()
            OUTPUT_PNG.write_bytes(png_data)
            print(f"✅ Saved PNG diagram to: {OUTPUT_PNG}")
            png_generated = True
            print()
        except Exception as png_error:
            print(f"⚠️  PNG generation skipped: {png_error}")
            print("   To enable PNG generation, install graphviz:")
            print("   brew install graphviz  # macOS")
            print("   sudo apt install graphviz  # Linux")
            print()

        # Convert PNG to JPG (if PNG was generated)
        if png_generated:
            try:
                print("📸 Converting PNG to JPG...")
                from PIL import Image

                # Open PNG and convert to RGB (JPG doesn't support transparency)
                img = Image.open(OUTPUT_PNG)

                # Create white background for transparency
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save as JPG with high quality
                img.save(OUTPUT_JPG, 'JPEG', quality=95)
                print(f"✅ Saved JPG diagram to: {OUTPUT_JPG}")
                print()
            except ImportError:
                print("⚠️  JPG conversion skipped: Pillow not installed")
                print("   To enable JPG conversion, install Pillow:")
                print("   pip install Pillow")
                print()
            except Exception as jpg_error:
                print(f"⚠️  JPG conversion failed: {jpg_error}")
                print()

        # Print helpful info
        print("=" * 70)
        print("📝 HOW TO USE THE DIAGRAM")
        print("=" * 70)
        print()
        print("1. View online: Copy the mermaid diagram above and paste it into:")
        print("   https://mermaid.live/")
        print()
        print("2. In VS Code: Install the 'Markdown Preview Mermaid Support' extension")
        print("   Then create a markdown file with:")
        print("   ```mermaid")
        print("   [paste diagram here]")
        print("   ```")
        print()
        print("3. In GitHub: Mermaid diagrams render automatically in .md files")
        print()
        print("=" * 70)
        print("✅ DIAGRAM GENERATION COMPLETE!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ DIAGRAM GENERATION FAILED!")
        print("=" * 70)
        print(f"\nError: {e}\n")

        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    generate_mermaid_diagram()
