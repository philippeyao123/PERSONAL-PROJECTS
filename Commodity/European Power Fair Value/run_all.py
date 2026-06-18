"""Run the full pipeline end-to-end.

    python run_all.py            # ingest -> QA -> models -> trading -> figures -> AI (dry-run)
    python run_all.py --live-ai  # same, with a live Anthropic API call at the end
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
STEPS = [
    ("Data ingestion", ["ingest.py"]),
    ("Quality assurance", ["qa.py"]),
    ("Walk-forward modelling", ["models.py"]),
    ("Prompt-curve translation", ["trading.py"]),
    ("Figures", ["make_figures.py"]),
    ("AI morning note", ["ai_commentary.py"]
     + ([] if "--live-ai" in sys.argv else ["--dry-run"])),
]

for name, args in STEPS:
    print(f"\n=== {name} ===")
    r = subprocess.run([sys.executable, str(SRC / args[0]), *args[1:]], cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"step failed: {name}")
print("\nPipeline complete.")
