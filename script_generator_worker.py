import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List

from shots_generator import generate_shots_for_beat
from beats_generator import generate_beats

def generate_script(summary: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(f"episode_{timestamp}")
    base.mkdir(parents=True, exist_ok=True)

    # 1) Beats
    raw_beats = generate_beats(summary)
    write_text(base / "beats.txt", raw_beats)

    # 2) Iterate beats -> shots
    beat_lines = beats_to_list(raw_beats)
    episode_script = ""

    for idx, beat in enumerate(beat_lines, start=1):
        context = episode_script.strip()

        shots = generate_shots_for_beat(summary, beat, context=context)
        ep_dir = base / f"{idx}"
        write_text(ep_dir / "script.txt", shots)

        episode_script += f"\n\n# Beat {idx}\n{beat}\n{shots}"

def beats_to_list(beats_text: str) -> List[str]:
    # Split by newline, remove blanks, strip numbering and normalize separator
    lines = [ln.strip() for ln in beats_text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^\d+[\).\-]\s*", "", ln)  # remove a leading "1) " or "1. " etc.
        ln = ln.replace(" - ", " — ").replace("–", " — ")
        if "—" not in ln and "-" in ln:
            # last resort
            parts = ln.split("-", 1)
            ln = parts[0].strip() + " — " + parts[1].strip()
        cleaned.append(ln)
    return cleaned

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Script generator")
    ap.add_argument("--prompt", required=True, help="Summary prompt of episode to generate")
    args = ap.parse_args()

    generate_script(args.prompt)

if __name__ == "__main__":
    main()

