import argparse
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from fishspeech_client import generate, generate_batch, voices

SPEAKER_KEYS = tuple(voices.keys())
SPEAKER_RE = re.compile(
    rf"^\s*(?P<speaker>{'|'.join(SPEAKER_KEYS)})\s*:\s*(?P<text>.+)$",
    re.IGNORECASE
)

def generate_voice_audio(path):
    episode = Path(path).resolve()
    if not episode.exists() or not episode.is_dir():
        raise SystemExit(f"Episode directory not found: {episode}")

    jobs = build_all_jobs(episode, 10, True)

    if not jobs:
        print("[done] Nothing to synthesize.")
        return

    print(f"[plan] {len(jobs)} lines to synthesize across episode: {episode}")
    # Pretty print a few
    for spk, text, outp in jobs[:10]:
        print(f"  - {spk}: {text[:80]}{'...' if len(text) > 80 else ''} -> {outp}.[auto-ext]")

    # generate_batch does concurrency under the hood (max 10)
    try:
        results = generate_batch(jobs)
    except Exception as e:
        print(f"[error] Batch synthesis failed: {e}")
        raise

    # Report
    print(f"[ok] Synthesized {len(results)} files:")
    for req, path in results.items():
        spk, text, outp = req
        print(f"  - {spk} -> {path}")

def parse_script(script_path: Path) -> List[Tuple[str, str]]:
    """
    Return list of (speaker, text) pairs found in script.txt.
    Ignores 'Shot:', 'Initial Frame:', 'Visual Description:' lines.
    """
    lines = script_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[Tuple[str, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # skip metadata rows
        if line.lower().startswith(("shot:", "initial frame:", "visual description:")):
            continue
        m = SPEAKER_RE.match(line)
        if m:
            spk = m.group("speaker").lower()
            text = m.group("text")
            if text:
                out.append((spk, text))
    return out

def find_shot_dirs(episode_dir: Path) -> List[Path]:
    """
    Return subdirectories that are purely numeric (1,2,3,...) sorted ascending.
    """
    dirs = []
    for p in episode_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            dirs.append(p)
    return sorted(dirs, key=lambda p: int(p.name))

def build_jobs_for_shot(shot_dir: Path, overwrite: bool) -> List[Tuple[str, str, str]]:
    """
    Build TTS jobs for a single shot directory.
    Each utterance gets a numeric filename (1, 2, 3...) without extension.
    If overwrite=False, skip any that already exist with common extensions.
    """
    script = shot_dir / "script.txt"
    if not script.exists():
        print(f"[skip] No script.txt in {shot_dir}")
        return []

    pairs = parse_script(script)
    if not pairs:
        print(f"[skip] No dialog lines in {script}")
        return []

    jobs: List[Tuple[str, str, str]] = []
    idx = 1
    for spk, text in pairs:
        base = shot_dir / f"{idx}"
        if not overwrite:
            # if any known extension already exists, skip
            for ext in (".wav", ".mp3", ".ogg", ".flac"):
                if (shot_dir / f"{idx}{ext}").exists():
                    print(f"[exists] {shot_dir / (str(idx)+ext)} -> skipping")
                    break
            else:
                jobs.append((spk, text, str(base)))
        else:
            jobs.append((spk, text, str(base)))
        idx += 1
    return jobs

def build_all_jobs(episode_dir: Path, max_shots: int, overwrite: bool) -> List[Tuple[str, str, str]]:
    jobs: List[Tuple[str, str, str]] = []
    shot_dirs = find_shot_dirs(episode_dir)
    if max_shots and max_shots > 0:
        shot_dirs = shot_dirs[:max_shots]
    for sd in shot_dirs:
        jobs.extend(build_jobs_for_shot(sd, overwrite))
    return jobs

def main():
    ap = argparse.ArgumentParser(description="Episode voice audio generator")
    ap.add_argument("--episode-dir", required=True, help="Path to episode_{timestamp} directory")
    args = ap.parse_args()

    generate_voice_audio(args.episode_dir)

if __name__ == "__main__":
    main()

