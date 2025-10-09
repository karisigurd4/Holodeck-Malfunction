# comfy_episode_worker.py (v2) — per-folder, multi-shot
# Usage:
#   python comfy_episode_worker.py \
#     --episode-dir ./episode_2025-10-02_212008 \
#     --frames-root ./frames \
#     --workflow ./workflow_template.json \
#     --comfy http://127.0.0.1:8188 \
#     --concurrency 3 \
#     --overwrite      # optional
#     --keep-all       # optional

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

import aiohttp

# ---------- parse beats ----------

def read_beats(episode_dir: Path) -> List[Tuple[str, str]]:
    """
    beats.txt lines like: 'Bridge - Picard announces ...'
    Returns [(scene, summary), ...] in order, 1-based indexing for folders.
    """
    beats_file = episode_dir / "beats.txt"
    if not beats_file.exists():
        raise FileNotFoundError(f"beats.txt not found at {beats_file}")
    out: List[Tuple[str, str]] = []
    for line in beats_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if " - " in line:
            scene, rest = line.split(" - ", 1)
            out.append((scene.strip(), rest.strip()))
        else:
            out.append((line, ""))
    return out

def find_scene_dirs(episode_dir: Path) -> List[Path]:
    """
    A 'scene dir' is a numeric subfolder (1, 2, 3, ...)
    """
    dirs = [p for p in episode_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))

# ---------- parse script.txt with multiple shots ----------

# Flexible, whitespace-insensitive
SHOT_HEADER_RE       = re.compile(r"^\s*Shot\s*:\s*(\d+)\s*$", re.IGNORECASE)
INITIAL_FRAME_RE     = re.compile(r"^\s*Initial\s*Frame\s*:\s*(.+)$", re.IGNORECASE)
VISUAL_DESCRIPTION_RE= re.compile(r"^\s*Visual\s*Description\s*:\s*(.+)$", re.IGNORECASE)

class ShotEntry:
    def __init__(self, number: int):
        self.number = number              # shot number within folder
        self.initial_frame: Optional[str] = None
        self.visual_desc: Optional[str] = None

    def complete(self) -> bool:
        return bool(self.initial_frame and self.visual_desc)

def parse_script_shots(script_path: Path) -> List[ShotEntry]:
    """
    Parse script.txt into a list of ShotEntry for this folder.
    Each 'Shot: N' block should contain 'Initial Frame:' and 'Visual Description:' lines.
    Dialogue lines are ignored here (this worker only cares about visuals).
    """
    lines = script_path.read_text(encoding="utf-8", errors="replace").splitlines()

    shots: List[ShotEntry] = []
    current: Optional[ShotEntry] = None

    for raw in lines:
        line = raw.rstrip("\n")

        m_shot = SHOT_HEADER_RE.match(line)
        if m_shot:
            # push previous if it exists (even if incomplete, we'll warn later)
            if current:
                shots.append(current)
            current = ShotEntry(number=int(m_shot.group(1)))
            continue

        if current:
            m_init = INITIAL_FRAME_RE.match(line)
            if m_init:
                current.initial_frame = m_init.group(1).strip()
                continue

            m_vis = VISUAL_DESCRIPTION_RE.match(line)
            if m_vis:
                current.visual_desc = m_vis.group(1).strip()
                continue

    if current:
        shots.append(current)

    # sort by shot number, just in case
    shots.sort(key=lambda s: s.number)
    return shots

# ---------- frame resolution ----------

def resolve_frame_path(frames_root: Path, scene: str, filename: str) -> Path:
    """
    Try frames/<scene>/<filename>, else same stem with alt extensions.
    """
    scene_dir = frames_root / scene
    direct = scene_dir / filename
    if direct.exists():
        return direct

    stem = Path(filename).stem
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = scene_dir / f"{stem}{ext}"
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Still not found for scene='{scene}' filename='{filename}' under {scene_dir}")

# ---------- comfy client ----------

class ComfyClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        self.base = base_url.rstrip("/")
        self.http = session

    async def upload_image(self, path: Path) -> str:
        url = f"{self.base}/upload/image"
        data = aiohttp.FormData()
        data.add_field("image", path.read_bytes(), filename=path.name, content_type="application/octet-stream")
        async with self.http.post(url, data=data) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"upload_image failed {resp.status}: {txt}")
            j = await resp.json()
            return j.get("name") or j.get("filename") or path.name

    async def run_workflow(self, workflow: Dict[str, Any]) -> str:
        url = f"{self.base}/prompt"
        payload = {"prompt": workflow}
        async with self.http.post(url, json=payload) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"/prompt failed {resp.status}: {txt}")
            j = await resp.json()
            pid = j.get("prompt_id")
            if not pid:
                raise RuntimeError(f"/prompt returned no prompt_id: {j}")
            return pid

    async def wait_for_outputs(self, prompt_id: str, poll_s: float = 1.0, timeout_s: int = 600) -> Dict[str, Any]:
        url = f"{self.base}/history/{prompt_id}"
        for _ in range(int(timeout_s / poll_s)):
            async with self.http.get(url) as resp:
                if resp.status == 200:
                    j = await resp.json()
                    if prompt_id in j:
                        entry = j[prompt_id]
                        if entry.get("outputs"):
                            return entry
                await asyncio.sleep(poll_s)
        raise TimeoutError(f"Comfy history timeout for {prompt_id}")

    async def download_output_file(self, filename: str, dest: Path, subfolder: str = "", ftype: str = "output") -> Path:
        url = f"{self.base}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": ftype}
        async with self.http.get(url, params=params) as resp:
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"/view failed {resp.status}: {txt}")
            data = await resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return dest

# ---------- workflow templating ----------

def load_workflow_template(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def deep_replace(obj: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: deep_replace(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_replace(v, mapping) for v in obj]
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    return obj

# ---------- shot execution ----------

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")

async def run_cmd(cmd):
    """
    Run an external command asynchronously.
    Returns (rc, stdout, stderr).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode("utf-8", errors="ignore"), err.decode("utf-8", errors="ignore")


async def ffprobe_duration_seconds(path: Path) -> Optional[float]:
    """
    Use ffprobe to get duration in seconds as float. Returns None if it fails.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    rc, out, err = await run_cmd(cmd)
    if rc != 0:
        print(f"[warn] ffprobe failed on {path}: {err}")
        return None
    try:
        return float(out.strip())
    except Exception:
        return None

def find_audio_for_shot(scene_dir: Path, shot_number: int) -> Optional[Path]:
    """
    Looks for <shot_number>.(mp3|wav|...) inside scene_dir.
    """
    for ext in AUDIO_EXTS:
        p = scene_dir / f"{shot_number}{ext}"
        if p.exists():
            return p
    return None

def extract_fps_from_workflow(wf: Dict[str, Any], default_fps: int = 24) -> int:
    """
    Try to find an 'fps' input in the workflow. If not found, default to 24.
    """
    try:
        for node in wf.values():
            if isinstance(node, dict):
                inputs = node.get("inputs", {})
                if isinstance(inputs, dict) and "fps" in inputs:
                    val = inputs["fps"]
                    if isinstance(val, (int, float)) and val > 0:
                        return int(val)
    except Exception:
        pass
    return default_fps

def deep_replace_preserve_types(obj: Any, mapping: Dict[str, Any]) -> Any:
    """
    Like your deep_replace, but if a value equals a placeholder exactly,
    replace with the mapped value preserving type. If it is a string that
    contains placeholders, do plain text replace.
    """
    if isinstance(obj, dict):
        return {k: deep_replace_preserve_types(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_replace_preserve_types(v, mapping) for v in obj]
    if isinstance(obj, str):
        # exact placeholder first
        if obj in mapping:
            return mapping[obj]
        # otherwise do string replace for any placeholders present
        out = obj
        for k, v in mapping.items():
            if isinstance(v, (int, float)):
                out = out.replace(k, str(v))
            else:
                out = out.replace(k, v)
        return out
    return obj

async def compute_length_frames_for_shot(base_workflow: Dict[str, Any], scene_dir: Path, shot_number: int,
                                         min_frames: int = 12, max_frames: Optional[int] = None) -> Optional[int]:
    """
    Find the audio for this shot, get duration, convert to frames based on workflow fps.
    Clamp if you want.
    """
    audio = find_audio_for_shot(scene_dir, shot_number)
    if not audio:
        return None

    secs = await ffprobe_duration_seconds(audio)
    if not secs or secs <= 0:
        return None

    fps = extract_fps_from_workflow(base_workflow, default_fps=24)
    frames = int(math.ceil(secs * fps))

    if max_frames is not None:
        frames = min(frames, max_frames)
    frames = max(frames, min_frames)
    return frames

async def run_single_shot(
    comfy: ComfyClient,
    base_workflow: Dict[str, Any],
    frames_root: Path,
    scene: str,
    scene_dir: Path,          # .../1
    shot: ShotEntry,
    overwrite: bool,
    keep_all: bool,
) -> Optional[Path]:
    if not shot.complete():
        print(f"[skip] {scene_dir.name} shot {shot.number}: missing Initial Frame or Visual Description")
        return None

    dest_base = scene_dir / f"{shot.number}"
    for ext in (".mp4", ".png", ".webp", ".jpg", ".jpeg"):
        if (scene_dir / f"{shot.number}{ext}").exists() and not overwrite:
            print(f"[exists] {scene_dir.name} shot {shot.number}: {shot.number}{ext} -> skipping")
            return scene_dir / f"{shot.number}{ext}"

    still_path = resolve_frame_path(frames_root, scene, shot.initial_frame)
    server_name = await comfy.upload_image(still_path)

    prefix = f"{scene_dir.parent.name}_{scene_dir.name}_{shot.number}"

    # Compute frames from audio, if present
    frames_len = await compute_length_frames_for_shot(base_workflow, scene_dir, shot.number,
                                                      min_frames=12, max_frames=None)

    # Build mapping. IMPORTANT: set '%%LENGTH%%' to an int so JSON stays numeric.
    mapping: Dict[str, Any] = {
        "%%IMAGE%%": server_name,
        "%%PROMPT%%": shot.visual_desc,
        "%%PREFIX%%": prefix,
    }
    if frames_len is not None:
        mapping["%%LENGTH%%"] = int(frames_len) + 56

    # If your workflow template has "length": "%%LENGTH%%", this will drop an int in there.
    wf = deep_replace_preserve_types(base_workflow, mapping)

    pid = await comfy.run_workflow(wf)
    hist = await comfy.wait_for_outputs(pid)

    produced: List[Tuple[str, str, str]] = []
    outputs = hist.get("outputs", {})
    for _, node_out in outputs.items():
        for key in ("videos", "images"):
            if key in node_out:
                for f in node_out[key]:
                    produced.append((f.get("filename"), f.get("subfolder",""), f.get("type","output")))

    if not produced:
        print(f"[warn] {scene_dir.name} shot {shot.number}: no produced files")
        return None

    saved_paths: List[Path] = []
    if keep_all:
        for fn, sub, typ in produced:
            dest = Path(str(dest_base) + Path(fn).suffix)
            saved_paths.append(await comfy.download_output_file(fn, dest, subfolder=sub, ftype=typ))
        print(f"[ok] {scene_dir.name} shot {shot.number}: saved {len(saved_paths)} files")
        return saved_paths[-1]

    chosen = None
    for fn, sub, typ in produced:
        if fn.lower().endswith(".mp4"):
            chosen = (fn, sub, typ); break
    if not chosen:
        chosen = produced[0]

    fn, sub, typ = chosen
    dest = Path(str(dest_base) + Path(fn).suffix)
    out_path = await comfy.download_output_file(fn, dest, subfolder=sub, ftype=typ)
    print(f"[ok] {scene_dir.name} shot {shot.number}: saved {out_path}")
    return out_path

# ---------- episode runner ----------

async def run_episode(
    episode_dir: Path,
    frames_root: Path,
    workflow_path: Path,
    comfy_url: str,
    concurrency: int,
    max_scenes: int,
    overwrite: bool,
    keep_all: bool,
):
    beats = read_beats(episode_dir)
    scene_dirs = find_scene_dirs(episode_dir)
    if max_scenes > 0:
        scene_dirs = scene_dirs[:max_scenes]

    # map scene folder index (1-based) -> scene name
    scene_for_index: Dict[int, str] = {i: scene for i, (scene, _) in enumerate(beats, start=1)}

    base_wf = load_workflow_template(workflow_path)

    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        comfy = ComfyClient(comfy_url, session)

        async def schedule_scene(scene_dir: Path):
            idx = int(scene_dir.name)
            scene_name = scene_for_index.get(idx)
            if not scene_name:
                print(f"[skip] {scene_dir.name}: no scene mapping from beats.txt")
                return 0, 0

            script = scene_dir / "script.txt"
            if not script.exists():
                print(f"[skip] {scene_dir.name}: no script.txt")
                return 0, 0

            shots = parse_script_shots(script)
            if not shots:
                print(f"[skip] {scene_dir.name}: no Shot blocks found")
                return 0, 0

            # schedule each shot
            tasks = []
            for shot in shots:
                async def one(shot=shot):  # capture
                    async with sem:
                        try:
                            await run_single_shot(
                                comfy=comfy,
                                base_workflow=base_wf,
                                frames_root=frames_root,
                                scene=scene_name,
                                scene_dir=scene_dir,
                                shot=shot,
                                overwrite=overwrite,
                                keep_all=keep_all,
                            )
                            return 1
                        except Exception as e:
                            print(f"[error] {scene_dir.name} shot {shot.number}: {e}")
                            return 0
                tasks.append(asyncio.create_task(one()))

            done = sum(await asyncio.gather(*tasks))
            return done, len(tasks)

        # run all scenes
        totals_done = 0
        totals_sched = 0
        for sd in scene_dirs:
            d, s = await schedule_scene(sd)
            totals_done += d
            totals_sched += s

        print(f"[done] Rendered {totals_done}/{totals_sched} shots across {len(scene_dirs)} scene folders.")

def main():
    ap = argparse.ArgumentParser(description="Episode → ComfyUI video generator (multi-shot per folder)")
    ap.add_argument("--episode-dir", required=True, help="Path to episode_{timestamp} directory")
    ap.add_argument("--frames-root", required=True, help="Path to frames/ (Bridge/, Ready Room/, ...)")
    ap.add_argument("--workflow", required=True, help="Path to workflow_template.json with placeholders (%%IMAGE%%, %%PROMPT%%, %%PREFIX%%)")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    ap.add_argument("--concurrency", type=int, default=3, help="Parallel Comfy jobs")
    ap.add_argument("--max-scenes", type=int, default=0, help="Limit number of scene folders (0=all)")
    ap.add_argument("--overwrite", action="store_true", help="Re-render even if numbered output exists")
    ap.add_argument("--keep-all", action="store_true", help="Download all produced files, not just the final mp4")
    args = ap.parse_args()

    asyncio.run(
        run_episode(
            episode_dir=Path(args.episode_dir).resolve(),
            frames_root=Path(args.frames_root).resolve(),
            workflow_path=Path(args.workflow).resolve(),
            comfy_url=args.comfy,
            concurrency=args.concurrency,
            max_scenes=args.max_scenes,
            overwrite=args.overwrite,
            keep_all=args.keep_all,
        )
    )

if __name__ == "__main__":
    main()
