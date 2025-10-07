#!/usr/bin/env python3
# wav2lip_worker_v5.py — mid-only Wav2Lip + deterministic concat
# FIX: Head was coming out as full video — we now trim the HEAD with *input* options
#      (-ss 0 -t <lead>) BEFORE the first -i, and also add -shortest on outputs.
#      All segments are normalized to identical codecs/params, 48 kHz AAC, unified fps.

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import uuid
import os

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
OUTPUT_AR = 48000            # force 48 kHz across segments to avoid AAC frame warnings
OUTPUT_ABR = "128k"         # speech-safe bitrate


def _posix(p: Path) -> str:
    return p.resolve().as_posix()

# ----------------------------
# Discovery
# ----------------------------

def find_scene_dirs(episode_dir: Path) -> List[Path]:
    return sorted([p for p in episode_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                  key=lambda p: int(p.name))


def find_pairs(scene_dir: Path) -> List[Tuple[Path, Path, Path]]:
    pairs: List[Tuple[Path, Path, Path]] = []
    vids = {}
    for p in scene_dir.iterdir():
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm") and p.stem.isdigit():
            vids[int(p.stem)] = p
    for n, vpath in sorted(vids.items()):
        audio = None
        for ext in AUDIO_EXTS:
            cand = scene_dir / f"{n}{ext}"
            if cand.exists():
                audio = cand
                break
        if audio:
            pairs.append((vpath, audio, scene_dir / f"{n}_final.mp4"))
    return pairs

# ----------------------------
# Subprocess helper
# ----------------------------
async def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    return proc.returncode, out_b.decode(errors="ignore"), err_b.decode(errors="ignore")

# ----------------------------
# Probe
# ----------------------------
async def ffprobe_duration(path: Path) -> float:
    rc, out, err = await run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path)
    ])
    if rc != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{err}")
    try:
        return float(out.strip())
    except Exception:
        return 0.0

async def ffprobe_fps(path: Path) -> Optional[str]:
    rc, out, _ = await run_cmd([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path)
    ])
    if rc != 0:
        return None
    s = out.strip()
    return s or None

# ----------------------------
# Audio prep for W2L
# ----------------------------
async def ensure_wav_16k(input_audio: Path, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_wav = tmp_dir / (input_audio.stem + "_16k.wav")
    rc, _, err = await run_cmd(["ffmpeg", "-y", "-i", str(input_audio), "-ac", "1", "-ar", "16000", str(out_wav)])
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed converting audio: {input_audio}\n{err}")
    return out_wav

# ----------------------------
# Wav2Lip
# ----------------------------
async def wav2lip_once(wav2lip_path: Path, checkpoint: Path, video_in: Path, audio_wav: Path, outfile_tmp: Path,
                       resize_factor: int = 1, fps: Optional[int] = None, nosmooth: bool = False) -> str:
    infer = wav2lip_path / "inference.py"
    if not infer.exists():
        raise FileNotFoundError(f"inference.py not found at {infer}")
    cmd = [sys.executable, str(infer),
           "--checkpoint_path", _posix(checkpoint),
           "--face", _posix(video_in),
           "--audio", _posix(audio_wav),
           "--outfile", _posix(outfile_tmp),
           "--resize_factor", str(resize_factor)]
    if nosmooth:
        cmd.append("--nosmooth")
    if fps:
        cmd += ["--fps", str(fps)]
    rc, out, err = await run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"Wav2Lip failed for {video_in.name}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out

# ----------------------------
# Split head/mid/tail (mid-only W2L)
# ----------------------------
async def split_head_mid_tail(video_in: Path, head_s: float, tail_s: float, tmp_dir: Path,
                              force_fps: Optional[int] = None) -> Tuple[Path, Path, Path, float, str]:
    dur = await ffprobe_duration(video_in)
    if dur <= head_s + tail_s + 0.05:
        # Too short; signal pass-through by returning originals and mid_len
        ref = str(force_fps) if force_fps else (await ffprobe_fps(video_in) or "24")
        return (video_in, video_in, video_in, max(0.0, dur), ref)

    mid_len = max(0.0, dur - head_s - tail_s)
    ref_fps = str(force_fps) if force_fps else (await ffprobe_fps(video_in) or "24")

    head = tmp_dir / "head.mp4"
    mid_in = tmp_dir / "mid_in.mp4"   # video-only to feed W2L
    tail = tmp_dir / "tail.mp4"

    # HEAD (CRITICAL FIX): use input trimming (-ss before -i and -t before -i), and -shortest
    rc, _, err = await run_cmd([
        "ffmpeg", "-y", "-fflags", "+genpts",
        "-ss", "0", "-t", f"{head_s:.3f}", "-i", str(video_in),
        "-f", "lavfi", "-t", f"{head_s:.3f}", "-i", f"anullsrc=channel_layout=mono:sample_rate={OUTPUT_AR}",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", ref_fps,
        "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
        "-shortest", "-movflags", "+faststart", str(head)
    ])
    if rc != 0:
        raise RuntimeError(f"split head failed: {err}")

    # MID input (video only): input-trim
    rc, _, err = await run_cmd([
        "ffmpeg", "-y", "-fflags", "+genpts",
        "-ss", f"{head_s:.3f}", "-t", f"{mid_len:.3f}", "-i", str(video_in),
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", ref_fps,
        "-movflags", "+faststart", str(mid_in)
    ])
    if rc != 0:
        raise RuntimeError(f"split mid failed: {err}")

    # TAIL (silent at 48 kHz): also -shortest for belt-and-suspenders
    rc, _, err = await run_cmd([
        "ffmpeg", "-y", "-fflags", "+genpts",
        "-sseof", f"-{tail_s:.3f}", "-i", str(video_in), "-t", f"{tail_s:.3f}",
        "-f", "lavfi", "-t", f"{tail_s:.3f}", "-i", f"anullsrc=channel_layout=mono:sample_rate={OUTPUT_AR}",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", ref_fps,
        "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
        "-shortest", "-movflags", "+faststart", str(tail)
    ])
    if rc != 0:
        raise RuntimeError(f"split tail failed: {err}")

    return head, mid_in, tail, mid_len, ref_fps

# ----------------------------
# Build final (concat demuxer)
# ----------------------------
async def build_final_from_mid(head: Path, mid_synced: Path, tail: Path, out_final: Path, ref_fps: str) -> Path:
    tmp = out_final.parent / f"_tmp_concat_{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=True)
    listfile = tmp / "list.txt"
    listfile.write_text(f"file '{head.as_posix()}'\nfile '{mid_synced.as_posix()}'\nfile '{tail.as_posix()}'\n", encoding="utf-8")

    rc, _, err = await run_cmd([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", ref_fps,
        "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
        "-movflags", "+faststart", str(out_final)
    ])
    shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0 or not out_final.exists():
        raise RuntimeError(f"concat failed: {err}")
    return out_final

# ----------------------------
# Per-pair pipeline
# ----------------------------
async def process_pair(wav2lip_path: Path, checkpoint: Path, pair: Tuple[Path, Path, Path], overwrite: bool,
                       fps: Optional[int], resize_factor: int, nosmooth: bool,
                       breathing_room: bool, lead_seconds: float, tail_seconds: float) -> Optional[Path]:
    video_in, audio_in, out_final = pair
    if out_final.exists() and not overwrite:
        print(f"[exists] {out_final} -> skipping")
        return out_final

    shot_num = out_final.stem.split("_")[0]
    tmp_dir = out_final.parent / f"_tmp_w2l_{shot_num}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    staging_root = wav2lip_path / "_worker_out"
    staging_root.mkdir(parents=True, exist_ok=True)

    try:
        audio_wav_16k = await ensure_wav_16k(audio_in, tmp_dir)

        if breathing_room:
            head, mid_in, tail, mid_len, ref_fps = await split_head_mid_tail(
                video_in, lead_seconds, tail_seconds, tmp_dir, force_fps=fps
            )
            if head is video_in:  # too short; fallback
                rc, _, err = await run_cmd([
                    "ffmpeg", "-y", "-i", str(video_in), "-i", str(audio_in),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
                    "-shortest", str(out_final)
                ])
                if rc != 0:
                    raise RuntimeError(f"plain mux failed: {err}")
                return out_final

            # Run W2L on mid only
            w2l_mid_raw = staging_root / f"{uuid.uuid4().hex}.avi"
            made_lipsync = False
            try:
                # Convert rational fps to int for W2L
                if "/" in ref_fps:
                    num, den = ref_fps.split("/")
                    fps_int = int(round(float(num) / float(den)))
                else:
                    fps_int = int(ref_fps)
                _ = await wav2lip_once(wav2lip_path, checkpoint, mid_in, audio_wav_16k, w2l_mid_raw,
                                       resize_factor=resize_factor, fps=fps_int, nosmooth=nosmooth)
                made_lipsync = True
            except Exception as e:
                print(f"[warn] Wav2Lip(mid) failed: {e}")
                made_lipsync = False

            # Choose mid source
            mid_src = mid_in
            if made_lipsync and w2l_mid_raw.exists():
                mid_synced = tmp_dir / "mid_synced.mp4"
                rc, _, err = await run_cmd([
                    "ffmpeg", "-y", "-fflags", "+genpts",
                    "-i", str(w2l_mid_raw),
                    "-i", str(audio_in), "-ss", "0", "-t", f"{mid_len:.3f}",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", ref_fps,
                    "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
                    "-t", f"{mid_len:.3f}",  # also cap output length
                    "-shortest", "-movflags", "+faststart", str(mid_synced)
                ])
                if rc == 0 and mid_synced.exists():
                    mid_src = mid_synced
                else:
                    print(f"[warn] mid audio swap failed; using raw W2L mid. {err}")
                    mid_raw_mp4 = tmp_dir / "mid_raw.mp4"
                    rc, _, err = await run_cmd([
                        "ffmpeg", "-y", "-i", str(w2l_mid_raw),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", ref_fps,
                        "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
                        "-movflags", "+faststart", str(mid_raw_mp4)
                    ])
                    if rc == 0 and mid_raw_mp4.exists():
                        mid_src = mid_raw_mp4

            else:
                # No W2L: overlay dialog on original mid_in
                mid_overlay = tmp_dir / "mid_overlay.mp4"
                rc, _, err = await run_cmd([
                    "ffmpeg", "-y", "-fflags", "+genpts",
                    "-i", str(mid_in), "-i", str(audio_in), "-ss", "0", "-t", f"{mid_len:.3f}",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
                    "-t", f"{mid_len:.3f}", "-shortest", "-movflags", "+faststart", str(mid_overlay)
                ])
                if rc == 0 and mid_overlay.exists():
                    mid_src = mid_overlay

            out = await build_final_from_mid(head, mid_src, tail, out_final, ref_fps)
            print(f"[ok] Breathing-room final -> {out}")
            return out

        # --------------- non-breathing room: simple full-clip flow ---------------
        w2l_raw = staging_root / f"{uuid.uuid4().hex}.avi"
        made_lipsync = False
        try:
            _ = await wav2lip_once(wav2lip_path, checkpoint, video_in, audio_wav_16k, w2l_raw,
                                   resize_factor=resize_factor, fps=fps, nosmooth=nosmooth)
            made_lipsync = True
        except Exception as e:
            print(f"[warn] Wav2Lip(full) failed: {e}")
            made_lipsync = False

        if made_lipsync and w2l_raw.exists():
            ref_fps = str(fps) if fps else (await ffprobe_fps(video_in) or "24")
            w2l_mp4 = tmp_dir / "w2l_full.mp4"
            rc, _, err = await run_cmd([
                "ffmpeg", "-y", "-i", str(w2l_raw),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", ref_fps,
                "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
                "-movflags", "+faststart", str(w2l_mp4)
            ])
            if rc != 0 or not w2l_mp4.exists():
                raise RuntimeError(f"transcode W2L->mp4 failed: {err}")
            muxed = out_final.with_name(f"{out_final.stem}_with_audio.mp4")
            rc, _, err = await run_cmd([
                "ffmpeg", "-y", "-i", str(w2l_mp4), "-i", str(audio_in),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
                "-shortest", "-movflags", "+faststart", str(muxed)
            ])
            if rc == 0 and muxed.exists():
                shutil.move(str(muxed), str(out_final))
            else:
                shutil.move(str(w2l_mp4), str(out_final))
            print(f"[ok] Wav2Lip produced video: {out_final}")
            return out_final

        # Plain fallback mux
        rc, _, err = await run_cmd([
            "ffmpeg", "-y", "-i", str(video_in), "-i", str(audio_in),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR),
            "-shortest", str(out_final)
        ])
        if rc != 0 or not out_final.exists():
            rc, _, err = await run_cmd([
                "ffmpeg", "-y", "-i", str(video_in), "-i", str(audio_in),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", OUTPUT_ABR, "-ar", str(OUTPUT_AR), "-shortest", str(out_final)
            ])
            if rc != 0 or not out_final.exists():
                raise RuntimeError(f"plain mux failed: {err}")
        print(f"[ok] Plain mux -> {out_final}")
        return out_final

    finally:
        try:
            for p in tmp_dir.iterdir():
                try: p.unlink()
                except Exception: pass
            tmp_dir.rmdir()
        except Exception:
            pass

# ----------------------------
# Episode runner & CLI
# ----------------------------
async def run_episode(episode_dir: Path, wav2lip_path: Path, checkpoint: Path, concurrency: int, overwrite: bool,
                      fps: Optional[int], resize_factor: int, nosmooth: bool, max_scenes: int,
                      breathing_room: bool, lead_seconds: float, tail_seconds: float) -> None:
    scenes = find_scene_dirs(episode_dir)
    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    sem = asyncio.Semaphore(concurrency)

    async def handle_scene(scene_dir: Path):
        pairs = find_pairs(scene_dir)
        if not pairs:
            print(f"[skip] {scene_dir}: no <N>.mp4 + <N>.(wav|mp3|...) pairs found")
            return 0, 0
        tasks = []
        for pair in pairs:
            async def one(pair=pair):
                async with sem:
                    try:
                        await process_pair(wav2lip_path, checkpoint, pair, overwrite, fps, resize_factor, nosmooth,
                                           breathing_room, lead_seconds, tail_seconds)
                        return 1
                    except Exception as e:
                        print(f"[error] {scene_dir.name} {pair[0].name}: {e}")
                        return 0
            tasks.append(asyncio.create_task(one()))
        done = sum(await asyncio.gather(*tasks))
        return done, len(tasks)

    totals_done = 0
    totals_sched = 0
    results = await asyncio.gather(*[handle_scene(sd) for sd in scenes])
    for d, s in results:
        totals_done += d
        totals_sched += s
    print(f"[done] Lip-synced {totals_done}/{totals_sched} shots across {len(scenes)} scene folders. (Lead {lead_seconds}s, Tail {tail_seconds}s, Breathing={breathing_room})")


def main():
    ap = argparse.ArgumentParser(description="Wav2Lip mid-only worker with fixed head trim and deterministic concat")
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--wav2lip-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--resize-factor", type=int, default=1)
    ap.add_argument("--nosmooth", action="store_true")
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--breathing-room", action="store_true")
    ap.add_argument("--lead-seconds", type=float, default=1.0)
    ap.add_argument("--tail-seconds", type=float, default=1.0)
    args = ap.parse_args()

    asyncio.run(run_episode(
        episode_dir=Path(args.episode_dir).resolve(),
        wav2lip_path=Path(args.wav2lip_path).resolve(),
        checkpoint=Path(args.checkpoint).resolve(),
        concurrency=args.concurrency,
        overwrite=args.overwrite,
        fps=args.fps,
        resize_factor=args.resize_factor,
        nosmooth=args.nosmooth,
        max_scenes=args.max_scenes,
        breathing_room=args.breathing_room,
        lead_seconds=args.lead_seconds,
        tail_seconds=args.tail_seconds,
    ))

if __name__ == "__main__":
    main()
