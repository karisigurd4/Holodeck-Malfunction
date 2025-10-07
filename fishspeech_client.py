# fishspeech_client.py
# Requires: aiohttp, asyncio
#   pip install aiohttp

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import aiohttp
import re
import hashlib
from keys import FISHSPEECH_KEY

EMOTION_TAG_RE = re.compile(r"\([^)]*\)")

# ----------------------------
# Config
# ----------------------------

voices: Dict[str, str] = {
    "picard": "91e2c6b114dd4defa3beade83d1ff288",
    "data":   "0bec16e069fc4ce69596402c90245122",
    "worf":   "ad44136c229241fea6cf88e00268dad8",
    "riker":  "be0b8d12122e429c998979e906a4b20f",
    "troi":   "7c1db7ba75bf4b738a2b1a9c6f4314b4",
    "geordi": "f90f57226f004adc8e67698064524522",
    "purple joe rogan": "b0de3c7574d7477ca5f849c32c28de72",
    "bill gates": "23ce12b507a84011be6a2f2674b24858",
    "matt berry": "ffdc2477f0d5455caf2a28dbdaa68ae6",
    "drphil": "f318352c0936462eb86b31ac1358178e"
}

ApiKey  = FISHSPEECH_KEY
ApiBase = "https://api.fish.audio"
Backend = "s1"  # used as model

# Concurrency cap
MAX_CONCURRENCY = 10

# Request defaults
REQUEST_TIMEOUT_SECONDS = 120
RETRIES = 3
RETRY_BACKOFF_BASE = 0.75  # seconds


# ----------------------------
# Internal helpers
# ----------------------------

# add near _choose_extension_from_content_type
KNOWN_AUDIO_EXTS = [".wav", ".mp3", ".ogg", ".flac"]

def _stable_slug(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _existing_variant(base: Path) -> Optional[Path]:
    """
    If 'base' has no suffix, check base with any known audio ext.
    If 'base' has a suffix, just check that exact path.
    Return the first existing path found, else None.
    """
    if base.suffix:
        return base if base.exists() else None
    for ext in KNOWN_AUDIO_EXTS:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None

def strip_emotion_tags(s: str) -> str:
    """Remove all (emotion) tags like (excited), (sad), etc. from the text."""
    return EMOTION_TAG_RE.sub("", s).strip()

def _tts_url() -> str:
    base = ApiBase.rstrip("/")
    return f"{base}/v1/tts"


def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ApiKey}",
    }


def _payload(text: str, reference_id: str) -> Dict[str, object]:
    return {
        "text": text,
        "reference_id": reference_id,
        "model": Backend,     # "s1"
        "temperature": 0.8,
        "top_p": 0.8,
    }


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _choose_extension_from_content_type(content_type: Optional[str]) -> str:
    # Be pragmatic
    if not content_type:
        return ".wav"
    ct = content_type.lower()
    if "wav" in ct:
        return ".wav"
    if "mpeg" in ct or "mp3" in ct:
        return ".mp3"
    if "ogg" in ct:
        return ".ogg"
    if "flac" in ct:
        return ".flac"
    if "x-wav" in ct:
        return ".wav"
    return ".wav"


async def _post_tts(session: aiohttp.ClientSession, text: str, reference_id: str) -> Tuple[bytes, Optional[str]]:
    url = _tts_url()
    payload = _payload(text, reference_id)
    print(payload)
    data = json.dumps(payload)

    # naive retry with backoff on 5xx and network hiccups
    for attempt in range(1, RETRIES + 1):
        try:
            async with session.post(
                url,
                data=data,
                headers=_headers(),
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get("Content-Type")
                    body = await resp.read()
                    return body, content_type
                # read error text for debugging
                err_txt = await resp.text()
                # Some APIs return 4xx with JSON error, surface it plainly
                if 400 <= resp.status < 500 and resp.status != 429:
                    raise RuntimeError(f"TTS request failed {resp.status}: {err_txt.strip()}")
                # For 5xx or 429, retry
                if attempt < RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF_BASE * attempt)
                    continue
                raise RuntimeError(f"TTS request failed {resp.status} after {attempt} attempts: {err_txt.strip()}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as ex:
            if attempt < RETRIES:
                await asyncio.sleep(RETRY_BACKOFF_BASE * attempt)
                continue
            raise RuntimeError(f"TTS network error after {attempt} attempts: {ex}") from ex

    # Should not reach here
    raise RuntimeError("Unexpected retry fall-through")


# ----------------------------
# Public API
# ----------------------------

class FishSpeechClient:
    """
    Async client that can run up to MAX_CONCURRENCY TTS requests at once.
    Use the convenience sync functions below if you don't want to touch asyncio.
    """

    def __init__(self, api_key: str = ApiKey, api_base: str = ApiBase, backend: str = Backend, max_concurrency: int = MAX_CONCURRENCY):
        self.api_key = api_key
        self.api_base = api_base
        self.backend = backend
        self._sem = asyncio.Semaphore(max_concurrency)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        # override globals for this instance if provided
        global ApiKey, ApiBase, Backend
        ApiKey = self.api_key
        ApiBase = self.api_base
        Backend = self.backend
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def generate(self, speaker: str, text: str, output_path: str) -> Path:
        if speaker not in voices:
            raise ValueError(f"Unknown speaker '{speaker}'. Known: {', '.join(voices.keys())}")
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if self._session is None:
            raise RuntimeError("Client session not started. Use `async with FishSpeechClient() as c:` or call sync wrapper.")

        ref_id = voices[speaker]
        clean_text = strip_emotion_tags(text)

        # -------- resolve target path deterministically (pre-flight) --------
        out = Path(output_path)

        if out.exists() and out.is_dir():
            # deterministic, compact filename so we can skip reliably
            fname = f"{speaker}_{_stable_slug(clean_text)}"
            out = out / fname

        # default to .wav if no suffix provided
        if out.suffix == "":
            out = out.with_suffix(".wav")

        # if any variant already exists, skip network and return it
        existing = _existing_variant(out if out.suffix == "" else out)
        if existing is not None:
            return existing

        # -------- hit API only if needed --------
        async with self._sem:
            audio_bytes, content_type = await _post_tts(self._session, clean_text, ref_id)

        # choose final suffix based on content-type; if different, adjust
        desired_ext = _choose_extension_from_content_type(content_type)
        final_out = out.with_suffix(desired_ext)

        _ensure_parent_dir(final_out)
        with open(final_out, "wb") as f:
            f.write(audio_bytes)

        return final_out

    async def generate_many(self, requests: Iterable[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], Path]:
        """
        Accepts an iterable of (speaker, text, output_path) and runs them concurrently (max MAX_CONCURRENCY).
        Returns a mapping from request tuple to saved Path.
        """

        async def _one(req: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str], Path]:
            spk, txt, outp = req
            path = await self.generate(spk, txt, outp)
            return req, path

        tasks = [asyncio.create_task(_one(r)) for r in requests]
        results: Dict[Tuple[str, str, str], Path] = {}
        errors: Dict[Tuple[str, str, str], Exception] = {}

        for t in asyncio.as_completed(tasks):
            try:
                req, path = await t
                results[req] = path
            except Exception as e:
                # Capture which input failed
                # You can log this if you want, but don't swallow it silently
                # Raise at the end if you want hard-fail on any error
                # For now we keep a best-effort strategy and return what succeeded
                # while surfacing the first error at the end.
                # You can tweak this behavior to your taste.
                # To keep it explicit, we record errors and raise an aggregate at the end.
                # If you prefer soft-fail, comment out the final raise.
                pass

        # If you want strict behavior, check for failed tasks:
        failed = [t for t in tasks if t.done() and t.exception()]
        if failed:
            # raise the first error with context
            exc = failed[0].exception()
            raise RuntimeError(f"{len(failed)} TTS jobs failed, first error: {exc}") from exc

        return results


# ----------------------------
# Sync convenience wrappers
# ----------------------------

def generate(speaker: str, text: str, output_path: str) -> Path:
    """
    Synchronous one-shot convenience.
    """
    async def _run():
        async with FishSpeechClient() as client:
            return await client.generate(speaker, text, output_path)

    return asyncio.run(_run())


def generate_batch(batch: Iterable[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], Path]:
    """
    Synchronous batch convenience.
    batch = [
        ("picard", "Make it so.", "out/"),
        ("data", "I am fully functional.", "out/"),
        ...
    ]
    Each output_path can be a folder or a full filename.
    """
    async def _run():
        async with FishSpeechClient() as client:
            return await client.generate_many(batch)

    return asyncio.run(_run())

# Example usage

# p = generate("picard", "Make it so, Mr. Data.", "out/picard_line.wav")
# print(f"Wrote: {p}")

# # Batch, up to MAX_CONCURRENCY at once
# reqs = [
#     ("data",  "Captain, I am detecting...", "out/"),
#     ("worf",  "Captain, request permission to fire.", "out/"),
#     ("riker", "You have the bridge, Number One.", "out/"),
#     ("troi",  "I sense uncertainty.", "out/"),
#     ("picard","Tea, Earl Grey, hot.", "out/"),
# ]
# res = generate_batch(reqs)
# for k, v in res.items():
#     print(f"{k} -> {v}")