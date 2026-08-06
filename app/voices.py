from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import soundfile as sf

from .config import LATENTS_DIR, VOICES_DIR

_SAFE = re.compile(r"[^A-Za-z0-9_\-\.]+")


def sanitize_name(name: str) -> str:
    cleaned = _SAFE.sub("_", name.strip())
    return cleaned.strip("_") or "voice"


def voice_wav_path(name: str) -> Path:
    return VOICES_DIR / f"{sanitize_name(name)}.wav"


def voice_transcript_path(name: str) -> Path:
    return VOICES_DIR / f"{sanitize_name(name)}.txt"


def list_voices() -> list[str]:
    return sorted(
        p.stem for p in VOICES_DIR.glob("*.wav")
    )


def save_reference(name: str, wav_bytes: bytes, transcript: str | None = None) -> Path:
    path = voice_wav_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes)
    if transcript:
        save_transcript(name, transcript)
    return path


def save_transcript(name: str, transcript: str) -> Path:
    path = voice_transcript_path(name)
    path.write_text(transcript.strip(), encoding="utf-8")
    return path


def get_transcript(name: str) -> str | None:
    path = voice_transcript_path(name)
    if path.exists():
        txt = path.read_text(encoding="utf-8").strip()
        return txt or None
    return None


def remove_voice(name: str) -> None:
    for p in (voice_wav_path(name), voice_transcript_path(name), latent_path(name)):
        p.unlink(missing_ok=True)


def latent_path(name: str) -> Path:
    return LATENTS_DIR / f"{sanitize_name(name)}.pt"


def latent_exists(name: str) -> bool:
    return latent_path(name).exists()


def cache_key(*parts: object) -> str:
    blob = "|".join(str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_wav(path: Path, target_sr: int | None = None) -> tuple[object, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if target_sr and sr != target_sr:
        import numpy as np
        from librosa import resample

        audio = resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def unique_output_path(prefix: str, ext: str = ".wav") -> Path:
    from .config import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{prefix}_{int(time.time() * 1000)}{ext}"


def voice_languages() -> dict[str, list[str]]:
    # Language list of each engine is handled by the engine itself.
    return {name: [] for name in list_voices()}
