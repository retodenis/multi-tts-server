from __future__ import annotations

import asyncio
import base64
import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .config import OUTPUT_DIR, Config
from .engines.base import Engine, SynthesisResult
from .text_processing import clean_text
from .voices import list_voices, save_reference, save_transcript

router = APIRouter()

# Runtime state injected by main.py
_state: dict = {}


def init_state(engine: Engine, config: Config) -> None:
    _state["engine"] = engine
    _state["config"] = config


def get_engine() -> Engine:
    return _state["engine"]


def get_config() -> Config:
    return _state["config"]


def _silence_wav(sr: int = 24000, dur_s: float = 0.1) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * dur_s), dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _result_to_wav(result: SynthesisResult) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, result.audio, result.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SynthesisRequest(BaseModel):
    text: str = "Hello, world!"
    speaker_wav: str | None = None
    language: str | None = None
    accent: str | None = None
    save_path: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    speed: float | None = None
    repetition_penalty: float | None = None
    override: bool = False


def _prepare_text(req: SynthesisRequest, config: Config) -> SynthesisRequest:
    text = req.text
    if config.strip_tags:
        text = clean_text(text, strip=True)
    if config.max_text_length and len(text) > config.max_text_length:
        text = text[: config.max_text_length]
    return SynthesisRequest(**{**req.model_dump(), "text": text})


def _synthesize_sync(engine: Engine, req: SynthesisRequest, config: Config) -> SynthesisResult:
    voice_key = req.speaker_wav or config.default_voice
    return engine.synthesize(
        text=req.text,
        voice_key=voice_key,
        language=req.language,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        speed=req.speed,
    )


# ---------------------------------------------------------------------------
# SkyrimNet contract
# ---------------------------------------------------------------------------


@router.post("/tts_to_audio")
@router.post("/tts_to_audio/")
async def tts_to_audio(req: SynthesisRequest):
    engine = get_engine()
    config = get_config()

    if req.text.strip().lower() == "ping":
        return Response(content=_silence_wav(), media_type="audio/wav")

    req = _prepare_text(req, config)
    if not req.text:
        raise HTTPException(status_code=400, detail="Empty text after cleaning.")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _synthesize_sync, engine, req, config)

    media = _result_to_wav(result)
    if req.save_path:
        path = Path(req.save_path)
        try:
            if not path.is_absolute():
                path = OUTPUT_DIR / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(media)
        except OSError:
            pass

    resp = Response(content=media, media_type="audio/wav")
    resp.headers["X-Gen-Time-Ms"] = str(int(result.gen_ms))
    resp.headers["X-Audio-Duration-S"] = str(round(result.audio_duration_s, 3))
    resp.headers["X-RTF"] = str(round(result.rtf, 4))
    return resp


@router.post("/create_and_store_latents")
@router.post("/create_and_store_latents/")
async def create_and_store_latents(
    speaker_name: str = Form(...),
    wav_file: UploadFile = File(...),
    language: str = Form("ru"),
):
    engine = get_engine()
    config = get_config()

    if not speaker_name or not speaker_name.strip():
        raise HTTPException(status_code=400, detail="speaker_name is required")
    wav_bytes = await wav_file.read()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="wav_file is empty")

    transcript = None
    save_reference(speaker_name, wav_bytes, transcript)
    engine.register_voice(speaker_name, wav_bytes, transcript)

    return {
        "message": f"Voice '{speaker_name}' registered.",
        "speaker_name": speaker_name,
        "language": language,
        "latent_shapes": {"gpt_cond_latent": [], "speaker_embedding": []},
    }


class TranscriptRequest(BaseModel):
    name: str
    transcript: str


@router.post("/save_transcript")
def save_transcript_endpoint(req: TranscriptRequest):
    save_transcript(req.name, req.transcript)
    return {"ok": True, "name": req.name}


@router.get("/health")
def health():
    engine = get_engine()
    config = get_config()
    model_loaded = getattr(engine, "_model", None) is not None or getattr(engine, "_tts", None) is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "supported_languages": engine.languages(),
        "voices_ready": len(list_voices()) > 0,
        "engine": engine.name,
        "model": config.model,
        "voices": list_voices(),
        "presets": engine.supported_voices() if engine.supported_voices() else [],
    }


# ---------------------------------------------------------------------------
# Convenience endpoints (UI / benchmarking)
# ---------------------------------------------------------------------------


@router.get("/voices")
def voices():
    engine = get_engine()
    return {
        "voices": list_voices(),
        "presets": engine.supported_voices() if engine.supported_voices() else [],
    }


@router.get("/languages")
def languages():
    return {"languages": get_engine().languages()}


@router.post("/tts")
async def tts(req: SynthesisRequest):
    engine = get_engine()
    config = get_config()
    req = _prepare_text(req, config)
    if not req.text:
        raise HTTPException(status_code=400, detail="Empty text after cleaning.")
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _synthesize_sync, engine, req, config)
    return {
        "audio_base64": base64.b64encode(_result_to_wav(result)).decode("ascii"),
        "sample_rate": result.sample_rate,
        "gen_ms": result.gen_ms,
        "audio_duration_s": result.audio_duration_s,
        "rtf": result.rtf,
    }


def _sse(data: dict) -> str:
    import json

    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/tts_stream")
async def tts_stream(
    text: str = "Привет! Как дела?",
    speaker_wav: str | None = None,
    language: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
):
    engine = get_engine()
    config = get_config()

    from .text_processing import split_sentences

    sentences = split_sentences(text, max_len=config.max_text_length)
    if not sentences:
        raise HTTPException(status_code=400, detail="Empty text.")

    async def gen():
        started = time.perf_counter()
        first_chunk_ms: float | None = None
        total_audio_s = 0.0
        try:
            yield _sse(
                {
                    "event": "meta",
                    "sr": engine.sample_rate,
                    "voice": speaker_wav or config.default_voice,
                    "language": language,
                    "sentences": len(sentences),
                }
            )
        except Exception:  # noqa: BLE001
            return

        loop = asyncio.get_running_loop()
        for i, sentence in enumerate(sentences):
            req = SynthesisRequest(
                text=sentence,
                speaker_wav=speaker_wav,
                language=language,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            result = await loop.run_in_executor(None, _synthesize_sync, engine, req, config)
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - started) * 1000.0
            total_audio_s += result.audio_duration_s
            b64 = base64.b64encode(_result_to_wav(result)).decode("ascii")
            yield _sse(
                {
                    "event": "chunk",
                    "index": i,
                    "total": len(sentences),
                    "gen_ms": round(result.gen_ms, 1),
                    "wav_b64": b64,
                }
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        yield _sse(
            {
                "event": "done",
                "elapsed_ms": round(elapsed_ms, 1),
                "first_chunk_ms": round(first_chunk_ms, 1) if first_chunk_ms is not None else None,
                "audio_sec": round(total_audio_s, 3),
                "rtf": round(elapsed_ms / 1000.0 / total_audio_s, 4) if total_audio_s else None,
            }
        )

    return StreamingResponse(gen(), media_type="text/event-stream")
