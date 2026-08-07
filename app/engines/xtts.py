from __future__ import annotations

import logging
import os
import time

import numpy as np

from ..config import Config, LATENTS_DIR
from ..voices import get_transcript, list_voices, save_reference, voice_wav_path
from .base import Engine, SynthesisResult

logger = logging.getLogger("engine.xtts")

_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS language codes. NOTE: zh -> zh-cn is required by the model.
_XTTS_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
]

_LANG_ALIASES = {
    "zh": "zh-cn",
}


def normalize_language(lang: str | None) -> str:
    if not lang:
        return "en"
    key = lang.strip().lower()
    if key in _LANG_ALIASES:
        return _LANG_ALIASES[key]
    if key in _XTTS_LANGUAGES:
        return key
    # full names -> codes
    full = {
        "english": "en", "spanish": "es", "french": "fr", "german": "de",
        "italian": "it", "portuguese": "pt", "polish": "pl", "turkish": "tr",
        "russian": "ru", "dutch": "nl", "czech": "cs", "arabic": "ar",
        "chinese": "zh-cn", "hungarian": "hu", "korean": "ko", "japanese": "ja",
        "hindi": "hi",
    }
    if key in full:
        return full[key]
    return "en"


class XttsEngine(Engine):
    name = "xtts"
    sample_rate = 24000

    def __init__(self) -> None:
        self._tts = None
        self._cfg: Config | None = None
        self._prompt_cache: dict[str, object] = {}

    # -- lifecycle ---------------------------------------------------------

    def load(self, cfg: Config) -> None:
        from TTS.api import TTS

        self._cfg = cfg
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        device = cfg.torch_device
        try:
            self._tts = TTS(_MODEL_NAME)
            if device != "cpu":
                self._tts.to(device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS API load failed (%s); retrying on CPU.", exc)
            self._tts = TTS(_MODEL_NAME)
            device = "cpu"
        self._prompt_cache.clear()
        logger.info("XTTS loaded: %s (device=%s)", _MODEL_NAME, device)

    def warmup(self) -> None:
        if self._tts is None:
            return
        voice = self._default_voice_key()
        if not voice:
            logger.info("XTTS warmup skipped (no registered/default voice).")
            return
        try:
            self.synthesize("Раз, два, три — проверка связи.", voice, language="ru")
            logger.info("XTTS warmup done")
        except Exception as exc:  # noqa: BLE001
            logger.warning("XTTS warmup failed: %s", exc)

    def _default_voice_key(self) -> str | None:
        dflt = self._cfg.default_voice if self._cfg else None
        if dflt and voice_wav_path(dflt).exists():
            return dflt
        if list_voices():
            return list_voices()[0]
        return None

    # -- voices ------------------------------------------------------------

    def register_voice(self, name: str, wav_bytes: bytes, transcript: str | None = None) -> None:
        save_reference(name, wav_bytes, transcript)
        self._prompt_cache.pop(name, None)
        # Precompute conditioning latents and cache them as XTTS voice .pth
        if self._tts is not None:
            try:
                self._tts.synthesizer.tts_model.clone_voice(
                    str(voice_wav_path(name)),
                    speaker_id=name,
                    voice_dir=str(LATENTS_DIR),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("XTTS latent precompute for %s failed: %s", name, exc)

    def supported_voices(self) -> list[str]:
        return list_voices()

    def languages(self) -> list[str]:
        return _XTTS_LANGUAGES

    # -- synthesis ---------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_key: str | None,
        language: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        if self._tts is None:
            raise RuntimeError("XTTS engine not loaded")

        lang = normalize_language(language)
        voice_key = voice_key or self._default_voice_key()
        if not voice_key:
            raise RuntimeError("XTTS requires a reference voice: register a voice or set default_voice.")

        kwargs: dict = {"split_sentences": False}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        if speed is not None:
            kwargs["speed"] = float(speed)

        start = time.perf_counter()
        cached = (LATENTS_DIR / f"{voice_key}.pth").exists()
        wav = self._tts.tts(
            text=text,
            language=lang,
            speaker=voice_key,
            speaker_wav=None if cached else str(voice_wav_path(voice_key)),
            voice_dir=str(LATENTS_DIR),
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            gen_ms=elapsed_ms,
            audio_duration_s=len(audio) / self.sample_rate,
        )

    def unload(self) -> None:
        self._tts = None
        self._prompt_cache.clear()
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
