from __future__ import annotations

import logging
import time

import numpy as np

from ..config import Config
from ..voices import get_transcript, latent_exists, latent_path, list_voices, save_reference, voice_wav_path
from .base import Engine, SynthesisResult

logger = logging.getLogger("engine.qwen3")

# SkyrimNet/XTTS-style codes -> Qwen full language names.
_LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

_CUSTOMVOICE_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]


def normalize_language(lang: str | None) -> str:
    if not lang:
        return "auto"
    key = lang.strip().lower()
    if key in _LANG_MAP:
        return _LANG_MAP[key]
    # already a full name?
    title = lang.strip().title()
    if title in {"Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"}:
        return title
    return "auto"


class Qwen3Engine(Engine):
    name = "qwen3"

    def __init__(self) -> None:
        self._model = None
        self._cfg: Config | None = None
        self._prompt_cache: dict[str, object] = {}

    # -- lifecycle ---------------------------------------------------------

    def load(self, cfg: Config) -> None:
        from qwen_tts import Qwen3TTSModel

        self._cfg = cfg
        device_map = "cuda:0" if cfg.device == "cuda" else "cpu"
        dtype = self._resolve_dtype(cfg.dtype)

        kwargs: dict = {"device_map": device_map, "dtype": dtype}
        if cfg.device == "cuda":
            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    cfg.model, **kwargs, attn_implementation="flash_attention_2"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Flash Attention 2 unavailable (%s); using SDPA.", exc)
                try:
                    self._model = Qwen3TTSModel.from_pretrained(
                        cfg.model, **kwargs, attn_implementation="sdpa"
                    )
                except Exception:  # noqa: BLE001
                    self._model = Qwen3TTSModel.from_pretrained(cfg.model, **kwargs)
        else:
            self._model = Qwen3TTSModel.from_pretrained(cfg.model, **kwargs)
        logger.info("Qwen3 loaded: %s on %s (%s)", cfg.model, device_map, dtype)

    def _resolve_dtype(self, dtype: str):
        import torch

        if self._cfg is None:
            return torch.float32
        d = (dtype or "").strip().lower()
        if d in ("fp16", "float16", "half"):
            return torch.float16
        if d in ("bf16", "bfloat16"):
            if self._cfg.device == "cpu":
                return torch.float32
            try:
                torch.zeros(1, device="cuda", dtype=torch.bfloat16)
                return torch.bfloat16
            except Exception:  # noqa: BLE001
                pass
        return torch.float32

    def warmup(self) -> None:
        if self._model is None:
            return
        try:
            text = "Раз, два, три — проверка связи."
            if "CustomVoice" in self._cfg.model:
                self._model.generate_custom_voice(
                    text=text, language="Russian", speaker=self._default_speaker()
                )
            else:
                self._model.generate_voice_clone(
                    text=text, language="Russian",
                    ref_audio=(np.zeros(24000, dtype=np.float32), 24000),
                    ref_text="",
                    x_vector_only_mode=True,
                )
            logger.info("Qwen3 warmup done")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen3 warmup failed: %s", exc)

    def _default_speaker(self) -> str:
        dflt = self._cfg.default_voice if self._cfg else None
        if dflt and dflt.lower() in {s.lower() for s in _CUSTOMVOICE_SPEAKERS}:
            return next(s for s in _CUSTOMVOICE_SPEAKERS if s.lower() == dflt.lower())
        return "Vivian"

    # -- voices ------------------------------------------------------------

    def register_voice(self, name: str, wav_bytes: bytes, transcript: str | None = None) -> None:
        save_reference(name, wav_bytes, transcript)
        self._prompt_cache.pop(name, None)
        latent_path(name).unlink(missing_ok=True)

    def supported_voices(self) -> list[str]:
        if self._is_customvoice():
            return self._customvoice_speakers()
        return list_voices()

    def _customvoice_speakers(self) -> list[str]:
        supported = self._model.get_supported_speakers() if self._model is not None else None
        return list(supported or _CUSTOMVOICE_SPEAKERS)

    def languages(self) -> list[str]:
        return ["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

    def _is_customvoice(self) -> bool:
        return bool(self._cfg and "CustomVoice" in self._cfg.model)

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
        if self._model is None:
            raise RuntimeError("Qwen3 engine not loaded")

        lang = normalize_language(language)
        start = time.perf_counter()

        if self._is_customvoice():
            wavs, sr = self._generate_custom_voice(text, voice_key, lang, temperature, top_p, repetition_penalty)
        else:
            wavs, sr = self._generate_clone(text, voice_key, lang, temperature, top_p, repetition_penalty)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return SynthesisResult(
            audio=audio,
            sample_rate=int(sr),
            gen_ms=elapsed_ms,
            audio_duration_s=len(audio) / sr,
        )

    def _generate_custom_voice(self, text, voice_key, lang, temperature, top_p, repetition_penalty=None):
        speaker = self._default_speaker()
        if voice_key and voice_key.lower() in {s.lower() for s in self._customvoice_speakers()}:
            speaker = next(s for s in self._customvoice_speakers() if s.lower() == voice_key.lower())
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        return self._model.generate_custom_voice(text=text, language=lang, speaker=speaker, **kwargs)

    def _get_prompt(self, voice_key: str | None):
        if not voice_key:
            return None, None
        if voice_key in self._prompt_cache:
            return self._prompt_cache[voice_key], True

        path = voice_wav_path(voice_key)
        if not path.exists():
            return None, None

        latent_p = latent_path(voice_key)
        if latent_exists(voice_key):
            import torch

            prompt = torch.load(latent_p, map_location="cpu", weights_only=False)
            self._prompt_cache[voice_key] = prompt
            return prompt, True

        transcript = get_transcript(voice_key)
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(path),
            ref_text=transcript or "",
            x_vector_only_mode=not transcript,
        )
        self._prompt_cache[voice_key] = prompt
        return prompt, False

    def _generate_clone(self, text, voice_key, lang, temperature, top_p, repetition_penalty=None):
        prompt, from_cache = self._get_prompt(voice_key)
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)

        if prompt is not None:
            return self._model.generate_voice_clone(
                text=text, language=lang, voice_clone_prompt=prompt, **kwargs
            )
        # No registered reference voice -> speak with an empty speaker prompt.
        return self._model.generate_voice_clone(
            text=text, language=lang,
            ref_audio=(np.zeros(24000, dtype=np.float32), 24000),
            ref_text="",
            x_vector_only_mode=True,
            **kwargs,
        )

    def unload(self) -> None:
        self._model = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
