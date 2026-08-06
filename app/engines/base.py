from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    gen_ms: float
    audio_duration_s: float

    @property
    def rtf(self) -> float:
        return self.gen_ms / 1000.0 / self.audio_duration_s if self.audio_duration_s else 0.0


class Engine(abc.ABC):
    name: str = "base"
    sample_rate: int = 24000

    @abc.abstractmethod
    def load(self, cfg) -> None:
        ...

    @abc.abstractmethod
    def warmup(self) -> None:
        ...

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
    def register_voice(self, name: str, wav_bytes: bytes, transcript: str | None = None) -> None:
        ...

    def languages(self) -> list[str]:
        return []

    def supported_voices(self) -> list[str]:
        return []

    def unload(self) -> None:
        pass