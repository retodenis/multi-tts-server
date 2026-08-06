import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
VOICES_DIR = ROOT / "voices"
LATENTS_DIR = ROOT / "latents"
OUTPUT_DIR = ROOT / "output"
STATIC_DIR = Path(__file__).resolve().parent / "static"

ENGINE_PRESETS = {
    "qwen3-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen3-0.6b-base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "qwen3-customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen3-0.6b-customvoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}


@dataclass
class Config:
    engine: str = "qwen3-base"
    model: str = ENGINE_PRESETS["qwen3-base"]
    dtype: str = "bfloat16"
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = 7860
    strip_tags: bool = True
    warmup: bool = True
    max_text_length: int = 500
    default_voice: str | None = None
    use_cache: bool = True
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_qwen(self) -> bool:
        return self.engine.startswith("qwen")

    @property
    def torch_device(self) -> str:
        return self.device


def parse_cli(argv: list[str]) -> dict:
    out: dict = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--"):
            i += 1
            continue
        key = a[2:].replace("-", "_")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            val = argv[i + 1]
            i += 2
        else:
            val = True
            i += 1
        out[key] = val
    return out


def load_config(argv: list[str] | None = None) -> Config:
    argv = sys.argv[1:] if argv is None else argv
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    cli = parse_cli(argv)
    path = Path(cli.get("config", str(ROOT / "config.yaml")))
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # engine/model presets from CLI take priority
    if "engine" in cli and cli["engine"] in ENGINE_PRESETS:
        data["engine"] = cli["engine"]
        data["model"] = ENGINE_PRESETS[cli["engine"]]
    elif "engine" in cli:
        data["engine"] = cli["engine"]
    if "model" in cli:
        data["model"] = cli["model"]

    merged = {**data, **{k: v for k, v in cli.items() if k != "config"}}
    if "engine" in merged and "model" not in merged and merged["engine"] in ENGINE_PRESETS:
        merged["model"] = ENGINE_PRESETS[merged["engine"]]

    extra_keys = set(merged) - set(Config.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = {k: merged.pop(k) for k in extra_keys}
    cfg = Config(**{k: v for k, v in merged.items() if k in Config.__dataclass_fields__})
    cfg.extra = extra
    for d in (VOICES_DIR, LATENTS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
