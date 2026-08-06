from __future__ import annotations

import logging

from ..config import Config
from .base import Engine

logger = logging.getLogger("engines")


def create_engine(cfg: Config) -> Engine:
    if cfg.engine.startswith("qwen"):
        from .qwen3 import Qwen3Engine

        logger.info("Selecting engine: qwen3 (%s)", cfg.model)
        return Qwen3Engine()
    raise ValueError(f"Unknown engine: {cfg.engine!r}")