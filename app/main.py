from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import api
from .config import STATIC_DIR, load_config
from .engines.factory import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = app.state.config
    logger.info("Loading engine: %s (%s) on %s", cfg.engine, cfg.model, cfg.device)
    engine = create_engine(cfg)
    engine.load(cfg)
    api.init_state(engine, cfg)
    if cfg.warmup:
        logger.info("Warming up model ...")
        engine.warmup()
    app.state.engine = engine
    logger.info("Server ready. Engine=%s Model=%s", engine.name, cfg.model)
    yield
    engine.unload()
    logger.info("Engine unloaded.")


def create_app(cfg=None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="Multi-TTS Server (SkyrimNet)", version="0.1.0", lifespan=lifespan)
    app.state.config = cfg
    app.include_router(api.router)
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


def main() -> None:
    import uvicorn

    cfg = load_config()
    app = create_app(cfg)
    logger.info("Starting uvicorn on %s:%s", cfg.host, cfg.port)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
