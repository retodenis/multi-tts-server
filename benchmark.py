from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from urllib.parse import urlencode

BASE = "http://127.0.0.1:7860"


def get(path: str, timeout: int = 180):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def post_tts(payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/tts",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_events(text: str, voice: str | None, language: str) -> list[dict]:
    params = {"text": text, "language": language}
    if voice:
        params["speaker_wav"] = voice
    req = urllib.request.Request(BASE + "/tts_stream?" + urlencode(params))
    events: list[dict] = []
    with urllib.request.urlopen(req, timeout=180) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                raw, buf = buf.split(b"\n\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    continue
    return events


def bench_full(text: str, voice: str | None, language: str, runs: int) -> dict:
    totals, rtfs = [], []
    audio = 0.0
    for _ in range(runs):
        payload = {
            "text": text,
            "speaker_wav": voice,
            "language": language,
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
        }
        t0 = time.perf_counter()
        r = post_tts(payload)
        totals.append((time.perf_counter() - t0) * 1000.0)
        rtfs.append(r.get("rtf") or 0)
        audio = r.get("audio_duration_s", 0)
    return {
        "median_ms": round(statistics.median(totals), 1),
        "min_ms": round(min(totals), 1),
        "max_ms": round(max(totals), 1),
        "rtf": round(statistics.median(rtfs), 3),
        "audio_sec": round(audio, 2),
    }


def bench_stream(text: str, voice: str | None, language: str) -> dict:
    t0 = time.perf_counter()
    events = stream_events(text, voice, language)
    elapsed = (time.perf_counter() - t0) * 1000.0
    first = next((e for e in events if e.get("event") == "chunk"), None)
    done = next((e for e in events if e.get("event") == "done"), None)
    audio = (done or {}).get("audio_sec", 0) or 0
    return {
        "first_chunk_ms": round((first or {}).get("gen_ms", 0), 1),
        "elapsed_ms": round(elapsed, 1),
        "audio_sec": round(audio, 2),
        "rtf": round(elapsed / 1000.0 / audio, 3) if audio else 0,
    }


def main() -> None:
    global BASE
    parser = argparse.ArgumentParser(description="Benchmark the Multi-TTS server.")
    parser.add_argument("--base", default=BASE, help="Server base URL")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    BASE = args.base.rstrip("/")

    try:
        health = json.loads(get("/health", timeout=30))
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot reach {BASE}/health: {exc}")
        return

    print(f"Engine: {health['engine']}  Model: {health['model']}")
    print(f"Voices ready: {health['voices_ready']}  Voice: {args.voice or 'builtin'}")
    print("-" * 64)

    short = "Привет, путник!"
    s = bench_stream(short, args.voice, args.language)
    print("[stream] short:")
    print(f"  first_chunk_ms={s['first_chunk_ms']}  elapsed_ms={s['elapsed_ms']}  "
          f"audio_sec={s['audio_sec']}  rtf={s['rtf']}")

    paragraph = (
        "Добро пожаловать в Скайрим. Некоторые называют это провинцией. "
        "Другие — землей свободы. Но у каждого своя история."
    )
    s = bench_stream(paragraph, args.voice, args.language)
    print("[stream] paragraph:")
    print(f"  first_chunk_ms={s['first_chunk_ms']}  elapsed_ms={s['elapsed_ms']}  "
          f"audio_sec={s['audio_sec']}  rtf={s['rtf']}")

    print(f"\n[full /tts] paragraph x{args.runs}:")
    print(" ", bench_full(paragraph, args.voice, args.language, args.runs))


if __name__ == "__main__":
    main()
