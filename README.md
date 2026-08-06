# Multi-TTS Server (SkyrimNet)

Локальный TTS-сервер для **SkyrimNet-GamePlugin** с подменяемыми движками. Говорит
только контрактом SkyrimNet (`/tts_to_audio`, `/create_and_store_latents`, `/health`,
ping→тишина), а «мотор» внутри выбирается из `config.yaml`.

## Возможности

- **Движок**: Qwen3-TTS (Base-клонирование + CustomVoice-пресеты, 1.7B / 0.6B).
- **Русский язык**: `Russian` (10 языков: en, zh, ja, ko, de, fr, ru, pt, es, it).
- **Голоса NPC**: клип + опциональный транскрипт → кеш промптов в `latents/`, переиспользование без пересчёта.
- **Стриминг** по предложениям: `GET /tts_stream` (SSE) для UI/бенчмарка.
- **Web-UI** по-русски на `http://127.0.0.1:7860`.
- **Бенчмарк**: `benchmark.py` (TTFB, elapsed, RTF).

## Установка

Требуется Python 3.12/3.13 и (желательно) NVIDIA GPU с 6+ ГБ VRAM. Если HF-модель
gated — создай `.env` с `HF_TOKEN=hf_...`:

```
setup.bat
```

`setup.bat` создаст `.venv`, поставит PyTorch 2.6.0+cu126 (рабочая сборка под
Python 3.13 на Windows) и зависимости `qwen-tts`.

## Запуск

```
start.bat                    # с конфигом из config.yaml (по умолчанию Qwen3-Base 1.7B)
start.bat --engine qwen3-customvoice
start_cpu.bat                # CPU (медленно)
```

`config.yaml`:

```yaml
engine: qwen3-base            # qwen3-base | qwen3-0.6b-base | qwen3-customvoice | qwen3-0.6b-customvoice
model: Qwen/Qwen3-TTS-12Hz-1.7B-Base   # переопределяется пресетом engine
dtype: bfloat16              # fp32 | bfloat16
device: cuda                 # cuda | cpu
host: 127.0.0.1
port: 7860
strip_tags: true
warmup: true
max_text_length: 500
default_voice: null          # голос, если SkyrimNet не указал speaker_wav
```

## Подключение к SkyrimNet

1. Запусти сервер (`start.bat`).
2. В SkyrimNet: движок **Chatterbox / server you choose**, URL **`http://127.0.0.1:7860`**.
3. Референсы NPC SkyrimNet присылает сам через `/create_and_store_latents` (Voice Samples).
4. Для Base-клонирования качество повысится, если рядом с клипом
   `voices/<name>.wav` лежит транскрипт `voices/<name>.txt` (можно добавить через UI).
   Без транскрипта используется только тембр (`x_vector_only_mode`).

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/tts_to_audio` | JSON `{text, speaker_wav, language, temperature, top_p, top_k, speed, repetition_penalty, ...}` → WAV; `text=="ping"` → тишина |
| POST | `/create_and_store_latents` | multipart `speaker_name`, `wav_file` → регистрация голоса NPC |
| GET | `/health` | статус, движок, языки, голоса |
| GET | `/voices`, `/languages` | справочники |
| POST | `/tts` | полный синтез → JSON base64 WAV + метрики |
| GET | `/tts_stream` | SSE-стриминг по предложениям (`meta` → `chunk` → `done`) |
| POST | `/save_transcript` | сохранить транскрипт для голоса |
