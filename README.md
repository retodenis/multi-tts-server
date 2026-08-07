# Multi-TTS Server (SkyrimNet)

Локальный TTS-сервер для **SkyrimNet-GamePlugin** с подменяемыми движками. Говорит
только контрактом SkyrimNet (`/tts_to_audio`, `/create_and_store_latents`, `/health`,
ping→тишина), а «мотор» внутри выбирается из `config.yaml`.

## Возможности

- **Движки**:
  - **XTTS-v2** (по умолчанию, `coqui/XTTS-v2`) — быстрое клонирование голоса по
    клипу, 17 языков (вкл. `ru`). **RTF ≈ 0.5–0.7** на RTX 50xx — примерно в 7 раз
    быстрее Qwen3.
  - **Qwen3-TTS** (Base-клонирование + CustomVoice-пресеты, 1.7B / 0.6B) — качественнее
    русская интонация, но RTF ≈ 4.
- **Русский язык**: XTTS (`ru`) и Qwen3 (`Russian`; 10 языков).
- **Голоса NPC**: клип + опциональный транскрипт → кеш латентов в `latents/`
  (XTTS) / кеш промптов (Qwen3), переиспользование без пересчёта.
- **Стриминг** по предложениям: `GET /tts_stream` (SSE) для UI/бенчмарка.
- **Web-UI** по-русски на `http://127.0.0.1:7860`.
- **Бенчмарк**: `benchmark.py` (TTFB, elapsed, RTF).

## Установка

Требуется Python 3.11/3.12/3.13 и NVIDIA GPU (см. матрицу ниже). Если HF-модель
gated — создай `.env` с `HF_TOKEN=hf_...`:

```
setup.bat
```

`setup.bat` → `setup.ps1` создаст `.venv`, определит GPU через `nvidia-smi`
(compute capability) и сам выберет подходящую сборку PyTorch:

| GPU (compute capability) | torch / torchaudio | индекс |
|---|---|---|
| RTX 50xx (≥ 12.0, Blackwell) | `2.7.0+cu128` | cu128 |
| RTX 30xx/40xx (8.x, Ampere/Ada) | `2.6.0+cu126` | cu126 |
| RTX 20xx (7.x, Turing) | `2.6.0+cu118` | cu118 |
| GTX 10xx/16xx (5.x–6.x) | `2.6.0+cu118` | cu118 |
| нет NVIDIA GPU | `2.6.0` (CPU) | cpu |

Опции установщика: `setup.bat -Recreate` (пересоздать venv),
`setup.bat -NoGpu` (только CPU), `setup.bat -SkipTorch` / `-SkipRequirements`.

> **XTTS и лицензия.** `coqui/XTTS-v2` распространяется под некоммерческой лицензией
> **CPML**. При первом скачивании сервер автоматически подтверждает согласие
> (файл `tos_agreed.txt` рядом с моделью). Используя XTTS, ты принимаешь условия
> https://coqui.ai/cpml; для коммерческого использования нужна платная лицензия
> Coqui.

## Запуск

```
start.bat                    # с конфигом из config.yaml (по умолчанию XTTS)
start.bat --engine qwen3-customvoice
start_cpu.bat                # CPU (медленно)
```

`config.yaml`:

```yaml
engine: xtts                 # xtts | qwen3-base | qwen3-0.6b-base | qwen3-customvoice | qwen3-0.6b-customvoice
model: coqui/XTTS-v2         # переопределяется пресетом engine
dtype: bfloat16              # fp32 | bfloat16 (XTTS работает в fp32 на CUDA)
device: cuda                 # cuda | cpu
host: 127.0.0.1
port: 7860
strip_tags: true
warmup: true
max_text_length: 500
default_voice: xtts_default  # голос, если SkyrimNet/запрос не указал speaker_wav
```

При первом запуске модель скачивается автоматически:
- XTTS-v2 (~1.9 ГБ) — в `%LOCALAPPDATA%\tts\...\xtts_v2`;
- Qwen3 — в кеш HuggingFace `~/.cache/huggingface/hub`.

## Как пользоваться (пошагово)

1. **Запусти сервер**: `start.bat`. Дождись в логе `Server ready`.
2. **Добавь голос NPC** (нужен клип от 3–6 секунд чистой речи):
   - через **Web-UI** на `http://127.0.0.1:7860`: форма регистрации голоса
     (имя + wav-файл);
   - или через SkyrimNet: он сам присылает референсы через
     `POST /create_and_store_latents`.
3. **Синтез**:
   - UI: выбери голос из выпадающего списка, введи текст, нажми «Озвучить».
   - SkyrimNet: см. раздел «Подключение к SkyrimNet» ниже.
   - Без указания `speaker_wav` используется `default_voice` из `config.yaml`.
4. **Проверка**: `GET /health` вернёт движок, языки и список голосов.

### Дополнительно

- **Транскрипт** (улучшает Base-клонирование Qwen3): рядом с `voices/<name>.wav`
  положи `voices/<name>.txt` или отправь через UI/`POST /save_transcript`.
  Для XTTS не обязателен.
- **Параметры речи**: `temperature`, `top_p`, `repetition_penalty`, `speed` —
  передаются в `/tts_to_audio`, `/tts` и `/tts_stream` (в UI настраиваются).
- **Смена движка**: отредактируй `config.yaml` (`engine`) и перезапусти, либо
  `start.bat --engine qwen3-0.6b-customvoice` без правки файла.

## Подключение к SkyrimNet

### Важно: какой движок выбрать в SkyrimNet

Этот сервер реализует **XTTS-контракт** (`/create_and_store_latents`,
`/tts_to_audio`, `/health`). Поэтому в SkyrimNet нужно выбрать движок **XTTS**,
**а не «Chatterbox»**:

- **XTTS** — общается с сервером по `/create_and_store_latents` и `/tts_to_audio`
  (наш контракт). ✅
- **Chatterbox** — общается по **Gradio API** (`/gradio_api/upload`,
  `/gradio_api/call/...`), которого у нас нет → сервер вернёт
  `405 {"detail":"Method Not Allowed"}`. ❌

### Пошаговая настройка

1. Запусти сервер (`start.bat`), дождись `Server ready`.
2. Запусти SkyrimNet и открой дашборд **`http://localhost:8080`**.
3. Перейди на страницу **Config** → раздел **TTS** (Text to Speech).
4. В поле движка (TTS engine) выбери **XTTS**.
5. В поле URL / endpoint укажи **`http://127.0.0.1:7860`**.
   (Если сервер на другой машине в LAN — `http://<IP-компьютера>:7860`.)
6. Нажми **Test TTS** — должен пройти тест: голос загрузится через
   `/create_and_store_latents`, синтез — через `/tts_to_audio`.
7. Для конкретных NPC: страница **Voice Samples** → для нужного голосового типа
   нажми **Select** (SkyrimNet пришлёт клип на сервер) и **🧪 Test TTS**.
   Если линий нет — заговори с NPC в игре, первая реплика станет референсом.
8. Поговори с NPC в игре.

### Если тест падает

**`Failed to upload voice sample: HTTP 405 - {"detail":"Method Not Allowed"}`**

Это значит, что в SkyrimNet выбран движок **Chatterbox** (он ходит на
`/gradio_api/upload`, которого у сервера нет). В логах SkyrimNet это видно так:
`GradioTTSInterface ... Uploading to: http://localhost:7860/gradio_api/upload`.

Решение: переключи движок в SkyrimNet на **XTTS** (см. шаг 4 выше). Ничего
переустанавливать/перезапускать на сервере не нужно.

### Как работает клонирование голоса

- **XTTS**: клип → латенты кешируются в `latents/<name>.pth` (первый раз ~0.7 с,
  далее мгновенно). Голос переиспользуется по имени без пересчёта.
- **Qwen3-Base**: качество повысится, если рядом с клипом лежит транскрипт
  `voices/<name>.txt`; без него используется только тембр (`x_vector_only_mode`).
- SkyrimNet ресемплит референс-клипы до 16 кГц перед отправкой — сервер принимает
  любую частоту.

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

## Бенчмарк

```
.venv\Scripts\python benchmark.py
```

Выводит для каждого блока текста: TTFB, elapsed (мс), длительность аудио и **RTF**
(Real-Time Factor, `< 1` — быстрее реального времени). Ориентиры на RTX 50xx:
XTTS ≈ **0.5–0.7**, Qwen3 ≈ **4**.
