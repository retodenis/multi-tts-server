# Chatterbox TTS-сервер для SkyrimNet — финальный план

Сервер текстового синтеза (TTS) на базе [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox), модель **Chatterbox-Multilingual V3**, работающий как API для движка TTS в [SkyrimNet-GamePlugin](https://github.com/MinLL/SkyrimNet-GamePlugin). Поддержка русского языка, стриминговой отдачи, клонирования голосов NPC, простого Web-UI и минимального бенчмарка.

---

## 1. Цели и нефункциональные требования

- **Совместимость со SkyrimNet**: сервер должен реализовывать HTTP-контракт «server you choose» движков SkyrimNet (см. раздел 4). Проверка связки с игрой — позже; сейчас сервер + UI + бенчмарк.
- **Русский язык**: модель `Chatterbox-Multilingual V3`, `language_id="ru"` (поддерживается 23 языка).
- **Отзывчивость и стриминг**:
  - SkyrimNet сам стримит по предложениям (шлёт по одной реплике в `/tts_to_audio`), поэтому сервер должен быть «горячим» и отвечать быстро на каждую реплику.
  - Отдельный `GET /tts_stream` (SSE) — разбивает текст на предложения и отдаёт каждый чанк по готовности (для UI и бенчмарка).
- **Клонирование голосов NPC**: при первом обращении к голосу — подсчёт и сохранение эмбеддинга, при повторных — переиспользование кеша (это же даёт главный выигрыш по задержке для повторных реплик).
- **Простота**: однокомандный setup, bat-скрипты для Windows, минимум зависимостий.

---

## 2. Исследование (что уже выяснено)

### 2.1 API-контракт SkyrimNet (подтверждён 3 независимыми community-бэкендами)

Все движки SkyrimNet типа «server you choose» (XTTS, Zonos, PocketTTS, QwenTTS и Chatterbox) используют общий контракт, построенный по образцу `xtts_api_server`:

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/tts_to_audio` (+ с `/`) | JSON `{text, speaker_wav, language, accent, save_path, temperature, top_p, top_k, speed, repetition_penalty, override}` → WAV |
| `POST` | `/create_and_store_latents` (+ с `/`) | multipart `speaker_name`, `language`, `wav_file` → регистрация голоса NPC |
| `GET` | `/health` | `{status, model_loaded, supported_languages}` |

Особенности контракта:
- `text == "ping"` → ответ файлом **тишины** (warmup-«рукопожатие» SkyrimNet).
- `speaker_wav` — **имя** голоса (не путь), соответствует загруженному через `create_and_store_latents` клипу.
- SkyrimNet ресемплит референс-клипы до **16 кГц** перед отправкой — сервер должен принимать любую частоту.
- Ответ `/tts_to_audio` — полноценный WAV (FileResponse), контракт не требует chunked HTTP.

### 2.2 Откуда берутся референсы NPC (система Voice Samples в SkyrimNet)

**Вручную собирать голоса не нужно** — SkyrimNet делает это сам:
1. Автоскан диалоговой озвучки из `Data/Sound/Voice/<plugin>/<VoiceType>.fuz` (реплики обычных разговоров; сцены/катсцены/боевые крики исключены; учитываются моды-реплейсеры).
2. Страница *Voice Samples* в дашборде SkyrimNet (`localhost:8080`) показывает все голосовые типы по плагам; для каждого — список реплик с ▶ Play / 🧪 Test TTS / 🎯 Select.
3. `Select` → SkyrimNet отправляет клип на сервер через `/create_and_store_latents` под именем голоса.
4. Fallback: если линий нет — заговорить с NPC в игре, первая реплика станет референсом.

**Следствие**: серверу НЕ нужен свой механизм извлечения голосов. Нужно хранить клипы (`voices/<name>.wav`) и клонировать по имени. UI-загрузка на сервере — только для ручного теста.

### 2.3 Chatterbox-Multilingual V3 (официальный пакет `chatterbox-tts`)

- Класс: `from chatterbox.mtl_tts import ChatterboxMultilingualTTS`.
- Загрузка: `ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")` (v2 — по умолчанию, нам нужен явный `t3_model="v3"`).
- Генерация:
  ```python
  wav = model.generate(
      text, language_id="ru", audio_prompt_path="voices/serana.wav",
      exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
      repetition_penalty=1.2, min_p=0.05, top_p=1.0,
  )
  ```
  Возвращает torch-тензор `(1, N)` на частоте `model.sr` (S3GEN_SR = 24000 Гц).
- Встроенный watermark (Perth) применяется автоматически (можно отключать флагом для скорости).
- `audio_prompt_path` при каждом вызове пересчитывает conditionals (референс-эмбеддинг) — дорого. Если `audio_prompt_path=None`, используется текущий `self.conds`. **Вывод**: кешируем `Conditionals` по имени голоса и подставляем перед генерацией без пересчёта.
- Кросс-языковое клонирование заявлено как сильная сторона V3 — англ. референс NPC + русский текст работает.

### 2.4 Окружение пользователя

- GPU: **RTX 3060 Laptop, 6 ГБ VRAM** (Ampere → bf16 поддерживается).
- CUDA: **12.8** уже установлена → torch ставить с индексом **cu128**.
- Python: **3.13** (3.12/3.11 не установлены). Chatterbox официально тестирован на 3.11 — есть риск; при проблемах — установить 3.12 или собрать из исходников.
- ffmpeg отсутствует и **не требуется** (librosa + soundfile).

---

## 3. Стек и структура проекта

### 3.1 Технологии

- Python 3.13, venv в проекте.
- `torch` (+ `torchaudio`) CUDA cu128; опция `--device cpu`.
- `chatterbox-tts` (официальный pip-пакет; фолбэк — `git clone` исходников).
- `fastapi`, `uvicorn`, `pydantic`, `python-multipart`.
- `librosa`, `soundfile`, `numpy`, `safetensors`, `huggingface_hub`.
- Порт по умолчанию **7860** (конвенция SkyrimNet), bind `127.0.0.1`, опция `0.0.0.0` для LAN.

### 3.2 Структура файлов

```
chatterboxtts-server/
├── PLAN.md                 # этот документ
├── README.md               # инструкции: установка, запуск, подключение к SkyrimNet
├── requirements.txt        # зависимости (torch cu128 отдельно)
├── setup.bat               # venv + зависимости + проверка CUDA
├── start.bat               # запуск на CUDA
├── start_cpu.bat           # запуск на CPU
├── benchmark.py            # бенчмарк задержек/RTF
├── app/
│   ├── __init__.py
│   ├── config.py           # аргументы/ENV: --device, --port, --host, --dtype,
│   │                       #   --strip-tags, --warmup-text, --default-voice
│   ├── tts_engine.py       # загрузка модели, warmup, generate(), кеш эмбеддингов
│   ├── text_processing.py  # чистка тегов, нормализация, split на предложения (RU)
│   ├── voices.py           # voices/ и latents/: хранение клипов, кеш conds
│   ├── api.py              # эндпоинты SkyrimNet-контракта + /tts_stream
│   ├── main.py             # FastAPI app, lifespan (фоновая загрузка модели)
│   └── static/
│       └── index.html      # простой UI (рус.)
├── voices/                 # референс-клипы <name>.wav (в gitignore)
├── latents/                # кеш эмбеддингов голосов <name>.pt (в gitignore)
└── output/                 # сгенерированные wav (в gitignore)
```

---

## 4. API сервера

### 4.1 SkyrimNet-контракт (обязательный)

**`POST /tts_to_audio`** и **`POST /tts_to_audio/`**
- JSON-тело (модель `SynthesisRequest`):
  ```json
  {
    "text": "Привет, странник.",
    "speaker_wav": "femalenord",
    "language": "ru",
    "accent": null,
    "save_path": null,
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "speed": null,
    "repetition_penalty": null,
    "override": false
  }
  ```
- Логика:
  - `text == "ping"` → вернуть WAV тишины (~0.1 c) как FileResponse.
  - `speaker_wav` не задан/не найден → `--default-voice` (встроенный голос модели или `voices/default.wav`).
  - Скорость: если пришёл `speed != null` и `!= 1.0` — через звуковой пост-процессинг (не модель; у Chatterbox speed_factor ломает голос). `top_k`/`min_p`/`cfg_weight`/`exaggeration` — маппинг параметров UI SkyrimNet на параметры модели.
- Ответ: `FileResponse` WAV + заголовки `X-Gen-Time-Ms`, `X-Audio-Duration-S`.

**`POST /create_and_store_latents`** и **`POST /create_and_store_latents/`**
- multipart: `speaker_name`, `language`, `wav_file` (WAV).
- Логика: сохранить клип в `voices/<speaker_name>.wav`, (ре)считать эмбеддинг на следующее использование; вернуть JSON:
  ```json
  { "message": "...", "speaker_name": "...", "language": "ru",
    "latent_shapes": { "gpt_cond_latent": [], "speaker_embedding": [] } }
  ```
- Файл-имя матчится гибко: точное имя, stem, регистронезависимо.

**`GET /health`**
```json
{ "status": "healthy", "model_loaded": true,
  "supported_languages": ["ar", "...", "ru", "..."], "voices_ready": true }
```

### 4.2 Вспомогательные эндпоинты (UI/бенчмарк)

- `GET /voices` → список доступных голосов из `voices/`.
- `GET /languages` → список языков модели.
- `GET /tts_stream?text=...&speaker_wav=...&language=ru&...` → **SSE-стриминг**:
  - `event: meta` — `{sr, voice, language}`
  - `event: chunk` — `{seq, index, total, gen_ms, wav_b64}` (WAV одного предложения)
  - `event: done` — `{elapsed_ms, first_chunk_ms, audio_sec, rtf}`
- `POST /tts` — полный синтез одним куском (для curl-проверки).

---

## 5. Ключевые решения по производительности

1. **Горячая модель**: загрузка в фоне на старте (lifespan); warmup = `ping`-тишина + реальная короткая генерация, чтобы CUDA «прогрелась» до первого запроса.
2. **Кеш эмбеддингов голосов** (`latents/<name>.pt`): `Conditionals` считаются один раз при первом обращении к NPC и сохраняются на диск; повторные вызовы идут с `audio_prompt_path=None`, используя готовый `self.conds`. Вызовы сериализуются через `asyncio.Lock` (одна GPU, модель не потокобезопасна), тяжёлая генерация — в `asyncio.to_thread`.
3. **Стриминг по предложениям** в `/tts_stream`: первый чанк приходит, как только готово первое предложение (TTFB ≈ генерация одного предложения), а не всего абзаца.
4. **Кеш результатов** (RAM): hash(text+voice+params) → WAV для повторных одинаковых реплик (опционально, включается флагом).
5. **Память 6 ГБ**: по умолчанию fp32; при OOM при загрузке/генерации — авто-фолбэк на `bfloat16` (Ampere поддерживает) через `--dtype`. Ограничение максимальной длины текста (`--max-text-length`), чтобы избегать OOM на длинных фразах.
6. **Чистка тегов**: по умолчанию вырезаем `[...]`/`<...>`-теги, которые может прислать LLM SkyrimNet (иначе модель их озвучивает). Опция `--strip-tags` и настраиваемый белый список.
7. **Водяной знак (Perth)**: включён по умолчанию (ответственная AI), флаг `--no-watermark` для минимальной задержки.

---

## 6. UI (`app/static/index.html`)

Один файл, по-русски, без сборки:

- Текстовое поле + выбор голоса (из `/voices`) + язык (`ru` по умолчанию).
- Слайдеры: `temperature`, `cfg_weight`, `exaggeration`, `top_p`, `repetition_penalty`, `speed`.
- **Стриминговое проигрывание** через `/tts_stream` (Web Audio; чанки играют по мере готовности), отображение `first_chunk_ms`.
- Загрузка референс-клипа (→ `/create_and_store_latents`), список голосов, кнопка «Проверить health».
- Панель последнего запроса: время генерации, длительность аудио, RTF.

---

## 7. Бенчмарк (`benchmark.py`)

- warmup (ping + короткая фраза);
- короткая RU-фраза → TTFB, total_ms, RTF;
- средняя фраза;
- абзац (3–4 предложения) через `/tts_stream` → first_chunk_ms, total;
- повтор той же фразы ×3 → демонстрация выигрыша кеша голоса/результата;
- табличный вывод метрик.

Ожидания на RTX 3060: короткая фраза ~0.5–1.5 c, first-chunk абзаца ~0.7–1.5 c, RTF < 1.

---

## 8. Порядок реализации

1. **Каркас**: `requirements.txt`, `setup.bat`, venv, установка torch cu128 + `chatterbox-tts`, проверка импорта и CUDA.
2. **Ядро**: `config.py`, `voices.py`, `tts_engine.py` (загрузка, warmup, кеш эмбеддингов, generate), `text_processing.py`.
3. **API**: `api.py` + `main.py` — контракт SkyrimNet, `ping`→тишина, `/tts_stream`, health/voices/languages.
4. **UI**: `index.html`.
5. **Бенчмарк**: `benchmark.py`.
6. **Документация**: `README.md` (установка, запуск, подключение к SkyrimNet: движок **Chatterbox**, URL `http://127.0.0.1:7860`, выбор референсов в Voice Samples), `start.bat` / `start_cpu.bat`.
7. **Проверка** (curl/UI):
   - `GET /health`;
   - `POST /tts_to_audio` c `text="ping"` → тишина;
   - RU-фраза с встроенным голосом (клипов нет);
   - `create_and_store_latents` с тестовым клипом → клонирование по имени;
   - `/tts_stream` (SSE) для абзаца;
   - `benchmark.py`.

---

## 9. Риски и митигации

| Риск | Митигация |
|---|---|
| `chatterbox-tts` не работает на Python 3.13 (офиц. 3.11) | Проверить на старте; фолбэк: установить Python 3.12 или собрать из исходников репозитория |
| 6 ГБ VRAM в fp32 (OOM на длинных фразах) | `--dtype bfloat16` авто-фолбэк, `--max-text-length` |
| Кросс-языковое клонирование (англ. референс → рус. текст) | Это заявленная сильная сторона Multilingual V3; проверка качеством на бенчмарке; рекомендация: референс 10–15 с чистой речи |
| Модель `t3_mtl23ls_v3.safetensors` + файлы ~2–3 ГБ скачиваются при первом запуске | Предупредить в README; `HF_TOKEN` не обязателен |
| Неизвестен точный маппинг движка Chatterbox в SkyrimNet (проверка позже) | Контракт `/tts_to_audio` подтверждён тремя бэкендами; при несовпадении добавить недостающий маршрут (общая движок-прослойка уже есть) |
| Скорость `speed` не поддерживается моделью | Применить ресемплинг/питч-пост-обработку или игнорировать с предупреждением (документировать) |
