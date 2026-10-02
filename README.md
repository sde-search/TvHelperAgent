# TVHelper Web Agent

Лёгкий RAG-агент с веб-интерфейсом. FastAPI + qwen3:8b (Ollama) + гибридный поиск.

## Возможности

- **Чат с RAG** — вопросы по документации с ответами от LLM и цитированием источников
- **Три режима**: RAG+LLM (чат), только поиск, только LLM
- **Провайдеры LLM**: локальный Ollama, OpenAI-совместимые, Anthropic (ключи в `.env`)
- **Адаптивный слайдер токенов** — подстраивается под лимиты выбранной модели
- **Ползунки** K чанков, температура, макс. токенов
- **Streaming SSE** — токены приходят по одному
- **История диалога** в localStorage
- **Экспорт диалога** в Markdown
- **Кнопка справки** ❓ — описание всех настроек
- **Статус-бар**: GPU, Ollama, search_server
- **Тёмная тема**
- **Защита от повторной отправки** — `sending` guard блокирует двойной Enter
- **Cancel стрима** — кнопка ⏹ прерывает генерацию через `AbortController`
- **История диалога** в localStorage — кап 50 сообщений, при превышении квоты — тримминг до 20
- **Кнопка очистки чата** 🗑️ (также двойной клик по заголовку)
- **Копирование ответа** с fallback: `navigator.clipboard` → `execCommand('copy')` для WireGuard/не-HTTPS
- **marked.js локально** — не зависит от CDN, работает офлайн

## Быстрый старт

```bash
cd ~/.hermes/profiles/tvhelper-web
.venv/bin/python3 app.py
```

Сервер стартует на `http://0.0.0.0:8080`.
Через WireGuard: `http://10.66.66.2:8080`

Требуется работающий **Ollama** (`http://localhost:11434`) с моделью (`qwen3:8b`),
и **search_server** для гибридного поиска (`http://localhost:11436`).

## Публичный доступ

Без домена — через Cloudflare Quick Tunnel:

```bash
cloudflared tunnel --url http://localhost:8080
```

## API

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Веб-интерфейс |
| `/api/models` | GET | Список провайдеров и их моделей (с `max_output`, `vram_estimate_mb`, `presets`) |
| `/api/chat` | POST | RAG + LLM (SSE-поток). Валидирует: question (max 8192), k (int 1–20), temperature (0–2), num_predict (1–131072), num_ctx (int, по модели), history (list), model (str 1–256) |
| `/api/search` | POST | Только поиск по RAG. Валидирует query (max 8192), k (1–20) |
| `/api/status` | GET | Статус GPU, Ollama, search_server |
| `/api/activate-model` | POST | Принудительная загрузка модели в Ollama. Валидирует model (str, non-empty, max 256) |
| `/api/chat` | POST | Доп. параметры: `search_backend` (tavily/searxng/off), `k` (1–20), `temperature` (0–2), `num_predict`, `num_ctx` |

## Веб-поиск

Поддерживаются три режима (поле `search_backend` в запросе `/api/chat` или выбор в UI):

| Режим | Описание |
|-------|----------|
| `tavily` | Поиск через Tavily API (требуется `TAVILY_API_KEY` в `.env`). **По умолчанию** при `WEB_SEARCH_ENABLED=1` |
| `searxng` | Поиск через локальный SearXNG (по умолч. `http://127.0.0.1:8888`, настраивается `SEARXNG_BASE_URL`) |
| `off` | Без веб-поиска |

При `WEB_SEARCH_ENABLED=0` (или отсутствии) режим по умолчанию — `off`.  
При ошибке Tavily — автоматический fallback на SearXNG.

## PDF документы

PDF-файлы монтируются на `/docs/` из `PDFS_DIR` (по умолч. `/home/hermes/share_pdf-rag/pdf-rag/data/pdfs/`).  
В каждом RAG-источнике в ответе `/api/chat` есть поле `pdf_url` — прямая ссылка на PDF для скачивания.

## Провайдеры LLM

Ключи только в `.env` (рядом с `app.py`). UI их не видит.

| Провайдер | Переменные `.env` |
|-----------|-------------------|
| Локальный (Ollama) | `LOCAL_ENABLED=1`, `OLLAMA_URL` |
| OpenAI-совместимые | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_LABEL`, `OPENAI_MODELS` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODELS` |

Чтобы включить провайдера — раскомментировать строки в `.env`, вписать ключ, перезапустить.

## Настройки UI

- **K** — количество чанков из RAG (1–10)
- **temp** — температура модели (0.0–1.0, шаг 0.1)
- **токенов** — макс. длина ответа (адаптивный слайдер под модель)
- **Провайдер** — откуда брать модель (локальная / облачная)
- **Модель** — выбор из списка + кнопка **OK** для применения
- **⚠️ VRAM** — предупреждение в статус-баре, если модель с текущим контекстом не влезает в видеопамять (наведение показывает детали)
- **🗑️** — очистка чата (подтверждение)
- **Автоматические presets** — при выборе модели автоматом выставляются: температура (0.1), K (5), num_ctx (под модель), num_predict. Значения подобраны под роль справочного агента и ограничения VRAM GPU (RTX 3060 12GB)
- **RAG-промпт в user message** — контекст из поиска передаётся в последнем сообщении пользователя, а не в system. Гарантирует, что документы видят все модели: Qwen, Gemma, Llama, Phi, DeepSeek
- **Статус-бар** 🟢/🔴 — GPU, Ollama, Search, VRAM

## Ошибки Ollama

Дифференцируются по типу:
- **Timeout** — Ollama не ответил за 120 с
- **404** — модель не найдена (название в сообщении)
- **ConnectError** — Ollama не запущен
- **ReadError** — разрыв соединения при стриме

## Зависимости

- Python 3.11+
- Ollama с моделью (по умолч. qwen3:8b)
- search_server (гибридный поиск, порт 11436)
- httpx, fastapi, uvicorn (в `.venv`)