# TVHelper Web Agent

Лёгкий RAG-агент с веб-интерфейсом для работы с технической документацией. 
FastAPI + qwen3:8b (Ollama) + гибридный поиск.

## Возможности

- **Чат с RAG** — вопросы по документации с ответами от LLM и цитированием источников
- **Три режима**: RAG+LLM (чат), только поиск, только LLM
- **Меню выбора модели** — быстрые кнопки + выпадающий список моделей Ollama
- **Streaming SSE** — токены приходят по одному
- **Тёмная тема** в стиле веб-терминала

## Быстрый старт

```bash
# Установка зависимостей
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn jinja2 httpx python-multipart

# Запуск
.venv/bin/python3 app.py
```

Сервер стартует на `http://0.0.0.0:8080`.

Требуется работающий **Ollama** (http://localhost:11434) с моделью, например `qwen3:8b`, и **search_server** для гибридного поиска (http://localhost:11436).

## Публичный доступ

Без домена — через Cloudflare Quick Tunnel:

```bash
cloudflared tunnel --url http://localhost:8080
```

## API

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/` | GET | Веб-интерфейс |
| `/api/models` | GET | Список моделей из Ollama |
| `/api/chat` | POST | RAG + LLM (SSE-поток) |
| `/api/search` | POST | Только поиск |

## Зависимости

- Python 3.11+
- Ollama с любой моделью (по умолч. qwen3:8b)
- search_server (гибридный поиск, порт 11436)