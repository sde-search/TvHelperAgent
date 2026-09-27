|# Hybrid RAG — сервис гибридного поиска (мультиязычный)

Гибридный поиск по технической документации: векторный (ChromaDB + bge-m3) + BM25 + bge-reranker-v2-m3 на GPU + фильтр по продукту.
Поддерживает 100+ языков.
Доступен через HTTP (постоянный сервер) либо как библиотека.

## Возможности

- Индексация PDF, DOCX, PPTX
- Гибридный поиск: векторный + BM25 + переранжирование bge-reranker-v2-m3 на GPU
- Поддержка 100+ языков (русский, английский, немецкий, французский, китайский...)
- Постоянный HTTP-сервер (без cold start, ~4 сек на запрос)
- Инкрементальное добавление — только новые и изменённые файлы (флаг `--incremental`)
- Метаданные продукта — при добавлении указывается `--product` и `--summary`, описание вшивается в каждый чанк
- Фильтрация по продукту в поиске (`?product=Имя`)
- Дедупликация по SHA256 при загрузке (эндпоинт `/upload` возвращает 409, если такой файл уже есть)
- Http-эндпоинты для всех операций (загрузка, индексация, валидация, статус, список файлов, информация о системе)
- Все поисковые запросы и события индексации логируются

## Структура

```
pdf-rag/
├── data/
│   ├── pdfs/           # файлы для индексации (PDF, DOCX, PPTX)
│   ├── pdfs_raw/       # исходные файлы с подпапками (опционально)
│   ├── chroma/         # ChromaDB + BM25 pickle + file_hashes.json
│   └── logs/           # JSONL-логи поиска и индексации
├── src/
│   ├── ingestion_hybrid.py   # индексация (--incremental, --product, --summary)
│   ├── hybrid_search.py      # библиотека поиска
│   └── search_server.py      # HTTP-сервер (FastAPI + эндпоинты)
├── scripts/            # вспомогательные скрипты
└── INSTRUCTIONS.md     # этот файл
```

## Переменные окружения (все опциональны, есть fallback)

| Переменная | По умолчанию | Что задаёт |
|---|---|---|
| `CORS_ORIGINS` | `*` (все) | Разрешённые origin через запятую, напр. `http://site1.com,http://site2.com` |
| `RAG_EMBED_MODEL` | `bge-m3` | Модель эмбеддингов в Ollama |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Модель реранжинга (CrossEncoder, GPU) |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Адрес Ollama для эмбеддингов |
| `RAG_LLM_MODEL` | `qwen3:8b` | Модель для генерации ответа (через `/v1/chat/completions`) |
| `RAG_LLM_ENDPOINT` | `http://localhost:11434/v1/chat/completions` | Полный URL OpenAI-совместимого API для LLM |
| `FLASHRANK_CACHE_DIR` | `~/.cache/flashrank` | (не используется с bge-reranker-v2-m3)

Пример `.env` или export перед запуском:
```bash
export CORS_ORIGINS="http://localhost:11436,http://10.66.66.2:8080"
export RAG_EMBED_MODEL=bge-m3
```

## Зависимости и среда

### Обязательно
- **Python 3.13+**
- **Ollama** с моделью эмбеддингов (`bge-m3`) — порт 11434
- **PyTorch (CUDA)** — для bge-reranker-v2-m3 на GPU

### Пакеты Python (устанавливаются в .venv)
```
chromadb
rank_bm25
pypdf
python-docx
python-pptx
fastapi
uvicorn
llama-index-embeddings-ollama
sentence-transformers
torch
nltk
python-multipart
```

### Важные нюансы по среде
- **Ollama обязателен** — эмбеддинги генерируются через Ollama, без него поиск не работает.
- **HTTP-сервер держит всё в памяти** — ChromaDB, BM25, модель ранжирования. Первый запуск может быть медленным из-за загрузки эмбеддинг-модели. После старта — ~4 сек на запрос.
- **bge-reranker-v2-m3 работает на GPU** — требуется CUDA и ~3.5 GB VRAM. В отличие от старого FlashRank (CPU, только английский), новая модель мультиязычная и значительно быстрее на GPU.
- **nltk punkt** — скачивается автоматически при первом запуске.

## Как добавить данные

### Через HTTP-эндпоинты (рекомендуемый способ)

#### Загрузка файла
```bash
curl -X POST http://localhost:11436/upload \
  -F "file=@/путь/к/файлу.pdf"
```
Если файл с таким содержимым уже проиндексирован — вернёт 409 с именем дубликата.

#### Инкрементальная индексация
```bash
curl -X POST http://localhost:11436/ingest \
  -H "Content-Type: application/json" \
  -d '{"product": "Название продукта", "summary": "Краткое описание"}'
```
Индекс перезагружается автоматически после индексации.

#### Полная индексация (CLI, первичная)
```bash
cd pdf-rag && .venv/bin/python src/ingestion_hybrid.py \
    --product "Название продукта" \
    --summary "Краткое описание продукта"
```

Все файлы из `data/pdfs/` индексируются с нуля. ChromaDB коллекция и BM25 создаются заново.

### Инкрементальное добавление (CLI, только новые/изменённые файлы)

```bash
cd pdf-rag && .venv/bin/python src/ingestion_hybrid.py --incremental \
    --product "Название продукта" \
    --summary "Краткое описание"
```

**Как работает:** сверяет SHA256 и mtime файлов с кэшем `file_hashes.json`. Добавляет только то, чего нет или что изменилось. Старые чанки удаляются, новые векторизуются, BM25 перестраивается из актуальных данных ChromaDB.

**Важно:** инкрементальный режим каждый раз перестраивает BM25 целиком (не дёшево, но необходимо для консистентности).

### Дедупликация
Перед индексацией файлы из `data/pdfs_raw/` копируются в `data/pdfs/` с дедупликацией по SHA256. Повторяющиеся файлы пропускаются.
При загрузке через `/upload` проверка хеша происходит на лету — дубликат отклоняется с HTTP 409.

## Поиск

### Через HTTP-сервер

```bash
# Базовый поиск
curl -s "http://localhost:11436/search?query=ваш+запрос"

# С фильтром по продукту
curl -s "http://localhost:11436/search?query=ваш+запрос&product=Название+продукта"

# Больше результатов (по умолчанию 10)
curl -s "http://localhost:11436/search?query=запрос&k=20"

# POST-версия (для сложных запросов)
curl -X POST http://localhost:11436/search \
  -H "Content-Type: application/json" \
  -d '{"query": "ваш запрос", "k": 10, "product": null}'
```

### Перезагрузка индекса (после CLI-инкремента)

Сервер автоматически перезагружается после `/ingest` эндпоинта.
Если индексация делалась через CLI, нужно вызвать:

```bash
curl http://localhost:11436/reload
```

### Через библиотеку (без сервера)

```bash
cd pdf-rag && .venv/bin/python -c "
from hybrid_search import HybridSearch
h = HybridSearch(
    chroma_path='data/chroma',
    collection_name='docs',
    bm25_pkl_path='data/chroma/bm25_data.pkl'
)
results = h.search('ваш запрос')
for r in results:
    print(f\"{r['source']} — {r['text'][:100]}\")"
```

## Проверка работоспособности

```bash
# Проверка сервера
curl http://localhost:11436/health
# → {"status":"ok","search_engine":true}

# Статус индекса
curl http://localhost:11436/status
# → {"chunks":8467,"sources":403,"bm25_ready":true,...}

# Список проиндексированных файлов
curl http://localhost:11436/files
# → {"files":[{"name":"manual.pdf","size_kb":1234.5},...],"count":5}

# Информация о системе
curl http://localhost:11436/info
# → {"name":"TvHelper Super Bot","commands":{...},...}

# Тестовый поиск
curl -s "http://localhost:11436/search?query=test+query"

# Количество чанков в базе
.venv/bin/python -c "import chromadb; c=chromadb.PersistentClient('data/chroma'); print(c.get_or_create_collection('docs').count())"

# Проверка BM25 индекса
.venv/bin/python -c "import pickle; d=pickle.load(open('data/chroma/bm25_data.pkl','rb')); print(f'BM25: {len(d[\"documents\"])} документов')"
```

## Валидация ответа

Проверяет, что ответ агента основан на реально найденных документах.

```bash
curl -X POST http://localhost:11436/validate \
  -H "Content-Type: application/json" \
  -d '{"query": "вопрос пользователя", "response": "ответ агента"}'
```

Возвращает:
- `stage1` — эвристическая проверка (удаление неподтверждённого)
- `stage2` — финальная проверка deepseek v4
- `cleaned` — очищенный ответ
- `issues` — список найденных проблем
- `verdict` — вердикт deepseek

## Логирование

Все события записываются в `data/logs/`:

| Файл | Формат | Содержание |
|---|---|---|
| `searches.jsonl` | JSONL | Каждый поисковый запрос: timestamp, query, product, количество результатов, время выполнения |
| `ingestion.jsonl` | JSONL | События индексации: полная (`full_index`), инкрементальная (`incremental_index`), перезагрузка (`reload`), загрузка файла (`upload`) — с метаданными |
| `server.log` | Текст | Лог работы сервера (уровень INFO) |
| `ingestion.log` | Текст | Лог работы индексатора (уровень INFO) |

```bash
# Последние 10 поисков
tail -10 data/logs/searches.jsonl | python -m json.tool --no-ensure-ascii

# Сколько всего поисков выполнено
wc -l data/logs/searches.jsonl

# Сколько индексаций было
grep -c '"event": "full_index\|incremental_index\|reload"' data/logs/ingestion.jsonl
```

## Как подключить к агенту (LLM-ассистенту)

1. Скопировать папку `pdf-rag/` на сервер, где работает агент
2. Установить зависимости и запустить сервер поиска
3. В prompt или SOUL.md агента указать команду поиска через curl:

```bash
curl -s "http://localhost:11436/search?query=${вопрос пользователя}"
```

4. Агент получает JSON с результатами — читает `results` (массив), из каждого `source`, `text`, `product`, `rerank_score`
5. Формирует ответ на основе найденных чанков с указанием источника

### Варианты подключения агента

**Вариант A (безопасный) — HTTP-эндпоинты:**
Агент использует только curl к API сервера:
- Поиск: `GET /search`
- Загрузка файла: `POST /upload`
- Индексация: `POST /ingest`
- Валидация: `POST /validate`
- Статус: `GET /status`
- Список файлов: `GET /files`
- Информация: `GET /info`

CLI-доступ агенту не нужен. Достаточно разрешить `curl` в `command_allowlist`.

**Вариант B (shell-доступ):**
Агент использует shell-команды для индексации и валидации:
- `cd pdf-rag && .venv/bin/python src/ingestion_hybrid.py ...`
- `rag-validate` (CLI-скрипт)

**В production рекомендуется Вариант A** — он не требует shell-доступа, все операции идут через HTTP.

### Важно для агента
- **Не давать агенту shell-доступ без ограничений** — достаточно разрешить только `curl` (см. command_allowlist в конфиге Hermes), либо вынести все действия агента в HTTP-эндпоинты
- **Для добавления файлов** агенту нужен доступ к HTTP-эндпоинтам или shell-командам

## Технические детали

- **Чанкование текста:** 768 слов, overlap 128, разделение по границам предложений `[.!?]`
- **Поиск:** ChromaDB (n_results=30) + BM25 (n_results=30) → RRF fusion (константа 60) → FlashRank reranker → сортировка по rerank_score
- **Эмбеддинги:** `nomic-embed-text` через Ollama, метрика cosine
- **Коллекция ChromaDB:** по умолчанию называется `docs`
- **BM25:** сериализуется в pickle-файл `bm25_data.pkl` (содержит documents, metadatas, ids)
- **Метаданные чанка:** `source` (имя файла), `product`, `summary`
- **Дедупликация:** SHA256 + mtime, кэш в `file_hashes.json`
- **Сервер:** FastAPI + uvicorn, порт 11436, CORS настраивается через `CORS_ORIGINS` (по умолчанию открыт для всех)
- **systemd:** сервис `pdf-rag-search.service` (user), автозапуск

## Эндпоинты сервера (полный список)

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Проверка сервера |
| GET | `/status` | Статистика индекса (чанки, источники, продукты) |
| GET | `/files` | Список проиндексированных файлов с размером |
| GET | `/info` | Описание системы и список команд |
| GET | `/search` | Поиск (?query=, &k=, &product=) |
| POST | `/search` | Поиск (JSON-тело) |
| POST | `/upload` | Загрузка файла (multipart), проверка дубликата по SHA256 |
| POST | `/ingest` | Инкрементальная индексация + перезагрузка |
| POST | `/validate` | Валидация ответа (эвристики + deepseek) |
| GET | `/reload` | Перезагрузка индекса из файлов |

## Типичные проблемы и их решение

| Проблема | Причина | Решение |
|---|---|---|
| Сервер не отвечает | Не запущен или порт занят | `systemctl --user restart pdf-rag-search.service` |
| `search_engine: false` в /health | Ошибка загрузки ChromaDB/BM25 | Проверить лог: `journalctl --user -u pdf-rag-search -n 30` |
| Поиск возвращает пустой результат | Неправильный параметр `?query=` (не `?q=`) | Использовать `?query=запрос` |
| `ollama: connection refused` | Ollama не запущен | `systemctl start ollama` |
| BM25 не обновился после CLI-добавления | Не вызван `/reload` | `curl http://localhost:11436/reload` |
| `/upload` падает с 500 | Не установлен python-multipart | `.venv/bin/pip install python-multipart` |
| `/upload` возвращает 409 | Дубликат содержимого | Файл с таким SHA256 уже проиндексирован под другим именем |
| Инкремент не видит новые файлы | Несовпадение путей (data/pdfs vs pdfs_raw) | Файлы класть в `data/pdfs/`, а `pdfs_raw` — для дедупликации |