#!/usr/bin/env python3
"""FastAPI search server for hybrid RAG with product filtering."""

import hashlib
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# — Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
HASH_CACHE = CHROMA_DIR / "file_hashes.json"
EXTRACTED_DIR = DATA_DIR / "extracted"
PDFS_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
VENV_PYTHON = sys.executable  # runs from .venv, so this is the venv python


# — Helper: get images for a source file
def get_images_for_source(source_name: str, page_number: int | None = None) -> list[str]:
    """Возвращает список URL картинок для указанного source и опционально — страницы."""
    source_stem = Path(source_name).stem
    img_dir = EXTRACTED_DIR / source_stem
    if not img_dir.exists():
        return []
    pattern = f"page*_img*.png"
    if page_number is not None:
        pattern = f"page{page_number}_img*.png"
    return [f"/extracted/{source_stem}/{f.name}" for f in sorted(img_dir.glob(pattern))]

# — Логирование поисковых запросов
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_LOG = str(LOG_DIR / "searches.jsonl")
INGEST_LOG = str(LOG_DIR / "ingestion.jsonl")

_logger = logging.getLogger("pdf-rag-search")
_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(str(LOG_DIR / "server.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_logger.addHandler(_fh)


def log_search(query: str, product: str | None, results_count: int, elapsed_ms: int, status: str = "ok"):
    """Запись поискового запроса в JSONL."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "search",
        "query": query,
        "product": product,
        "results": results_count,
        "elapsed_ms": elapsed_ms,
        "status": status,
    }
    with open(SEARCH_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _logger.info(f"SEARCH query={query[:60]} product={product} results={results_count} {elapsed_ms}ms")


def log_ingestion(event: str, **kwargs):
    """Запись события индексации в JSONL."""
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event}
    entry.update(kwargs)
    with open(INGEST_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _logger.info(f"INGEST {event} {kwargs}")


app = FastAPI(title="Hybrid RAG Search")
# — CORS: настраивается через переменную окружения CORS_ORIGINS
#   Формат: список origin через запятую. По умолчанию — все (разрешено любому клиенту).
#   Пример: CORS_ORIGINS="http://localhost:11436,http://10.66.66.2:8080"
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    cors_origins = ["*"]
else:
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.mount("/extracted", StaticFiles(directory=str(EXTRACTED_DIR)), name="extracted")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# — Globals, reloaded by /reload.
searcher = None
bm25_data = None
chroma_collection = None
SEARCHER_LOCK = [None]  # mutable box, lets /reload swap atomically


def load_searcher():
    base = str(BASE_DIR)
    data_dir = str(DATA_DIR)
    chroma_path = str(CHROMA_DIR)
    bm25_pkl = str(CHROMA_DIR / "bm25_data.pkl")

    sys.path.insert(0, str(BASE_DIR / "src"))
    from hybrid_search import HybridSearch

    hs = HybridSearch(
        chroma_path=chroma_path,
        collection_name="docs",
        bm25_pkl_path=bm25_pkl,
        embedding_model=os.environ.get("RAG_EMBED_MODEL", "bge-m3"),
        ollama_base_url=os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434"),
    )
    return hs


@app.on_event("startup")
async def startup():
    global SEARCHER_LOCK
    try:
        SEARCHER_LOCK[0] = load_searcher()
    except Exception as e:
        print(f"[WARN] Failed to load searcher at startup: {e}", file=sys.stderr)
        SEARCHER_LOCK[0] = None


@app.get("/health")
async def health():
    return {"status": "ok", "search_engine": SEARCHER_LOCK[0] is not None}


@app.get("/reload")
async def reload_search():
    try:
        t0 = time.time()
        SEARCHER_LOCK[0] = load_searcher()
        elapsed = int((time.time() - t0) * 1000)
        log_ingestion("reload", elapsed_ms=elapsed, status="ok")
        return {"status": "ok", "message": "Searcher reloaded"}
    except Exception as e:
        log_ingestion("reload", status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search(
    query: str = Query(..., description="Search query"),
    k: int = Query(10, ge=1, le=100),
    product: str = Query(None, description="Filter by product name"),
    format: str = Query(None, description="DEPRECATED — kept for compat"),
):
    hs = SEARCHER_LOCK[0]
    if hs is None:
        log_search(query, product, 0, 0, status="engine_not_loaded")
        raise HTTPException(503, "Search engine not loaded")

    t0 = time.time()
    docs = hs.search(query, k=k, product_filter=product)
    elapsed = int((time.time() - t0) * 1000)
    log_search(query, product, len(docs), elapsed)

    # Обогащаем результаты ссылками на картинки
    for d in docs:
        d["images"] = get_images_for_source(
            d.get("source", ""),
            page_number=d.get("page", None),
        )

    return {
        "query": query,
        "k": k,
        "product": product,
        "results": docs,
        "count": len(docs),
    }


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    product: str = None


@app.post("/search")
async def search_post(body: SearchRequest):
    return await search(
        query=body.query, k=body.k, product=body.product
    )


# =====================================================================
# NEW ENDPOINTS — заменяют shell-команды на HTTP
# =====================================================================

# --- Статус индекса ---

@app.get("/status")
async def index_status():
    """Возвращает статистику по индексу: чанки, файлы, продукты."""
    hs = SEARCHER_LOCK[0]
    if hs is None:
        raise HTTPException(503, "Search engine not loaded")

    try:
        all_data = hs.collection.get()
        doc_count = len(all_data["ids"])
    except Exception as e:
        doc_count = 0

    # Собираем уникальные продукты, файлы
    products = set()
    sources = set()
    if doc_count > 0:
        for m in all_data["metadatas"]:
            if m.get("product"):
                products.add(m["product"])
            if m.get("source"):
                sources.add(m["source"])

    pdf_files = len([f for f in PDFS_DIR.glob("*") if f.suffix.lower() in ('.pdf', '.docx', '.doc', '.pptx')])
    bm25_ok = (CHROMA_DIR / "bm25_data.pkl").exists()

    return {
        "chunks": doc_count,
        "sources": len(sources),
        "sources_list": sorted(sources),
        "products": sorted(products),
        "pdf_files": pdf_files,
        "bm25_ready": bm25_ok,
    }


# --- Список проиндексированных файлов ---

@app.get("/files")
async def list_files():
    """Возвращает список проиндексированных файлов с деталями."""
    if not HASH_CACHE.exists():
        return {"files": []}
    try:
        with open(HASH_CACHE) as f:
            hashes = json.load(f)
    except Exception:
        return {"files": []}

    result = []
    for name in sorted(hashes.keys()):
        fpath = PDFS_DIR / name
        size_kb = round(fpath.stat().st_size / 1024, 1) if fpath.exists() else 0
        result.append({"name": name, "size_kb": size_kb})
    return {"files": result, "count": len(result)}


# --- Информация о системе ---

@app.get("/info")
async def system_info():
    """Возвращает описание системы для отображения пользователю."""
    return {
        "name": "TvHelper Super Bot",
        "description": "Помогаю с документацией по Nexio, IconStation и другим продуктам.",
        "commands": {
            "start": "Показать это приветствие",
            "help": "Инструкции по использованию",
            "files": "Список проиндексированных документов",
            "upload": "Загрузить документ (PDF/DOCX)",
            "search": "Поиск по документации",
        },
        "skills": True,
        "files_available": HASH_CACHE.exists(),
    }


# --- Загрузка файла ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Сохраняет загруженный файл в data/pdfs/."""
    if file.filename is None or file.filename == "":
        raise HTTPException(400, "No filename provided")

    # Безопасность: только разрешённые расширения
    allowed = {'.pdf', '.docx', '.doc', '.pptx', '.txt', '.md'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Extension '{ext}' not allowed. Allowed: {', '.join(sorted(allowed))}")

    # Лимит размера: 50 MB
    content = await file.read()
    max_size = 50 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(413, f"File too large ({len(content)} bytes). Max: {max_size} bytes")

    # Защита от path traversal
    safe_name = Path(file.filename).name
    file_path = PDFS_DIR / safe_name

    # Проверка на дубликат по SHA256
    file_hash = hashlib.sha256(content).hexdigest()
    if HASH_CACHE.exists():
        try:
            with open(HASH_CACHE) as f:
                hashes = json.load(f)
            for name, h in hashes.items():
                if h == file_hash:
                    raise HTTPException(409, f"Дубликат: файл с таким содержимым уже проиндексирован как «{name}»")
        except HTTPException:
            raise
        except Exception:
            pass  # кэш битый — игнорируем, сохраняем как есть

    with open(file_path, "wb") as f:
        f.write(content)

    log_ingestion("upload", filename=file.filename, size=len(content))
    return {"status": "ok", "filename": file.filename, "size": len(content), "path": str(file_path)}


# --- Индексация ---

class IngestRequest(BaseModel):
    product: str = ""
    summary: str = ""


@app.post("/ingest")
async def ingest(body: IngestRequest):
    """Запускает инкрементальную индексацию + перезагрузку."""
    ingest_script = str(BASE_DIR / "src" / "ingestion_hybrid.py")
    cmd = [
        VENV_PYTHON,
        ingest_script,
        "--incremental",
        "--product", body.product or "",
        "--summary", body.summary or "",
    ]
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        log_ingestion("ingest", status="timeout", product=body.product)
        raise HTTPException(504, "Ingestion timed out after 300s")
    except Exception as e:
        log_ingestion("ingest", status="error", error=str(e))
        raise HTTPException(500, f"Ingestion failed: {e}")

    if result.returncode != 0:
        log_ingestion("ingest", status="error", returncode=result.returncode,
                       stderr=result.stderr[-300:])
        raise HTTPException(500, f"Ingestion failed (code {result.returncode}): {result.stderr[:500]}")

    # Перезагрузка индекса
    try:
        SEARCHER_LOCK[0] = load_searcher()
    except Exception as e:
        log_ingestion("ingest", status="reload_error", error=str(e))
        # Индексация прошла, но индекс не перезагрузился
        elapsed = int((time.time() - t0) * 1000)
        log_ingestion("ingest", status="reload_failed", elapsed_ms=elapsed, product=body.product)
        return {
            "status": "partial",
            "message": "Документы проиндексированы, но перезагрузка не удалась",
            "error": str(e),
            "stdout": result.stdout[-500:],
        }

    elapsed = int((time.time() - t0) * 1000)
    log_ingestion("ingest", status="ok", elapsed_ms=elapsed, product=body.product)
    return {
        "status": "ok",
        "message": "Индексация завершена, индекс перезагружен",
        "elapsed_ms": elapsed,
        "stdout": result.stdout[-500:],
    }


# --- Валидация ответа ---

class ValidateRequest(BaseModel):
    query: str
    response: str


# Паттерны галлюцинаций (из rag-validate)
HALLUCINATION_PATTERNS = [
    r'(?:рекомендуется|необходимо)\s+(?:использовать|установить|настроить)\s+(?:Docker|Java|Ubuntu|PostgreSQL|Nginx)',
    r'(?:системные|минимальные)\s+требования.*?(?:RAM|CPU|GHz|GB)',
    r'(?:установка|инсталляция)\s+(?:занимает|требует|производится)',
    r'(?:команда|выполните)\s+(?:sudo|apt|yum|dnf|brew)',
    r'(?:ссылайтесь|обращайтесь)\s+к\s+официальной\s+документации',
]


def _call_deepseek(messages: list) -> str:
    """Вызов Polza deepseek v4 (как в rag-validate)."""
    api_key = os.environ.get("HERMES_CUSTOM_POLZA_AI_API_KEY", "")
    if not api_key:
        return "⚠️ API ключ не найден"

    body = json.dumps({
        "model": "deepseek/deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        "https://polza.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _heuristic_validate(response: str, chunks: list[str]) -> tuple[str, list[str]]:
    """Эвристическая проверка. Удаляет неподтверждённые утверждения."""
    sentences = re.split(r'(?<=[.!?])\s+', response)
    validated = []
    issues = []

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 15:
            validated.append(sent)
            continue

        sent_lower = sent.lower()

        # Пропускаем заголовки, ссылки, списки
        if sent.startswith(('-', '•', '*', '📎', 'Источник', '##', '**')):
            validated.append(sent)
            continue

        # Проверка паттернов галлюцинаций
        hallucinated = False
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, sent_lower):
                issues.append(f"- ⚠️ Галлюцинация: {sent[:80]}...")
                hallucinated = True
                break
        if hallucinated:
            continue

        # Проверка ключевых слов в чанках
        words = set(re.findall(r'\w+', sent_lower))
        stopwords = {
            'для', 'что', 'это', 'как', 'не', 'по', 'на', 'с', 'от', 'в',
            'и', 'или', 'из', 'к', 'о', 'об', 'при', 'без', 'да', 'нет',
            'уже', 'еще', 'ещё', 'так', 'же', 'бы', 'ли',
        }
        keywords = words - stopwords

        backed = False
        for chunk in chunks:
            chunk_lower = chunk.lower()
            matched = [w for w in keywords if w in chunk_lower]
            if len(matched) >= max(2, len(keywords) * 0.3):
                backed = True
                break

        if backed:
            validated.append(sent)
        else:
            issues.append(f"- ⚠️ Не подтверждено чанками: {sent[:80]}...")

    return '\n'.join(validated), issues


@app.post("/validate")
async def validate(body: ValidateRequest):
    """Двухэтапная валидация: эвристики + deepseek v4."""
    hs = SEARCHER_LOCK[0]
    if hs is None:
        raise HTTPException(503, "Search engine not loaded")

    # Получаем чанки из поиска (как rag-validate через HybridSearch)
    docs = hs.search(body.query, k=10)
    chunks_text = [d["text"] for d in docs]
    chunks = [c for c in chunks_text if len(c.strip()) > 50]

    if not chunks:
        return {
            "query": body.query,
            "cleaned": body.response,
            "issues": ["⚠️ Не удалось получить чанки для проверки"],
            "verdict": "",
            "stage1": "no_chunks",
        }

    # ЭТАП 1: Эвристики
    cleaned, issues = _heuristic_validate(body.response, chunks)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "*(Ответ не содержал информации, подтверждаемой документами.)*"

    # ЭТАП 2: deepseek v4
    verdict = ""
    if issues or cleaned:
        chunks_for_ds = '\n---\n'.join(chunks[:5])[:3000]
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты финальный верификатор ответов RAG-агента.\n"
                    "Ответ уже прошёл эвристическую проверку (удалены неподтверждённые утверждения).\n"
                    "Твоя задача — перепроверить что всё в порядке.\n"
                    "Если всё норм — ответь: ✅ ВСЁ ЧИСТО\n"
                    "Если найдены проблемы — напиши что именно."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Запрос: {body.query}\n"
                    f"Чанки:\n{chunks_for_ds}\n\n"
                    f"Очищенный ответ:\n{cleaned}\n\n"
                    "Проверь."
                ),
            },
        ]
        try:
            verdict = _call_deepseek(messages)
        except Exception as e:
            verdict = f"⚠️ Ошибка deepseek: {e}"

    return {
        "query": body.query,
        "cleaned": cleaned,
        "issues": issues,
        "verdict": verdict,
        "stage1": "issues_found" if issues else "clean",
        "stage2": verdict.startswith("✅") if verdict else False,
    }


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11436
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")