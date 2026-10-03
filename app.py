#!/usr/bin/env python3
"""TVHelper Web Agent — RAG-агент с веб-интерфейсом и поддержкой внешних API."""

import asyncio
import json
import os
import re
import subprocess
import urllib.parse
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="TVHelper Web Agent")
app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("tvhelper")

# Монтируем PDF-документы для прямого доступа (кликабельные ссылки на источники)
PDFS_DIR = Path(os.getenv("PDFS_DIR", str(Path(__file__).resolve().parent / "pdfs")))
if PDFS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(PDFS_DIR)), name="docs")
    logger.info(f"PDF docs mounted at /docs/ from {PDFS_DIR}")
else:
    logger.warning(f"PDFS_DIR {PDFS_DIR} not found, /docs/ not mounted")

# Подробный лог всех HTTP-запросов
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {response.status_code}")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Дефолтный макс. вывод по модели (можно переопределить в .env) ===
# Формат: MODEL_MAX_OUTPUT = {"gpt-4o": 16384, "deepseek-chat": 8192, ...}
_MODEL_MAX_OUTPUT_STR = os.getenv("MODEL_MAX_OUTPUT", "")
MODEL_MAX_OUTPUT_OVERRIDES = {}
if _MODEL_MAX_OUTPUT_STR:
    for part in _MODEL_MAX_OUTPUT_STR.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            MODEL_MAX_OUTPUT_OVERRIDES[k.strip()] = int(v.strip())
DEFAULT_MAX_OUTPUT = int(os.getenv("DEFAULT_MAX_OUTPUT", "16384"))
MAX_QUESTION_LEN = int(os.getenv("MAX_QUESTION_LEN", "8192"))
MIN_K = 1
MAX_K = int(os.getenv("MAX_K", "20"))
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:11436/search")
SEARCH_SERVER_BASE = os.getenv("SEARCH_SERVER_BASE", "http://localhost:11436")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
DEFAULT_NUM_PREDICT = int(os.getenv("DEFAULT_NUM_PREDICT", "2048"))
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "16384"))

# === Web Search ===
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "0")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8888")


# === Web Search (Tavily API) ===
async def _web_search(query: str, max_results: int = 5, timeout: int = 15) -> list[dict]:
    """Поиск в интернете через Tavily API. Возвращает [{title, url, snippet, score}, ...]."""
    if not query or not query.strip():
        return []
    api_key = TAVILY_API_KEY
    if not api_key:
        logger.warning("TAVILY_API_KEY not set, web search disabled")
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query.strip(),
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            out = []
            for item in results:
                out.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "score": item.get("score", 0),
                })
            logger.info(f"Tavily search for '{query[:80]}': {len(out)} results")
            return out
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return []


# === Web Search (SearXNG) ===
_GARBAGE_TITLE_PATTERNS = re.compile(
    r'^(error|404|not found|access denied|forbidden|redirect|loading|'
    r'just a moment|please wait|attention required|verify|bot|captcha|'
    r'503|502|500|page not found|this page|sorry|enable javascript|'
    r'javascript required|реклама|купить|продажа|цена|магазин|интернет-магазин|'
    r'\[pdf\]|\[doc\]|download)', re.I
)
_GARBAGE_DOMAINS = {
    'youtube.com', 'youtu.be', 'facebook.com', 'instagram.com',
    'twitter.com', 'x.com', 'tiktok.com', 'pinterest.com',
    'reddit.com', 'linkedin.com', 'habr.com',
    'market.yandex.ru', 'ozon.ru', 'wildberries.ru',
    'aliexpress.com', 'avito.ru',
}


def _clean_url(url: str) -> str:
    """Убирает tracking-параметры и нормализует URL."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    # Дропаем utm_*, fbclid, ref, и пр tracking
    clean = parsed._replace(
        query='&'.join(
            kv for kv in (parsed.query.split('&') if parsed.query else [])
            if not kv.startswith(('utm_', 'fbclid', 'gclid', 'ref', 'source', 'si'))
        )
    )
    return urlunparse(clean).rstrip('/')


async def _web_search_searxng(query: str, max_results: int = 5, timeout: int = 15) -> list[dict]:
    """Поиск в интернете через локальный SearXNG. Возвращает [{title, url, snippet, score}, ...]."""
    if not query or not query.strip():
        return []
    base_url = SEARXNG_BASE_URL
    try:
        params = {
            "q": query.strip(),
            "format": "json",
            "language": "ru",
            "categories": "general",
            "pageno": 1,
        }
        async with httpx.AsyncClient(timeout=timeout) as c:
            resp = await c.get(f"{base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            out = []
            seen_urls = set()
            for item in results:
                title = (item.get("title") or "").strip()
                url = _clean_url((item.get("url") or "").strip())
                snippet = (item.get("content") or "").strip()
                score = item.get("score", 1.0)

                # Фильтр мусора
                if not title or not snippet or len(snippet) < 20:
                    continue
                if _GARBAGE_TITLE_PATTERNS.search(title):
                    continue
                try:
                    domain = url.split('/')[2] if '//' in url else url
                except IndexError:
                    continue
                if domain in _GARBAGE_DOMAINS or any(d in url for d in ('youtube.com', 'facebook.com')):
                    continue

                # Дедупликация по URL
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                out.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "score": score,
                })
                if len(out) >= max_results:
                    break
            logger.info(f"SearXNG search for '{query[:80]}': {len(results)} raw → {len(out)} after filter+dedup")
            return out
    except httpx.ConnectError:
        logger.warning(f"SearXNG at {base_url} unavailable, falling back")
        return []
    except Exception as e:
        logger.warning(f"SearXNG search failed: {e}")
        return []


# === Провайдеры LLM ===
PROVIDERS = []
DEFAULT_PROVIDER = "local"
DEFAULT_MODEL = "qwen3:8b"


def add_provider(pid, label, ptype, models, api_key=None, base_url=None, is_default=False):
    global DEFAULT_PROVIDER, DEFAULT_MODEL
    PROVIDERS.append({
        "id": pid,
        "label": label,
        "type": ptype,
        "models": models,
        "api_key": api_key,
        "base_url": base_url,
    })
    if is_default:
        DEFAULT_PROVIDER = pid
        if models:
            DEFAULT_MODEL = models[0]


# Local Ollama
if os.getenv("LOCAL_ENABLED", "1") == "1":
    add_provider("local", "Локальная (Ollama)", "ollama",
                 models=[],
                 base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                 is_default=True)

# OpenAI-совместимые
_openai_key = os.getenv("OPENAI_API_KEY", "")
if _openai_key:
    add_provider("openai",
                 label=os.getenv("OPENAI_LABEL", "OpenAI API"),
                 ptype="openai",
                 models=os.getenv("OPENAI_MODELS", "gpt-4o-mini,gpt-4o").split(","),
                 api_key=_openai_key,
                 base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

# Anthropic
_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
if _anthropic_key:
    add_provider("anthropic",
                 label="Anthropic (Claude)",
                 ptype="anthropic",
                 models=os.getenv("ANTHROPIC_MODELS", "claude-sonnet-4-20250514,claude-haiku-3-5-20241022").split(","),
                 api_key=_anthropic_key,
                 base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"))


# ======================= API Endpoints =======================


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models():
    result = {"providers": [], "default_provider": DEFAULT_PROVIDER, "default_model": DEFAULT_MODEL}
    for p in PROVIDERS:
        entry = {"id": p["id"], "label": p["label"], "type": p["type"], "models": []}
        if p["type"] == "ollama":
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.get(f"{p['base_url']}/api/tags")
                    r.raise_for_status()
                    data = r.json()
                for m in data.get("models", []):
                    name = m["name"]
                    if name == "nomic-embed-text":
                        continue
                    model_info = m.get("model_info", {})
                    ctx = (
                        model_info.get("llama.context_length") or
                        16384
                    )
                    max_out = MODEL_MAX_OUTPUT_OVERRIDES.get(name, ctx)
                    model_size = m.get("size", 0)
                    entry["models"].append({
                        "name": name,
                        "size": model_size,
                        "max_output": max_out,
                        "vram_estimate_mb": _estimate_vram_mb(model_size, max_out),
                        "presets": _recommend_presets(model_size, max_out, is_ollama=True)
                    })
            except Exception as e:
                entry["error"] = str(e)
        else:
            for mname in p["models"]:
                mname = mname.strip()
                if mname:
                    max_out = MODEL_MAX_OUTPUT_OVERRIDES.get(mname, DEFAULT_MAX_OUTPUT)
                    entry["models"].append({"name": mname, "max_output": max_out, "vram_estimate_mb": 0, "presets": _recommend_presets(0, max_out, is_ollama=False)})
        result["providers"].append(entry)
    return result


def _estimate_vram_mb(model_size_bytes: int, num_ctx: int) -> int:
    """Примерная оценка требуемой VRAM для модели (в MB).
    Веса + KV-кеш (грубая прикидка для Qwen/Llama/Gemma в Q4)."""
    model_mb = model_size_bytes / (1024 * 1024)
    # KV-кеш ~ 1 байт на параметр на токен при Q4, упрощённо:
    kv_overhead = model_mb * 0.3 * (num_ctx / 16384)
    return int(model_mb + kv_overhead)


def _get_gpu_total_mb() -> int:
    """Общий объём VRAM в MB."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    return 12288  # fallback RTX 3060


def _recommend_presets(model_size_bytes: int, max_output: int, is_ollama: bool = True) -> dict:
    """Рекомендованные параметры для модели-справочного агента с учётом железа."""
    # Температура: минимальная для фактологичных RAG-ответов
    rec_temperature = 0.1

    # K: сколько чанков — зависит от доступного контекста
    if max_output <= 8192:
        rec_k = 3
    elif max_output <= 16384:
        rec_k = 5
    else:
        rec_k = 7

    # num_predict: ~80% от max_output, чтобы модель могла выдать развёрнутый ответ
    rec_num_predict = min(max(2048, int(max_output * 0.8 / 100) * 100), 32768)

    # num_ctx: баланс между контекстным окном и VRAM
    if is_ollama and model_size_bytes > 0:
        total_vram_mb = _get_gpu_total_mb()
        model_mb = model_size_bytes / (1024 * 1024)
        usable_vram = total_vram_mb * 0.9  # 10% запас для системы
        if model_mb >= usable_vram:
            rec_num_ctx = 4096  # модель едва влезает — минимум контекста
        else:
            kv_budget = usable_vram - model_mb
            kv_per_token = (model_mb * 0.3) / 16384
            if kv_per_token > 0:
                max_safe_ctx = int(kv_budget / kv_per_token)
            else:
                max_safe_ctx = 32768
            rec_num_ctx = min(max(4096, max_safe_ctx), max_output, 32768)
            rec_num_ctx = max(4096, (rec_num_ctx // 1024) * 1024)
    else:
        rec_num_ctx = min(max_output, 16384)

    return {
        "temperature": rec_temperature,
        "num_predict": rec_num_predict,
        "k": rec_k,
        "num_ctx": rec_num_ctx,
    }


# === Query Rewrite ===
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "")




@app.get("/api/status")
async def system_status():
    status = {"gpu": {}, "ollama": False, "search": False}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            if len(parts) == 3:
                status["gpu"] = {"used_mb": int(parts[0]), "total_mb": int(parts[1]), "util_pct": int(parts[2])}
    except Exception:
        status["gpu"] = {"error": "nvidia-smi failed"}
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:11434/api/tags")
            status["ollama"] = r.status_code == 200
    except Exception:
        status["ollama"] = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:11436/health")
            status["search"] = r.status_code == 200
    except Exception:
        status["search"] = False
    return status


@app.post("/api/activate-model")
async def activate_model(body: dict):
    """Принудительно загрузить модель в VRAM (вытесняет предыдущую)."""
    model = body.get("model", "")
    if not model or not isinstance(model, str):
        return {"error": "Model name is required"}
    if len(model) > 256:
        return {"error": "Model name too long"}
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("http://localhost:11434/api/generate", json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_ctx": 16384}
            })
        if r.status_code == 200:
            return {"ok": True, "model": model}
        if r.status_code == 404:
            return {"error": f"Model '{model}' not found in Ollama"}
        return {"error": f"Ollama returned {r.status_code}: {r.text[:200]}"}
    except httpx.TimeoutException:
        return {"error": "Ollama did not respond within 30s"}
    except httpx.ConnectError:
        return {"error": "Cannot connect to Ollama (is it running?)"}


# ======================= File Management =======================

@app.get("/api/files")
async def list_indexed_files(search: str = ""):
    """Прокси к search_server: список проиндексированных файлов."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{SEARCH_SERVER_BASE}/files")
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError:
        return {"files": [], "count": 0, "error": "Search server недоступен"}
    except Exception as e:
        return {"files": [], "count": 0, "error": str(e)}

    files = data.get("files", [])
    if search:
        q = search.lower()
        files = [f for f in files if q in f.get("name", "").lower()]
    # Добавляем human-readable размер
    for f in files:
        kb = f.get("size_kb", 0)
        if kb >= 1024:
            f["size_human"] = f"{kb / 1024:.1f} MB"
        else:
            f["size_human"] = f"{kb:.0f} KB"
    return {"files": files, "count": len(files)}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Прокси для загрузки документа в search_server с проверкой дубликатов."""
    # Валидация расширения
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"}
    if ext not in allowed:
        raise HTTPException(400, f"Неподдерживаемый формат: {ext}. Допустимы: {', '.join(sorted(allowed))}")
    # Проверка размера (50 MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(413, "Файл слишком большой (макс. 50 MB)")
    await file.seek(0)

    # Отправка в search_server
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{SEARCH_SERVER_BASE}/upload",
                files={"file": (file.filename, contents, file.content_type or "application/octet-stream")},
            )
            if r.status_code == 409:
                return {"status": "duplicate", "message": r.json().get("detail", "Такой файл уже есть в базе")}
            r.raise_for_status()
            return {"status": "ok", "message": f"Файл «{file.filename}» загружен", "detail": r.json()}
    except httpx.ConnectError:
        raise HTTPException(503, "Search server недоступен")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"Ошибка search server: {e.response.text[:200]}")


@app.post("/api/ingest")
async def trigger_ingest():
    """Запуск инкрементальной индексации новых файлов."""
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{SEARCH_SERVER_BASE}/ingest", json={"incremental": True})
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(503, "Search server недоступен")
    except httpx.TimeoutException:
        return {"status": "timeout", "message": "Индексация запущена, но ещё не завершена"}
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"Ошибка: {e.response.text[:200]}")


@app.post("/api/search")
async def search_only(body: dict):
    query = body.get("query", "").strip()
    k = body.get("k", 5)
    if not query:
        return {"error": "Query is required"}
    if len(query) > MAX_QUESTION_LEN:
        return {"error": f"Query too long (max {MAX_QUESTION_LEN} chars)"}
    if not isinstance(k, int) or k < MIN_K or k > MAX_K:
        return {"error": f"k must be int between {MIN_K} and {MAX_K}"}
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(SEARCH_URL, params={"query": query, "k": k})
            r.raise_for_status()
            d = r.json()
        return {"query": d.get("query", query), "count": d.get("count", 0), "results": d.get("results", [])}
    except Exception as e:
        return {"error": f"Search failed: {e}"}


@app.post("/api/chat")
async def chat(body: dict):
    question = body.get("question", "").strip()
    provider_id = body.get("provider", DEFAULT_PROVIDER)
    model = body.get("model", DEFAULT_MODEL)
    k = body.get("k", 5)
    temperature = body.get("temperature", DEFAULT_TEMPERATURE)
    num_predict = body.get("num_predict", DEFAULT_NUM_PREDICT)
    num_ctx = body.get("num_ctx", DEFAULT_NUM_CTX)
    history = body.get("history", [])
    search_backend = body.get("search_backend", "tavily" if WEB_SEARCH_ENABLED == "1" else "off")
    if search_backend not in ("tavily", "searxng", "off"):
        search_backend = "off"

    if not question:
        return {"error": "Question is required"}
    if len(question) > MAX_QUESTION_LEN:
        return {"error": f"Question too long (max {MAX_QUESTION_LEN} chars)"}
    if not isinstance(k, int) or k < MIN_K or k > MAX_K:
        return {"error": f"k must be int between {MIN_K} and {MAX_K}"}
    if not isinstance(temperature, (int, float)) or temperature < MIN_TEMPERATURE or temperature > MAX_TEMPERATURE:
        return {"error": f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"}
    if not isinstance(num_predict, int) or num_predict < 1 or num_predict > 131072:
        return {"error": "num_predict must be int between 1 and 131072"}
    if not isinstance(history, list):
        return {"error": "history must be a list"}

    provider = None
    for p in PROVIDERS:
        if p["id"] == provider_id:
            provider = p
            break
    if not provider:
        return {"error": f"Provider '{provider_id}' not found"}

    # === RAG search ===
    rag_context = ""
    rag_sources = []
    # Обогащаем RAG-запрос контекстом последнего вопроса из истории
    rag_query = question
    if history:
        for msg in reversed(history):
            if msg.get("role") == "user":
                prev_q = msg["content"].strip()
                if prev_q and prev_q != question:
                    rag_query = f"{prev_q} {question}"
                    break
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(SEARCH_URL, params={"query": rag_query, "k": k})
            r.raise_for_status()
            search_data = r.json()
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    results = search_data.get("results", [])
    context_parts = []
    for r_item in results:
        text = r_item.get("text", "")
        source = r_item.get("source", "unknown")
        product = r_item.get("product", "")
        score = r_item.get("rerank_score", r_item.get("score", 0))
        images_raw = r_item.get("images", [])
        images = [f"{SEARCH_SERVER_BASE}{img}" for img in images_raw] if images_raw else []
        page = r_item.get("page")
        page_info = f" (стр. {page})" if page is not None else ""
        context_parts.append(f"[Source: {source}{page_info}, product: {product}, score: {score}]\n{text}")
        # Ссылка на конкретную страницу PDF (#page=N)
        pdf_path = f"/docs/{urllib.parse.quote(source)}"
        if page is not None:
            pdf_path += f"#page={page}"
        rag_sources.append({
            "source": source,
            "product": product,
            "score": round(score, 3),
            "images": images,
            "page": page,
            "pdf_url": pdf_path,
        })

    if context_parts:
        rag_context = "\n\n---\n\n".join(context_parts)

    # === Веб-поиск с обогащением из RAG ===
    # Сначала RAG: находим тему, потом ищем веб в контексте этой темы
    web_context = ""
    web_sources = []
    enriched_query = question
    if search_backend != "off" and rag_sources:
        rag_terms = set()
        for r_item in results[:5]:
            product = r_item.get("product", "")
            if product and len(product) > 2:
                rag_terms.add(product)
            source = r_item.get("source", "")
            stem = source.replace(".pdf", "").replace(".PDF", "").replace("_", " ").replace("-", " ")
            for term in stem.split():
                term = term.strip()
                if len(term) > 3 and term.lower() not in ("the", "and", "for", "with", "user", "guide", "edition", "manual", "document", "pdf", "version"):
                    rag_terms.add(term)
        if rag_terms:
            enriched_query = f"{question} {' '.join(sorted(rag_terms))}"
            logger.info(f"RAG-enriched web query: '{question}' → '{enriched_query[:120]}'")
    if search_backend != "off":
        try:
            if search_backend == "tavily":
                web_results = await _web_search(enriched_query, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT)
                if not web_results:
                    logger.info("Tavily вернул пусто, fallback → SearXNG")
                    web_results = await _web_search_searxng(enriched_query, WEB_SEARCH_MAX_RESULTS, min(WEB_SEARCH_TIMEOUT, 10))
            else:
                web_results = await _web_search_searxng(enriched_query, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT)
            if web_results:
                web_parts = []
                for wr in web_results:
                    title = wr.get("title", "")
                    url = wr.get("url", "")
                    snippet = wr.get("snippet", "")
                    web_parts.append(f"[Web: {title}]({url})\n{snippet}")
                web_context = "\n\n---\n\n".join(web_parts)
                web_sources = [{
                    "source": wr.get("url", ""),
                    "product": wr.get("title", ""),
                    "score": round(wr.get("score", 1.0), 3),
                    "images": [],
                    "page": None,
                    "url": wr.get("url", ""),
                    "pdf_url": None,
                } for wr in web_results]
                logger.info(f"Web search{' (enriched)' if rag_sources else ''} added {len(web_results)} results")
        except Exception as e:
            logger.warning(f"Web search error: {e}")

    # === Собираем контекст: RAG приоритетный, веб — дополнительный ===
    parts = []
    if rag_context:
        parts.append("=== Техническая документация ===\n" + rag_context)
    if web_context:
        parts.append("=== Результаты веб-поиска ===\n" + web_context)
    context = "\n\n".join(parts) if parts else "Нет релевантного контекста."

    # sources для UI: RAG (PDF) всегда первыми, веб — после
    sources = rag_sources[:5] + web_sources[:5]

    system_prompt = (
            "Ты — технический ассистент поддержки. У тебя два источника информации:\n"
            "1) Техническая документация (раздел «=== Техническая документация ===») — приоритетный источник.\n"
            "2) Результаты веб-поиска (раздел «=== Результаты веб-поиска ===») — дополнительный источник.\n\n"
            "Правила:\n"
            "- ВСЕГДА сначала смотри в техническую документацию. Используй её как основной источник.\n"
            "- Веб-результаты используй только для дополнения или актуализации (даты, события).\n"
            "- Никогда не игнорируй документацию в пользу веба.\n"
            "- Для каждого утверждения из документации указывай имя файла и номер страницы "
            "в формате: «[имя_файла, стр. N]».\n"
            "- Для каждого утверждения из веба указывай кликабельную ссылку "
            "в формате: «[заголовок](url)».\n"
            "- Если ответ можно дать из нескольких источников — укажи все.\n"
            "- НЕ выдумывай URL-адреса. Никогда не используй example.com.\n"
            "- Для RAG-документов используй только формат «[имя_файла, стр. N]», не делай markdown-ссылок.\n"
            "- Для веб-ссылок копируй URL из контекста как есть, не переписывай.\n"
            "- Выдавай максимально полную информацию."
    )

    user_content = f"Контекст:\n{context}\n\nВопрос: {question}"

    # Ограничиваем историю до 3 последних оборотов (6 сообщений)
    if len(history) > 6:
        history = history[-6:]

    if provider["type"] == "openai":
        # OpenAI-провайдеры (deepseek и т.п.): контекст в system для лучшего восприятия
        enhanced_system = system_prompt + f"\n\nКонтекст документации:\n{context}"
        messages = [{"role": "system", "content": enhanced_system}]
        messages.extend(history)
        if not history or history[-1].get("content") != question:
            messages.append({"role": "user", "content": f"Вопрос: {question}"})
    else:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        if not history or history[-1].get("content") != question:
            messages.append({"role": "user", "content": user_content})

    # Динамический token budgeting: вычитаем размер входного промпта из num_ctx
    all_text = sum(len(m.get("content", "")) for m in messages)
    input_tokens_est = max(1, all_text // 4)
    token_budget = num_ctx - input_tokens_est - 300
    token_budget = max(128, min(num_predict, token_budget))
    if token_budget < num_predict:
        logger.info(f"→ budget: num_ctx={num_ctx}, input≈{input_tokens_est} токенов, " +
                     f"num_predict урезан {num_predict}→{token_budget}")
    num_predict = token_budget

    async def generate():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
        if provider["type"] == "ollama":
            async for chunk in _stream_ollama(provider, model, messages, temperature, num_predict, num_ctx):
                yield chunk
        elif provider["type"] == "openai":
            async for chunk in _stream_openai(provider, model, messages, temperature, num_predict):
                yield chunk
        elif provider["type"] == "anthropic":
            async for chunk in _stream_anthropic(provider, model, messages, temperature, num_predict, system_prompt):
                yield chunk
        else:
            ptype = provider["type"]
            yield f"data: {json.dumps({'type': 'error', 'data': f'Unknown provider type: {ptype}'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ======================= Stream handlers =======================


async def _stream_ollama(provider, model, messages, temperature, num_predict, num_ctx=None):
    url = f"{provider['base_url']}/api/chat"
    if num_ctx is None:
        num_ctx = DEFAULT_NUM_CTX
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": -1,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "thinking": False,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload, timeout=120) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        content = msg.get("content", "")
                        thinking = msg.get("thinking", "")
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'data': content}, ensure_ascii=False)}\n\n"
                        if chunk.get("done", False):
                            done_data = {
                                "total_duration": chunk.get("total_duration", 0),
                                "eval_count": chunk.get("eval_count", 0),
                                "eval_duration": chunk.get("eval_duration", 0),
                            }
                            yield f"data: {json.dumps({'type': 'done', 'data': done_data}, ensure_ascii=False)}\n\n"
                    except json.JSONDecodeError:
                        pass
    except httpx.TimeoutException:
        yield f"data: {json.dumps({'type': 'error', 'data': 'LLM request timed out — Ollama took too long.'})}\n\n"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            err_msg = f"Model '{model}' not found in Ollama."
            yield f"data: {json.dumps({'type': 'error', 'data': err_msg})}\n\n"
        else:
            err_msg = f"Ollama HTTP {e.response.status_code}"
            yield f"data: {json.dumps({'type': 'error', 'data': err_msg})}\n\n"
    except httpx.ConnectError:
        yield f"data: {json.dumps({'type': 'error', 'data': 'Cannot connect to Ollama — is the service running?'})}\n\n"
    except httpx.ReadError:
        yield f"data: {json.dumps({'type': 'error', 'data': 'Connection lost while reading Ollama response.'})}\n\n"


async def _stream_openai(provider, model, messages, temperature, num_predict):
    url = f"{provider['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    base_url_lower = (provider.get("base_url") or "").lower()
    if "openrouter" in base_url_lower:
        headers["HTTP-Referer"] = "http://localhost:8080"
        headers["X-Title"] = "TVHelper Web Agent"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=120) as resp:
                resp.raise_for_status()
                eval_count = 0
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    if line == "data: [DONE]":
                        done_data = {"eval_count": eval_count, "eval_duration": 0, "total_duration": 0}
                        yield f"data: {json.dumps({'type': 'done', 'data': done_data}, ensure_ascii=False)}\n\n"
                        return
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("delta", {}).get("content"):
                                content = choices[0]["delta"]["content"]
                                eval_count += 1
                                yield f"data: {json.dumps({'type': 'token', 'data': content}, ensure_ascii=False)}\n\n"
                        except json.JSONDecodeError:
                            pass
    except httpx.TimeoutException:
        yield f"data: {json.dumps({'type': 'error', 'data': 'LLM request timed out.'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': f'LLM error: {e}'})}\n\n"


async def _stream_anthropic(provider, model, messages, temperature, num_predict, system_prompt):
    url = f"{provider['base_url'].rstrip('/')}/messages"
    headers = {
        "x-api-key": provider["api_key"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    anthro_messages = [m for m in messages if m["role"] != "system"]
    payload = {
        "model": model,
        "messages": anthro_messages,
        "system": system_prompt,
        "stream": True,
        "max_tokens": num_predict,
        "temperature": temperature,
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload, headers=headers, timeout=120) as resp:
                resp.raise_for_status()
                event_type = None
                eval_count = 0
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    content = delta.get("text", "")
                                    if content:
                                        eval_count += 1
                                        yield f"data: {json.dumps({'type': 'token', 'data': content}, ensure_ascii=False)}\n\n"
                            elif event_type == "message_done":
                                msg = data.get("message", {})
                                usage = msg.get("usage", {})
                                done_data = {
                                    "eval_count": usage.get("output_tokens", eval_count),
                                    "eval_duration": 0,
                                    "total_duration": 0,
                                }
                                yield f"data: {json.dumps({'type': 'done', 'data': done_data}, ensure_ascii=False)}\n\n"
                        except json.JSONDecodeError:
                            pass
    except httpx.TimeoutException:
        yield f"data: {json.dumps({'type': 'error', 'data': 'LLM request timed out.'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': f'LLM error: {e}'})}\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)