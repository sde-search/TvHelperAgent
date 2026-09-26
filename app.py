#!/usr/bin/env python3
"""TVHelper Web Agent — RAG-агент с веб-интерфейсом и поддержкой внешних API."""

import json
import os
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"


def _load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) > 1 and val[0] in '"\'' and val[-1] == val[0]:
            val = val[1:-1]
        os.environ.setdefault(key, val)


_load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="TVHelper Web Agent")

# === Настройки ===
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:11436/search")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
DEFAULT_NUM_PREDICT = int(os.getenv("DEFAULT_NUM_PREDICT", "2048"))
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "16384"))

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
                    entry["models"].append({"name": name, "size": m.get("size", 0)})
            except Exception as e:
                entry["error"] = str(e)
        else:
            for mname in p["models"]:
                mname = mname.strip()
                if mname:
                    entry["models"].append({"name": mname})
        result["providers"].append(entry)
    return result


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


@app.post("/api/search")
async def search_only(body: dict):
    query = body.get("query", "").strip()
    k = body.get("k", 5)
    if not query:
        return {"error": "Query is required"}
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
    history = body.get("history", [])

    if not question:
        return {"error": "Question is required"}

    provider = None
    for p in PROVIDERS:
        if p["id"] == provider_id:
            provider = p
            break
    if not provider:
        return {"error": f"Provider '{provider_id}' not found"}

    # RAG search
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(SEARCH_URL, params={"query": question, "k": k})
            r.raise_for_status()
            search_data = r.json()
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    results = search_data.get("results", [])
    sources = []
    context_parts = []
    for r_item in results:
        text = r_item.get("text", "")
        source = r_item.get("source", "unknown")
        product = r_item.get("product", "")
        score = r_item.get("rerank_score", r_item.get("score", 0))
        sources.append({"source": source, "product": product, "score": round(score, 3)})
        context_parts.append(f"[Source: {source} (product: {product}, score: {score})]\n{text}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    system_prompt = (
        "Ты — технический ассистент поддержки. У тебя есть база знаний технической документации.\n"
        "Отвечай на вопрос пользователя ТОЛЬКО на основе предоставленного контекста. "
        "Если контекста недостаточно — так и скажи.\n"
        "Обязательно указывай имя файла-источника и название продукта.\n"
        "Отвечай КРАТКО и ПО СУЩЕСТВУ.\n"
        f"Контекст:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    if not history or history[-1].get("content") != question:
        messages.append({"role": "user", "content": question})

    async def generate():
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
        if provider["type"] == "ollama":
            async for chunk in _stream_ollama(provider, model, messages, temperature, num_predict):
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


async def _stream_ollama(provider, model, messages, temperature, num_predict):
    url = f"{provider['base_url']}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": -1,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": DEFAULT_NUM_CTX,
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
                        if chunk.get("message", {}).get("content"):
                            content = chunk["message"]["content"]
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
        yield f"data: {json.dumps({'type': 'error', 'data': 'LLM request timed out.'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': f'LLM error: {e}'})}\n\n"


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