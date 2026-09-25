#!/usr/bin/env python3
"""TVHelper Web Agent — минимальный RAG-агент с веб-интерфейсом."""

import json
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="TVHelper Web Agent")

# === Настройки ===
SEARCH_URL = "http://localhost:11436/search"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:8b"
SEARCH_K = 5


# === API endpoints ===

@app.get("/", response_class=HTMLResponse)
async def index():
    """Главная страница — читаем HTML напрямую."""
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models():
    """Список доступных чат-моделей из Ollama."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            data = resp.json()
        models = []
        for m in data.get("models", []):
            name = m["name"]
            # Показываем только чат-модели (не эмбеддинги)
            if "embed" not in name.lower():
                models.append({
                    "name": name,
                    "size": m.get("size", 0),
                    "modified": m.get("modified_at", ""),
                })
        return {"models": models, "default": DEFAULT_MODEL}
    except Exception as e:
        return {"models": [], "default": DEFAULT_MODEL, "error": str(e)}


@app.post("/api/search")
async def search_only(body: dict):
    """Только поиск по RAG-базе (без LLM)."""
    query = body.get("query", "").strip()
    k = body.get("k", SEARCH_K)
    if not query:
        return {"error": "Query is required"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(SEARCH_URL, params={"query": query, "k": k})
            resp.raise_for_status()
            data = resp.json()
        return {
            "query": data.get("query", query),
            "count": data.get("count", 0),
            "results": data.get("results", []),
        }
    except Exception as e:
        return {"error": f"Search failed: {e}"}


@app.post("/api/chat")
async def chat(body: dict):
    """Ответ с RAG-контекстом от LLM (streaming)."""
    question = body.get("question", "").strip()
    model = body.get("model", DEFAULT_MODEL)
    k = body.get("k", SEARCH_K)

    if not question:
        return {"error": "Question is required"}

    # 1. Поиск по RAG-базе
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(SEARCH_URL, params={"query": question, "k": k})
            resp.raise_for_status()
            search_data = resp.json()
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    results = search_data.get("results", [])
    sources = []
    context_parts = []

    for r in results:
        text = r.get("text", "")
        source = r.get("source", "unknown")
        product = r.get("product", "")
        score = r.get("rerank_score", r.get("score", 0))
        sources.append({"source": source, "product": product, "score": round(score, 3)})
        context_parts.append(f"[Source: {source} (product: {product}, score: {score})]\n{text}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

    # 2. Формируем промпт
    system_prompt = """Ты — технический ассистент поддержки. У тебя есть база знаний технической документации.
Отвечай на вопрос пользователя ТОЛЬКО на основе предоставленного контекста. Если контекста недостаточно — так и скажи.
Обязательно указывай имя файла-источника и название продукта.
Отвечай КРАТКО и ПО СУЩЕСТВУ.
ВАЖНО: Отвечай на том же языке, на котором задан вопрос пользователя."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    # 3. Отправляем в Ollama (streaming)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "num_predict": 2048,
        },
    }

    async def generate():
        # Сначала шлём информацию об источниках в формате JSON-события
        sources_data = json.dumps({"type": "sources", "data": sources}, ensure_ascii=False)
        yield f"data: {sources_data}\n\n"

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload, timeout=120) as resp:
                    resp.raise_for_status()
                    full_text = ""
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                full_text += content
                                event = json.dumps({"type": "token", "data": content}, ensure_ascii=False)
                                yield f"data: {event}\n\n"
                            if chunk.get("done", False):
                                # Финальное сообщение с метриками
                                done_event = json.dumps({
                                    "type": "done",
                                    "data": {
                                        "total_duration": chunk.get("total_duration", 0),
                                        "eval_count": chunk.get("eval_count", 0),
                                        "eval_duration": chunk.get("eval_duration", 0),
                                    }
                                }, ensure_ascii=False)
                                yield f"data: {done_event}\n\n"
                        except json.JSONDecodeError:
                            pass
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'type': 'error', 'data': 'LLM request timed out. Try a smaller model.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': f'LLM error: {e}'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)