#!/usr/bin/env python3
"""Гибридный поиск: векторный (ChromaDB) + BM25 + реранкер + фильтр по продукту."""

import os, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE / "data" / "chroma"
COLLECTION_NAME = "docs"
TOP_K_VECTOR = 30
TOP_K_BM25 = 20


class HybridSearch:
    """Гибридный поиск с поддержкой фильтрации по продукту."""

    def __init__(
        self,
        chroma_path,
        collection_name,
        bm25_pkl_path,
        embedding_model="bge-m3",
        ollama_base_url="http://localhost:11434",
    ):
        import chromadb
        import nltk
        from llama_index.embeddings.ollama import OllamaEmbedding
        from rank_bm25 import BM25Okapi

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.chroma_client = chromadb.PersistentClient(str(chroma_path))
        self.collection = self.chroma_client.get_collection(collection_name)

        self.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=ollama_base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )
        self.embed_query_prefix = "query: "  # bge-m3: prefix for queries per docs

        with open(bm25_pkl_path, "rb") as f:
            self.bm25_data = pickle.load(f)

        tokenized_corpus = [
            nltk.word_tokenize(d.lower()) for d in self.bm25_data["documents"]
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Reranker (CrossEncoder multilingual on GPU)
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder

            reranker_model = os.environ.get(
                "RAG_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            )
            self.reranker = CrossEncoder(
                reranker_model,
                device="cuda",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  [WARN] Reranker не загружен: {e}")

    def search(self, query: str, k: int = 10, product_filter: str = None) -> list[dict]:
        """Поиск с опциональным фильтром по product."""
        import nltk

        where_filter = None
        if product_filter:
            where_filter = {"product": product_filter}

        # --- 1. Векторный поиск (ChromaDB) ---
        # bge-m3 требует query: prefix для запросов
        q_embed = self.embed_model.get_text_embedding(self.embed_query_prefix + query)
        vec_results = self.collection.query(
            query_embeddings=[q_embed],
            n_results=TOP_K_VECTOR,
            where=where_filter,
        )

        vec_docs = []
        for i in range(len(vec_results["ids"][0])):
            meta = vec_results["metadatas"][0][i]
            vec_docs.append(
                {
                    "id": vec_results["ids"][0][i],
                    "text": vec_results["documents"][0][i],
                    "source": meta.get("source", ""),
                    "page": meta.get("page"),
                    "product": meta.get("product", ""),
                    "score": 1 - vec_results["distances"][0][i]
                    if vec_results["distances"]
                    else 0,
                }
            )

        # --- 2. BM25 поиск ---
        tokenized_query = nltk.word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Запрашиваем больше кандидатов при фильтре, т.к. часть отсеется
        bm25_candidate_k = TOP_K_BM25 * 3 if product_filter else TOP_K_BM25
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:bm25_candidate_k]

        bm25_docs = []
        for idx in top_bm25_indices:
            if bm25_scores[idx] > 0:
                meta = self.bm25_data["metadatas"][idx]
                prod = meta.get("product", "")
                if product_filter and prod != product_filter:
                    continue
                bm25_docs.append(
                    {
                        "id": self.bm25_data["ids"][idx],
                        "text": self.bm25_data["documents"][idx],
                        "source": meta.get("source", ""),
                        "product": prod,
                        "score": bm25_scores[idx],
                    }
                )

        # --- 3. Fusion (RSF — Reciprocal Rank Fusion) ---
        seen = {}
        for rank, d in enumerate(vec_docs):
            seen[d["id"]] = seen.get(d["id"], 0) + 1.0 / (rank + 60)
        for rank, d in enumerate(bm25_docs):
            seen[d["id"]] = seen.get(d["id"], 0) + 1.0 / (rank + 60)

        fused = []
        for did, rrf_score in sorted(seen.items(), key=lambda x: -x[1]):
            for d in vec_docs + bm25_docs:
                if d["id"] == did:
                    d["rrf_score"] = rrf_score
                    fused.append(d)
                    break

        # --- 4. Реренкинг (CrossEncoder на GPU) ---
        if self.reranker and fused:
            passages = [d["text"] for d in fused]
            query_passage_pairs = [(query, p) for p in passages]
            scores = self.reranker.predict(query_passage_pairs, show_progress_bar=False)
            for i, score in enumerate(scores):
                fused[i]["rerank_score"] = float(score)
            fused.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return fused[:k]


def build_query_engine():
    """Backward-compat alias (standalone use)."""
    return HybridSearch(
        chroma_path=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        bm25_pkl_path=str(CHROMA_DIR / "bm25_data.pkl"),
    ).search


def format_answer(results: list[dict], query: str) -> tuple[str, list[str]]:
    """Формирует контекст для LLM."""
    context_parts = []
    sources = []
    for r in results:
        ctx = (
            f"[{r['source']} (chunk {r.get('rerank_score', 0):.3f})]"
            f"\n{r['text'][:1500]}"
        )
        context_parts.append(ctx)
        if r["source"] not in sources:
            sources.append(r["source"])

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def ask(query: str, top_k: int = 5) -> str:
    """Полный цикл: поиск + LLM ответ (только для локального теста)."""
    import httpx
    import json

    search_fn = build_query_engine()
    results = search_fn(query, top_k=top_k)

    if not results:
        return "Ничего не найдено по вашему запросу."

    context, sources = format_answer(results, query)

    prompt = f"""Ты — технический эксперт. Отвечай на русском языке, опираясь ТОЛЬКО на предоставленный контекст.

Контекст:
{context[:12000]}

Вопрос пользователя: {query}

Дай точный, структурированный ответ со ссылками на источники."""

    llm_model = os.environ.get("RAG_LLM_MODEL", "qwen3:8b")
    llm_endpoint = os.environ.get("RAG_LLM_ENDPOINT", "http://localhost:11434/v1/chat/completions")
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(llm_endpoint, json=payload)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]

    return f"{answer}\n\n📎 Источники: {', '.join(sources[:5])}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(ask(q))
    else:
        print('Использование: python hybrid_search.py "твой вопрос"')