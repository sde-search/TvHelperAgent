#!/usr/bin/env python3
"""Wrapper: запускает полную индексацию с прогресс-трекингом."""
import sys, os, re, json, time, builtins
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

os.chdir(str(Path(__file__).resolve().parent))
os.environ.pop("RAG_EMBED_MODEL", None)  # используем bge-m3 по умолчанию

from ingestion_hybrid import index_documents, DATA_PDFS

PROGRESS_FILE = Path("data/progress.json")
last_pct = 0

def emit_progress(pct: int, phase: str, detail: str):
    global last_pct
    if pct >= last_pct + 5 or pct == 100:
        last_pct = (pct // 5) * 5
        line = f"PROGRESS: {pct}% — {phase}: {detail}"
        print(line, flush=True)
    # always write file for status
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"percent": pct, "phase": phase, "detail": detail, "ts": time.time()}, f)

original_print = builtins.print
def tracked_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    # Parse progress from standard messages
    # File-level: "  [3/435] filename.pdf"
    m_file = re.search(r'^\s*\[(\d+)/(\d+)\]', msg)
    if m_file:
        done = int(m_file.group(1))
        total = int(m_file.group(2))
        pct = int(60 * done / total)
        emit_progress(pct, "Извлечение текста/картинок", f"файл {done}/{total}")
        original_print(*args, **kwargs)
        return
    
    # Embedding batch: "  векторизовано 32/1500"
    if "векторизовано" in msg:
        m_emb = re.search(r'(\d+)/(\d+)$', msg)
        if m_emb:
            done = int(m_emb.group(1))
            total = int(m_emb.group(2))
            pct = 60 + int(35 * done / total)
            emit_progress(pct, "Генерация эмбеддингов", f"чанк {done}/{total}")
        else:
            # first embedding message
            emit_progress(60, "Генерация эмбеддингов", "начало")
        original_print(*args, **kwargs)
        return
    
    if "Векторная БД:" in msg:
        emit_progress(95, "BM25", "сохранение индекса")
    elif "BM25 данные" in msg:
        emit_progress(97, "BM25", "сохранение данных")
    elif "завершена" in msg.lower() and "✅" in msg:
        emit_progress(100, "Готово", "индексация завершена")
    
    original_print(*args, **kwargs)

builtins.print = tracked_print

# --- MAIN ---
print("Запуск полной индексации...", flush=True)
emit_progress(0, "Старт", "начало индексации")
index_documents(DATA_PDFS)
emit_progress(100, "Готово", "✅ Полная индексация завершена!")