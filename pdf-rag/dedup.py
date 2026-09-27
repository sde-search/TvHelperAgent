#!/usr/bin/env python3
"""Dedup PDFs: keep 1 per hash, prefer base name (no _NUMBER suffix)."""
import hashlib
from pathlib import Path

pdfs = sorted(Path("data/pdfs").glob("*.pdf"))
seen = {}  # hex -> (path, is_preferred)
removed = 0

for p in pdfs:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    # prefer: no _digits before .pdf, or shorter name
    import re
    is_pref = not re.search(r'_\d+\.pdf$', p.name)
    if h not in seen:
        seen[h] = (p, is_pref)
    else:
        existing, existing_pref = seen[h]
        if is_pref and not existing_pref:
            # current is better — remove existing
            print(f"REMOVE {existing.name} (dup, kept {p.name})")
            existing.unlink()
            seen[h] = (p, is_pref)
            removed += 1
        else:
            print(f"REMOVE {p.name} (dup, kept {existing.name})")
            p.unlink()
            removed += 1

left = len(list(Path("data/pdfs").glob("*.pdf")))
print(f"\nDedup done: {removed} removed, {left} remain")