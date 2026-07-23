"""One-off: import pre-existing reports from html_output/ into the webapp cache.

Extracts the HF model ID from each report's meta line ("org/name · arch · ...")
and registers the file under data/reports/. Skips files whose model ID cannot
be determined and slugs that are already registered (later files win among
duplicates of the same model, so glm-5.2-fp8-v3 overrides v1/v2).

Usage: python3 import_existing.py [source_dir]
"""

import re
import shutil
import sys
from pathlib import Path

from app import DB_PATH, REPORTS_DIR, db, init_db, now, slug_for

MODEL_ID_META_RE = re.compile(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+) · ")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "html_output"
    init_db()
    imported, skipped = [], []
    for f in sorted(src.glob("*.html")):
        head = f.read_text(errors="ignore")[:200_000]
        m = MODEL_ID_META_RE.search(head)
        if not m:
            skipped.append((f.name, "no model id found"))
            continue
        model_id = m.group(1)
        lang = "zh" if '<html lang="zh"' in head[:200] else "en"
        slug = slug_for(model_id, lang)
        dest = REPORTS_DIR / f"{slug}.html"
        shutil.copyfile(f, dest)
        with db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reports"
                "(slug, model_id, lang, created_at, created_by, size_bytes, last_access)"
                " VALUES(?,?,?,?,?,?,?)",
                (slug, model_id, lang, now(), "import", dest.stat().st_size, now()),
            )
        imported.append((f.name, slug))
    for name, slug in imported:
        print(f"imported {name} -> {slug}")
    for name, why in skipped:
        print(f"SKIPPED {name}: {why}")
    print(f"\n{len(imported)} imported, {len(skipped)} skipped; db={DB_PATH}")


if __name__ == "__main__":
    main()
