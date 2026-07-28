"""Disk cache for the safetensors header sweep, keyed on the repo commit sha.

The sweep is by far the slowest thing the analyzer does: DeepSeek-V3 has 163
shards, each needing a range request whose latency against the HF CDN is ~2 s,
so a cold catalog costs 25-90 s depending on concurrency. It depends only on
the repo's commit sha, so it belongs on disk rather than in a process:

  * the webapp's in-process cache dies with the container (every redeploy made
    every model cold again),
  * report-generating subprocesses never shared that cache at all — each one
    re-swept the same shards the API had just swept.

`config.json` is deliberately NOT cached: it costs ~1 s, it is the authoritative
input to every formula, and caching it would put a staleness window in front of
the one file users edit when they re-quantize a checkpoint.

Layout: one gzipped JSON per model under $ANALYZER_CACHE_DIR (default
webapp/data/hf_cache, which is the container's persistent volume). Entries are
self-describing — a payload whose `sha` no longer matches the repo is refetched,
so a stale file is a slow request, never a wrong number.
"""

import gzip
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("ANALYZER_CACHE_DIR") or os.path.join(
    BASE_DIR, os.pardir, "data", "hf_cache")
# bound disk use: catalogs run ~0.5-2 MB gzipped (91,991 tensors for DSv3)
MAX_ENTRIES = int(os.environ.get("HF_CACHE_MAX_ENTRIES", "200"))

FORMAT = 1  # bump when the payload shape changes; old entries then miss


def _path(model_id: str) -> str:
    slug = model_id.lower().replace("/", "--")
    return os.path.join(CACHE_DIR, f"{slug}.json.gz")


def load(model_id: str, sha: str | None) -> dict | None:
    """Cached payload for `model_id`, or None on miss/mismatch/corruption.

    `sha` is the repo's current commit sha. None means "could not determine"
    (HF unreachable or rate-limited) — the entry is then accepted as-is, since
    a possibly-stale catalog beats failing the request outright.
    """
    try:
        with gzip.open(_path(model_id), "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):     # absent, truncated, or not gzip/JSON
        return None
    if payload.get("format") != FORMAT:
        return None
    if sha is not None and payload.get("sha") != sha:
        return None
    # JSON has no tuples: restore the (name, shard) pairs the evidence tab
    # unpacks, so a cache hit is indistinguishable from a live sweep
    info = payload.get("index_info")
    if info and "weight_map_sample" in info:
        info["weight_map_sample"] = [tuple(p) for p in info["weight_map_sample"]]
    return payload


def store(model_id: str, payload: dict) -> None:
    """Write `payload` for `model_id`. Best-effort: a read-only or full disk
    degrades to no caching, never to a failed analysis."""
    payload = dict(payload, format=FORMAT, stored_at=time.time())
    path = _path(model_id)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, path)         # atomic: concurrent readers see old or new
        prune()
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def prune() -> None:
    """Drop the least-recently-written entries past MAX_ENTRIES."""
    try:
        names = [n for n in os.listdir(CACHE_DIR) if n.endswith(".json.gz")]
    except OSError:
        return
    if len(names) <= MAX_ENTRIES:
        return
    stamped = []
    for n in names:
        p = os.path.join(CACHE_DIR, n)
        try:
            stamped.append((os.path.getmtime(p), p))
        except OSError:
            continue
    for _mtime, p in sorted(stamped)[:len(stamped) - MAX_ENTRIES]:
        try:
            os.unlink(p)
        except OSError:
            pass
