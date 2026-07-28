"""Server-side what-if engine: the parallel-tab and roofline-tab math.

This is the SINGLE implementation of the interactive what-if calculators
(gpuMemory / kvUtil / rooflineKernels / verdict cards). template.js is a pure
renderer: every dropdown change fetches /api/v1/whatif, which is built from
whatif_payload() below. There is deliberately NO math in the browser.

Everything here is pure arithmetic over the dicts produced by main.analyze()
and main.per_layer_breakdown(). No i18n: numbers and short English notes only
(the consuming LLM/report does the language).

Parity notes (mirrors template.js line-for-line where it matters):
  * attn-TP sharding divisor: dp-attention sets the attention TP group to 1,
    so attention projections, embed, MLA w_kc/w_vc and the draft embed become
    one full copy per rank; lm_head stays /tp (enable_dp_lm_head off).
  * KV shard: dp-attention 1/tp; pure TP: MLA replicates the full cache per
    rank (no head dim), GQA shards by min(tp, kv_heads).
  * SGLang mem-fraction-static semantics: static region = frac × cap is
    pre-allocated as weights + KV pool; activation lives outside it.
  * Roofline: kernel time = max(bytes/BW, FLOPs/peak); replicated components
    scale HBM bytes ×factor so the shared ÷TP stays honest.
"""

import hashlib
import math
import os

import main as core

GIB = 1024 ** 3


# ---------------------------------------------------------------- versioning

def _files_hash(*fnames: str) -> str:
    h = hashlib.sha256()
    base = os.path.dirname(os.path.abspath(__file__))
    for fname in fnames:
        with open(os.path.join(base, fname), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def engine_version() -> str:
    """Content hash over the compute sources — changes whenever the math does."""
    return _files_hash("main.py", "engine.py")


def report_version() -> str:
    """Hash over everything baked into a report page (math + markup + i18n) —
    a cached report whose stored version differs is stale and gets regenerated
    on next view."""
    return _files_hash("main.py", "engine.py", "i18n.py",
                       "template.html", "template.js")


# ---------------------------------------------------------------- config prep

def normalize_config(cfg: dict) -> dict:
    """Un-nest multimodal configs (LLM under text_config) keeping quantization,
    vision tower, and mm token count — same treatment as main.main()."""
    if "num_hidden_layers" not in cfg and "text_config" in cfg:
        qc, vc, mm_tok = (cfg.get("quantization_config"), cfg.get("vision_config"),
                          cfg.get("mm_tokens_per_image"))
        cfg = cfg["text_config"]
        if qc and "quantization_config" not in cfg:
            cfg["quantization_config"] = qc
        if vc and "vision_config" not in cfg:
            cfg["vision_config"] = vc
        if mm_tok and "mm_tokens_per_image" not in cfg:
            cfg["mm_tokens_per_image"] = mm_tok
    return cfg


def resolve_kv_auto(cfg: dict) -> str:
    """SGLang's kv-dtype default: fp8 for DSA/V4-style sparse-attention models,
    else the activation dtype (bf16)."""
    arch = (cfg.get("architectures") or [""])[0]
    is_dsa = cfg.get("index_topk") is not None or arch in (
        "DeepseekV4ForCausalLM", "DeepseekV32ForCausalLM")
    return "fp8" if is_dsa else "bf16"


# ---------------------------------------------------------------- KV cell math

def kv_bytes_per(dtype: str) -> float:
    """Bytes per stored KV element. fp4 = mxfp4: 0.5 B data + 1 uint8 scale / 16."""
    return {"fp8": 1.0, "fp4": 0.5625, "fp16": 2.0, "bf16": 2.0}.get(dtype, 2.0)


def kv_cell_bytes(D: dict, dtype: str) -> float:
    """Bytes per token per KV-bearing layer; the DSA index-key add-on does not
    follow the KV dtype."""
    return D["kvElemsPerLayer"] * kv_bytes_per(dtype) + (D.get("kvIndexerBytes") or 0)


def kv_layer_tokens(D: dict, ctx: int) -> float:
    """Stored token-positions summed over layers, honoring sliding-window caps."""
    groups = D.get("kvGroups") or []
    if groups:
        return sum(n * (min(ctx, w) if w else ctx) for n, w in groups)
    return (D.get("nKvLayers") or D["L"]) * ctx


# ---------------------------------------------------------------- deploy data

def deploy_data(a: dict, cfg: dict) -> dict:
    """The per-model constants the parallel/roofline math needs — the Python
    twin of the `D` object render_html() serializes for template.js."""
    p = a["p"]
    layer = core.per_layer_breakdown(a, cfg)
    elems, _ = core.kv_per_token_elems(cfg)
    comps = {c["key"]: c for c in a["comps"]}
    n_routed = (cfg.get("n_routed_experts") or cfg.get("num_experts")
                or cfg.get("num_local_experts") or 0)
    return {
        "L": cfg["num_hidden_layers"],
        "nDense": p["dense_layers"],
        "nMoe": p["moe_layers"],
        "nExperts": n_routed,
        "topk": cfg.get("num_experts_per_tok", 0),
        "layer": layer,
        "embed": comps.get("embed", {}).get("bytes", 0),
        "lmHead": comps.get("lm_head", {}).get("bytes", 0),
        "tied": bool(cfg.get("tie_word_embeddings")),
        "kvIsMla": bool(p["is_mla"]),
        "kvNKvHeads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "kvElemsPerLayer": elems,
        "kvIndexerBytes": a.get("kv_indexer_bytes", 0),
        "kvGroups": a["kv_struct"]["kv_groups"],
        "nKvLayers": a["kv_struct"]["n_kv_layers"],
        "linLayers": a["kv_struct"]["linear_layers"],
        "linStateBytes": a["kv_struct"]["lin_state_per_req"],
        "absorbPerLayer": a.get("absorb_per_layer", 0),
        "draftEmbedBytes": a.get("draft_embed_bytes", 0),
        "visionBytes": comps.get("vision", {}).get("bytes", 0),
        "visionAttnFrac": (round(a["vision"]["attn_params"] / a["vision"]["params"], 4)
                           if a.get("vision") else 0),
        "visionActBytes": a.get("vision_act", 0),
        "actBytes": a["act_total"],
        # per-token transient coefficients split by TP-sharding behavior
        # (see core.activation_parts docstring); actTokens is the F the
        # estimate was built at — gpu_memory() recombines these per rank
        "actParts": core.activation_parts(cfg, a["p"]),
        "actTokens": a["batch_tokens"],
        "weightsBytes": a["total_bytes"],
        "weightDtype": ("fp4" if "fp4" in a["wname"] else
                        "fp8" if "fp8" in a["wname"] else "bf16"),
    }


# ---------------------------------------------------------------- parallel fit

def dp_available(D: dict, tp: int) -> bool:
    """DP attention only matters when KV would otherwise replicate across TP."""
    return D["kvIsMla"] or tp > (D["kvNKvHeads"] or tp)


def stage_range(s: int, pp: int, L: int) -> tuple[int, int]:
    """Layers [lo, hi) of pipeline stage s; later stages get the remainder."""
    base, rem = L // pp, L % pp
    lo = sum(base + (1 if i >= pp - rem else 0) for i in range(s))
    return lo, lo + base + (1 if s >= pp - rem else 0)


def gpu_memory(D: dict, P: dict, s: int) -> dict:
    """Bytes on one GPU of pipeline stage s (all tp ranks of a stage are
    identical). Direct port of template.js gpuMemory()."""
    L_total = D["L"]
    lo, hi = stage_range(s, P["pp"], L_total)
    n = hi - lo
    nd = max(0, min(hi, D["nDense"]) - min(lo, D["nDense"]))
    nm = n - nd
    Ld = D["layer"]
    tp, dp_attn = P["tp"], P["dpAttn"]
    attn_tp_div = 1 if dp_attn else tp

    w = {}
    w["embed"] = D["embed"] / attn_tp_div if s == 0 else 0
    w["vision"] = 0
    if s == 0 and D["visionBytes"]:
        v_attn = D["visionAttnFrac"] or 0
        w["vision"] = D["visionBytes"] * (v_attn / attn_tp_div + (1 - v_attn))
    w["lmHead"] = 0
    if s == P["pp"] - 1:
        w["lmHead"] = (D["embed"] / tp if P["pp"] > 1 else 0) if D["tied"] else D["lmHead"] / tp
    if dp_attn:
        w["attention"] = n * (Ld["attnQo"] + Ld["attnKvProj"] + Ld["attnRepl"])
    else:
        w["attention"] = n * (Ld["attnQo"] / tp
                              + Ld["attnKvProj"] / min(tp, D["kvNKvHeads"] or tp)
                              + Ld["attnRepl"])
    w["attention"] += n * (D["absorbPerLayer"] or 0) / attn_tp_div
    # leading dense FFN (first_k_dense_replace layers): normally /tp, but with
    # moe_dense_tp_size=1 (P["denseRepl"], auto-forced by CP) each rank keeps a
    # full copy. Only meaningful for a MoE model's dense prefix.
    w["denseFfn"] = nd * Ld["denseFfn"] / (1 if P.get("denseRepl") else tp)
    w["moeRouted"] = nm * Ld["moeRouted"] / tp
    # shared expert: sharded /tp with the plain `none` MoE backend, but the EP
    # backends (deepep/mooncake/nixl, which force ep_size=tp_size) build it
    # REPLICATED (tp1). We proxy "EP backend in use" by ep>1: ep>1 -> replicated
    # (/1), ep==1 -> /tp. Verified against SGLang deepseek_v2.py
    # (_shared_expert_use_tp1); modeling it as /tp under EP under-counts memory
    # (GLM tp8/ep8: ~2.5 GiB/card).
    w["moeShared"] = nm * Ld["moeShared"] / (1 if P["ep"] > 1 else tp)
    w["mtp"] = 0
    if s == P["pp"] - 1 and Ld["mtpTotal"]:
        if dp_attn:
            mtp_ffn_frac = Ld["mtpSlicedFrac"] - (Ld.get("mtpAttnFrac") or 0)
            w["mtp"] = Ld["mtpTotal"] * (mtp_ffn_frac / tp + (1 - mtp_ffn_frac))
        else:
            w["mtp"] = Ld["mtpTotal"] * (Ld["mtpSlicedFrac"] / tp + (1 - Ld["mtpSlicedFrac"]))
        w["mtp"] += ((D["draftEmbedBytes"] or 0) + (D["absorbPerLayer"] or 0)) / attn_tp_div
    w["others"] = n * (Ld["norms"] + Ld["indexer"]) + nm * Ld["moeGate"]

    weights = sum(w.values())

    if dp_attn:
        kv_shard = 1 / tp
    elif D["kvIsMla"]:
        kv_shard = 1
    else:
        kv_shard = 1 / min(tp, D["kvNKvHeads"])
    cell = kv_cell_bytes(D, P["kvDtype"])
    groups = D["kvGroups"] or [[D["nKvLayers"] or L_total, 0]]
    # per storage group (full-context vs sliding-capped) for the split fill in
    # the kv-utilization bar; kv is their sum
    kv_parts = [{"layers": g_layers, "window": g_win,
                 "b": cell * g_layers * (min(P["ctx"], g_win) if g_win else P["ctx"])
                      * (n / L_total) * P["req"] * kv_shard}
                for g_layers, g_win in groups]
    kv = sum(g["b"] for g in kv_parts)

    # Serving transient per rank. TP slices weight matrices, NOT the token
    # dimension: every rank runs all F tokens, so hidden-sized buffers, MoE
    # dispatch staging, and the DSA indexer workspace are full-size per rank
    # (unshard); only the FFN/expert intermediate follows its sliced weights
    # (shard / tp). The old `actBytes / tp` divided everything by tp — with
    # the missing MoE/DSA terms that compounded to a ~60x underestimate
    # (0.19 vs 13 GiB measured, S1M-S). The base floor is per-rank as-is.
    # F follows the chunked-prefill-size control (P["chunk"]) so the parallel
    # tab reacts to it; D["actTokens"] is the report-build default.
    ap = D["actParts"]
    F = P.get("chunk") or D["actTokens"]
    act = (ap["base"]
           + F * (ap["unshard_per_tok"] + ap["shard_per_tok"] / tp)
           + (D["visionActBytes"] or 0))
    lin_state = (D["linStateBytes"] or 0) * P["req"] * n / L_total / tp
    cap = P["memGib"] * GIB
    fixed = P["fixedGib"] * GIB
    static_budget = P["frac"] * cap
    kv_cap = static_budget - fixed - weights - lin_state
    can_start = kv_cap > 0
    used = static_budget + act if can_start else weights + fixed + lin_state + act
    cap_shard = tp if dp_attn else 1
    max_req = math.floor(kv_cap * P["req"] / kv / cap_shard) if (can_start and kv > 0) else 0
    max_tokens = kv_cap * P["ctx"] * P["req"] / kv / cap_shard if (can_start and kv > 0) else 0

    return {"w": w, "weights": weights, "kv": kv, "kvParts": kv_parts,
            "kvCap": max(kv_cap, 0.0),
            "kvCapRaw": kv_cap, "canStart": can_start, "act": act, "used": used,
            "free": cap - used, "maxReq": max_req, "maxTokens": max_tokens,
            "linState": lin_state, "layers": [lo, hi], "nDense": nd, "nMoe": nm}


def _gib(b: float) -> float:
    return round(b / GIB, 2)


def _parallel_compute(D: dict, P: dict) -> dict | None:
    """Per-stage memory + cluster aggregates for one deployment shape — the
    single computation behind parallel_verdict() (prose) and whatif_payload()
    (renderer shape). Returns None when pp exceeds the layer count."""
    if P["pp"] > D["L"]:
        return None
    stages = [gpu_memory(D, P, s) for s in range(P["pp"])]
    return {
        "stages": stages,
        "world": P["tp"] * P["pp"],
        "oom": sum(1 for m in stages for _ in range(P["tp"])
                   if not m["canStart"] or m["free"] < 0),
        # startup and serving fail differently (S1M/S1M-H): the KV pool is
        # sized to whatever the static budget leaves (canStart), but the
        # serving transient lives OUTSIDE the static region — a config can
        # start cleanly and still crash on the first full-chunk prefill.
        # The act estimate itself is best-effort: on the calibration model
        # (DSv4-Pro) it lands within ±2%, but the blind test on DSv4-Flash
        # (S1M-B) under-predicted by 17% — new architectures carry unnamed
        # allocations. Hence the risk trigger keeps a 25% act margin rather
        # than free < 0 exactly.
        "serving_risk": sum(1 for m in stages for _ in range(P["tp"])
                            if m["canStart"] and m["free"] < 0.25 * m["act"]),
        "all_start": all(m["canStart"] for m in stages),
        "kv_fits": all(m["canStart"] and m["kv"] <= m["kvCap"] for m in stages),
        "max_used": max(m["used"] for m in stages),
        "cluster_weights": sum(m["weights"] for m in stages) * P["tp"],
        "cluster_kv": sum(m["kv"] for m in stages) * P["tp"],
        "kv_single": kv_cell_bytes(D, P["kvDtype"])
                     * kv_layer_tokens(D, P["ctx"]) * P["req"],
        "min_max_req": min((m["maxReq"] for m in stages if m["canStart"]), default=0),
        "min_max_tok": min((m["maxTokens"] for m in stages if m["canStart"]), default=0),
    }


def parallel_verdict(D: dict, P: dict) -> dict:
    """Full fit verdict for one deployment shape — the parallel tab's summary
    numbers as structured data."""
    pc = _parallel_compute(D, P)
    if pc is None:
        return {"error": f"pp({P['pp']}) exceeds layer count {D['L']}"}
    stages = pc["stages"]
    world, oom, all_start, kv_fits = pc["world"], pc["oom"], pc["all_start"], pc["kv_fits"]
    max_used, cluster_weights, cluster_kv = pc["max_used"], pc["cluster_weights"], pc["cluster_kv"]
    kv_single = pc["kv_single"]
    min_max_req, min_max_tok = pc["min_max_req"], pc["min_max_tok"]
    dp_mult = P["tp"] if P["dpAttn"] else 1

    if P["dpAttn"]:
        kv_note = "dp-attention on: KV sharded by TP, no replication; pool numbers are per-rank"
    elif D["kvIsMla"]:
        kv_note = (f"MLA latent has no head dim: KV fully replicated per GPU, "
                   f"cluster holds {cluster_kv / kv_single:.1f}x one KV copy — dp-attention removes this")
    elif P["tp"] > D["kvNKvHeads"]:
        kv_note = (f"TP({P['tp']}) > kv heads({D['kvNKvHeads']}): KV fully sharded, "
                   f"extra {P['tp'] / min(P['tp'], D['kvNKvHeads']):.1f}x is replication — dp-attention removes this")
    else:
        kv_note = f"KV sharded by kv head 1/{P['tp']}, no replication"

    per_stage = []
    for s, m in enumerate(stages):
        per_stage.append({
            "pp_stage": s, "layers": f"L{m['layers'][0]}-L{m['layers'][1] - 1}",
            "weights_gib": _gib(m["weights"]), "kv_pool_gib": _gib(m["kvCap"]),
            "kv_demand_gib": _gib(m["kv"]),
            "kv_utilization_pct": round(m["kv"] / m["kvCap"] * 100, 1) if m["kvCap"] > 0 else None,
            "activation_gib": _gib(m["act"]),
            "linear_state_gib": _gib(m["linState"]) if m["linState"] else 0,
            "used_gib": _gib(m["used"]), "free_gib": _gib(m["free"]),
            "can_start": m["canStart"],
            "short_by_gib": _gib(-m["kvCapRaw"]) if not m["canStart"] else None,
        })

    return {
        "tp": P["tp"], "pp": P["pp"], "ep": P.get("ep") or P["tp"],
        "dp_attention": P["dpAttn"],
        "world_gpus": world,
        "gpu_mem_gib": P["memGib"],
        "mem_fraction_static": P["frac"],
        "fixed_overhead_gib": P["fixedGib"],
        "fits": all_start and oom == 0 and kv_fits,
        # startability only (SKILL.md contract: false = engine can't even
        # start). Serving-transient overflow is NOT startup failure — it
        # reports through serving_oom_risk, and through fits via oom_gpus.
        "weights_fit": all_start,
        "kv_fits_demand": kv_fits,
        "oom_gpus": oom,
        # engine starts (KV pool > 0) but the serving transient exceeds what
        # the static region left free: crashes on the first full-chunk
        # prefill, not at startup. Validated S1M-H: frac >= 0.94 on 8xB200
        # launched READY at every value and died serving. When true, quote
        # serving_note, don't just say "fits=false".
        "serving_oom_risk": pc["serving_risk"] > 0,
        "serving_note": (
            f"starts but serving transient (~{_gib(stages[0]['act'])} GiB "
            f"peak per forward at {P.get('chunk') or D['actTokens']:,} tokens) "
            f"exceeds the non-static headroom — lower mem_fraction_static or "
            f"chunked_prefill_size" if pc["serving_risk"] > 0 else None),
        "per_gpu": per_stage,  # one entry per pp stage; all tp ranks of a stage are identical
        "max_used_gib": _gib(max_used),
        "cluster_weights_gib": _gib(cluster_weights),
        "weight_replication_overhead_gib": _gib(max(cluster_weights - D["weightsBytes"], 0)),
        "kv_note": kv_note,
        # bottleneck-stage pool capacity; per-rank under dp (matches SGLang's
        # logged max_total_num_tokens), with the cluster equivalent alongside
        "max_concurrency": min_max_req,
        "max_concurrency_cluster": min_max_req * dp_mult,
        "kv_pool_tokens": int(min_max_tok),
        "kv_pool_tokens_cluster": int(min_max_tok * dp_mult),
    }


def tp_sweep(D: dict, P: dict, gpn: int) -> list[dict]:
    """Compact comparison across TP degrees at pp=1 on the chosen GPU — the
    'how should I shard this' answer. MLA/over-sharded shapes get a dp-attention
    row too."""
    tps = sorted({t for t in (1, 2, 4, 8, 16, 32) if t <= max(gpn, P["tp"])} | {P["tp"]})
    rows = []
    for tp in tps:
        for dp in ((False, True) if dp_available(D, tp) and tp > 1 else (False,)):
            Q = dict(P, tp=tp, pp=1, dpAttn=dp)
            m = gpu_memory(D, Q, 0)
            dp_mult = tp if dp else 1
            rows.append({
                "tp": tp, "dp_attention": dp,
                "per_gpu_weights_gib": _gib(m["weights"]),
                "kv_pool_gib": _gib(m["kvCap"]),
                "kv_demand_gib": _gib(m["kv"]),
                "used_gib": _gib(m["used"]), "free_gib": _gib(m["free"]),
                "fits": m["canStart"] and m["free"] >= 0 and m["kv"] <= m["kvCap"],
                "can_start": m["canStart"],
                "max_concurrency": m["maxReq"] * dp_mult,
            })
    return rows


# ---------------------------------------------------------------- roofline

def _phase(flops: float, bytes_: float, note: str) -> dict:
    return {"flops": flops, "bytes": bytes_,
            "intensity": flops / bytes_ if bytes_ else 0.0, "note": note}


def _gemm_work(kernel: dict, wl: dict) -> dict:
    B, T = wl["B"], wl["T"]
    return {
        "dec": _phase(2 * kernel["params"] * B, kernel["bytes"],
                      f"weights read once per step, shared by {B} tokens"),
        "pre": _phase(2 * kernel["params"] * T, kernel["bytes"],
                      f"{T} tokens share one weight read"),
    }


def _moe_experts_work(kernel: dict, wl: dict) -> dict:
    B, T = wl["B"], wl["T"]
    active = wl["topk"] / wl["nExperts"]
    touched = min(1.0, B * active)
    return {
        "dec": _phase(2 * kernel["params"] * active * B, kernel["bytes"] * touched,
                      f"decode batch={B} touches ~{touched:.0%} of experts"),
        "pre": _phase(2 * kernel["params"] * active * T, kernel["bytes"],
                      f"prefill {T} tokens touches nearly all experts"),
    }


def _attention_geometry(spec: dict) -> dict:
    kind = spec["kind"]
    if kind == "mla":
        dc, dr = spec["kvLoraRank"], spec["ropeHeadDim"]
        return {"label": f"MLA absorbed (latent {dc} + rope {dr})",
                "flopsPerPair": 2 * spec["qHeads"] * (2 * dc + dr),
                "kvElementsPerKey": dc + dr}
    if kind in ("mha", "gqa"):
        return {"label": f"{kind.upper()} ({spec['qHeads']} Q / {spec['kvHeads']} KV heads)",
                "flopsPerPair": 2 * spec["qHeads"] * (spec["qkHeadDim"] + spec["valueHeadDim"]),
                "kvElementsPerKey": spec["kvHeads"] * (spec["qkHeadDim"] + spec["valueHeadDim"])}
    raise ValueError(f"unsupported attention geometry: {kind}")


def _dense_causal_pairs(t: float) -> float:
    return t * (t + 1) / 2


def _capped_causal_pairs(t: float, limit: float) -> float:
    dense = min(t, limit)
    return _dense_causal_pairs(dense) + max(0, t - limit) * limit


def _attention_pattern(wl: dict) -> dict:
    pat = wl["pattern"]
    B, S, T, Lkv = wl["B"], wl["S"], wl["T"], wl["kvLayers"]
    if pat["kind"] == "dense":
        dec_pairs = B * S * Lkv
        return {"label": "dense", "decodePairs": dec_pairs,
                "prefillPairs": _dense_causal_pairs(T) * Lkv,
                "decodeKvReads": dec_pairs, "prefillKvReads": T * Lkv,
                "decodeNote": "each request reads the full context",
                "prefillNote": "dense causal within chunk; HBM = ideal one-pass bound"}
    if pat["kind"] == "dsa":
        attended = min(S, pat["topk"])
        return {"label": f"DSA top-{pat['topk']}",
                "decodePairs": B * attended * Lkv,
                "prefillPairs": _capped_causal_pairs(T, pat["topk"]) * Lkv,
                "decodeKvReads": B * attended * Lkv, "prefillKvReads": T * Lkv,
                "decodeNote": f"each request reads top-{attended} selected KV; "
                              "indexer full-context scoring not modeled",
                "prefillNote": f"top-{pat['topk']} causal within chunk; ideal bound, "
                               "excludes gather reloads"}
    if pat["kind"] == "capped":
        cap_l = min(pat["capLayers"], Lkv)
        dense_l = Lkv - cap_l
        attended = min(S, pat["cap"])
        dec_reads = B * (attended * cap_l + S * dense_l)
        return {"label": f"capped-{pat['cap']}",
                "decodePairs": dec_reads,
                "prefillPairs": (_capped_causal_pairs(T, pat["cap"]) * cap_l
                                 + _dense_causal_pairs(T) * dense_l),
                "decodeKvReads": dec_reads, "prefillKvReads": T * Lkv,
                "decodeNote": f"{cap_l} layers capped at {attended} tokens, "
                              f"{dense_l} dense",
                "prefillNote": "capped causal within chunk; ideal bound"}
    raise ValueError(f"unsupported attention pattern: {pat['kind']}")


def _attention_core_work(wl: dict) -> dict:
    geo = _attention_geometry(wl["geometry"])
    pat = _attention_pattern(wl)
    prefix = geo["label"] + " + " + pat["label"] + ": "
    return {
        "dec": _phase(geo["flopsPerPair"] * pat["decodePairs"],
                      geo["kvElementsPerKey"] * pat["decodeKvReads"] * wl["kvBytes"],
                      prefix + pat["decodeNote"]),
        "pre": _phase(geo["flopsPerPair"] * pat["prefillPairs"],
                      geo["kvElementsPerKey"] * pat["prefillKvReads"] * wl["kvBytes"],
                      prefix + pat["prefillNote"]),
    }


def _linear_state_work(wl: dict) -> dict:
    B, T = wl["B"], wl["T"]
    elems = wl["linStateBytes"] / 2
    return {
        "dec": _phase(2 * elems * B, 2 * wl["linStateBytes"] * B,
                      f"{wl['linLayers']} linear layers: fixed state read+written per step"),
        "pre": _phase(2 * elems * T, 2 * wl["linStateBytes"],
                      f"{wl['linLayers']} linear layers: fixed state read+written per chunk"),
    }


def _kernel_bytes_replication(key: str, wl: dict) -> float:
    """Per-rank replication multiplier for a kernel's HBM weight bytes; keeps
    the shared ÷TP honest for components each rank reads in full."""
    if key in ("indexer", "moe_gate"):
        return wl["tp"]
    if key == "attention":
        return wl["tp"] if wl["dpAttn"] else 1
    return 1


def _attn_core_kv_replication(wl: dict) -> float:
    """Opposite polarity of the weights term: pure-TP MLA streams the full KV
    per rank; GQA shards reads by min(tp, kvHeads); dp-attention shards evenly."""
    if wl["dpAttn"] or wl["tp"] <= 1:
        return 1
    if wl["kvIsMla"]:
        return wl["tp"]
    return wl["tp"] / min(wl["tp"], wl["kvHeads"] or wl["tp"])


def _scale_bytes(work: dict, mult: float) -> dict:
    if mult == 1:
        return work
    return {ph: _phase(w["flops"], w["bytes"] * mult,
                       w["note"] + f"; replicated per rank, HBM bytes x{mult:g}")
            for ph, w in work.items()}


_CALCULATORS = {
    "attention": _gemm_work, "indexer": _gemm_work, "dense_ffn": _gemm_work,
    "moe_gate": _gemm_work, "moe_routed": _moe_experts_work,
    "moe_shared": _gemm_work, "lm_head": _gemm_work,
}


def _build_roofline_rows(a: dict, cfg: dict, D: dict, P: dict,
                         chunk_tokens: int,
                         weight_dtype: str | None = None) -> tuple[list, dict, bool]:
    """One row per kernel with {dec, pre} phase work plus renderer extras —
    the shared core of roofline_verdict() and whatif_payload()."""
    wl = {
        "B": P["req"], "T": chunk_tokens, "S": P["ctx"],
        "kvBytes": kv_bytes_per(P["kvDtype"]),
        "kvLayers": D["nKvLayers"] or D["L"],
        "linLayers": D["linLayers"] or 0,
        "linStateBytes": D["linStateBytes"] or 0,
        "tp": P["tp"], "dpAttn": P["dpAttn"],
        "kvIsMla": D["kvIsMla"], "kvHeads": D["kvNKvHeads"] or 0,
        "geometry": None, "pattern": None,
        "topk": D["topk"], "nExperts": D["nExperts"],
    }
    core_spec = core.attention_core_spec(cfg)
    wl["geometry"], wl["pattern"] = core_spec["geometry"], core_spec["pattern"]

    # weight-dtype what-if: an explicit override replaces exact stored bytes
    # with ideal params × bytes/param (real quant checkpoints keep some bf16)
    ckpt_dtype = D["weightDtype"]
    w_override = (kv_bytes_per(weight_dtype)
                  if (weight_dtype and weight_dtype != ckpt_dtype) else 0)

    rows = []
    for kernel in core.roofline_kernels(a, cfg):
        k = dict(kernel)
        if w_override:
            k["bytes"] = k["params"] * w_override
        repl = _kernel_bytes_replication(k["key"], wl)
        work = _scale_bytes(_CALCULATORS[k["key"]](k, wl), repl)
        extras = {"replMult": repl if repl != 1 else None}
        if k["key"] == "moe_routed":
            extras["touchedPct"] = round(min(1.0, wl["B"] * wl["topk"] / wl["nExperts"]) * 100)
        rows.append({"key": k["key"], "label": k["label"],
                     "color": k.get("color"), "kind": k.get("kind"),
                     "extras": extras, **work})
    kv_repl = _attn_core_kv_replication(wl)
    pat = wl["pattern"]
    rows.append({"key": "attn_core", "label": "Attn core (KV/score)",
                 "color": "var(--s5)", "kind": "attn",
                 "extras": {"replMult": kv_repl if kv_repl != 1 else None,
                            "attended": (min(wl["S"], pat["topk"])
                                         if pat["kind"] == "dsa" else None)},
                 **_scale_bytes(_attention_core_work(wl), kv_repl)})
    if wl["linStateBytes"] > 0:
        rows.append({"key": "linear_state", "label": "Linear/SSM state",
                     "color": "var(--s4)", "kind": "attn", "extras": {},
                     **_linear_state_work(wl)})
    return rows, wl, bool(w_override)


def _resolve_perf(gpu_name: str | None, D: dict,
                  weight_dtype: str | None) -> dict | None:
    """GPU peak/bandwidth at the effective weight dtype, with bf16 fallback."""
    perf_spec = core.GPU_PERF.get(gpu_name) if gpu_name else None
    if not perf_spec:
        return None
    dt = weight_dtype or D["weightDtype"]
    peak, used_dt = perf_spec.get(dt), dt
    if not peak:
        peak, used_dt = perf_spec["bf16"], "bf16"
    return {"gpu": gpu_name, "peak": peak, "bw": perf_spec["bw"],
            "dtype": used_dt, "fallback": dt if used_dt != dt else None}


def _comp_time_s(c: dict, perf: dict) -> float:
    """Roofline execution model: kernel time = max(bytes/BW, FLOPs/peak).
    THE single definition — every consumer times kernels through this."""
    return max(c["bytes"] / (perf["bw"] * 1e12), c["flops"] / (perf["peak"] * 1e12))


def _phase_aggregate(comps: list, perf: dict, tp: int) -> dict:
    """Aggregate one phase's kernels: totals, bound verdict, and the ÷TP phase
    time. Shared by roofline_verdict() and whatif_payload() so the execution
    model cannot drift between the JSON API and the report page."""
    knee = perf["peak"] / perf["bw"]
    agg_f = sum(c["flops"] for c in comps)
    agg_b = sum(c["bytes"] for c in comps)
    intensity = agg_f / agg_b if agg_b else 0
    sum_s = sum(_comp_time_s(c, perf) for c in comps)
    return {
        "agg_flops": agg_f, "agg_bytes": agg_b, "intensity": intensity,
        "is_mem": intensity < knee,
        "knee_ratio": (knee / intensity if intensity < knee
                       else intensity / knee) if intensity else 0,
        "sum_s": sum_s, "phase_s": sum_s / tp,
    }


def roofline_verdict(a: dict, cfg: dict, D: dict, P: dict,
                     gpu_name: str, chunk_tokens: int,
                     weight_dtype: str | None = None) -> dict | None:
    """Decode/prefill roofline verdicts on one GPU type — the roofline tab's
    verdict cards as structured data. Returns None when the GPU has no
    compute/bandwidth spec."""
    perf = _resolve_perf(gpu_name, D, weight_dtype)
    if not perf:
        return None
    peak, bw, used_dt = perf["peak"], perf["bw"], perf["dtype"]
    knee = peak / bw
    dt = weight_dtype or D["weightDtype"]

    rows, wl, w_override = _build_roofline_rows(a, cfg, D, P, chunk_tokens, weight_dtype)

    out = {"gpu": gpu_name, "peak_tflops": peak, "peak_dtype": used_dt,
           "hbm_tbs": bw, "knee_flops_per_byte": round(knee, 1),
           "tp": P["tp"], "dp_attention": P["dpAttn"],
           "weight_dtype": dt if not w_override else f"{dt} (idealized what-if)",
           "note": "theoretical upper bounds assuming perfect overlap and no comm cost; "
                   "real systems typically reach 50-70%"}

    for ph, label in (("dec", "decode"), ("pre", "prefill")):
        comps = [{"label": r["label"], "key": r["key"], **r[ph]} for r in rows
                 if r[ph]["flops"] > 0 or r[ph]["bytes"] > 0]
        agg = _phase_aggregate(comps, perf, P["tp"])
        is_mem, ph_time, total_t = agg["is_mem"], agg["phase_s"], agg["sum_s"]
        kernels = [{
            "kernel": c["label"], "tflops": round(c["flops"] / 1e12, 3),
            "hbm_gib": _gib(c["bytes"]),
            "intensity": round(c["intensity"], 1),
            "bound": "memory" if c["intensity"] < knee else "compute",
            "time_share_pct": round(_comp_time_s(c, perf) / total_t * 100, 1) if total_t else 0,
            "note": c["note"],
        } for c in comps]

        entry = {
            "bound": "memory" if is_mem else "compute",
            "aggregate_intensity": round(agg["intensity"], 1),
            "knee_ratio": round(agg["knee_ratio"], 2),
            "kernels": kernels,
        }
        if ph == "dec":
            step_ms = ph_time * 1000
            entry.update({
                "batch_tokens_per_step": wl["B"],
                "step_ms": round(step_ms, 2),
                "tokens_per_s_per_request": round(1000 / step_ms, 1) if step_ms else None,
                "tokens_per_s_tp_group": round(1000 / step_ms * wl["B"], 0) if step_ms else None,
                "guidance": ("memory-bound: concurrency is nearly free until intensity "
                             f"reaches the knee (~{entry['knee_ratio']}x away); quantize the "
                             "dominant kernel to speed up" if is_mem else
                             "compute-bound: more concurrency lengthens the step linearly "
                             "with no throughput gain"),
            })
        else:
            entry.update({
                "chunk_tokens": wl["T"],
                "chunk_ms": round(ph_time * 1000, 1),
                "ttft_s_full_context": round(ph_time * P["ctx"] / wl["T"], 2),
                "guidance": ("prefill compute-bound (typical): weight quantization alone "
                             "does not cut TTFT unless low-precision compute is used"
                             if not is_mem else "prefill memory-bound (uncommon: small "
                             "chunk / very large model)"),
            })
        out[label] = entry
    return out


# ---------------------------------------------------------------- whatif payload
#
# The render-shaped bundle the report page fetches on every control change.
# Field names deliberately match what template.js's render functions consume;
# human-language strings stay in the page's JS I18N table — kernels carry
# noteRefs = [[i18nKey, arg...], ...] instead of prose.

def _expert_info(D: dict, P: dict, t: int) -> dict | None:
    """Experts held by tp rank t (EP grouping) — mirrors template.js expertInfo."""
    if not D["nMoe"] or not D["nExperts"]:
        return None
    grp = P["tp"] // P["ep"] if P["ep"] else P["tp"]
    grp = max(grp, 1)
    ep_rank = t // grp
    base, rem = D["nExperts"] // P["ep"], D["nExperts"] % P["ep"]
    cnt = base + (1 if ep_rank < rem else 0)
    lo = ep_rank * base + min(ep_rank, rem)
    return {"cnt": cnt, "lo": lo, "hi": lo + cnt - 1, "sliceDenom": grp}


def _note_refs(row: dict, ph: str, wl: dict) -> list:
    """i18n references for the detail-table note column, rendered in-page."""
    key, x = row["key"], row["extras"]
    B, T = wl["B"], wl["T"]
    refs = []
    if key == "moe_routed":
        refs.append(["moeDecodeNote", B, x.get("touchedPct", 0)] if ph == "dec"
                    else ["moePrefillNote", f"{T:,}"])
    elif key == "attn_core":
        geo, pat = wl["geometry"], wl["pattern"]
        if geo["kind"] == "mla":
            refs.append(["mlaLabel", geo["kvLoraRank"], geo["ropeHeadDim"]])
        elif geo["kind"] == "gqa":
            refs.append(["gqaLabel", geo["qHeads"], geo["kvHeads"]])
        else:
            refs.append(["mhaLabel", geo["qHeads"]])
        refs.append(["_raw", " + "])
        ctx_lbl = (f"{wl['S'] // 1048576}M" if wl["S"] % 1048576 == 0
                   else f"{wl['S'] // 1024}K")
        if pat["kind"] == "dsa":
            refs.append(["dsaDecodeNote", ctx_lbl, x.get("attended")] if ph == "dec"
                        else ["dsaPrefillNote", pat["topk"]])
        else:
            refs.append(["denseDecodeNote", ctx_lbl] if ph == "dec"
                        else ["densePrefillNote"])
    elif key == "linear_state":
        refs.append(["linStateDecodeNote" if ph == "dec" else "linStatePrefillNote",
                     wl["linLayers"]])
    else:  # weight GEMMs
        refs.append(["gemmDecodeNote", B] if ph == "dec"
                    else ["gemmPrefillNote", f"{T:,}"])
    if x.get("replMult"):
        refs.append(["replNote", f"{x['replMult']:g}"])
    return refs


def whatif_payload(a: dict, cfg: dict, D: dict, P: dict,
                   gpu_name: str | None, chunk_tokens: int,
                   weight_dtype: str | None = None) -> dict:
    """Everything the report page needs to re-render all three dynamic tabs
    for one parameter combination."""
    # ---- estimate tab
    cell = kv_cell_bytes(D, P["kvDtype"])
    kv_per_tok = cell * (D["nKvLayers"] or D["L"])
    kv_per_req = cell * kv_layer_tokens(D, P["ctx"])
    kv_total = kv_per_req * P["req"]
    lin_total = (D["linStateBytes"] or 0) * P["req"]
    runtime = kv_total + lin_total + D["actBytes"] + (D["visionActBytes"] or 0)
    total = D["weightsBytes"] + runtime
    estimate = {
        "kvPerTok": kv_per_tok, "kvPerReq": kv_per_req, "kvTotal": kv_total,
        "linTotal": lin_total, "runtime": runtime, "total": total,
        "grand": total * (1 + a["overhead"]),
        "mhaTotal": kv_total * a["mha_ratio"] if a.get("mha_ratio") else None,
    }

    # ---- parallel tab (same _parallel_compute core as parallel_verdict)
    pc = _parallel_compute(D, P)
    if pc is None:
        parallel = {"error": "ppExceeds", "pp": P["pp"], "L": D["L"]}
    else:
        parallel = {
            "stages": pc["stages"],
            "experts": [_expert_info(D, P, t) for t in range(P["tp"])],
            "world": pc["world"],
            "clusterWeights": pc["cluster_weights"], "clusterKv": pc["cluster_kv"],
            "kvSingle": pc["kv_single"], "oomTotal": pc["oom"],
            "allStart": pc["all_start"], "maxUsed": pc["max_used"],
            "minMaxReq": pc["min_max_req"], "minMaxTok": pc["min_max_tok"],
            # CP's distinctive payoff: one sequence's KV is split across the CP
            # ranks. We model attn_cp_size=tp (the common case), so a single
            # sequence spans all tp ranks: longest single sequence = tp x (one
            # rank's pool tokens). minMaxTok is already the per-rank pool under
            # dp/CP (CP is memory-equivalent to dp-attention), so scale it by tp.
            # Only surfaced when CP is on (else a sequence lives on one rank).
            "maxSingleSeq": (P["tp"] if P.get("cp") else 1) * pc["min_max_tok"],
        }

    # ---- roofline tab (same _comp_time_s/_phase_aggregate core as roofline_verdict)
    perf = _resolve_perf(gpu_name, D, weight_dtype)
    roofline = None
    if perf:
        knee = perf["peak"] / perf["bw"]
        rows, wl, w_override = _build_roofline_rows(a, cfg, D, P, chunk_tokens,
                                                    weight_dtype)

        out_rows, phases = [], {}
        for r in rows:
            entry = {"key": r["key"], "label": r["label"], "color": r["color"]}
            for ph in ("dec", "pre"):
                c = r[ph]
                entry[ph] = {"flops": c["flops"], "bytes": c["bytes"],
                             "intensity": c["intensity"],
                             "timeMs": _comp_time_s(c, perf) * 1000,
                             "noteRefs": _note_refs(r, ph, wl)}
            out_rows.append(entry)
        for ph in ("dec", "pre"):
            comps = [r[ph] for r in out_rows
                     if r[ph]["flops"] > 0 or r[ph]["bytes"] > 0]
            agg = _phase_aggregate(comps, perf, P["tp"])
            ph_ms = agg["phase_s"] * 1000
            e = {"aggFlops": agg["agg_flops"], "aggBytes": agg["agg_bytes"],
                 "aggIntensity": agg["intensity"], "isMem": agg["is_mem"],
                 "sumMs": agg["sum_s"] * 1000, "phaseMs": ph_ms,
                 "kneeRatio": agg["knee_ratio"]}
            if ph == "dec":
                e["tpsPerReq"] = 1000 / ph_ms if ph_ms else 0
                e["tpsGroup"] = (1000 / ph_ms) * wl["B"] if ph_ms else 0
            else:
                e["ttftS"] = ph_ms / 1000 * P["ctx"] / wl["T"]
            phases[ph] = e
        roofline = {"perf": perf, "knee": knee, "rows": out_rows,
                    "phases": phases, "B": wl["B"], "T": wl["T"],
                    "wdtypeOverride": (weight_dtype if w_override else None)}

    return {
        "echo": {"ctx": P["ctx"], "req": P["req"], "kvDtype": P["kvDtype"],
                 "tp": P["tp"], "pp": P["pp"], "ep": P.get("ep") or P["tp"],
                 "dpAttn": P["dpAttn"], "dpAvailable": dp_available(D, P["tp"]),
                 "cp": bool(P.get("cp")),
                 # CP splits the latent-KV sequence — only meaningful for MLA/DSA
                 "cpApplies": bool(D["kvIsMla"]),
                 "denseRepl": bool(P.get("denseRepl")),
                 "denseReplApplies": D["nDense"] > 0 and D["nMoe"] > 0,
                 "frac": P["frac"], "memGib": P["memGib"], "gpn": P["gpn"],
                 "fixedGib": P["fixedGib"], "chunk": chunk_tokens,
                 "weightDtype": weight_dtype or D["weightDtype"]},
        "estimate": estimate,
        "parallel": parallel,
        "roofline": roofline,
        "engineVersion": engine_version(),
    }
