#!/usr/bin/env python3
"""
Estimate GPU memory footprint of an LLM from its HuggingFace config.json,
and optionally render a breakdown diagram as a self-contained HTML page
(markup/style and JavaScript live in template.html/template.js).

Memory is split into two groups:
  * static  — model weights: paid once at load time, independent of traffic
  * runtime — KV cache (grows with context_length x running_requests) and
              activation workspace (grows with tokens in flight per forward pass)

Usage:
    python3 vram_estimate.py zai-org/GLM-5.2-FP8
    python3 vram_estimate.py zai-org/GLM-5.2-FP8 --context 120000 --requests 8 --html glm.html
    python3 vram_estimate.py Qwen/Qwen3-32B --context 32768 --requests 16 --kv-dtype fp8

Only stdlib is used. For gated repos set HF_TOKEN env var.
"""

import argparse
import json
import math
import os
import re
import string
import struct
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from i18n import t, set_lang, get_lang

GIB = 1024 ** 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "template.html")
SCRIPT_PATH = os.path.join(BASE_DIR, "template.js")

# categorical slots in fixed order (dataviz reference palette; values live in template.html)
COMP_SLOT = {"embed": 0, "lm_head": 0, "attention": 1, "dense_ffn": 2,
             "moe_routed": 3, "moe_shared": 3, "moe_gate": 3,
             "mtp": 4, "indexer": 7, "norms": 7, "vision": 8,
             "kv": 5, "act": 6}

# GPU instance types offered in the parallel-tab dropdown (order = display order).
# AWS EC2 names resolve via `aws ec2 describe-instance-types`; the h800-*/h20-*
# pseudo-instances are non-AWS bare-metal nodes (China-market GPUs) that only
# ever come from the static table below.
INSTANCE_TYPES = [
    "p6-b300.48xlarge", "p6-b200.48xlarge", "p5en.48xlarge", "p5.48xlarge",
    "p4de.24xlarge", "p4d.24xlarge", "g6e.48xlarge", "g6e.12xlarge", "g5.48xlarge",
    "h800-8gpu", "h20-8gpu",
]

# static fallback, from `aws ec2 describe-instance-types` on 2026-07-09; the
# h800-*/h20-* entries are hand-authored bare-metal nodes (SXM, 8×GPU/node).
STATIC_INSTANCES = {  # name: (gpu, count per node, MiB per GPU)
    "p6-b300.48xlarge": ("B300", 8, 275040),
    "p6-b200.48xlarge": ("B200", 8, 183359),
    "p5en.48xlarge":    ("H200", 8, 144384),
    "p5.48xlarge":      ("H100", 8, 81920),
    "p4de.24xlarge":    ("A100", 8, 81920),
    "p4d.24xlarge":     ("A100", 8, 40960),
    "g6e.48xlarge":     ("L40S", 8, 45776),
    "g6e.12xlarge":     ("L40S", 4, 45776),
    "g5.48xlarge":      ("A10G", 8, 22888),
    "h800-8gpu":        ("H800", 8, 81920),   # 80 GiB SXM
    "h20-8gpu":         ("H20",  8, 98304),   # 96 GiB SXM
}


# GPU peak performance for the roofline tab. Dense Tensor Core TFLOPs (no
# sparsity) and HBM bandwidth in TB/s — approximate NVIDIA datasheet values,
# keyed by the `gpu` field of the instance table. None = precision unsupported.
GPU_PERF = {
    "H100": {"bf16": 989,  "fp8": 1979, "fp4": None,  "bw": 3.35},
    "H200": {"bf16": 989,  "fp8": 1979, "fp4": None,  "bw": 4.8},
    "H800": {"bf16": 989,  "fp8": 1979, "fp4": None,  "bw": 3.35},  # H100 die, NVLink capped
    "H20":  {"bf16": 148,  "fp8": 296,  "fp4": None,  "bw": 4.0},   # compute-cut, HBM3 96GB
    "B200": {"bf16": 2250, "fp8": 4500, "fp4": 9000,  "bw": 8.0},
    "B300": {"bf16": 2250, "fp8": 4500, "fp4": 13500, "bw": 8.0},
    "A100": {"bf16": 312,  "fp8": None, "fp4": None,  "bw": 2.0},  # 80G SXM; 40G is 1.6
    "L40S": {"bf16": 362,  "fp8": 733,  "fp4": None,  "bw": 0.864},
    "A10G": {"bf16": 125,  "fp8": None, "fp4": None,  "bw": 0.6},
}


def fetch_instance_specs(names: list) -> dict:
    """{name: {gpu, count, memGib}} via aws cli, falling back to the static table."""
    specs = {}
    try:
        out = subprocess.run(
            ["aws", "ec2", "describe-instance-types",
             "--filters", f"Name=instance-type,Values={','.join(names)}",
             "--query", "InstanceTypes[].{t:InstanceType,g:GpuInfo.Gpus[0].Name,"
                        "c:GpuInfo.Gpus[0].Count,m:GpuInfo.Gpus[0].MemoryInfo.SizeInMiB}",
             "--output", "json"],
            capture_output=True, text=True, timeout=30, check=True)
        for it in json.loads(out.stdout):
            specs[it["t"]] = {"gpu": it["g"], "count": it["c"],
                              "memGib": round(it["m"] / 1024, 1)}
    except Exception as e:
        print(f"note: aws describe-instance-types unavailable ({e}); "
              f"using static spec table (2026-07 snapshot)", file=sys.stderr)
    for name in names:
        if name not in specs and name in STATIC_INSTANCES:
            g, c, m = STATIC_INSTANCES[name]
            specs[name] = {"gpu": g, "count": c, "memGib": round(m / 1024, 1)}
    return {n: specs[n] for n in names if n in specs}


# ---------------------------------------------------------------- fetch

def fetch_config(model_id: str) -> dict:
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "vram-estimate/1.0"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


# ---------------------------------------------------------------- exact sizes from safetensors

def _fetch_range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "vram-estimate/1.0",
        "Range": f"bytes={start}-{start + length - 1}",
    })
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_safetensors_catalog(model_id: str) -> tuple[dict, int | None]:
    """Per-tensor {name: {dtype, shape, bytes}} from safetensors headers.

    Only range-requests each shard's JSON header (a few hundred KB), never the
    weights. This is the ground truth for mixed-precision checkpoints where a
    single bytes-per-param number is wrong.
    """
    base = f"https://huggingface.co/{model_id}/resolve/main"
    declared = None
    try:
        # resolve/ follows git-lfs; raw/ would return the LFS pointer for big indexes
        req = urllib.request.Request(
            f"{base}/model.safetensors.index.json",
            headers={"User-Agent": "vram-estimate/1.0"})
        token = os.environ.get("HF_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            idx = json.load(resp)
        files = sorted(set(idx["weight_map"].values()))
        declared = (idx.get("metadata") or {}).get("total_size")
    except urllib.error.HTTPError:
        files = ["model.safetensors"]  # single-file checkpoint

    def header(fname):
        url = f"{base}/{fname}"
        n = struct.unpack("<Q", _fetch_range(url, 0, 8))[0]
        h = json.loads(_fetch_range(url, 8, n))
        h.pop("__metadata__", None)
        return h

    catalog = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for h in ex.map(header, files):
            for name, meta in h.items():
                b = meta["data_offsets"][1] - meta["data_offsets"][0]
                catalog[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "bytes": b}
    return catalog, declared


def classify_tensor(name: str, num_layers: int) -> str:
    """Map a tensor name to a component key. Handles HF-standard naming
    (model.layers.N.self_attn...) and DeepSeek-style (layers.N.attn.wkv...)."""
    n = name.lower()
    # vision tower + multimodal projector, checked first: ViT tensors contain
    # substrings ("embed" in patch_embed, "norm" in layernorms) that would
    # otherwise leak into text components.
    if re.search(r"(?:^|\.)(?:vision_tower|vision_model|vision_encoder|visual"
                 r"|mm_projector|multi_modal_projector)\.", n):
        return "vision"
    m = re.search(r"(?:^|\.)(?:mtp|nextn)\.", n)
    if m:
        return "mtp"
    lm = re.search(r"layers\.(\d+)\.", n)
    if lm and int(lm.group(1)) >= num_layers:   # extra layers beyond L are MTP
        return "mtp"
    if re.search(r"(?:^|\.)hc_", n):            # hyper-connection / highway params
        return "norms"
    if "layernorm" in n or re.search(r"(?:^|[._])norm\b", n) or n.endswith("norm.weight"):
        return "norms"
    if "embed" in n:
        return "embed"
    if "lm_head" in n or re.match(r"(?:model\.)?head\.", n):
        return "lm_head"
    if "indexer" in n:
        return "indexer"
    if "shared_expert" in n:
        return "moe_shared"
    if re.search(r"\.experts\.", n):
        return "moe_routed"
    if re.search(r"\.(?:mlp|ffn)\.gate\.", n):  # router (gate_proj is dense FFN)
        return "moe_gate"
    if re.search(r"\.(?:mlp|ffn)\.", n):
        return "dense_ffn"
    if "attn" in n or "attention" in n:
        return "attention"
    return "norms"


# quantization metadata tensors: storage overhead, not model parameters
QUANT_META = re.compile(r"(?:^|\.|_)(scales?(?:_inv)?|qzeros|zeros|g_idx|zero_point)(?:$|\.)")
INT_DTYPES = {"I8": 8, "U8": 8, "I16": 16, "U16": 16, "I32": 32, "U32": 32}


def _sub_byte_name(cfg: dict, bits: int) -> str:
    """fp4 vs int4 for packed weights, from config conventions."""
    qc = cfg.get("quantization_config") or {}
    text = (json.dumps(qc) + " " + str(cfg.get("expert_dtype", ""))).lower()
    if bits == 4:
        if "fp4" in text or "mxfp4" in text or "nvfp4" in text:
            return "fp4"
        if qc.get("quant_method") in ("gptq", "awq") or "int" in text:
            return "int4"
        return "4bit"
    return f"{bits}bit"


def exact_components(catalog: dict, cfg: dict) -> tuple[dict, dict, dict, dict]:
    """Aggregate the tensor catalog into per-component bytes/params/dtypes.

    Sub-byte packing (e.g. two fp4 values per int8 byte, or eight int4 per
    int32 word) is detected WITHOUT any model-specific config hint: for each
    component, the shape-derived param count is reconciled against the
    config-formula count. An integer-dtype component whose apparent count is
    1/2, 1/4 or 1/8 of the formula is unpacked by that factor.
    """
    L = cfg["num_hidden_layers"]
    p = count_params(cfg)
    # components whose closed-form param count is reliable enough to reconcile
    formula = {"embed": p["embed"], "lm_head": p["lm_head"], "attention": p["attention"],
               "dense_ffn": p["dense_ffn"], "moe_routed": p["moe_routed"],
               "moe_shared": p["moe_shared"], "mtp": p["mtp"], "vision": p["vision"]}

    groups = {}
    for name, t in catalog.items():
        groups.setdefault(classify_tensor(name, L), []).append((name, t))

    # config-declared hint (e.g. DeepSeek expert_dtype) as fallback confirmation
    hint_fp4 = str(cfg.get("expert_dtype", "")).lower() in ("fp4", "mxfp4", "nvfp4")

    by_key, dtype_hist, comp_dtypes, recon = {}, {}, {}, {}
    for key, tensors in groups.items():
        apparent = 0
        has_int_weight = False
        int_bits = None
        for name, t in tensors:
            if QUANT_META.search(name):
                continue
            apparent += math.prod(t["shape"]) if t["shape"] else 1
            if t["dtype"] in INT_DTYPES:
                has_int_weight = True
                int_bits = INT_DTYPES[t["dtype"]]

        # ---- packing factor: formula/apparent ≈ 2, 4 or 8 (±6%)
        pack = 1
        expected = formula.get(key)
        if has_int_weight and expected and apparent:
            ratio = expected / apparent
            for cand in (2, 4, 8):
                if abs(ratio - cand) <= 0.06 * cand:
                    pack = cand
                    break
        if pack == 1 and has_int_weight and hint_fp4 \
                and key in ("moe_routed", "moe_shared", "mtp"):
            pack = 2

        d = {"bytes": 0, "params": 0}
        dts = {}
        for name, t in tensors:
            is_meta = bool(QUANT_META.search(name))
            if is_meta:
                dt = "QSCALE"
            elif pack > 1 and t["dtype"] in INT_DTYPES:
                dt = _sub_byte_name(cfg, INT_DTYPES[t["dtype"]] // pack)
            else:
                dt = t["dtype"]
            d["bytes"] += t["bytes"]
            dts[dt] = dts.get(dt, 0) + t["bytes"]
            dtype_hist[dt] = dtype_hist.get(dt, 0) + t["bytes"]
            if not is_meta:
                params = math.prod(t["shape"]) if t["shape"] else 1
                if pack > 1 and t["dtype"] in INT_DTYPES:
                    params *= pack
                d["params"] += params
        by_key[key] = d
        comp_dtypes[key] = dts
        # reconciliation trail for the evidence tab: the two independent param
        # counts (config formula vs safetensors shapes) and the packing verdict
        recon[key] = {
            "formula": expected,               # None when no reliable closed form
            "apparent": apparent,              # from real tensor shapes
            "pack": pack,                      # 1 = unpacked; 2/4/8 = sub-byte packed
            "storageBits": int_bits,           # int8/int32 the packed values live in
            "trueBits": (int_bits // pack) if (int_bits and pack > 1) else None,
        }
    return by_key, dtype_hist, comp_dtypes, recon


def dtype_label(dtype_hist: dict) -> str:
    """Human name for the weight storage: 'fp8' or 'mixed: I8(fp4) 87% + ...'."""
    qscale = t("dtype.quant_scale")
    nice = {"F8_E4M3": "fp8", "F8_E5M2": "fp8", "BF16": "bf16", "F16": "fp16",
            "F32": "fp32", "I8": "int8", "U8": "uint8", "I64": "int64",
            "F8_E8M0": qscale, "QSCALE": qscale, "F4": "fp4", "U4": "fp4"}
    total = sum(dtype_hist.values())
    ranked = sorted(dtype_hist.items(), key=lambda kv: -kv[1])
    if ranked[0][1] / total >= 0.97:
        return nice.get(ranked[0][0], ranked[0][0])
    parts = [f"{nice.get(dt, dt)} {b / total:.0%}"
             for dt, b in ranked if b / total >= 0.02][:4]
    return t("dtype.mixed_prefix") + " + ".join(parts)


# ---------------------------------------------------------------- dtype

def weight_bytes_per_param(cfg: dict) -> tuple[float, str]:
    qc = cfg.get("quantization_config") or {}
    method = (qc.get("quant_method") or "").lower()
    if method == "fp8":
        return 1.0, "fp8"
    if method in ("awq", "gptq"):
        bits = qc.get("bits", 4)
        return bits / 8, f"{method}-int{bits}"
    if method == "compressed-tensors":
        for g in (qc.get("config_groups") or {}).values():
            w = g.get("weights") or {}
            if w.get("num_bits"):
                return w["num_bits"] / 8, f"ct-int{w['num_bits']}"
        return 1.0, "compressed-tensors"
    dtype = (cfg.get("dtype") or cfg.get("torch_dtype") or "bfloat16").lower()
    if "float32" in dtype:
        return 4.0, "fp32"
    if "float16" in dtype or "bfloat16" in dtype:
        return 2.0, dtype.replace("torch.", "")
    return 2.0, dtype


# ---------------------------------------------------------------- params

def count_params(cfg: dict) -> dict:
    """Return per-component parameter counts (not bytes)."""
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    p = {}

    # ---- embeddings
    p["embed"] = V * H
    p["lm_head"] = 0 if cfg.get("tie_word_embeddings") else V * H

    # ---- attention: MLA vs GQA/MHA
    is_mla = bool(cfg.get("kv_lora_rank"))
    if is_mla:
        q_lora = cfg.get("q_lora_rank")
        qk_nope = cfg["qk_nope_head_dim"]
        qk_rope = cfg["qk_rope_head_dim"]
        v_dim = cfg["v_head_dim"]
        kv_lora = cfg["kv_lora_rank"]
        qk_head = qk_nope + qk_rope
        if q_lora:  # low-rank Q: q_a + q_b
            attn = H * q_lora + q_lora * n_heads * qk_head
        else:
            attn = H * n_heads * qk_head
        attn += H * (kv_lora + qk_rope)                # kv_a (+ decoupled rope k)
        attn += kv_lora * n_heads * (qk_nope + v_dim)  # kv_b
        attn += n_heads * v_dim * H                    # o_proj
    else:
        head_dim = cfg.get("head_dim") or H // n_heads
        n_kv = cfg.get("num_key_value_heads", n_heads)
        attn = (H * n_heads * head_dim             # q
                + 2 * H * n_kv * head_dim          # k, v
                + n_heads * head_dim * H)          # o
        if cfg.get("attention_bias"):
            attn += (n_heads + 2 * n_kv) * head_dim
    p["attention"] = attn * L
    p["attention_per_layer"] = attn
    p["is_mla"] = is_mla

    # ---- DSA/NSA indexer (lightning indexer, e.g. GLM-5.x / DeepSeek-V3.2)
    idx = 0
    if cfg.get("index_n_heads"):
        d_i = cfg.get("index_head_dim", 128)
        n_i = cfg["index_n_heads"]
        idx = (H * n_i * d_i + H * d_i + H * n_i) * L
    p["indexer"] = idx

    # ---- FFN: dense layers vs MoE layers
    inter = cfg.get("intermediate_size") or 4 * H  # some configs omit it (all-MoE nets)
    n_routed = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts") or 0
    if n_routed:
        # Field-alias trap: most models use `intermediate_size` for dense layers
        # and `moe_intermediate_size` for experts. MiniMax reverses it —
        # `intermediate_size` is the expert size and `dense_intermediate_size`
        # holds the dense-layer size — so pick the dense size explicitly.
        dense_inter = cfg.get("dense_intermediate_size") or inter
        moe_inter = (cfg.get("moe_intermediate_size")
                     or cfg.get("intermediate_size") or 4 * H)
        # dense-layer count: `first_k_dense_replace`, or the leading 0s of a
        # per-layer `moe_layer_freq` array (MiniMax-style), else 0.
        mlf = cfg.get("moe_layer_freq")
        if isinstance(mlf, list) and mlf:
            first_dense = sum(1 for x in mlf if not x)
        else:
            first_dense = cfg.get("first_k_dense_replace", 0)
        moe_layers = L - first_dense
        n_shared = cfg.get("n_shared_experts", 0) or 0
        expert = 3 * H * moe_inter                 # gate/up/down
        p["dense_ffn"] = 3 * H * dense_inter * first_dense
        p["moe_routed"] = expert * n_routed * moe_layers
        p["moe_shared"] = expert * n_shared * moe_layers
        p["moe_gate"] = (H * n_routed + n_routed) * moe_layers  # router + e_score bias
        p["moe_layers"] = moe_layers
        p["dense_layers"] = first_dense
    else:
        p["dense_ffn"] = 3 * H * inter * L
        p["moe_routed"] = p["moe_shared"] = p["moe_gate"] = 0
        p["moe_layers"] = 0
        p["dense_layers"] = L

    # ---- norms & misc
    p["norms"] = (2 * H) * L + H

    # ---- MTP (multi-token prediction) extra layer(s)
    n_mtp = cfg.get("num_nextn_predict_layers", 0) or 0
    if n_mtp:
        moe_inter = cfg.get("moe_intermediate_size", inter)
        n_shared = cfg.get("n_shared_experts", 0) or 0
        mtp_ffn = 3 * H * moe_inter * (n_routed + n_shared) if n_routed else 3 * H * inter
        mtp = attn + (idx // L if idx else 0) + mtp_ffn + 2 * H * H
        p["mtp"] = mtp * n_mtp
    else:
        p["mtp"] = 0

    # ---- vision tower + mm projector (VLMs)
    vspec = vision_tower_spec(cfg)
    p["vision"] = vspec["params"] if vspec else 0

    return p


# ---------------------------------------------------------------- vision tower

def vision_tower_spec(cfg: dict) -> dict | None:
    """Vision tower (ViT) + multimodal projector: weight formula, encoder
    activation workspace, and image-token arithmetic.

    Field aliases across the four in-scope VLMs:
      hidden: vt_hidden_size (Kimi) | hidden_size (Qwen/MiniMax/Gemma)
      inter : vt_intermediate_size | intermediate_size
      layers: vt_num_hidden_layers | num_hidden_layers | depth (Qwen)
      heads : vt_num_attention_heads | num_attention_heads | num_heads (Qwen)

    Weight model (standard pre-norm ViT block, qkv/o with bias):
      per layer qkv+o = 4H²+4H, MLP = 2HI+I+H, 2 layernorms = 4H; plus the
      patch-embed conv (in_ch·ps²·tps·H), learned pos-emb when dims are in the
      config, final layernorm, and the projector (Kimi 'patchmerger':
      pre_norm + (H·merge)² + (H·merge)·text_H; generic fallback: one linear
      to the text hidden). Verified byte-exact against moonshotai/Kimi-K2.6
      safetensors (471,143,920 params).

    Runtime model: image tokens enter the text KV cache as ordinary tokens
    (per-token KV cell identical to text), so KV needs no separate pool —
    each image just consumes tokens_per_image context positions. What IS
    extra is the ViT encoder's transient activation over max_patches tokens.
    """
    vc = cfg.get("vision_config") or {}
    H = vc.get("vt_hidden_size") or vc.get("hidden_size")
    inter = vc.get("vt_intermediate_size") or vc.get("intermediate_size")
    Lv = (vc.get("vt_num_hidden_layers") or vc.get("num_hidden_layers")
          or vc.get("depth"))
    if not (H and inter and Lv):
        return None
    heads = (vc.get("vt_num_attention_heads") or vc.get("num_attention_heads")
             or vc.get("num_heads") or 16)
    ps = vc.get("patch_size") or 14
    in_ch = vc.get("num_channels") or vc.get("in_channels") or 3
    tps = vc.get("temporal_patch_size") or 1
    mk = vc.get("merge_kernel_size") or vc.get("spatial_merge_size") or 1
    merge = mk[0] * mk[1] if isinstance(mk, (list, tuple)) else int(mk) ** 2

    attn = (4 * H * H + 4 * H) * Lv
    mlp = (2 * H * inter + inter + H) * Lv
    norms = 4 * H * Lv + 2 * H
    pos = (vc.get("init_pos_emb_height", 0) * vc.get("init_pos_emb_width", 0)
           or vc.get("num_position_embeddings", 0)
           or ((vc["image_size"] // ps) ** 2 if vc.get("image_size") else 0)) * H
    patch_embed = in_ch * ps * ps * tps * H + H + pos
    text_h = (vc.get("text_hidden_size") or vc.get("out_hidden_size")
              or cfg.get("hidden_size") or 0)
    merged = H * merge
    if vc.get("mm_projector_type") == "patchmerger":
        proj = 2 * H + merged * merged + merged + merged * text_h + text_h
    elif text_h:
        proj = merged * text_h + text_h    # generic single-linear projector (approx)
    else:
        proj = 0
    params = patch_embed + attn + mlp + norms + proj

    # max ViT sequence per image: fixed-resolution models expose image_size;
    # dynamic-resolution models (Kimi/Qwen) are capped by the preprocessor's
    # patch limit (Kimi media_proc in_patch_limit = 16384; verified: a 7168²
    # image yields exactly 16384/merge = 4096 image tokens).
    max_patches = ((vc["image_size"] // ps) ** 2 if vc.get("image_size") else 16384)
    # LLM-side tokens per image: models with a pooling projector declare it
    # directly (Gemma mm_tokens_per_image: avg-pool to a fixed 256); otherwise
    # it is the merge-kernel reduction of the patch count.
    tokens_per_image = cfg.get("mm_tokens_per_image") or max_patches // merge
    # same per-token workspace shape as the LLM estimate: residual/attn
    # buffers (~8·H) + MLP intermediate (2·I), bf16, one layer live at a time
    act_per_patch = 2 * (8 * H + 2 * inter)
    return {
        "hidden": H, "inter": inter, "layers": Lv, "heads": heads,
        "patch_size": ps, "merge": merge,
        "params": params, "attn_params": attn,
        "max_patches": max_patches,
        "act_per_patch": act_per_patch,
        "act_bytes": max_patches * act_per_patch,
        "tokens_per_image": tokens_per_image,
    }


# ---------------------------------------------------------------- runtime memory

def kv_per_token_elems(cfg: dict) -> tuple[int, str]:
    """Elements stored per token per layer, and a description.

    Elements only — the DSA indexer cache is a *byte* add-on independent of the
    KV dtype, so it lives in indexer_kv_bytes_per_token_layer(), not here. The
    description does mention it so every surface that prints the formula shows
    the full cell.
    """
    if cfg.get("kv_lora_rank"):
        elems = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
        desc = t("kv.mla_desc", kv_lora=cfg["kv_lora_rank"], rope=cfg["qk_rope_head_dim"], elems=elems)
    else:
        n_heads = cfg["num_attention_heads"]
        n_kv = cfg.get("num_key_value_heads", n_heads)
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
        elems = 2 * n_kv * head_dim
        kind = t("kv.kind_mha") if n_kv == n_heads else t("kv.kind_gqa", n_kv=n_kv)
        desc = t("kv.gqa_desc", kind=kind, n_kv=n_kv, head_dim=head_dim, elems=elems)
    idx_b = indexer_kv_bytes_per_token_layer(cfg)
    if idx_b:
        desc += t("kv.indexer_suffix", d_i=idx_b - 4, b=idx_b)
    return elems, desc


def indexer_kv_bytes_per_token_layer(cfg: dict) -> int:
    """DSA indexer per-token index-key cache, bytes per token per layer.

    DSA/NSA models (index_topk set) cache one fp8 index-key vector of
    index_head_dim per token per layer plus a 4-byte scale, paged alongside the
    KV pool but sized independently of --kv-dtype. Verified on 8xB200 (S1 E2):
    GLM-5.x cell = layers x (576 x kv_bytes + 132), the +132 = 128 + 4 held
    across fp8/bf16 KV, GLM-FP8/NVFP4, and dp-attention runs.
    """
    if cfg.get("index_topk") is None:
        return 0
    return (cfg.get("index_head_dim") or 128) + 4


def kv_structure(cfg: dict) -> dict:
    """Classify layers into KV-storage groups and derive a decode read cap.

    Two independent things matter, and the tool used to conflate them:

    * **KV storage (VRAM pool)** — how many token-positions each layer keeps.
      Global full-attention layers store the whole context; sliding-window
      layers store only `min(context, window)`; linear/SSM layers keep a fixed
      recurrent state (no paged KV). Block-sparse layers still store the full
      context (all blocks must be retained to select top-k), so they save no
      storage — only reads.
    * **decode read (roofline)** — how many KV positions each query attends per
      step. Sliding, block-sparse, and DSA all cap this below full context.

    Returns:
      n_layers        — total transformer blocks
      n_kv_layers     — layers that hold paged KV (storage side)
      linear_layers   — layers with fixed-size state instead of paged KV
      kv_groups       — [[count, window], ...] storage model; window=0 means
                        full context, window>0 means capped at min(ctx, window)
      lin_state_per_req — bytes of fixed conv+ssm state per concurrent request
                        summed over linear layers (0 when none/unknown dims)
      read_cap        — decode per-layer read cap in tokens (0 = full context)
      read_cap_layers — how many layers the read cap applies to
      warnings        — notes for structures whose numbers are still upper bounds
    """
    L = cfg["num_hidden_layers"]
    warnings = []
    sliding_window = cfg.get("sliding_window") or 0
    swp = cfg.get("sliding_window_pattern")
    layer_types = cfg.get("layer_types")

    # --- classify each layer into: 'full', 'sliding', or 'linear' ------------
    # 'full' = holds full-context paged KV (incl. sparse-attention variants like
    # deepseek_sparse_attention — DSA caps *reads*, not storage). 'linear' = a
    # fixed-size recurrent/SSM state, no paged KV. 'sliding' = capped KV storage.
    kinds = []
    if isinstance(layer_types, list) and layer_types:
        for x in layer_types:
            xs = str(x).lower()
            if "sliding" in xs:
                kinds.append("sliding")
            elif any(k in xs for k in ("linear", "mamba", "recurrent", "conv", "ssm")):
                kinds.append("linear")
            else:                       # full_attention / *_sparse_attention / attention
                kinds.append("full")
    elif sliding_window and isinstance(swp, int) and swp > 1:
        # Gemma-style: every swp-th layer is global, the rest sliding.
        kinds = ["full" if (i + 1) % swp == 0 else "sliding" for i in range(L)]
    else:
        kinds = ["full"] * L

    n_full = kinds.count("full")
    n_sliding = kinds.count("sliding")
    n_linear = kinds.count("linear")
    n_kv_layers = n_full + n_sliding    # both hold paged KV; sliding is capped

    # --- storage groups ------------------------------------------------------
    kv_groups = []
    if n_full:
        kv_groups.append([n_full, 0])
    if n_sliding:
        kv_groups.append([n_sliding, sliding_window])

    # --- linear/SSM per-request fixed state (SGLang mamba pool) ---------------
    # Qwen3.5/GDN-style: conv state (bf16) + ssm state per linear layer per
    # request; grows with concurrency, not context. Verified against S0 g5
    # startup log (conv 48 KiB + ssm 2 MiB per layer per slot).
    lin_state_per_req = 0
    if n_linear:
        k_hd, n_k = cfg.get("linear_key_head_dim"), cfg.get("linear_num_key_heads")
        v_hd, n_v = cfg.get("linear_value_head_dim"), cfg.get("linear_num_value_heads")
        kernel = cfg.get("linear_conv_kernel_dim") or 0
        if k_hd and n_k and v_hd and n_v:
            conv_b = (2 * k_hd * n_k + v_hd * n_v) * max(kernel - 1, 0) * 2
            ssm_bytes = 4 if str(cfg.get("mamba_ssm_dtype", "")).startswith("float32") else 2
            ssm_b = n_v * k_hd * v_hd * ssm_bytes
            lin_state_per_req = n_linear * (conv_b + ssm_b)

    if n_linear and lin_state_per_req:
        warnings.append(
            f"混合架构：{n_kv_layers}/{L} 层为 attention 存 KV，"
            f"其余 {n_linear} 层为 linear/SSM 定长 state ≈ "
            f"{lin_state_per_req / 2**20:.1f} MiB/请求（随并发不随 context 增长，"
            f"并行页已计入静态区）。按 槽位数=并发数 的最小需求计；"
            f"SGLang 默认启发式可能预分配更多槽，部署时建议显式设 --max-mamba-cache-size。")
    elif n_linear:
        warnings.append(
            f"混合架构：{n_kv_layers}/{L} 层为 attention 存 KV，"
            f"其余 {n_linear} 层为 linear/SSM 定长 state（不计入 KV 池）；"
            f"config 缺 linear_* 维度字段，state 显存未建模。")
    if n_sliding:
        warnings.append(
            f"滑窗注意力：{n_sliding} 层 KV 存储上限已按 min(context, {sliding_window}) 计。")

    # sliding_window present but layers not identifiable → do NOT cap storage
    if sliding_window and not n_sliding:
        warnings.append(
            f"检出 sliding_window={sliding_window} 但无法从 config 判定哪些层滑窗；"
            f"KV 存储未封顶（保守按全 context 计，可能高估）。")

    # --- decode read cap (roofline) ------------------------------------------
    read_cap = 0
    read_cap_layers = 0
    sparse = cfg.get("sparse_attention_config")
    if isinstance(sparse, dict) and sparse.get("use_sparse_attention"):
        cap = (sparse.get("sparse_topk_blocks", 0)
               * sparse.get("sparse_block_size", 0)) or 0
        freq = sparse.get("sparse_attention_freq")
        n_sparse = sum(freq) if isinstance(freq, list) else L
        read_cap, read_cap_layers = cap, n_sparse
        warnings.append(
            f"块稀疏注意力：{n_sparse} 层 decode 读取封顶 min(context, {cap}) tokens；"
            f"KV 存储仍为全量（块稀疏需保留全部块）。")
    elif cfg.get("index_topk") is not None:
        read_cap, read_cap_layers = int(cfg["index_topk"]), L
        warnings.append(
            f"DSA top-k 稀疏：decode 读取封顶 min(context, {cfg['index_topk']})；"
            f"逐层稀疏频率（index_topk_freq 等）未区分。")
    elif n_sliding:
        read_cap, read_cap_layers = sliding_window, n_sliding

    return {
        "n_layers": L,
        "n_kv_layers": n_kv_layers,
        "linear_layers": n_linear,
        "lin_state_per_req": lin_state_per_req,
        "sliding_layers": n_sliding,
        "sliding_window": sliding_window,
        "kv_groups": kv_groups,
        "read_cap": read_cap,
        "read_cap_layers": read_cap_layers,
        "warnings": warnings,
    }


def attention_core_spec(cfg: dict) -> dict:
    """Raw dimensions needed by the in-page attention-core roofline model.

    Geometry and access pattern are intentionally independent:

    * geometry describes work per attended query/key pair and KV elements per
      key position. It is one of MHA, GQA, or absorbed MLA.
    * pattern describes how many key positions each query attends. DSA is a
      sparse pattern layered on top of a geometry (for example MLA + DSA), not
      a fourth KV representation.

    The JS keeps the formulas next to the chart calculation; this function only
    serializes model config fields without deriving FLOPs.
    """
    n_q = cfg["num_attention_heads"]
    if cfg.get("kv_lora_rank"):
        geometry = {
            "kind": "mla",
            "qHeads": n_q,
            "kvLoraRank": cfg["kv_lora_rank"],
            "ropeHeadDim": cfg["qk_rope_head_dim"],
        }
    else:
        n_kv = cfg.get("num_key_value_heads", n_q)
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_q
        geometry = {
            "kind": "mha" if n_kv == n_q else "gqa",
            "qHeads": n_q,
            "kvHeads": n_kv,
            "qkHeadDim": head_dim,
            "valueHeadDim": head_dim,
        }

    # decode read pattern: DSA / block-sparse / sliding all cap the per-step KV
    # read below full context. ks carries the cap and how many of the L layers
    # it applies to (capLayers); the JS treats the remaining layers as dense.
    ks = kv_structure(cfg)
    L = cfg["num_hidden_layers"]
    index_topk = cfg.get("index_topk")
    if index_topk is not None:
        pattern = {"kind": "dsa", "topk": int(index_topk),
                   "capLayers": ks["read_cap_layers"] or L, "totalLayers": L}
    elif ks["read_cap"]:
        pattern = {"kind": "capped", "cap": ks["read_cap"],
                   "capLayers": ks["read_cap_layers"] or L, "totalLayers": L}
    else:
        pattern = {"kind": "dense"}
    return {"geometry": geometry, "pattern": pattern}


def activation_bytes(cfg: dict, p: dict, batch_tokens: int) -> tuple[int, str]:
    """Peak activation workspace for one forward pass over `batch_tokens` tokens.

    Layers execute sequentially, so only one layer's intermediates are live at a
    time (plus residual/hidden buffers and the logits for the sampled positions).
    Per token, in bf16 (2 bytes): a few hidden-sized buffers (residual, attn in/out,
    double-buffering) + the FFN up/gate intermediate. MoE: each token runs top-k
    experts, so the intermediate is top_k x moe_intermediate. This matches the
    order of what vLLM's memory profiler reserves; it is a workspace estimate,
    not an exact number.
    """
    H = cfg["hidden_size"]
    inter = cfg.get("intermediate_size") or 4 * H
    if p["moe_layers"]:
        topk = cfg.get("num_experts_per_tok", 1)
        n_shared = cfg.get("n_shared_experts", 0) or 0
        inter_eff = (topk + n_shared) * cfg.get("moe_intermediate_size", inter)
    else:
        inter_eff = inter
    per_token = 2 * (8 * H + 2 * inter_eff)   # bf16: ~8 hidden-size buffers + gate/up
    total = batch_tokens * per_token
    moe_note = t("act.desc_moe_note") if p["moe_layers"] else ""
    desc = t("act.desc", H=H, inter_eff=f"{inter_eff:,}",
             per_token_kib=f"{per_token / 1024:,.0f}", moe_note=moe_note)
    return total, desc


# ---------------------------------------------------------------- analyze

def analyze(model_id: str, cfg: dict, ctx: int, requests: int, kv_dtype: str,
            batch_tokens: int, overhead: float, catalog: dict | None = None) -> dict:
    """All numbers the text report and the HTML diagram share.

    With a safetensors `catalog`, component bytes come from the real tensor
    sizes (exact, handles mixed precision); otherwise from formula x dtype.
    """
    wbytes, wname = weight_bytes_per_param(cfg)
    p = count_params(cfg)
    L = cfg["num_hidden_layers"]

    names = {
        "embed": "embed", "lm_head": "lm_head",
        "attention": "attention" + (" (MLA)" if p["is_mla"] else ""),
        "indexer": "attn indexer (DSA)",
        "dense_ffn": f"dense FFN ({p['dense_layers']} layers)",
        "moe_routed": f"MoE routed experts ({p['moe_layers']} layers)",
        "moe_shared": "MoE shared experts", "moe_gate": "MoE router/gate",
        "norms": "norms & misc", "mtp": "MTP layer(s)",
        "vision": "vision tower + projector",
    }
    order = ["embed", "lm_head", "attention", "indexer", "dense_ffn",
             "moe_routed", "moe_shared", "moe_gate", "norms", "mtp", "vision"]

    exact = comp_dtypes = recon = None
    if catalog:
        exact, dtype_hist, comp_dtypes, recon = exact_components(catalog, cfg)
        wname = dtype_label(dtype_hist)

    comps = []
    for key in order:
        if exact is not None:
            if key not in exact:
                continue
            comps.append({"key": key, "name": names[key],
                          "params": exact[key]["params"], "bytes": exact[key]["bytes"]})
        else:
            params = p.get(key, 0)
            if key in ("indexer", "mtp", "vision") and not params:
                continue
            if key in ("moe_routed", "moe_shared", "moe_gate") and not p["moe_layers"]:
                continue
            # vision towers stay bf16 even in quantized checkpoints (quant
            # configs ignore vision_tower/mm_projector), so never sub-2-byte
            kb = max(wbytes, 2.0) if key == "vision" else wbytes
            comps.append({"key": key, "name": names[key],
                          "params": params, "bytes": params * kb})

    total_params = sum(c["params"] for c in comps)
    total_bytes = sum(c["bytes"] for c in comps)
    for c in comps:
        c["share"] = c["bytes"] / total_bytes

    # active params (MoE)
    active = None
    if p["moe_layers"]:
        topk = cfg.get("num_experts_per_tok", 0)
        H = cfg["hidden_size"]
        moe_inter = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or 4 * H
        moe_routed_params = next((c["params"] for c in comps if c["key"] == "moe_routed"), 0)
        mtp_params = next((c["params"] for c in comps if c["key"] == "mtp"), 0)
        active = (total_params - moe_routed_params - mtp_params
                  + 3 * H * moe_inter * topk * p["moe_layers"])

    # ---- runtime: KV cache scales with context x running requests
    elems, kv_desc = kv_per_token_elems(cfg)
    kv_struct = kv_structure(cfg)
    kv_layers = kv_struct["n_kv_layers"]   # layers holding paged KV (full + sliding)
    # fp4 (mxfp4/e2m1): 0.5 B data + 1 uint8 scale per 16 elements (SGLang memory_pool)
    kvb = {"fp16": 2, "bf16": 2, "fp8": 1, "fp4": 0.5 + 1 / 16}[kv_dtype]
    # sum stored token-positions across storage groups: full layers keep ctx,
    # sliding layers keep min(ctx, window). layer_tokens is the ctx-weighted
    # layer count that a single request's KV occupies.
    layer_tokens = sum(n * (min(ctx, w) if w else ctx) for n, w in kv_struct["kv_groups"])
    # DSA models cache an fp8 index key + scale per token per layer alongside
    # the KV pool; its size does not follow kv_dtype, so it adds bytes/token/layer
    # rather than elements.
    idx_kv_b = indexer_kv_bytes_per_token_layer(cfg)
    cell_b = elems * kvb + idx_kv_b         # bytes per token per KV-bearing layer
    kv_per_tok = cell_b * kv_layers         # nominal (all KV layers at full ctx), for display
    kv_per_req = cell_b * layer_tokens
    kv_total = kv_per_req * requests
    mha_total = mha_ratio = None
    if p["is_mla"]:
        n_heads = cfg["num_attention_heads"]
        mha_elems = n_heads * (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"] + cfg["v_head_dim"])
        mha_total = mha_elems * layer_tokens * kvb * requests
        mha_ratio = mha_elems / elems

    # ---- runtime weight materializations (SGLang-specific, beyond safetensors)
    # MLA absorption: at load SGLang dequantizes kv_b_proj into bf16 w_kc/w_vc
    # for the absorbed decode path and KEEPS the original (MHA prefill still
    # uses it). Shape carries num_local_heads, so it shards by attn-TP: /tp in
    # pure TP, one full copy per rank under dp-attention. Verified on S1 E4:
    # GLM-5.2 28 MiB/layer full -> +1.86 GiB/GPU going TP8 -> TP8+DP8.
    absorb_per_layer = 0
    if p["is_mla"] and cfg.get("qk_nope_head_dim") and cfg.get("v_head_dim"):
        absorb_per_layer = (cfg["num_attention_heads"] * cfg["kv_lora_rank"]
                            * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]) * 2)
    # NextN/EAGLE draft allocates its own bf16 embed at load (not in the
    # checkpoint — the weight loader skips it and later aliases it to the
    # target's), but the KV pool is sized at the pre-release watermark, so it
    # costs pool capacity. Shards like the embedding: by attn-TP.
    n_mtp = cfg.get("num_nextn_predict_layers", 0) or 0
    draft_embed_bytes = cfg["vocab_size"] * cfg["hidden_size"] * 2 if n_mtp else 0

    # ---- runtime: activation workspace scales with tokens in flight
    act_total, act_desc = activation_bytes(cfg, p, batch_tokens)

    # hybrid models: linear/SSM layers hold a fixed per-request state (SGLang
    # mamba pool) — scales with concurrency, not context
    lin_state_total = kv_struct["lin_state_per_req"] * requests

    # ---- vision tower runtime (VLMs): image tokens are ordinary KV-cache
    # tokens (same cell as text — they consume context positions, already
    # covered by ctx above); the extra cost is the ViT encoder's transient
    # activation while encoding one max-size image batch. It lives in the
    # non-static region, alongside the LLM activation workspace.
    vision = vision_tower_spec(cfg)
    vision_act = vision["act_bytes"] if vision else 0
    if vision:
        kv_struct["warnings"].append(
            f"多模态：vision tower {vision['layers']} 层（hidden {vision['hidden']}），"
            f"每图最多 {vision['max_patches']:,} patches → merge 后 "
            f"{vision['tokens_per_image']:,} 个图像 token 进入 KV cache（与文本 token 同 cell，"
            f"占用 context 位置，已含在 context 预算内）；"
            f"ViT encoder 峰值 activation ≈ {vision_act / GIB:.2f} GiB（编码期瞬时，非常驻）。")

    runtime_total = kv_total + act_total + lin_state_total + vision_act
    grand = (total_bytes + runtime_total) * (1 + overhead)

    return {
        "model_id": model_id, "cfg": cfg, "arch": (cfg.get("architectures") or ["?"])[0],
        "wbytes": wbytes, "wname": wname, "p": p, "comps": comps,
        "total_params": total_params, "total_bytes": total_bytes, "active": active,
        "ctx": ctx, "requests": requests, "kv_dtype": kv_dtype, "kv_desc": kv_desc,
        "kv_struct": kv_struct,
        "kv_elems_total": elems * kv_layers,  # dtype-independent: elements/token over KV-bearing layers
        "kv_indexer_bytes": idx_kv_b,  # DSA index-key cache, bytes/token/layer (0 if no indexer)
        "absorb_per_layer": absorb_per_layer,   # MLA w_kc/w_vc bf16, full-heads bytes/layer
        "draft_embed_bytes": draft_embed_bytes,  # NextN draft's own bf16 embed (load-time)
        "kv_per_tok": kv_per_tok, "kv_per_req": kv_per_req, "kv_total": kv_total,
        "lin_state_total": lin_state_total,
        "vision": vision, "vision_act": vision_act,
        "mha_total": mha_total, "mha_ratio": mha_ratio,
        "batch_tokens": batch_tokens, "act_total": act_total, "act_desc": act_desc,
        "runtime_total": runtime_total, "overhead": overhead, "grand": grand,
        "exact": exact is not None, "comp_dtypes": comp_dtypes, "recon": recon,
    }


# ---------------------------------------------------------------- parallel partition (design_2)

def attn_tp_partition(cfg: dict) -> tuple:
    """Per-layer attention params split by TP behavior:
    (qo_sliced, kv_proj, replicated)
      qo_sliced  — has a head dim, cut 1/TP (q/o proj; MLA q_b/kv_b/o_proj)
      kv_proj    — GQA k/v proj, cut by min(TP, n_kv_heads) then replicated
      replicated — no head dim, every rank keeps a full copy (MLA q_a/kv_a)
    """
    H = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    if cfg.get("kv_lora_rank"):  # MLA
        q_lora = cfg.get("q_lora_rank")
        qk_head = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
        kv_lora = cfg["kv_lora_rank"]
        if q_lora:
            repl = H * q_lora
            sliced = q_lora * n_heads * qk_head
        else:
            repl = 0
            sliced = H * n_heads * qk_head
        repl += H * (kv_lora + cfg["qk_rope_head_dim"])            # kv_a + rope k
        sliced += kv_lora * n_heads * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"])  # kv_b
        sliced += n_heads * cfg["v_head_dim"] * H                  # o_proj
        return sliced, 0, repl
    head_dim = cfg.get("head_dim") or H // n_heads
    n_kv = cfg.get("num_key_value_heads", n_heads)
    qo = 2 * H * n_heads * head_dim
    kv_proj = 2 * H * n_kv * head_dim
    if cfg.get("attention_bias"):
        qo += n_heads * head_dim
        kv_proj += 2 * n_kv * head_dim
    return qo, kv_proj, 0


def per_layer_breakdown(a: dict, cfg: dict) -> dict:
    """Amortize analyze()'s per-component bytes into the two layer prototypes
    (dense layer, MoE layer). Exact-mode bytes stay exact in aggregate; the
    split of attention into sliced/replicated uses the formula param ratio."""
    p = a["p"]
    L = cfg["num_hidden_layers"]
    cb = {c["key"]: c["bytes"] for c in a["comps"]}

    qo, kvp, repl = attn_tp_partition(cfg)
    tot = qo + kvp + repl
    attn_b = cb.get("attention", 0) / L
    layer = {
        "attnQo": attn_b * qo / tot,
        "attnKvProj": attn_b * kvp / tot,
        "attnRepl": attn_b * repl / tot,
        "indexer": cb.get("indexer", 0) / L,
        "norms": cb.get("norms", 0) / L,
        "denseFfn": cb.get("dense_ffn", 0) / max(p["dense_layers"], 1),
        "moeRouted": cb.get("moe_routed", 0) / max(p["moe_layers"], 1),
        "moeShared": cb.get("moe_shared", 0) / max(p["moe_layers"], 1),
        "moeGate": cb.get("moe_gate", 0) / max(p["moe_layers"], 1),
    }

    # MTP: fraction of one MTP copy that is TP-sliceable (attention sliced part
    # + expert FFN); eh_proj / norms / indexer treated as replicated. The
    # attention share is tracked separately because dp-attention flips it to
    # replicated while the expert FFN stays TP-sliced.
    mtp_b = cb.get("mtp", 0)
    sliced_frac = attn_frac = 0.0
    if mtp_b:
        H = cfg["hidden_size"]
        inter = cfg.get("intermediate_size") or 4 * H
        n_routed = cfg.get("n_routed_experts") or cfg.get("num_experts") or 0
        moe_inter = cfg.get("moe_intermediate_size", inter)
        n_shared = cfg.get("n_shared_experts", 0) or 0
        ffn = 3 * H * moe_inter * (n_routed + n_shared) if n_routed else 3 * H * inter
        idx = p["indexer"] // L if p["indexer"] else 0
        per_copy = tot + idx + ffn + 2 * H * H
        sliced_frac = (qo + kvp + ffn) / per_copy
        attn_frac = (qo + kvp) / per_copy
    layer["mtpTotal"] = mtp_b
    layer["mtpSlicedFrac"] = round(sliced_frac, 4)
    layer["mtpAttnFrac"] = round(attn_frac, 4)
    return layer


def roofline_kernels(a: dict, cfg: dict) -> list:
    """Per-kernel weight params/bytes for the roofline tab.

    Emits one entry per weight-bearing matmul kernel (attention projections,
    DSA indexer, dense FFN, MoE router / routed experts / shared experts,
    lm_head). The in-page JS turns each into a FLOPs/HBM point for both the
    decode phase (B tokens/step) and the prefill phase (T tokens/chunk); the
    attention-core (KV read / causal) kernel is synthesized in JS from
    kvElemsPerLayer since it carries no weights. embed (a gather) and norms
    (elementwise) are omitted — they are not GEMM kernels and have negligible
    arithmetic intensity. Bytes are exact when safetensors headers were read.
    """
    cb = {c["key"]: c for c in a["comps"]}
    is_mla = a["p"]["is_mla"]
    # (comp key, label, color var, kind). kind drives the JS FLOPs/HBM model:
    #   gemm — read weights once, 2·params·tokens FLOPs
    #   moe  — routed experts: decode touches only a fraction of experts
    spec = [
        ("attention",  "Attn QKVO proj" + (" · MLA" if is_mla else ""), "var(--s2)", "gemm"),
        ("indexer",    "DSA indexer",        "var(--s4)", "gemm"),
        ("dense_ffn",  "Dense FFN",          "var(--s1)", "gemm"),
        ("moe_gate",   "MoE router",         "var(--s8)", "gemm"),
        ("moe_routed", "MoE experts",        "var(--s3)", "moe"),
        ("moe_shared", "MoE shared experts", "var(--s7)", "gemm"),
        ("lm_head",    "lm_head",            "var(--lmh)", "gemm"),
    ]
    out = []
    for key, label, color, kind in spec:
        c = cb.get(key)
        if not c or c["params"] <= 0:
            continue
        out.append({"key": key, "label": label, "color": color,
                    "params": c["params"], "bytes": c["bytes"], "kind": kind})
    return out


def parallel_self_check(a: dict, layer: dict, cfg: dict):
    """Prototype bytes x counts must reproduce the aggregate weight total."""
    p = a["p"]
    L = cfg["num_hidden_layers"]
    cb = {c["key"]: c["bytes"] for c in a["comps"]}
    rebuilt = (cb.get("embed", 0) + cb.get("lm_head", 0) + layer["mtpTotal"]
               + cb.get("vision", 0)
               + L * (layer["attnQo"] + layer["attnKvProj"] + layer["attnRepl"]
                      + layer["indexer"] + layer["norms"])
               + p["dense_layers"] * layer["denseFfn"]
               + p["moe_layers"] * (layer["moeRouted"] + layer["moeShared"] + layer["moeGate"]))
    dev = abs(rebuilt - a["total_bytes"]) / a["total_bytes"]
    print(f"self-check: layer prototypes rebuild {rebuilt / GIB:,.1f} GiB "
          f"vs total {a['total_bytes'] / GIB:,.1f} GiB (dev {dev:.3%})",
          file=sys.stderr)
    if dev > 0.005:
        print("warning: prototype decomposition deviates >0.5%", file=sys.stderr)


# ---------------------------------------------------------------- text report

def human(nbytes: float) -> str:
    return f"{nbytes / GIB:,.1f} GiB"


def report(a: dict):
    cfg, p = a["cfg"], a["p"]
    L = cfg["num_hidden_layers"]
    print(f"\n{'=' * 68}")
    print(f"Model     : {a['model_id']}  ({a['arch']})")
    print(f"Layers    : {L}  hidden={cfg['hidden_size']}  heads={cfg['num_attention_heads']}  vocab={cfg['vocab_size']}")
    if p["moe_layers"]:
        n_routed = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts")
        print(f"MoE       : {n_routed} experts, top-{cfg.get('num_experts_per_tok', '?')}, "
              f"moe_inter={cfg.get('moe_intermediate_size')}, first {p['dense_layers']} layers dense")
    eff = a["total_bytes"] / a["total_params"]
    print(f"Weights   : {a['wname']} ({eff:.2f} byte/param effective"
          f"{', exact from safetensors' if a.get('exact') else ''})")
    print(f"{'=' * 68}")
    print(f"\n-- STATIC: weights {a['total_params'] / 1e9:,.1f} B params -> {human(a['total_bytes'])} --\n")
    for w in a.get("weight_warnings", []):
        print(f"  ⚠ {w}")
    print(f"  {'component':<38}{'params':>12}{'memory':>12}{'share':>7}")
    for c in sorted(a["comps"], key=lambda x: -x["params"]):
        if c["params"] == 0:
            continue
        print(f"  {c['name']:<38}{c['params'] / 1e9:>10.2f}B{c['bytes'] / GIB:>10.1f}G{c['share']:>6.1%}")
    if a["active"]:
        print(f"\n  active params per token ~ {a['active'] / 1e9:,.0f}B "
              f"(top-{cfg.get('num_experts_per_tok')} + {cfg.get('n_shared_experts', 0)} shared)")
    print(f"\n-- RUNTIME: context {a['ctx']:,} x {a['requests']} running requests --\n")
    kv_layers = a["kv_struct"]["n_kv_layers"]
    layers_note = (f"all {L} layers" if kv_layers == L
                   else f"{kv_layers} of {L} full-attn layers")
    print(f"  KV cache ({a['kv_dtype']}): {a['kv_desc']}")
    print(f"    per token ({layers_note}) {a['kv_per_tok'] / 1024:,.1f} KiB"
          f" -> per request {human(a['kv_per_req'])} -> x{a['requests']} = {human(a['kv_total'])}")
    if a["mha_total"]:
        print(f"    (uncompressed MHA equivalent {human(a['mha_total'])}, MLA saves {a['mha_ratio']:,.0f}x)")
    if a["lin_state_total"]:
        print(f"  Linear/SSM state ({a['kv_struct']['linear_layers']} layers, fixed per request): "
              f"{a['kv_struct']['lin_state_per_req'] / 2**20:,.1f} MiB/req"
              f" -> x{a['requests']} = {human(a['lin_state_total'])}")
    print(f"  Activation workspace ({a['batch_tokens']:,} tokens/forward): {human(a['act_total'])}")
    print(f"    {a['act_desc']}")
    if a.get("vision"):
        v = a["vision"]
        print(f"  Vision encoder activation (ViT {v['layers']}L, hidden {v['hidden']}, "
              f"{v['max_patches']:,} patches/image): {human(a['vision_act'])}")
        print(f"    每图 {v['tokens_per_image']:,} 个图像 token 进 KV cache（与文本同 cell，占 context 位置）")
    print(f"  runtime total: {human(a['runtime_total'])}")
    for w in a["kv_struct"]["warnings"]:
        print(f"  ⚠ {w}")
    print(f"\n-- TOTAL --\n")
    print(f"  weights {human(a['total_bytes'])} + runtime {human(a['runtime_total'])} "
          f"+ fragmentation ~{a['overhead']:.0%} = ~{human(a['grand'])}")
    print()


# ---------------------------------------------------------------- html diagram

def _gib(nbytes: float) -> str:
    v = nbytes / GIB
    return f"{v:,.1f}" if v >= 0.95 else f"{v:.2f}"


def _b(params: float) -> str:
    v = params / 1e9
    return f"{v:,.1f}B" if v >= 10 else f"{v:.2f}B"


def _var(i: int) -> str:
    return f"var(--s{i + 1})"


def _card(slot, title, value, share, lines, dtype=None):
    rows = "".join(f"<div class='cl'>{ln}</div>" for ln in lines)
    pct = f"<span class='pct'>{share:.1%}</span>" if share is not None else ""
    dt = f"<span class='dt'>{dtype}</span>" if dtype else ""
    return f"""<div class='card' style='border-left-color:{_var(slot)}'>
      <div class='ch'><i class='dot' style='background:{_var(slot)}'></i><span class='ct'>{title}</span>{dt}
      <span class='cv'>{value}</span>{pct}</div>{rows}</div>"""


def _stacked_bar(segs):
    """segs: list of {label, bytes, share, slot} -> (bar_html, legend_html)."""
    bar = ""
    for s in segs:
        pct = s["share"] * 100
        inner = ""
        if pct >= 30:  # label inside only when it comfortably fits
            inner = (f"<span class='seg-label'>{s['label'].split('（')[0].split('(')[0].strip()} "
                     f"{_gib(s['bytes'])} GiB{t('bar.pct_paren', pct=pct)}</span>")
        bar += (f"<div class='seg' style='flex:{s['share']:.6f};"
                f"background:{_var(s['slot'])}' title='{s['label']}: {_gib(s['bytes'])} GiB'>{inner}</div>")
    legend = "".join(
        f"<span class='lg'><i style='background:{_var(s['slot'])}'></i>"
        f"{s['label']}&ensp;<b>{_gib(s['bytes'])}</b></span>"
        for s in segs)
    return bar, legend


def build_parallel_struct(a: dict, cfg: dict) -> str:
    """Parallel-tab left column, design_2 style: bordered blocks holding rows
    of small squares (yellow embed / green attention / blue FFN / pink MTP /
    gray lm_head)."""
    p = a["p"]
    L = cfg["num_hidden_layers"]
    is_moe = bool(p["moe_layers"])
    dense_n, moe_n = p["dense_layers"], p["moe_layers"]
    n_routed = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts")
    EMB, ATT, FFN = "var(--s3)", "var(--s2)", "var(--s1)"
    MTP_C, LMH = "var(--s7)", "var(--lmh)"

    def sqrow(color):
        return "<div class='sqrow'>" + \
            f"<i class='sq' style='background:{color}'></i>" * 8 + "</div>"

    def block(label, rows, border):
        return (f"<div class='sblk' style='border-color:{border}'>"
                f"<div class='sblab'>{label}</div>{rows}</div>")

    emb_label = "Embedding" if p["lm_head"] else t("pstruct.emb_lmhead_shared")
    h = ""
    if p.get("vision"):
        VIS = "var(--s9)"
        h += block(t("pstruct.vision_block"), sqrow(VIS), VIS)
    h += block(emb_label, sqrow(EMB), EMB)
    if is_moe:
        if dense_n:
            h += block(f"L0–L{dense_n - 1} · Attention + Dense FFN ×{dense_n}",
                       sqrow(ATT) + sqrow(FFN), ATT)
        h += block(t("pstruct.moe_layer_block", dense_n=dense_n, n_routed=n_routed),
                   sqrow(ATT) + sqrow(FFN), ATT)
        h += f"<div class='ell'>{t('pstruct.moe_ellipsis', moe_n=moe_n, lo=dense_n, hi=L - 1)}</div>"
        h += block(f"L{L - 1} · Attention + MoE FFN", sqrow(ATT) + sqrow(FFN), ATT)
    else:
        h += block("L0 · Attention + FFN", sqrow(ATT) + sqrow(FFN), ATT)
        h += f"<div class='ell'>{t('pstruct.same_layers_ellipsis', L=L)}</div>"
        h += block(f"L{L - 1} · Attention + FFN", sqrow(ATT) + sqrow(FFN), ATT)
    if p["mtp"]:
        h += block(t("pstruct.mtp_block", n_mtp=cfg.get('num_nextn_predict_layers')), sqrow(MTP_C), MTP_C)
    if p["lm_head"]:
        h += block(t("pstruct.lmhead_block"), sqrow(LMH), LMH)
    return h


def build_evidence(a: dict, cfg: dict, p: dict) -> str:
    """Evidence tab (first tab): why the weight numbers are trustworthy.

    Three stacked sections, top-to-bottom = the reasoning order:
      A · the config.json fields the formulas consume (the input)
      B · per-component parameter formula: symbolic = substituted = result
      C · reconciliation — config formula vs safetensors shapes, as paired bars.
          Equal bars = two independent sources agree. A short apparent bar +
          a ⚡ badge = sub-byte packing (e.g. two fp4 packed into one int8),
          explained inline. Grey/absent when safetensors was not read.
    """
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    is_moe = bool(p["moe_layers"])
    is_mla = p["is_mla"]
    recon = a.get("recon")

    # ---------- Fused A · config → formula → param count (one row per component) ----------
    # Each row pairs the config fields a component's formula consumes (left column)
    # with that formula rendered symbolic = substituted = result (right column), so
    # the causal chain "these fields feed this formula → this many params" reads
    # left-to-right on a single line. Config chips carry their plain-language meaning
    # as a hover tooltip, keeping the row compact. Three inputs are NOT
    # component-specific — hidden_size/num_hidden_layers (every formula uses them),
    # the quant byte-multiplier, and the DSA index (compute-only, yields no weight) —
    # so they sit in a global band above the table instead of being forced into a
    # row. Each row's left border + dot reuse the same slot color the component gets
    # everywhere else (bars, layer diagram, table), so color reads as a legend.
    def chip(field, val, meaning):
        if val is None:
            return ""
        return f"<code class='ev-chip' data-tip='{meaning}'>{field}</code>&nbsp;{val}"

    def abrow(slot, name, chips, symbolic, substituted, params):
        cfg_cell = " ".join(c for c in chips if c) or "—"
        return (f"<tr><td class='ev-abname' style='border-left-color:{_var(slot)}'>"
                f"<i class='dot' style='background:{_var(slot)}'></i>{name}</td>"
                f"<td class='ev-abcfg'>{cfg_cell}</td>"
                f"<td class='ev-abf'><div class='ev-fsym'>{symbolic}</div>"
                f"<div class='ev-fsub'>= {substituted}</div>"
                f"<div class='ev-fres'>= <b>{_b(params)}</b> {t('ev.params')}</div></td></tr>")

    # global band: inputs shared by every formula (not tied to one component)
    _bpp = f"{a['total_bytes'] / a['total_params']:.2f}"
    gchips = [
        chip("hidden_size", H, t("ev.f.hidden_size")),
        chip("num_hidden_layers", L, t("ev.f.num_hidden_layers")),
        chip(a["wname"], f"{_bpp} B/param", t("ev.f.quant", bpp=_bpp))]
    # compute-only fields (affect flops, not stored weight) — shown here rather
    # than in a component row, since no weight formula consumes them.
    if is_moe and cfg.get("num_experts_per_tok"):
        gchips.append(chip("num_experts_per_tok", cfg["num_experts_per_tok"], t("ev.f.num_experts_per_tok")))
    if cfg.get("index_n_heads"):
        gchips.append(chip("index_n_heads", cfg["index_n_heads"], t("ev.f.index_n_heads")))
    global_band = (f"<div class='ev-gband'><span class='ev-glab'>{t('ev.grp_global')}</span>"
                   + " · ".join(c for c in gchips if c) + "</div>")

    rows_ab = ""
    mult = "2" if p["lm_head"] else "1"
    rows_ab += abrow(0, "embed + lm_head",
                     [chip("vocab_size", f"{V:,}", t("ev.f.vocab_size"))],
                     f"{mult} × vocab × hidden", f"{mult} × {V:,} × {H:,}",
                     p["embed"] + p["lm_head"])
    if is_mla:
        q_lora = cfg.get("q_lora_rank")
        qk_nope, qk_rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
        v_dim, kv_lora = cfg["v_head_dim"], cfg["kv_lora_rank"]
        qk_head = qk_nope + qk_rope
        if q_lora:
            sym = "L × (q_a + q_b + kv_a + kv_b + o)"
            sub = (f"{L} × ({H}×{q_lora} + {q_lora}×{n_heads}×{qk_head} "
                   f"+ {H}×({kv_lora}+{qk_rope}) + {kv_lora}×{n_heads}×({qk_nope}+{v_dim}) "
                   f"+ {n_heads}×{v_dim}×{H})")
        else:
            sym = "L × (q + kv_a + kv_b + o)"
            sub = (f"{L} × ({H}×{n_heads}×{qk_head} + {H}×({kv_lora}+{qk_rope}) "
                   f"+ {kv_lora}×{n_heads}×({qk_nope}+{v_dim}) + {n_heads}×{v_dim}×{H})")
        attn_chips = [
            chip("q_lora_rank", q_lora, t("ev.f.q_lora_rank")),
            chip("kv_lora_rank", kv_lora, t("ev.f.kv_lora_rank")),
            chip("qk_nope_head_dim", qk_nope, t("ev.f.qk_nope_head_dim")),
            chip("qk_rope_head_dim", qk_rope, t("ev.f.qk_rope_head_dim")),
            chip("v_head_dim", v_dim, t("ev.f.v_head_dim")),
            chip("num_attention_heads", n_heads, t("ev.f.num_attention_heads"))]
        rows_ab += abrow(1, "attention (MLA)", attn_chips, sym, sub, p["attention"])
    else:
        n_kv = cfg.get("num_key_value_heads", n_heads)
        hd = cfg.get("head_dim") or H // n_heads
        kind = "MHA" if n_kv == n_heads else "GQA"
        attn_chips = [
            chip("num_attention_heads", n_heads, t("ev.f.num_attention_heads")),
            chip("num_key_value_heads", n_kv, t("ev.f.num_key_value_heads")),
            chip("head_dim", hd, t("ev.f.head_dim"))]
        rows_ab += abrow(1, f"attention ({kind})", attn_chips, "L × (q + k + v + o)",
                         f"{L} × ({H}×{n_heads}×{hd} + 2×{H}×{n_kv}×{hd} + {n_heads}×{hd}×{H})",
                         p["attention"])
    if is_moe:
        n_routed = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts")
        # same field-alias fallback as analyze(): MiniMax puts the expert size
        # in `intermediate_size` and the dense size in `dense_intermediate_size`
        moe_inter = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size")
        dense_n, moe_n = p["dense_layers"], p["moe_layers"]
        if dense_n and p.get("dense_ffn"):
            inter = cfg.get("dense_intermediate_size") or cfg.get("intermediate_size")
            dense_chips = [
                chip("intermediate_size", f"{inter:,}", t("ev.f.intermediate_size")),
                chip("first_k_dense_replace", dense_n, t("ev.f.first_k_dense_replace"))]
            rows_ab += abrow(2, "dense FFN", dense_chips,
                             "3 × hidden × intermediate × dense_layers",
                             f"3 × {H:,} × {inter:,} × {dense_n}", p["dense_ffn"])
        moe_chips = [
            chip("n_routed_experts", n_routed, t("ev.f.n_routed_experts")),
            chip("moe_intermediate_size", f"{moe_inter:,}", t("ev.f.moe_intermediate_size"))]
        rows_ab += abrow(3, "MoE routed experts ★", moe_chips,
                         "n_routed × 3 × hidden × moe_inter × moe_layers",
                         f"{n_routed} × 3 × {H:,} × {moe_inter:,} × {moe_n}", p["moe_routed"])
        if p.get("moe_shared"):
            shared_chips = [
                chip("n_shared_experts", cfg.get("n_shared_experts"), t("ev.f.n_shared_experts")),
                chip("moe_intermediate_size", f"{moe_inter:,}", t("ev.f.moe_intermediate_size"))]
            rows_ab += abrow(3, "MoE shared experts", shared_chips,
                             "n_shared × 3 × hidden × moe_inter × moe_layers",
                             f"{cfg.get('n_shared_experts')} × 3 × {H:,} × {moe_inter:,} × {moe_n}",
                             p["moe_shared"])
    else:
        inter = cfg.get("intermediate_size")
        ffn_chips = [chip("intermediate_size", f"{inter:,}", t("ev.f.intermediate_size"))]
        rows_ab += abrow(2, "FFN", ffn_chips, "3 × hidden × intermediate × L",
                         f"3 × {H:,} × {inter:,} × {L}", p["dense_ffn"])
    if p.get("mtp"):
        n_mtp = cfg.get("num_nextn_predict_layers")
        mtp_chips = [chip("num_nextn_predict_layers", n_mtp, t("ev.f.num_nextn_predict_layers"))]
        rows_ab += abrow(4, f"MTP × {n_mtp}", mtp_chips,
                         "n_mtp × (attention + MoE expert stack + eh_proj 2·H·H)",
                         t("ev.mtp_composite"), p["mtp"])
    if p.get("vision"):
        v = a.get("vision") or vision_tower_spec(cfg)
        vis_chips = [
            chip("vt_hidden_size", v["hidden"], t("ev.f.vt_hidden_size")),
            chip("vt_intermediate_size", v["inter"], t("ev.f.vt_intermediate_size")),
            chip("vt_num_hidden_layers", v["layers"], t("ev.f.vt_num_hidden_layers")),
            chip("patch_size", v["patch_size"], t("ev.f.vt_patch_size"))]
        rows_ab += abrow(8, "vision tower + projector", vis_chips,
                         "L_v × (qkv/o 4H² + MLP 2HI + norms) + patch_embed + projector",
                         t("ev.vision_composite"), p["vision"])

    ab_table = (f"{global_band}<table class='ev-abtab'><thead><tr>"
                f"<th>{t('ev.col_component')}</th><th>{t('ev.col_config')}</th>"
                f"<th>{t('ev.col_paramformula')}</th></tr></thead>"
                f"<tbody>{rows_ab}</tbody></table>")

    # ---------- Section B · reconciliation bars ----------
    if not recon:
        sec_c = f"<div class='ev-noexact'>{t('ev.no_exact')}</div>"
    else:
        cb = {c["key"]: c for c in a["comps"]}
        order = ["embed", "lm_head", "attention", "dense_ffn",
                 "moe_routed", "moe_shared", "mtp", "vision"]
        rows = [(k, recon[k]) for k in order
                if k in recon and recon[k].get("formula") and recon[k].get("apparent")]
        max_f = max((r["formula"] for _, r in rows), default=1)
        bars = ""
        for key, r in rows:
            fw = max(2.0, r["formula"] / max_f * 100)
            aw = max(2.0, r["apparent"] / max_f * 100)
            color = _var(COMP_SLOT.get(key, 7))
            packed = r["pack"] > 1
            if packed:
                true_name = _sub_byte_name(cfg, r["trueBits"]) if r["trueBits"] else "?"
                badge = f"<span class='ev-badge pack'>⚡ {r['pack']}× {t('ev.verdict_pack')} → {true_name}</span>"
            else:
                badge = f"<span class='ev-badge ok'>✓ 1.00× {t('ev.verdict_match')}</span>"
            name = cb.get(key, {}).get("name", key)
            bars += (
                f"<div class='ev-rrow'><div class='ev-rname'>{name}</div>"
                f"<div class='ev-rbars'>"
                f"<div class='ev-rline'><span class='ev-rtag'>{t('ev.col_formula')}</span>"
                f"<span class='ev-bar' style='width:{fw:.2f}%;background:{color}'></span>"
                f"<span class='ev-rval'>{_b(r['formula'])}</span></div>"
                f"<div class='ev-rline'><span class='ev-rtag'>{t('ev.col_apparent')}</span>"
                f"<span class='ev-bar' style='width:{aw:.2f}%;background:{color};opacity:.55'></span>"
                f"<span class='ev-rval'>{_b(r['apparent'])}</span></div>"
                f"</div><div class='ev-rverdict'>{badge}</div></div>")
            if packed:
                bars += (f"<div class='ev-packbox'><b>{t('ev.pack_box_title')}</b> "
                         f"{t('ev.pack_box_body', bits=r['storageBits'], pack=r['pack'], true=r['trueBits'])}"
                         f"<div class='ev-packviz'>"
                         + "".join("<span class='ev-nib'>fp"
                                   + str(r['trueBits']) + "</span>" for _ in range(r['pack']))
                         + f"<span class='ev-nibeq'>= 1 byte ({r['storageBits']}-bit)</span></div></div>")

        # Σ cross-check against the official index.json total_size
        idx = a.get("index_total")
        tot = a["total_bytes"]
        if idx:
            dev = abs(tot - idx) / idx
            sigma = t("ev.sigma_ok", sigma=_gib(tot), idx=_gib(idx), dev=f"{dev:.1%}")
        else:
            sigma = t("ev.sigma_line", sigma=_gib(tot))
        caption = f"<div class='ev-caption'>{t('ev.match_caption')}</div>"
        sec_c = (f"<div class='ev-recon'>{bars}</div>{caption}"
                 f"<div class='ev-sigma'>{sigma}</div>")

    # ---------- KV footnote (runtime; formula only, no reconciliation) ----------
    _, kv_desc = kv_per_token_elems(cfg)
    kv_foot = (f"<div class='ev-foot'><b>{t('ev.kv_foot_title')}</b> {kv_desc}"
               f"　<span class='ev-dim'>{t('ev.kv_foot_body')}</span></div>")

    return (
        f"<div class='ev-sec'><div class='ev-h'>{t('ev.sec_ab_title')}</div>"
        f"<div class='ev-note'>{t('ev.sec_ab_note')}</div>{ab_table}</div>"
        f"<div class='ev-sec'><div class='ev-h'>{t('ev.sec_c_title')}</div>"
        f"<div class='ev-note'>{t('ev.sec_c_note')}</div>{sec_c}</div>"
        f"{kv_foot}")


def _build_lang_fragments(a: dict, cfg: dict, p: dict, is_moe: bool, short: str,
                          n_routed, L: int, dense_n: int, moe_n: int,
                          instances: dict, kv_auto: str, kv_choice: str, tot: float) -> dict:
    """Build every Python-rendered (i.e. not recomputed by JS on filter change)
    string/HTML fragment for whichever language is currently set via set_lang().
    Called once per language by render_html() so the in-page switcher can swap
    a whole fragment's innerHTML instead of re-deriving it in JS."""

    # ---- left structure column
    attn_geo = "MLA" if p["is_mla"] else "GQA" if \
        cfg.get("num_key_value_heads", cfg["num_attention_heads"]) < cfg["num_attention_heads"] else "MHA"

    # attention-type annotation: same weights, different runtime KV behavior.
    # DSA/block-sparse cap reads; sliding caps stored KV. Shown as a suffix on
    # the attention sub-block so heterogeneous models don't look uniform.
    ks = a["kv_struct"]
    attn_note = ""
    pat = a.get("attention_core", {}).get("pattern") if a.get("attention_core") else None
    pat = pat or attention_core_spec(cfg)["pattern"]
    if pat.get("kind") == "dsa":
        attn_note = t("struct.attn_dsa", topk=pat["topk"])
    elif pat.get("kind") == "capped" and ks["sliding_layers"] == 0:
        attn_note = t("struct.attn_sparse", cap=pat["cap"])
    attn_label = f"① Attention ({attn_geo}){attn_note}"

    def layer_block(ffn_label, ffn_slot, tag, alabel=None):
        alabel = alabel if alabel is not None else attn_label
        return f"""<div class='layer'><span class='ltag'>{tag}</span>
          <div class='sub' style='border-color:{_var(1)}'><i class='dot' style='background:{_var(1)}'></i>{alabel}</div>
          <div class='sub' style='border-color:{_var(ffn_slot)}'><i class='dot' style='background:{_var(ffn_slot)}'></i>② {ffn_label}</div>
        </div>"""

    struct = f"<div class='io' style='border-color:{_var(0)}'><i class='dot' style='background:{_var(0)}'></i>{t('struct.embed_entry')}</div>"
    if ks["linear_layers"]:
        # hybrid (e.g. Qwen3.5): full-attention layers hold KV, linear layers a
        # fixed-size state. Show both as distinct blocks so it doesn't read as
        # one uniform attention stack.
        ffn_lbl = "MoE FFN" if is_moe else "FFN"
        ffn_slot = 3 if is_moe else 2
        struct += layer_block(ffn_lbl, ffn_slot, "full",
                              alabel=f"① Attention ({attn_geo})")
        struct += layer_block(ffn_lbl, ffn_slot, "linear",
                              alabel=t("struct.linear_block"))
        struct += (f"<div class='ell'>{t('struct.hybrid_ellipsis', L=L, n_full=ks['n_kv_layers'] - ks['sliding_layers'], n_linear=ks['linear_layers'])}</div>")
    elif ks["sliding_layers"]:
        # sliding-window model (e.g. Gemma): global vs sliding layers share
        # weights but differ in KV cap. Show one of each, annotated.
        ffn_lbl = "MoE FFN" if is_moe else "FFN"
        ffn_slot = 3 if is_moe else 2
        n_full = ks["n_kv_layers"] - ks["sliding_layers"]
        struct += layer_block(ffn_lbl, ffn_slot, "global",
                              alabel=f"① Attention ({attn_geo})")
        struct += layer_block(ffn_lbl, ffn_slot, "sliding",
                              alabel=f"① Attention ({attn_geo})" + t("struct.attn_swa", window=ks["sliding_window"]))
        struct += (f"<div class='ell'>{t('struct.swa_ellipsis', L=L, n_full=n_full, n_sliding=ks['sliding_layers'], window=ks['sliding_window'])}</div>")
    elif is_moe:
        if dense_n:
            struct += layer_block("Dense FFN", 2, "L0")
            if dense_n > 1:
                struct += f"<div class='ell'>{t('struct.dense_prefix_ellipsis', dense_n=dense_n, last=dense_n - 1)}</div>"
        struct += layer_block("MoE FFN", 3, f"L{dense_n}")
        which = t("struct.which_rest") if dense_n else t("struct.which_all")
        struct += (f"<div class='ell'>{t('struct.moe_ellipsis', which=which, moe_n=moe_n, n_routed=n_routed, lo=dense_n, hi=L - 1)}</div>")
        struct += layer_block("MoE FFN", 3, f"L{L - 1}")
    else:
        struct += layer_block("FFN", 2, "L0")
        struct += f"<div class='ell'>{t('struct.same_layers_ellipsis', L=L)}</div>"
        struct += layer_block("FFN", 2, f"L{L - 1}")
    if a.get("vision"):
        struct = (f"<div class='io' style='border-color:{_var(8)}'><i class='dot' style='background:{_var(8)}'></i>"
                  f"{t('struct.vision_entry', Lv=a['vision']['layers'])}</div>") + struct
    if p["mtp"]:
        struct += f"<div class='io' style='border-color:{_var(4)}'><i class='dot' style='background:{_var(4)}'></i>{t('struct.mtp_entry', n_mtp=cfg.get('num_nextn_predict_layers'))}</div>"
    head_note = t("struct.lmhead_exit") if p["lm_head"] else t("struct.lmhead_shared")
    struct += f"<div class='io' style='border-color:{_var(0)}'><i class='dot' style='background:{_var(0)}'></i>{head_note}</div>"

    # ---- static cards (weights); bytes come from comps (exact when available)
    cb = {c["key"]: c for c in a["comps"]}

    def cbytes(*keys):
        return sum(cb[k]["bytes"] for k in keys if k in cb)

    def cshare(*keys):
        return cbytes(*keys) / a["total_bytes"]

    def cdtype(*keys):
        """dominant storage dtype label for one or more components, e.g. 'fp4'
        or 'fp8 + bf16'; None when exact dtypes are unavailable."""
        cd = a.get("comp_dtypes") or {}
        merged = {}
        for k in keys:
            for dt, b in (cd.get(k) or {}).items():
                if dt in ("QSCALE", "F8_E8M0"):   # scales are overhead, not identity
                    continue
                merged[dt] = merged.get(dt, 0) + b
        if not merged:
            return None if a.get("exact") else a["wname"]
        label = dtype_label(merged)
        return label.replace(t("dtype.mixed_prefix"), "")

    def cdt(key):
        """inline note like '(fp4 storage)' for card body lines."""
        label = cdtype(key)
        return t("card.dtype_storage_note", label=label) if label and "+" not in label and "%" not in label else ""

    cards = ""
    embed_mult = "2 ×" if p["lm_head"] else t("card.embed_tied_mult")
    cards += _card(0, "embed + lm_head", f"{_gib(cbytes('embed', 'lm_head'))} GiB", cshare("embed", "lm_head"),
                   [t("card.embed_lmhead_line", mult=embed_mult, vocab=f"{cfg['vocab_size']:,}", H=cfg['hidden_size'])],
                   dtype=cdtype("embed", "lm_head"))

    attn_lines = []
    if "attention" in cb:
        attn_lines.append(t("card.attention_params_line", params=_b(cb['attention']['params']), L=L, dt=cdt('attention')))
    if p["is_mla"]:
        attn_lines.append(
            t("card.attention_mla_line", q_lora=cfg.get('q_lora_rank'), kv_lora=cfg['kv_lora_rank'],
              heads=cfg['num_attention_heads'], qk_nope=cfg['qk_nope_head_dim'],
              qk_rope=cfg['qk_rope_head_dim'], v_dim=cfg['v_head_dim']))
    else:
        n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        extra = t("card.attention_lora_extra") if cfg.get("q_lora_rank") or cfg.get("o_lora_rank") else ""
        attn_lines.append(t("card.attention_gqa_line", heads=cfg['num_attention_heads'], n_kv=n_kv, hd=hd, extra=extra))
    cards += _card(1, t("card.attention_title", L=L),
                   f"{_gib(cbytes('attention'))} GiB", cshare("attention"), attn_lines,
                   dtype=cdtype("attention"))

    if "indexer" in cb:
        cards += _card(7, t("card.indexer_title"),
                       f"{_gib(cbytes('indexer'))} GiB", cshare("indexer"),
                       [t("card.indexer_line", n_i=cfg['index_n_heads'], d_i=cfg.get('index_head_dim'))],
                       dtype=cdtype("indexer"))

    inter = cfg.get("dense_intermediate_size") or cfg.get("intermediate_size")
    if is_moe:
        if dense_n and "dense_ffn" in cb:
            cards += _card(2, t("card.dense_ffn_title", dense_n=dense_n),
                           f"{_gib(cbytes('dense_ffn'))} GiB", cshare("dense_ffn"),
                           [t("card.dense_ffn_line", H=cfg['hidden_size'], inter=inter, dense_n=dense_n)],
                           dtype=cdtype("dense_ffn"))
        moe_lines = [
            t("card.moe_routed_line", n_routed=n_routed, H=cfg['hidden_size'],
              moe_inter=cfg.get('moe_intermediate_size') or cfg.get('intermediate_size'), moe_n=moe_n,
              gib=_gib(cbytes('moe_routed')), dt=cdt('moe_routed')),
        ]
        if cbytes("moe_shared"):
            moe_lines.append(t("card.moe_shared_line", n_shared=cfg.get('n_shared_experts'), moe_n=moe_n,
                                gib=_gib(cbytes('moe_shared')), dt=cdt('moe_shared')))
        if a["active"]:
            moe_lines.append(t("card.moe_active_line", topk=cfg.get('num_experts_per_tok'),
                                active=f"{a['active'] / 1e9:,.0f}", n_routed=n_routed))
        scope = t("card.moe_scope_last", moe_n=moe_n) if dense_n else t("card.moe_scope_all", moe_n=moe_n)
        cards += _card(3, t("card.moe_ffn_title", scope=scope),
                       f"{_gib(cbytes('moe_routed', 'moe_shared', 'moe_gate'))} GiB",
                       cshare("moe_routed", "moe_shared", "moe_gate"), moe_lines,
                       dtype=cdtype("moe_routed"))
    else:
        cards += _card(2, t("card.ffn_title_dense", L=L),
                       f"{_gib(cbytes('dense_ffn'))} GiB", cshare("dense_ffn"),
                       [t("card.ffn_line_dense", H=cfg['hidden_size'], inter=inter, L=L)],
                       dtype=cdtype("dense_ffn"))

    if "mtp" in cb and cbytes("mtp"):
        cards += _card(4, t("card.mtp_title", n_mtp=cfg.get('num_nextn_predict_layers')),
                       f"{_gib(cbytes('mtp'))} GiB", cshare("mtp"),
                       [t("card.mtp_line")],
                       dtype=cdtype("mtp"))

    if "vision" in cb and cbytes("vision"):
        v = a["vision"]
        vlines = ([t("card.vision_line", Lv=v['layers'], H=v['hidden'], inter=v['inter'])]
                  if v else [])
        cards += _card(8, t("card.vision_title"),
                       f"{_gib(cbytes('vision'))} GiB", cshare("vision"), vlines,
                       dtype=cdtype("vision"))

    if a.get("exact") and cbytes("norms") / a["total_bytes"] >= 0.005:
        cards += _card(7, t("card.norms_title"),
                       f"{_gib(cbytes('norms'))} GiB", cshare("norms"), [],
                       dtype=cdtype("norms"))

    # ---- runtime cards (dynamic values wrapped in id'd spans; JS rewrites them)
    # kv_desc/act_desc are re-derived here (not read from `a`) because analyze()
    # computes them once before this function's per-language loop runs
    _, kv_desc = kv_per_token_elems(cfg)
    _, act_desc = activation_bytes(cfg, p, a["batch_tokens"])
    kv_lines = [kv_desc, t("card.kv_derivation_line", L=L)]
    if a["mha_total"]:
        kv_lines.append(t("card.mha_compare_line", kv_lora=cfg['kv_lora_rank'], ratio=f"{a['mha_ratio']:.0f}"))
    runtime_cards = _card(5, t("card.kv_title"),
                          "<span id='d-kv-total'>…</span> GiB", None, kv_lines,
                          dtype="<span id='d-kv-dtype'>…</span>")

    if a["kv_struct"]["lin_state_per_req"]:
        runtime_cards += _card(7, t("card.lin_title", n_linear=a["kv_struct"]["linear_layers"]),
                               "<span id='d-lin-total'>…</span> GiB", None,
                               [t("card.lin_line",
                                  mib=f"{a['kv_struct']['lin_state_per_req'] / 2**20:.1f}",
                                  n_linear=a["kv_struct"]["linear_layers"]),
                                t("card.lin_line2")])

    act_lines = [act_desc,
                 t("card.act_line2", batch_tokens=f"{a['batch_tokens']:,}"),
                 t("card.act_line3")]
    runtime_cards += _card(6, t("card.act_title", batch_tokens=f"{a['batch_tokens']:,}"),
                           f"{_gib(a['act_total'])} GiB", None, act_lines, dtype="bf16")

    if a.get("vision"):
        v = a["vision"]
        runtime_cards += _card(8, t("card.vision_act_title", Lv=v['layers']),
                               f"{_gib(a['vision_act'])} GiB", None,
                               [t("card.vision_act_line",
                                  patches=f"{v['max_patches']:,}",
                                  kib=f"{v['act_per_patch'] / 1024:.0f}"),
                                t("card.vision_act_line2",
                                  tokens=f"{v['tokens_per_image']:,}", merge=v['merge'])],
                               dtype="bf16")

    # ---- stacked bars
    ranked = sorted((c for c in a["comps"] if c["bytes"] > 0), key=lambda c: -c["bytes"])
    big = [c for c in ranked if c["share"] >= 0.015][:5]
    tail = [c for c in ranked if c not in big]
    wsegs = [{"label": c["name"], "bytes": c["bytes"], "share": c["share"],
              "slot": COMP_SLOT.get(c["key"], 7)} for c in big]
    if tail:
        tb = sum(c["bytes"] for c in tail)
        wsegs.append({"label": t("bar.other_label"), "bytes": tb, "share": tb / a["total_bytes"], "slot": 7})
    weights_segs, weights_legend = _stacked_bar(wsegs)

    tsegs = [{"label": t("bar.weights_static_label"), "bytes": a["total_bytes"], "share": a["total_bytes"] / tot, "slot": 3 if is_moe else 2},
             {"label": "KV Cache", "bytes": a["kv_total"], "share": a["kv_total"] / tot, "slot": 5}]
    if a["lin_state_total"]:
        tsegs.append({"label": "linear/SSM state", "bytes": a["lin_state_total"],
                      "share": a["lin_state_total"] / tot, "slot": 7})
    tsegs.append({"label": "Activation", "bytes": a["act_total"], "share": a["act_total"] / tot, "slot": 6})
    if a.get("vision_act"):
        tsegs.append({"label": t("bar.vision_act_label"), "bytes": a["vision_act"],
                      "share": a["vision_act"] / tot, "slot": 8})
    total_segs, total_legend = _stacked_bar(tsegs)

    lin_part = t("bar.lin_part", lin=_gib(a['lin_state_total'])) if a["lin_state_total"] else ""
    total_line = t("bar.total_line", w=_gib(a['total_bytes']), kv=_gib(a['kv_total']),
                   lin_part=lin_part,
                   act=_gib(a['act_total']), ov=f"{a['overhead']:.0%}".rstrip('%'),
                   grand=_gib(a['grand']), kv_per_req=_gib(a['kv_per_req']))

    # ---- table view
    trows = "".join(
        f"<tr><td><i class='dot' style='background:{_var(COMP_SLOT.get(c['key'], 7))}'></i>{c['name']}</td>"
        f"<td>{cdtype(c['key']) or '—'}</td>"
        f"<td class='num'>{_b(c['params'])}</td><td class='num'>{_gib(c['bytes'])}</td>"
        f"<td class='num'>{c['share']:.1%}</td></tr>"
        for c in sorted(a["comps"], key=lambda x: -x["params"]) if c["params"] > 0)
    trows += (f"<tr class='sep'><td><i class='dot' style='background:{_var(5)}'></i>"
              f"<span id='d-tbl-kv-label'>KV cache</span></td>"
              f"<td><span id='d-tbl-kv-dtype'>—</span></td>"
              f"<td class='num'>—</td><td class='num'><span id='d-tbl-kv-val'>—</span></td><td class='num'>—</td></tr>")
    if a["kv_struct"]["lin_state_per_req"]:
        trows += (f"<tr><td><i class='dot' style='background:{_var(7)}'></i>"
                  f"linear/SSM state ({a['kv_struct']['linear_layers']} layers)</td><td>—</td>"
                  f"<td class='num'>—</td><td class='num'><span id='d-tbl-lin-val'>—</span></td><td class='num'>—</td></tr>")
    trows += (f"<tr><td><i class='dot' style='background:{_var(6)}'></i>"
              f"activation ({a['batch_tokens']:,} tokens/forward)</td><td>bf16</td>"
              f"<td class='num'>—</td><td class='num'>{_gib(a['act_total'])}</td><td class='num'>—</td></tr>")
    if a.get("vision_act"):
        trows += (f"<tr><td><i class='dot' style='background:{_var(8)}'></i>"
                  f"vision encoder activation ({a['vision']['max_patches']:,} patches/image)</td><td>bf16</td>"
                  f"<td class='num'>—</td><td class='num'>{_gib(a['vision_act'])}</td><td class='num'>—</td></tr>")

    if is_moe:
        subtitle = t("html.subtitle_moe", dense_n=dense_n, moe_n=moe_n,
                     pct=p['moe_routed'] / a['total_params'])
    else:
        subtitle = t("html.subtitle_dense", L=L)

    kv_options_html = "".join(
        f"<option value='{v}'{' selected' if v == kv_choice else ''}>{lbl}</option>"
        for v, lbl in [("auto", t("html.kv_auto_label", kv_auto=kv_auto)), ("bf16", "bf16"), ("fp8", "fp8"),
                       ("fp4", "fp4 (mxfp4)")])

    # roofline what-if: weight dtype. Defaults to the checkpoint's own dtype
    # (exact safetensors bytes); picking another dtype switches to ideal
    # params × bytes/param + the matching peak line.
    wdtype_model = ("fp4" if "fp4" in a["wname"] else
                    "fp8" if "fp8" in a["wname"] else "bf16")
    wdtype_options_html = "".join(
        f"<option value='{v}'{' selected' if v == wdtype_model else ''}>"
        f"{t('html.wdtype_model_label', wdtype=lbl) if v == wdtype_model else lbl}</option>"
        for v, lbl in [("bf16", "bf16"), ("fp8", "fp8"), ("fp4", "fp4 (mxfp4)")])

    inst_options = "".join(
        f"<option value='{n}'{' selected' if n == '' else ''}>"
        f"{n} · {s['count']}×{s['gpu']} {s['memGib']:g} GiB</option>"
        for n, s in (instances or {}).items())
    inst_options += f"<option value='custom'>{t('html.instance_custom')}</option>"

    exact_note = t("html.meta_exact") if a.get("exact") else t("html.meta_formula")
    exact_note_bar = t("html.weights_exact_note") if a.get("exact") else ""

    return {
        "page_title": t("html.page_title", short=short),
        "title": t("html.title", short=short, b=f"{a['total_params'] / 1e9:,.0f}",
                    moe_tag=" MoE" if is_moe else "", gib=_gib(a['total_bytes'])),
        "subtitle": subtitle,
        "meta": t("html.meta", model_id=a['model_id'], arch=a['arch'], H=cfg['hidden_size'],
                  heads=cfg['num_attention_heads'], vocab=f"{cfg['vocab_size']:,}",
                  batch_tokens=f"{a['batch_tokens']:,}", est_note=exact_note),
        "kv_options": kv_options_html,
        "wdtype_options": wdtype_options_html,
        "struct_title": t("html.struct_title", L=L),
        "struct": struct,
        "static_cards": cards,
        "runtime_cards": runtime_cards,
        "weights_bar_head": t("html.weights_bar_head", gib=_gib(a['total_bytes']), wname=a['wname'],
                              bpp=a['total_bytes'] / a['total_params'], exact_note=exact_note_bar),
        "weights_segs": weights_segs,
        "weights_legend": weights_legend,
        "total_bar_head": t("html.total_bar_head", gib=_gib(tot), ctx=f"{a['ctx']:,}", req=a['requests']),
        "total_segs": total_segs,
        "total_legend": total_legend,
        "total_line": total_line,
        "table_rows": trows,
        # parallel tab
        "pstruct_title": t("html.pstruct_title", L=L),
        "pstruct": build_parallel_struct(a, cfg),
        "instance_options": inst_options,
        # static i18n labels (template.html placeholders)
        "lbl_ctx": t("html.lbl_ctx"),
        "lbl_req": t("html.lbl_req"),
        "lbl_kv": t("html.lbl_kv"),
        "lbl_dp": t("html.lbl_dp"),
        "lbl_inst": t("html.lbl_inst"),
        "lbl_frac": t("html.lbl_frac"),
        "lbl_wdtype": t("html.lbl_wdtype"),
        "lbl_chunk": t("html.lbl_chunk"),
        "lbl_custom_mem": t("html.lbl_custom_mem"),
        "lbl_custom_gpn": t("html.lbl_custom_gpn"),
        "lbl_custom_cards": t("html.lbl_custom_cards"),
        "fnote": t("html.fnote"),
        "tab_estimate": t("html.tab_estimate"),
        "tab_parallel": t("html.tab_parallel"),
        "tab_roofline": t("html.tab_roofline"),
        "grp_static": t("html.grp_static"),
        "grp_runtime": t("html.grp_runtime"),
        "details_table_all": t("html.details_table_all"),
        "th_component": t("html.th_component"),
        "th_dtype": t("html.th_dtype"),
        "th_params": t("html.th_params"),
        "th_share": t("html.th_share"),
        "parallel_arrow": t("html.parallel_arrow"),
        "details_table_gpu": t("html.details_table_gpu"),
        # evidence tab (first tab)
        "evidence": build_evidence(a, cfg, p),
        "tab_evidence": t("html.tab_evidence"),
    }


def render_html(a: dict, out_path: str, ctx_options: list, req_options: list,
                layer: dict = None, instances: dict = None, pargs=None):
    cfg, p = a["cfg"], a["p"]
    L = cfg["num_hidden_layers"]
    is_moe = bool(p["moe_layers"])
    short = a["model_id"].split("/")[-1]
    n_routed = cfg.get("n_routed_experts") or cfg.get("num_experts") or cfg.get("num_local_experts")
    dense_n, moe_n = p["dense_layers"], p["moe_layers"]
    tot = a["total_bytes"] + a["runtime_total"]

    # ---- filter options; language-independent (numbers / K / M labels)
    ctx_opts = sorted(set(ctx_options) | {a["ctx"]})
    req_opts = sorted(set(req_options) | {a["requests"]})

    def _ctx_label(v):
        if v % (1024 * 1024) == 0:
            return f"{v // (1024 * 1024)}M"
        return f"{v // 1024}K" if v % 1024 == 0 else f"{v:,}"

    ctx_options_html = "".join(
        f"<option value='{v}'{' selected' if v == a['ctx'] else ''}>{_ctx_label(v)}</option>"
        for v in ctx_opts)
    req_options_html = "".join(
        f"<option value='{v}'{' selected' if v == a['requests'] else ''}>{v}</option>"
        for v in req_opts)
    kv_auto = a.get("kv_auto", a["kv_dtype"])
    kv_choice = a.get("kv_choice", a["kv_dtype"])

    # data the in-page JS needs to recompute the runtime side (both tabs)
    elems_per_layer, _ = kv_per_token_elems(cfg)

    # ---- parallel-tab option lists (language-independent)
    def _popts(values, selected, labeler=str):
        return "".join(f"<option value='{v}'{' selected' if v == selected else ''}>"
                       f"{labeler(v)}</option>" for v in values)

    # roofline chunked-prefill-size choices (sglang --chunked-prefill-size);
    # default selection = --batch-tokens so the tab opens on the generated value
    chunk_opts = sorted({1024, 2048, 4096, 8192, 16384, 32768} | {a["batch_tokens"]})
    chunk_options_html = "".join(
        f"<option value='{v}'{' selected' if v == a['batch_tokens'] else ''}>{v:,}</option>"
        for v in chunk_opts)

    tp_init = pargs.tp if pargs else 8
    pp_init = pargs.pp if pargs else 1
    inst_init = pargs.instance if pargs else "p5en.48xlarge"
    tp_opts = sorted(set(int(v) for v in (pargs.tp_options if pargs else "1,2,4,8,16,32").split(",")) | {tp_init})
    pp_opts = sorted(set(int(v) for v in (pargs.pp_options if pargs else "1,2,3,4,6,8").split(",")) | {pp_init})
    ep_init = 8
    if pargs:
        ep_init = pargs.ep if pargs.ep != "auto" else (tp_init if is_moe else 1)
        ep_init = int(ep_init)

    # ---- build both languages' static fragments up front; the page ships with
    # the CLI-selected language rendered into the $-placeholders below, and the
    # rest travel in viz.frags so the in-page switcher can swap innerHTML with
    # no server round-trip.
    display_lang = get_lang()
    frags = {}
    for lang in ("zh", "en"):
        set_lang(lang)
        frags[lang] = _build_lang_fragments(
            a, cfg, p, is_moe, short, n_routed, L, dense_n, moe_n,
            instances, kv_auto, kv_choice, tot)
    set_lang(display_lang)

    # instance <option> labels need the actual selected instance re-applied
    # per language (the neutral build above leaves nothing selected)
    for lang, fr in frags.items():
        fr["instance_options"] = "".join(
            f"<option value='{n}'{' selected' if n == inst_init else ''}>"
            f"{n} · {s['count']}×{s['gpu']} {s['memGib']:g} GiB</option>"
            for n, s in (instances or {}).items())
        set_lang(lang)
        fr["instance_options"] += f"<option value='custom'>{t('html.instance_custom')}</option>"
    set_lang(display_lang)

    # ---- initial what-if payload: the page's first paint renders this baked
    # combination; every control change afterwards refetches /api/v1/whatif.
    # Lazy import: engine imports this module back at module load.
    import engine
    D0 = engine.deploy_data(a, cfg)
    fixed_gib = pargs.fixed_overhead_gib if pargs else 1.0
    D0["fixedGib"] = fixed_gib
    inst_spec = (instances or {}).get(inst_init) or {"gpu": None, "count": 8, "memGib": 80}
    P0 = {"tp": tp_init, "pp": pp_init, "ep": ep_init, "dpAttn": False,
          "memGib": inst_spec["memGib"], "gpn": inst_spec["count"],
          "ctx": a["ctx"], "req": a["requests"], "kvDtype": a["kv_dtype"],
          "frac": pargs.mem_fraction_static if pargs else 0.9,
          "fixedGib": fixed_gib}
    whatif0 = engine.whatif_payload(a, cfg, D0, P0, inst_spec["gpu"],
                                    a["batch_tokens"])

    viz = {
        # shared chrome / static-fragment state
        "lang": display_lang,
        "frags": frags,
        "model": a["model_id"],
        "whatif0": whatif0,
        "kvWarnings": a.get("weight_warnings", []) + a["kv_struct"]["warnings"],
        # model constants the renderer needs between payloads (labels, legend
        # visibility, and the static segments of the estimate bar)
        "weightsBytes": a["total_bytes"],
        "actBytes": a["act_total"],
        "visionActBytes": a.get("vision_act", 0),
        "overhead": a["overhead"],
        "weightsSlot": 3 if is_moe else 2,
        "nMoe": p["moe_layers"],
        "kvGroups": a["kv_struct"]["kv_groups"],   # sliding-legend visibility
        "linStateBytes": a["kv_struct"]["lin_state_per_req"],
        "kvIsMla": bool(p["is_mla"]),
        "kvNKvHeads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "fixedGib": fixed_gib,
        "batchTokens": a["batch_tokens"],
        "instances": instances or {},
    }
    viz_json = json.dumps(viz)

    fields = dict(frags[display_lang])
    fields.update({
        "html_lang": display_lang,
        "ctx_options": ctx_options_html,
        "req_options": req_options_html,
        "chunk_options": chunk_options_html,
        "tp_options": _popts(tp_opts, tp_init),
        "pp_options": _popts(pp_opts, pp_init),
        "ep_init": str(ep_init),
        "mem_frac_init": f"{pargs.mem_fraction_static if pargs else 0.9:g}",
        "viz_json": viz_json,
    })

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        tpl = string.Template(f.read())
    with open(SCRIPT_PATH, encoding="utf-8") as f:
        fields["app_js"] = f.read()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tpl.substitute(fields))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Estimate LLM VRAM from HF config.json")
    ap.add_argument("model_id", help="HuggingFace model ID, e.g. zai-org/GLM-5.2-FP8")
    ap.add_argument("--context", type=int, default=131_072,
                    help="context length per request (default 128K)")
    ap.add_argument("--requests", "--batch", type=int, default=16, dest="requests",
                    help="concurrent running requests (default 16)")
    ap.add_argument("--ctx-options", default="32768,65536,131072,262144,524288,1048576",
                    help="comma-separated context choices for the HTML dropdown "
                         "(default 32K,64K,128K,256K,512K,1M)")
    ap.add_argument("--req-options", default="1,8,16,32,64,128,256,512,1024",
                    help="comma-separated concurrency choices for the HTML dropdown "
                         "(default 1,8,16,32,64,128,256,512,1024)")
    ap.add_argument("--kv-dtype", choices=["auto", "bf16", "fp16", "fp8", "fp4"], default="auto",
                    help="KV cache dtype; auto mirrors SGLang: fp8 for DSA/V4-style "
                         "sparse-attention models on new GPUs, else bf16. fp4 = mxfp4 "
                         "(SGLang --kv-cache-dtype fp4_e2m1, CUDA 12.8+, incl. block-16 scale overhead)")
    ap.add_argument("--batch-tokens", type=int, default=8192,
                    help="max tokens per forward pass, for activation estimate (default 8192, ~vLLM chunked prefill)")
    ap.add_argument("--overhead", type=float, default=0.05,
                    help="extra fraction for fragmentation/CUDA context (default 5%%)")
    ap.add_argument("--html", metavar="FILE", help="also write a breakdown diagram to this HTML file")
    ap.add_argument("--no-exact", action="store_true",
                    help="skip reading safetensors headers; use formula estimate only")
    # parallel tab (design_2) initial values
    ap.add_argument("--tp", type=int, default=8, help="initial tensor-parallel size (default 8)")
    ap.add_argument("--pp", type=int, default=1, help="initial pipeline-parallel size (default 1)")
    ap.add_argument("--ep", default="auto", help="initial expert-parallel size (default auto = TP for MoE)")
    ap.add_argument("--instance", default="p5en.48xlarge",
                    help="initial AWS instance type (default p5en.48xlarge)")
    ap.add_argument("--fixed-overhead-gib", type=float, default=1.0,
                    help="per-GPU fixed overhead: CUDA context / NCCL buffers (default 1 GiB)")
    ap.add_argument("--mem-fraction-static", type=float, default=0.9,
                    help="initial mem-fraction-static for the parallel tab: fraction of GPU "
                         "memory pre-allocated as weights + KV pool, mirroring SGLang "
                         "--mem-fraction-static (default 0.9)")
    ap.add_argument("--tp-options", default="1,2,4,8,16,32")
    ap.add_argument("--pp-options", default="1,2,3,4,6,8")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="language for the terminal report and generated HTML (default zh)")
    args = ap.parse_args()
    set_lang(args.lang)

    try:
        cfg = fetch_config(args.model_id)
    except Exception as e:
        sys.exit(f"failed to fetch config for {args.model_id}: {e}")

    # multimodal configs nest the LLM under text_config; keep quantization and
    # the vision tower config (vision weights/activation are modeled too)
    if "num_hidden_layers" not in cfg and "text_config" in cfg:
        qc = cfg.get("quantization_config")
        vc = cfg.get("vision_config")
        mm_tok = cfg.get("mm_tokens_per_image")
        cfg = cfg["text_config"]
        if qc and "quantization_config" not in cfg:
            cfg["quantization_config"] = qc
        if vc and "vision_config" not in cfg:
            cfg["vision_config"] = vc
        if mm_tok and "mm_tokens_per_image" not in cfg:
            cfg["mm_tokens_per_image"] = mm_tok

    # resolve kv-dtype "auto" the way SGLang does (server_args + deepseek_v4_hook):
    # DSA/V4 sparse-attention models default to fp8_e4m3 KV, everything else
    # keeps KV in the activation dtype (bf16)
    arch = (cfg.get("architectures") or [""])[0]
    is_dsa = cfg.get("index_topk") is not None or arch in (
        "DeepseekV4ForCausalLM", "DeepseekV32ForCausalLM")
    kv_auto = "fp8" if is_dsa else "bf16"
    kv_choice = args.kv_dtype                    # what the user picked (may be "auto")
    if args.kv_dtype == "auto":
        args.kv_dtype = kv_auto
        print(f"kv-dtype auto -> {args.kv_dtype}"
              f"{t('cli.kv_dsa_note') if is_dsa else ''}",
              file=sys.stderr)

    catalog = None
    declared = None
    weight_warnings = []
    if not args.no_exact:
        try:
            catalog, declared = fetch_safetensors_catalog(args.model_id)
            got = sum(tv["bytes"] for tv in catalog.values())
            if declared and abs(got - declared) / declared > 0.01:
                msg = (f"权重总量存疑：safetensors 头求和 {got / GIB:,.1f} GiB "
                       f"vs index 声明 {declared / GIB:,.1f} GiB"
                       f"（差 {abs(got - declared) / declared:.1%}，可能有分片头未读到或含未加载张量）。")
                weight_warnings.append(msg)
                print(f"warning: {msg}", file=sys.stderr)
        except Exception as e:
            msg = f"未能读取 safetensors 头（{e}），回退到公式估算，权重为估算值。"
            weight_warnings.append(msg)
            print(f"note: {msg}", file=sys.stderr)

    a = analyze(args.model_id, cfg, args.context, args.requests, args.kv_dtype,
                args.batch_tokens, args.overhead, catalog=catalog)
    a["kv_auto"] = kv_auto
    a["kv_choice"] = kv_choice
    a["index_total"] = declared            # official total_size for the Σ cross-check
    if a.get("absorb_per_layer"):
        n_mtp = cfg.get("num_nextn_predict_layers", 0) or 0
        full = a["absorb_per_layer"] * (cfg["num_hidden_layers"] + n_mtp)
        weight_warnings.append(
            f"MLA 权重吸收：SGLang 加载时将 kv_b_proj 反量化为 bf16 w_kc/w_vc（fp8 原件保留），"
            f"全尺寸 ≈ {full / GIB:.2f} GiB，随 attention-TP 切分（纯 TP÷tp；dp-attention 每卡整份）。"
            f"未计入上方 safetensors 权重表；并行 tab 已计入。")
    a["weight_warnings"] = weight_warnings
    report(a)
    if args.html:
        ctx_options = [int(v) for v in args.ctx_options.split(",")]
        req_options = [int(v) for v in args.req_options.split(",")]
        layer = per_layer_breakdown(a, cfg)
        parallel_self_check(a, layer, cfg)
        instances = fetch_instance_specs(INSTANCE_TYPES)
        if args.instance not in instances:
            sys.exit(f"unknown instance {args.instance}; known: {', '.join(instances)}")
        render_html(a, args.html, ctx_options, req_options,
                    layer=layer, instances=instances, pargs=args)
        print(f"diagram written to {args.html}")


if __name__ == "__main__":
    main()
