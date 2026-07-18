#!/usr/bin/env python3
"""Bilingual (zh/en) string tables for the terminal report and generated HTML.

User-facing text that isn't already plain English lives here as
{"zh": ..., "en": ...} pairs, keyed by a short dotted name. `t(key, **kwargs)`
looks up the string for the language set via `set_lang()` and fills in
`{name}`-style placeholders. Each language gets its own literal template
(not a word-for-word substitution) so punctuation/spacing/word order can
differ naturally between zh and en.
"""

MSG = {
    # ---- dtype labelling (dtype_label)
    "dtype.quant_scale": {"zh": "量化scale", "en": "quant scale"},
    "dtype.mixed_prefix": {"zh": "混合精度: ", "en": "mixed: "},

    # ---- activation workspace description (activation_bytes)
    "act.desc": {
        "zh": "每 token ≈ 2B × (8×{H} + 2×{inter_eff}) = {per_token_kib} KiB"
              "（残差/attn 缓冲 + FFN 中间层{moe_note}）",
        "en": "per token ≈ 2B × (8×{H} + 2×{inter_eff}) = {per_token_kib} KiB"
              " (residual/attn buffers + FFN intermediate{moe_note})",
    },
    "act.desc_moe_note": {"zh": "，MoE 按 top-k 有效宽度", "en": ", MoE uses top-k effective width"},

    # ---- roofline kernel labels are fixed English technical terms (like their
    # siblings "DSA indexer", "Dense FFN"); no i18n needed — the kernels array
    # is baked once at generation time and consumed directly by chart JS.

    # ---- KV cache per-token description (kv_per_token_elems)
    "kv.mla_desc": {"zh": "MLA latent：kv_lora {kv_lora} + rope {rope} = {elems}/token/layer",
                     "en": "MLA latent: kv_lora {kv_lora} + rope {rope} = {elems}/token/layer"},
    "kv.gqa_desc": {"zh": "{kind}：2 × {n_kv} × {head_dim} = {elems}/token/layer",
                     "en": "{kind}: 2 x {n_kv} x {head_dim} = {elems}/token/layer"},
    "kv.kind_mha": {"zh": "MHA", "en": "MHA"},
    "kv.kind_gqa": {"zh": "GQA（{n_kv} kv heads）", "en": "GQA ({n_kv} kv heads)"},

    # ---- parallel-tab left column (build_parallel_struct)
    "pstruct.emb_lmhead_shared": {"zh": "Embedding + LM Head（共享）", "en": "Embedding + LM Head (shared)"},
    "pstruct.moe_layer_block": {"zh": "L{dense_n} · Attention + MoE FFN（{n_routed} 专家）",
                                 "en": "L{dense_n} · Attention + MoE FFN ({n_routed} experts)"},
    "pstruct.moe_ellipsis": {"zh": "⋯ MoE 层 ×{moe_n}（L{lo}–L{hi}）", "en": "⋯ MoE layers ×{moe_n} (L{lo}–L{hi})"},
    "pstruct.same_layers_ellipsis": {"zh": "⋯ 共 {L} 层（每层结构相同）", "en": "⋯ {L} layers total (identical structure)"},
    "pstruct.mtp_block": {"zh": "MTP 预测层 ×{n_mtp}", "en": "MTP predict layer ×{n_mtp}"},
    "pstruct.lmhead_block": {"zh": "LM head 层", "en": "LM head layer"},

    # ---- estimate-tab left structure column (render_html: struct)
    "struct.embed_entry": {"zh": "embed（入口）", "en": "embed (input)"},
    "struct.dense_prefix_ellipsis": {"zh": "⋮ 前 {dense_n} 层：FFN 是 Dense（L0–L{last}）",
                                      "en": "⋮ first {dense_n} layers: FFN is Dense (L0–L{last})"},
    "struct.moe_ellipsis": {"zh": "⋮ {which} {moe_n} 层：FFN 是 MoE，{n_routed} 专家（L{lo}–L{hi}）",
                             "en": "⋮ {which} {moe_n} layers: FFN is MoE, {n_routed} experts (L{lo}–L{hi})"},
    "struct.which_rest": {"zh": "后", "en": "last"},
    "struct.which_all": {"zh": "全部", "en": "all"},
    "struct.same_layers_ellipsis": {"zh": "⋮ 共 {L} 层（每层结构相同）", "en": "⋮ {L} layers total (identical structure)"},
    "struct.mtp_entry": {"zh": "MTP 预测层 ×{n_mtp}", "en": "MTP predict layer ×{n_mtp}"},
    "struct.lmhead_exit": {"zh": "lm_head（出口）", "en": "lm_head (output)"},
    "struct.lmhead_shared": {"zh": "lm_head（与 embed 共享）", "en": "lm_head (shared with embed)"},

    # ---- static weight cards (render_html)
    "card.embed_lmhead_line": {"zh": "{mult} {vocab}（词表）× {H}", "en": "{mult} {vocab} (vocab) × {H}"},
    "card.embed_tied_mult": {"zh": "1 ×（共享）", "en": "1 × (shared)"},
    "card.attention_title": {"zh": "① Attention 子层 — {L} 层每层都有", "en": "① Attention sublayer — every one of {L} layers"},
    "card.attention_params_line": {"zh": "共 {params} 参数 × {L} 层{dt}", "en": "{params} params total × {L} layers{dt}"},
    "card.attention_mla_line": {
        "zh": "MLA 低秩：q_lora={q_lora}, kv_lora={kv_lora}, {heads} head（qk {qk_nope}+{qk_rope} / v {v_dim}）",
        "en": "MLA low-rank: q_lora={q_lora}, kv_lora={kv_lora}, {heads} heads (qk {qk_nope}+{qk_rope} / v {v_dim})",
    },
    "card.attention_gqa_line": {"zh": "{heads} q head / {n_kv} kv head，head_dim {hd}{extra}",
                                 "en": "{heads} q heads / {n_kv} kv heads, head_dim {hd}{extra}"},
    "card.attention_lora_extra": {"zh": "，q/o 低秩分解", "en": ", q/o low-rank factorized"},
    "card.indexer_title": {"zh": "attn indexer（DSA 稀疏注意力索引）", "en": "attn indexer (DSA sparse-attention index)"},
    "card.indexer_line": {"zh": "index_n_heads={n_i}, index_head_dim={d_i}，省的是计算，非存储",
                           "en": "index_n_heads={n_i}, index_head_dim={d_i} — saves compute, not memory"},
    "card.dense_ffn_title": {"zh": "② Dense FFN（前 {dense_n} 层）", "en": "② Dense FFN (first {dense_n} layers)"},
    "card.dense_ffn_line": {"zh": "3 矩阵 × {H} × {inter}（胖）× {dense_n} 层", "en": "3 matrices × {H} × {inter} (wide) × {dense_n} layers"},
    "card.moe_routed_line": {"zh": "routed experts：{n_routed} × 3 矩阵 × {H} × {moe_inter}（瘦）× {moe_n} 层 = <b>{gib} GiB</b>{dt}",
                              "en": "routed experts: {n_routed} × 3 matrices × {H} × {moe_inter} (narrow) × {moe_n} layers = <b>{gib} GiB</b>{dt}"},
    "card.moe_shared_line": {"zh": "shared expert（每层 {n_shared} 个 × {moe_n}）= {gib} GiB{dt}",
                              "en": "shared expert ({n_shared} per layer × {moe_n}) = {gib} GiB{dt}"},
    "card.moe_active_line": {"zh": "每 token 只激活 top-{topk}（~{active}B 计算）—— 显存按 {n_routed} 存（大），计算按 {topk} 跑（省）",
                              "en": "only top-{topk} activated per token (~{active}B compute) — memory sized for all {n_routed} (large), compute runs top-{topk} (cheap)"},
    "card.moe_ffn_title": {"zh": "② MoE FFN（{scope}）★ 绝对大头", "en": "② MoE FFN ({scope}) ★ the big one"},
    "card.moe_scope_last": {"zh": "后 {moe_n} 层", "en": "last {moe_n} layers"},
    "card.moe_scope_all": {"zh": "全部 {moe_n} 层", "en": "all {moe_n} layers"},
    "card.ffn_title_dense": {"zh": "② FFN（{L} 层）★ 大头", "en": "② FFN ({L} layers) ★ the big one"},
    "card.ffn_line_dense": {"zh": "3 矩阵 × {H} × {inter} × {L} 层", "en": "3 matrices × {H} × {inter} × {L} layers"},
    "card.mtp_title": {"zh": "MTP（multi-token prediction）×{n_mtp}", "en": "MTP (multi-token prediction) ×{n_mtp}"},
    "card.mtp_line": {"zh": "一整套额外的 Attention + FFN，投机解码用；不需要可不加载",
                       "en": "a full extra Attention + FFN, used for speculative decoding; skip loading it if unneeded"},
    "card.norms_title": {"zh": "norms & 其他（layernorm、hyper-connection、量化 scale 等）",
                          "en": "norms & misc (layernorm, hyper-connection, quant scales, etc.)"},
    "card.dtype_storage_note": {"zh": "（{label} 存储）", "en": " ({label} storage)"},

    # ---- runtime cards
    "card.kv_title": {"zh": "KV Cache — context <span id='d-ctx'>…</span> × <span id='d-req'>…</span> 并发请求",
                       "en": "KV Cache — context <span id='d-ctx'>…</span> × <span id='d-req'>…</span> concurrent requests"},
    "card.kv_derivation_line": {
        "zh": "每 token（全 {L} 层）= <span id='d-kv-per-tok'>…</span> KiB → 单请求（context <span id='d-ctx2'>…</span>）"
              "= <b><span id='d-kv-per-req'>…</span> GiB</b> → × <span id='d-req2'>…</span> 并发 = <b><span id='d-kv-total2'>…</span> GiB</b>",
        "en": "per token (all {L} layers) = <span id='d-kv-per-tok'>…</span> KiB → per request (context <span id='d-ctx2'>…</span>) "
              "= <b><span id='d-kv-per-req'>…</span> GiB</b> → × <span id='d-req2'>…</span> concurrent = <b><span id='d-kv-total2'>…</span> GiB</b>",
    },
    "card.mha_compare_line": {
        "zh": "假如用 MHA 全存：<b><span id='d-mha'>…</span> GiB</b> —— MLA 压成 {kv_lora} 维共享 latent，省 <b>{ratio}×</b>",
        "en": "if stored as full MHA: <b><span id='d-mha'>…</span> GiB</b> — MLA compresses to a {kv_lora}-dim shared latent, saving <b>{ratio}×</b>",
    },
    "card.act_title": {"zh": "Activation 工作区 — {batch_tokens} tokens/forward", "en": "Activation workspace — {batch_tokens} tokens/forward"},
    "card.act_line2": {"zh": "一次 forward 最多 {batch_tokens} tokens（chunked prefill 上限），逐层执行，只有当前层的中间结果存活",
                        "en": "one forward pass handles at most {batch_tokens} tokens (chunked-prefill cap); layers execute sequentially, only the current layer's intermediates are live"},
    "card.act_line3": {"zh": "工作区估算（vLLM profile 的量级），与请求数无关、与单次批处理 token 数有关",
                        "en": "workspace estimate (same order as vLLM's profiler); independent of request count, depends on tokens per forward batch"},

    # ---- stacked bars / totals
    "bar.weights_static_label": {"zh": "权重（静态）", "en": "Weights (static)"},
    "bar.other_label": {"zh": "其它", "en": "other"},
    "bar.pct_paren": {"zh": "（{pct:.0f}%）", "en": " ({pct:.0f}%)"},
    "bar.total_head": {"zh": "静态 + 动态 · 部署总占用 ≈ {gib} GiB（context {ctx} × {req} 并发）",
                        "en": "Static + dynamic · total deployment footprint ≈ {gib} GiB (context {ctx} × {req} concurrent)"},
    "bar.total_line": {
        "zh": "权重 {w}（静态） + KV {kv} + Activation {act}（动态） + 碎片 ~{ov}% ≈ <b>{grand} GiB</b>"
              "<span class='pct'>　·　KV 随 context × 并发线性增长：每并发 +{kv_per_req} GiB</span>",
        "en": "Weights {w} (static) + KV {kv} + Activation {act} (dynamic) + ~{ov}% fragmentation ≈ <b>{grand} GiB</b>"
              "<span class='pct'> · KV scales linearly with context × concurrency: +{kv_per_req} GiB per concurrent request</span>",
    },

    # ---- filter row / static labels (template.html)
    "html.lbl_ctx": {"zh": "context / 请求", "en": "context / request"},
    "html.lbl_req": {"zh": "并发请求数", "en": "concurrent requests"},
    "html.lbl_kv": {"zh": "KV cache 精度", "en": "KV cache dtype"},
    "html.lbl_dp": {"zh": "DP attention（KV 切 TP，attention 权重复制）", "en": "DP attention (KV split by TP, attention weights replicated)"},
    "html.lbl_inst": {"zh": "机型", "en": "instance"},
    "html.lbl_frac": {"zh": "mem-fraction-static", "en": "mem-fraction-static"},
    "html.lbl_custom_mem": {"zh": "单卡", "en": "per GPU"},
    "html.lbl_custom_gpn": {"zh": "每节点", "en": "per node"},
    "html.lbl_custom_cards": {"zh": "卡", "en": "GPUs"},
    "html.fnote": {"zh": "KV Cache 与总占用随选择实时重算；权重（静态）部分不变",
                   "en": "KV cache and totals recompute live on selection; weights (static) stay fixed"},
    "html.tab_estimate": {"zh": "显存拆解", "en": "Memory Breakdown"},
    "html.tab_parallel": {"zh": "并行切分", "en": "Parallel Sharding"},
    "html.tab_roofline": {"zh": "性能 Roofline", "en": "Perf Roofline"},
    "html.grp_static": {"zh": "静态 · 模型权重（加载即占用，与流量无关）", "en": "Static · Model Weights (paid at load, independent of traffic)"},
    "html.grp_runtime": {"zh": "动态 · 运行时内存（随 context × 并发请求增长）", "en": "Dynamic · Runtime Memory (grows with context × concurrent requests)"},
    "html.details_table_all": {"zh": "表格视图（全部部件）", "en": "Table view (all components)"},
    "html.th_component": {"zh": "部件", "en": "component"},
    "html.th_dtype": {"zh": "精度", "en": "dtype"},
    "html.th_params": {"zh": "参数量", "en": "params"},
    "html.th_share": {"zh": "占比", "en": "share"},
    "html.parallel_arrow": {"zh": "⟶ 模型加载（TP/PP/EP 切分）⟶", "en": "⟶ model loading (TP/PP/EP sharded) ⟶"},
    "html.details_table_gpu": {"zh": "表格视图（每卡显存明细）", "en": "Table view (per-GPU memory detail)"},

    # ---- render_html top-level fields
    "html.page_title": {"zh": "{short} 显存拆解与并行切分", "en": "{short} VRAM Breakdown & Parallel Sharding"},
    "html.title": {"zh": "{short}（{b}B{moe_tag}）：显存拆解 —— 权重 {gib} GiB + 运行时 <span id='d-runtime'>…</span> GiB",
                   "en": "{short} ({b}B{moe_tag}): VRAM Breakdown — weights {gib} GiB + runtime <span id='d-runtime'>…</span> GiB"},
    "html.meta": {"zh": "{model_id} · {arch} · hidden {H} · {heads} heads · vocab {vocab} · activation 按 {batch_tokens} tokens/forward · {est_note} · AWS 机型规格来自 describe-instance-types，H800/H20 为内置静态表",
                  "en": "{model_id} · {arch} · hidden {H} · {heads} heads · vocab {vocab} · activation @ {batch_tokens} tokens/forward · {est_note} · AWS instance specs from describe-instance-types; H800/H20 from a built-in static table"},
    "html.meta_exact": {"zh": "权重为 safetensors 精确值", "en": "weights are exact from safetensors"},
    "html.meta_formula": {"zh": "由 config.json 解析式估算", "en": "weights estimated from config.json formula"},
    "html.struct_title": {"zh": "{L} 层 Transformer（每层：上 Attention + 下 FFN）", "en": "{L}-layer Transformer (per layer: Attention above, FFN below)"},
    "html.weights_bar_head": {"zh": "静态 · 权重合计 ≈ {gib} GiB（{wname}，{bpp:.2f} 字节/参数{exact_note}）",
                               "en": "Static · total weights ≈ {gib} GiB ({wname}, {bpp:.2f} bytes/param{exact_note})"},
    "html.weights_exact_note": {"zh": "，safetensors 精确值", "en": ", exact from safetensors"},
    "html.total_bar_head": {"zh": "静态 + 动态 · 部署总占用 ≈ {gib} GiB（context {ctx} × {req} 并发）",
                             "en": "Static + dynamic · total deployment footprint ≈ {gib} GiB (context {ctx} × {req} concurrent)"},
    "html.subtitle_moe": {"zh": "每层 = ① Attention + ② FFN；前 {dense_n} 层 Dense、后 {moe_n} 层 MoE —— {pct:.0%} 的权重压在 MoE 专家上",
                           "en": "each layer = ① Attention + ② FFN; first {dense_n} layers Dense, last {moe_n} layers MoE — {pct:.0%} of weights sit in MoE experts"},
    "html.subtitle_dense": {"zh": "每层 = ① Attention + ② FFN，共 {L} 层", "en": "each layer = ① Attention + ② FFN, {L} layers total"},
    "html.kv_auto_label": {"zh": "auto（{kv_auto}）", "en": "auto ({kv_auto})"},
    "html.instance_custom": {"zh": "自定义…", "en": "Custom…"},
    "html.pstruct_title": {"zh": "{L} 层 Transformer", "en": "{L}-layer Transformer"},

    # ---- main() console notes
    "cli.kv_dsa_note": {"zh": "（DSA/稀疏注意力模型，对齐 SGLang 默认 fp8_e4m3）", "en": " (DSA/sparse-attention model, matches SGLang default fp8_e4m3)"},
}

_lang = "zh"


def set_lang(lang: str) -> None:
    global _lang
    _lang = lang


def get_lang() -> str:
    return _lang


def t(key: str, **kwargs) -> str:
    entry = MSG[key]
    s = entry.get(_lang, entry["zh"])
    return s.format(**kwargs) if kwargs else s
