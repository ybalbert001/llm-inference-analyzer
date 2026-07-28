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
        "zh": "每 token ≈ {unshard_kib} KiB 不随 TP 分片（hidden 缓冲{moe_note}{dsa_note}）"
              " + {shard_kib} KiB ÷ TP（FFN/expert 中间层）+ 底数 {base_gib} GiB/卡；"
              "含 ×{mult} 运行时系数（B200 实测标定 1.4–1.9）{mtp_note}",
        "en": "per token ≈ {unshard_kib} KiB unsharded across TP (hidden buffers{moe_note}{dsa_note})"
              " + {shard_kib} KiB ÷ TP (FFN/expert intermediate) + {base_gib} GiB/GPU floor;"
              " includes a ×{mult} runtime factor (measured 1.4–1.9 on B200){mtp_note}",
    },
    "act.desc_mtp_note": {
        "zh": "，另 ×1.15 MTP/投机解码系数（模型带 MTP 层，默认按开启计）",
        "en": " and ×1.15 for MTP/speculative decoding (model ships MTP layers, assumed on)",
    },
    "act.desc_moe_note": {"zh": "、MoE dispatch 每 token 复制 top-k+shared 份",
                          "en": ", MoE dispatch copies each token top-k+shared times"},
    "act.desc_dsa_note": {"zh": "、DSA indexer 工作区", "en": ", DSA indexer workspace"},

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
    "kv.indexer_suffix": {
        "zh": "；+ DSA indexer 索引缓存 fp8×{d_i} + scale = {b} B/token/layer（不随 kv-dtype 变）",
        "en": "; + DSA indexer index-key cache fp8×{d_i} + scale = {b} B/token/layer (kv-dtype-independent)"},

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
    "struct.vision_entry": {"zh": "vision tower（ViT {Lv} 层）+ projector（图像入口）",
                            "en": "vision tower (ViT {Lv} layers) + projector (image input)"},
    "struct.lmhead_exit": {"zh": "lm_head（出口）", "en": "lm_head (output)"},
    "struct.lmhead_shared": {"zh": "lm_head（与 embed 共享）", "en": "lm_head (shared with embed)"},
    # attention-type annotations on the layer's attention sub-block
    "struct.attn_swa": {"zh": " · 滑窗 KV≤{window}", "en": " · sliding KV≤{window}"},
    "struct.attn_dsa": {"zh": " · DSA 稀疏读 top-{topk}", "en": " · DSA sparse read top-{topk}"},
    "struct.attn_sparse": {"zh": " · 块稀疏读 top-{cap}", "en": " · block-sparse read top-{cap}"},
    # hybrid models: linear-attention layers are a structurally distinct block
    "struct.linear_block": {"zh": "① Linear attention（定长 state，无 KV）",
                            "en": "① Linear attention (fixed state, no KV)"},
    "struct.hybrid_ellipsis": {"zh": "⋮ 共 {L} 层：{n_full} 层 full attention（存 KV）+ {n_linear} 层 linear（无 KV）交替",
                               "en": "⋮ {L} layers: {n_full} full-attention (KV) + {n_linear} linear (no KV), interleaved"},
    "struct.swa_ellipsis": {"zh": "⋮ 共 {L} 层：{n_full} 层全局 + {n_sliding} 层滑窗（KV≤{window}）",
                            "en": "⋮ {L} layers: {n_full} global + {n_sliding} sliding (KV≤{window})"},

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
    "card.indexer_line": {"zh": "index_n_heads={n_i}, index_head_dim={d_i}，省 attention 计算；"
                                "另有 index-key 缓存 {d_i}+4 B/token/layer 计入 KV cell",
                           "en": "index_n_heads={n_i}, index_head_dim={d_i} — saves attention compute; "
                                 "its index-key cache adds {d_i}+4 B/token/layer to the KV cell"},
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
    "card.vision_title": {"zh": "vision tower + projector（多模态图像入口）",
                          "en": "vision tower + projector (multimodal image input)"},
    "card.vision_line": {"zh": "ViT {Lv} 层 × (attention 4H² + MLP 2×{H}×{inter}) + patch embed + projector；量化 checkpoint 中也保持 bf16",
                         "en": "ViT {Lv} layers × (attention 4H² + MLP 2×{H}×{inter}) + patch embed + projector; stays bf16 even in quantized checkpoints"},
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
    "card.lin_title": {"zh": "Linear/SSM state — {n_linear} 层定长 state × <span id='d-lin-req'>…</span> 并发",
                        "en": "Linear/SSM state — {n_linear} layers × <span id='d-lin-req'>…</span> concurrent requests"},
    "card.lin_line": {"zh": "每请求 {mib} MiB（conv + ssm state，{n_linear} 层合计）——随并发增长，与 context 无关",
                       "en": "{mib} MiB per request (conv + ssm state over {n_linear} layers) — grows with concurrency, independent of context"},
    "card.lin_line2": {"zh": "启动时随静态区一次性预分配（SGLang mamba 池），真实并发只占用/释放槽位；此处按 槽位数 = 并发数 的最小需求计，"
                             "SGLang 默认启发式可能预分配更多槽（建议显式设 --max-mamba-cache-size）",
                        "en": "pre-allocated once at startup with the static region (SGLang mamba pool); real traffic only claims/releases slots. "
                              "Sized as slots = concurrency (minimum need) — SGLang's default heuristic may pre-allocate more "
                              "(set --max-mamba-cache-size explicitly when deploying)"},
    "card.act_title": {"zh": "Activation 工作区 — {batch_tokens} tokens/forward", "en": "Activation workspace — {batch_tokens} tokens/forward"},
    "card.act_line2": {"zh": "一次 forward 最多 {batch_tokens} tokens（chunked prefill 上限），逐层执行，只有当前层的中间结果存活",
                        "en": "one forward pass handles at most {batch_tokens} tokens (chunked-prefill cap); layers execute sequentially, only the current layer's intermediates are live"},
    "card.act_line3": {"zh": "工作区估算（vLLM profile 的量级），与请求数无关、与单次批处理 token 数有关",
                        "en": "workspace estimate (same order as vLLM's profiler); independent of request count, depends on tokens per forward batch"},
    "card.vision_act_title": {"zh": "Vision encoder 工作区 — ViT {Lv} 层编码一张最大图",
                              "en": "Vision encoder workspace — ViT {Lv} layers encoding one max-size image"},
    "card.vision_act_line": {"zh": "每图最多 {patches} patches × ≈{kib} KiB/patch（残差/attn 缓冲 + MLP 中间层，逐层执行）——编码期瞬时占用，非常驻",
                             "en": "up to {patches} patches per image × ≈{kib} KiB/patch (residual/attn buffers + MLP intermediate, layer-by-layer) — transient during encoding, not resident"},
    "card.vision_act_line2": {"zh": "merge {merge}:1 后每图 {tokens} 个图像 token 进入 KV cache —— 与文本 token 同 cell、占用 context 位置，KV 无需单列",
                              "en": "after {merge}:1 merging each image yields {tokens} image tokens into the KV cache — same cell as text tokens, consuming context positions; no separate KV pool"},

    # ---- stacked bars / totals
    "bar.weights_static_label": {"zh": "权重（静态）", "en": "Weights (static)"},
    "bar.other_label": {"zh": "其它", "en": "other"},
    "bar.pct_paren": {"zh": "（{pct:.0f}%）", "en": " ({pct:.0f}%)"},
    "bar.total_head": {"zh": "静态 + 动态 · 部署总占用 ≈ {gib} GiB（context {ctx} × {req} 并发）",
                        "en": "Static + dynamic · total deployment footprint ≈ {gib} GiB (context {ctx} × {req} concurrent)"},
    "bar.lin_part": {"zh": " + linear state {lin}", "en": " + linear state {lin}"},
    "bar.vision_act_label": {"zh": "Vision encoder", "en": "Vision encoder"},
    "bar.total_line": {
        "zh": "权重 {w}（静态） + KV {kv}{lin_part} + Activation {act}（动态） + 碎片 ~{ov}% ≈ <b>{grand} GiB</b>"
              "<span class='pct'>　·　KV 随 context × 并发线性增长：每并发 +{kv_per_req} GiB</span>",
        "en": "Weights {w} (static) + KV {kv}{lin_part} + Activation {act} (dynamic) + ~{ov}% fragmentation ≈ <b>{grand} GiB</b>"
              "<span class='pct'> · KV scales linearly with context × concurrency: +{kv_per_req} GiB per concurrent request</span>",
    },

    # ---- filter row / static labels (template.html)
    "html.lbl_ctx": {"zh": "context / 请求", "en": "context / request"},
    "html.lbl_req": {"zh": "并发请求数", "en": "concurrent requests"},
    "html.lbl_kv": {"zh": "KV cache 精度", "en": "KV cache dtype"},
    "html.lbl_dp": {"zh": "DP attention（KV 切 TP，attention 权重复制）", "en": "DP attention (KV split by TP, attention weights replicated)"},
    "html.lbl_dense_repl": {"zh": "dense 复制（moe_dense_tp=1，前置 dense 层每卡整份）", "en": "dense replicate (moe_dense_tp=1, leading dense layers kept whole per rank)"},
    "html.lbl_cp": {"zh": "Prefill CP（--enable-prefill-cp）", "en": "Prefill CP (--enable-prefill-cp)"},
    "html.cp_title": {"zh": "开启后按最常用的 attn_cp_size=tp 建模：attention 权重每卡整份复制、序列 KV 跨全部 TP 卡分摊（等价 dp-attention 显存，但可服务单卡装不下的超长 context），并强制 dp-attention + moe_dense_tp=1。不建模 1<cp<tp 的部分复制中间态。",
                       "en": "When on, modeled at the most common attn_cp_size=tp: attention weights replicated per rank, one sequence's KV split across all TP ranks (same per-rank memory as dp-attention, but serves a single context too long for one GPU), forcing dp-attention + moe_dense_tp=1. The 1<cp<tp partial-replication middle ground is not modeled."},
    "html.lbl_inst": {"zh": "机型", "en": "instance"},
    "html.lbl_frac": {"zh": "mem-fraction-static", "en": "mem-fraction-static"},
    "html.lbl_wdtype": {"zh": "权重精度", "en": "weight dtype"},
    "html.wdtype_model_label": {"zh": "{wdtype}（本模型，实测字节）", "en": "{wdtype} (this model, exact bytes)"},
    "html.lbl_chunk": {"zh": "chunked-prefill-size", "en": "chunked-prefill-size"},
    "html.lbl_custom_mem": {"zh": "单卡", "en": "per GPU"},
    "html.lbl_custom_gpn": {"zh": "每节点", "en": "per node"},
    "html.lbl_custom_cards": {"zh": "卡", "en": "GPUs"},
    "html.fnote": {"zh": "KV Cache 与总占用随选择实时重算；权重（静态）部分不变",
                   "en": "KV cache and totals recompute live on selection; weights (static) stay fixed"},
    "html.tab_evidence": {"zh": "推导依据", "en": "How We Know"},
    "html.tab_estimate": {"zh": "显存拆解", "en": "Memory Breakdown"},
    "html.tab_parallel": {"zh": "并行切分", "en": "Parallel Sharding"},
    "html.tab_roofline": {"zh": "性能 Roofline", "en": "Perf Roofline"},

    # ---- evidence tab · section 0: the raw source files beside parsed facts
    "ev.raw_title": {"zh": "原始证据 · 一切数字的三个来源文件",
                      "en": "Primary evidence · the three source files behind every number"},
    "ev.raw_note": {"zh": "左侧是我们读取的三个原始文件，串成一条链：config.json 定义结构 → index.json 把每个张量映射到分片（桥接）→ safetensors 存实际权重字节。右侧是工具从中解析出的事实。左右可拖动分隔条调整宽度，文件区可折叠。张量太多，只按结构挑代表性的层展示（重复的同构层与专家已省略）。",
                     "en": "Left: the three raw files we read, forming a chain: config.json defines the structure → index.json maps each tensor to its shard (the bridge) → safetensors stores the actual weight bytes. Right: the facts the tool parsed out of them. Drag the divider to resize; the file panes collapse. There are too many tensors to list, so we show one representative layer per distinct structure (duplicate identical layers and experts are elided)."},
    "ev.raw_title2": {"zh": "原始证据 · 一切数字的两个来源文件",
                       "en": "Primary evidence · the two source files behind every number"},
    "ev.raw_note2": {"zh": "左侧是我们读取的两个原始文件：config.json 定义结构、safetensors 存实际权重字节（本模型为单文件权重，无 index.json 分片目录）。右侧是工具从中解析出的事实。左右可拖动分隔条调整宽度，文件区可折叠。",
                      "en": "Left: the two raw files we read — config.json defines the structure, safetensors stores the actual weight bytes (this model ships as a single weight file, with no index.json shard directory). Right: the facts the tool parsed out of them. Drag the divider to resize; the file panes collapse."},
    "ev.raw_left_title": {"zh": "① 原始文件（客户可自行核对）", "en": "① Raw files (independently verifiable)"},
    "ev.raw_right_title": {"zh": "② 工具解析出的事实", "en": "② What the tool parsed"},
    "ev.raw_cfg_summary": {"zh": "config.json — 模型结构定义（关键字段）", "en": "config.json — model structure (key fields)"},
    "ev.raw_cfg_full": {"zh": "展开完整 config.json 原文", "en": "Expand full config.json"},
    "ev.raw_idx_summary": {"zh": "model.safetensors.index.json — 桥接文件（张量 → 分片 的目录）",
                           "en": "model.safetensors.index.json — bridge file (tensor → shard directory)"},
    "ev.raw_idx_bridge_note": {"zh": "承上启下：上接 config 定义的结构，下接实际权重分片。metadata.total_size 是全模型总字节；weight_map 把每个张量名映射到它所在的分片文件。",
                               "en": "The bridge: it links the config-defined structure to the actual weight shards. metadata.total_size is the whole-model byte total; weight_map maps each tensor name to the shard file that holds it."},
    "ev.raw_idx_metalab": {"zh": "metadata（全局）", "en": "metadata (global)"},
    "ev.raw_idx_wmaplab": {"zh": "weight_map（张量 → 分片，示例 {shown}/{total:,}）",
                           "en": "weight_map (tensor → shard, showing {shown}/{total:,})"},
    "ev.raw_idx_shardlab": {"zh": "分片文件（{n} 个）", "en": "shard files ({n})"},
    "ev.raw_st_summary": {"zh": "safetensors — 权重文件张量清单（代表性采样）",
                          "en": "safetensors — weight tensor listing (representative sample)"},
    "ev.raw_idx_meta": {"zh": "index.json 声明：total_size = {total} GiB · {shards} 个分片 · {tensors:,} 个张量",
                        "en": "index.json declares: total_size = {total} GiB · {shards} shards · {tensors:,} tensors"},
    "ev.raw_idx_meta_noshard": {"zh": "index.json 声明：total_size = {total} GiB · {tensors:,} 个张量",
                                "en": "index.json declares: total_size = {total} GiB · {tensors:,} tensors"},
    "ev.raw_st_globals": {"zh": "全局张量（embed / 最终 norm / lm_head）", "en": "global tensors (embed / final norm / lm_head)"},
    "ev.raw_st_expertnote": {"zh": "（专家 {kept}/{total}，其余 {rest} 个同构专家已省略）",
                             "en": "(experts {kept}/{total}; the other {rest} identical experts elided)"},
    "ev.raw_st_samenote": {"zh": "· 另有 {n} 层与此结构相同", "en": "· {n} layers share this structure"},
    "ev.raw_st_trunc": {"zh": "（还有 {n} 种结构不同的层未展示）", "en": "({n} more structurally-distinct layers not shown)"},
    "ev.raw_col_tensor": {"zh": "张量名", "en": "tensor"},
    "ev.raw_col_dtype": {"zh": "精度", "en": "dtype"},
    "ev.raw_col_shape": {"zh": "形状", "en": "shape"},
    "ev.raw_col_bytes": {"zh": "字节", "en": "bytes"},
    "ev.raw_fact_arch": {"zh": "架构", "en": "architecture"},
    "ev.raw_fact_struct": {"zh": "结构", "en": "structure"},
    "ev.raw_fact_layers": {"zh": "{L} 层", "en": "{L} layers"},
    "ev.raw_fact_dtypes": {"zh": "权重精度构成（按文件真实字节，未做 sub-byte 还原）",
                           "en": "weight dtype mix (file-literal bytes, no sub-byte unpacking)"},
    "ev.raw_fact_scale_note": {"zh": "含 fp8/fp4 量化的缩放因子（scale）——每块权重配一个，属存储开销非模型参数",
                               "en": "includes fp8/fp4 scale factors — one per block, storage overhead not model params"},
    "ev.raw_sigma_ok": {"zh": "Σ 逐张量字节加总 = {sigma} GiB　✓ 与 index.json 声明的 total_size {idx} GiB 一致（偏差 {dev}）",
                        "en": "Σ over all tensor bytes = {sigma} GiB　✓ matches index.json total_size {idx} GiB (dev {dev})"},
    "ev.raw_sigma_line": {"zh": "Σ 逐张量字节加总 = {sigma} GiB", "en": "Σ over all tensor bytes = {sigma} GiB"},
    "ev.raw_note_globals": {"zh": "全局", "en": "global"},

    # ---- evidence tab (build_evidence): why the weight numbers are trustworthy
    # fused A+B: one row per component (config it uses → formula → param count)
    "ev.sec_ab_title": {"zh": "A · 从 config 到参数量（每部件：用到哪些配置 → 套什么公式 → 得出多少参数）",
                         "en": "A · From config to parameter count (per component: which fields → which formula → how many params)"},
    "ev.sec_ab_note": {"zh": "每行 = 一个部件：左列是它在公式里用到的 config 字段（悬停看含义），右列套结构公式：符号 → 代入 config 数值 → 参数量。顶部一条是所有公式共用的全局输入。",
                        "en": "Each row = one component: the left column lists the config fields its formula consumes (hover a chip for its meaning); the right column applies the structural formula: symbolic → config substituted → param count. The band on top holds the global inputs every formula shares."},
    "ev.grp_global": {"zh": "全局输入（所有公式共用）", "en": "global inputs (shared by all formulas)"},
    "ev.col_component": {"zh": "部件", "en": "component"},
    "ev.col_config": {"zh": "用到的 config（悬停看含义）", "en": "config used (hover for meaning)"},
    "ev.col_paramformula": {"zh": "参数量公式：符号 = 代入 = 结果", "en": "param formula: symbolic = substituted = result"},
    "ev.sec_c_title": {"zh": "B · 双来源对账（独立第二证据）★",
                        "en": "B · Two-source reconciliation (independent second evidence) ★"},
    "ev.sec_c_note": {"zh": "每个部件有两个独立算出的参数量：① 上面的 config 公式；② 直接读真实权重文件（safetensors）的张量形状。两条等长 = 对上了 = 可信。",
                       "en": "Each component has two independently derived parameter counts: ① the config formula above; ② the real tensor shapes read from the weight files (safetensors). Two equal bars = they agree = trustworthy."},
    "ev.params": {"zh": "参数", "en": "params"},
    # per-field hover explanations (Section A): <b>是什么</b> · 原理/好处（面向初学者）.
    # Rendered inside the shared #tip bubble via innerHTML, so <b> works; text sits
    # in a single-quoted data-tip='...' attribute, so NO apostrophes in the en strings.
    "ev.f.hidden_size": {
        "zh": "<b>每个 token 的向量宽度 H</b> · 几乎每个权重矩阵都有一条边是它，H 越大模型越「宽」，参数量随 H（常是 H²）增长。",
        "en": "<b>Per-token vector width (H)</b> · one edge of almost every weight matrix, so params grow with H (often H squared); the single biggest size knob."},
    "ev.f.num_hidden_layers": {
        "zh": "<b>Transformer 层数 L</b> · 每层结构相同，权重 ≈ 单层 × L；层数越多越「深」、越强，但显存随 L 线性增长。",
        "en": "<b>Number of layers (L)</b> · layers are identical, so weights are about one layer times L; deeper means stronger but memory grows linearly."},
    "ev.f.vocab_size": {
        "zh": "<b>词表大小</b> · embedding 和输出 lm_head 都是 vocab×hidden 的大矩阵；词表越大这两块越吃显存，小模型尤其明显。",
        "en": "<b>Vocabulary size</b> · both the embedding and output lm_head are vocab×hidden matrices; a large vocab dominates memory, especially in small models."},
    "ev.f.q_lora_rank": {
        "zh": "<b>query 的低秩压缩维度</b> · MLA 先把 query 压到这个小维度再展开，省掉巨大的 Q 投影权重（原理：低秩分解 W ≈ A·B）。",
        "en": "<b>Query low-rank dim</b> · MLA compresses the query to this small rank then expands it, shrinking the huge Q projection (principle: low-rank factorization W = A times B)."},
    "ev.f.kv_lora_rank": {
        "zh": "<b>KV 压成的共享 latent 宽度</b> · MLA 只缓存这个几百维的 latent、而非完整 K/V，KV cache 缩小约 10 倍 —— 长上下文省显存的关键。",
        "en": "<b>Shared KV latent width</b> · MLA caches only this few-hundred-dim latent instead of full K/V, shrinking the KV cache roughly 10x — the key to cheap long context."},
    "ev.f.qk_nope_head_dim": {
        "zh": "<b>不带 RoPE 的 QK 分量</b> · 负责「内容」匹配；与 rope 分量相加得到每个 head 的 QK 总维度。",
        "en": "<b>QK dim without RoPE</b> · carries content-based matching; added to the rope part to form each head total QK width."},
    "ev.f.qk_rope_head_dim": {
        "zh": "<b>带 RoPE 旋转位置编码的 QK 分量</b> · 让注意力感知 token 的「位置」；MLA 把位置与内容拆开分别处理。",
        "en": "<b>QK dim carrying RoPE</b> · gives attention its positional sense; MLA separates position from content and handles each apart."},
    "ev.f.v_head_dim": {
        "zh": "<b>每个 value head 的维度</b> · 决定注意力输出宽度，进而决定 o 投影大小；MLA 里可与 qk 维度不同。",
        "en": "<b>Per value-head dim</b> · sets the attention output width and thus the o-projection size; in MLA it may differ from the qk dim."},
    "ev.f.num_attention_heads": {
        "zh": "<b>注意力头数</b> · 把 attention 拆成多个并行子空间，各自关注不同模式；总维度 = heads × head_dim。",
        "en": "<b>Number of attention heads</b> · splits attention into parallel subspaces that each focus on different patterns; total dim = heads times head_dim."},
    "ev.f.num_key_value_heads": {
        "zh": "<b>KV 头数</b> · GQA 让多个 query 头共享少数 KV 头，KV cache 按 KV 头数缩小（原理：K/V 比 Q 更可压缩）。",
        "en": "<b>KV heads</b> · GQA lets many query heads share a few KV heads, shrinking the KV cache in proportion (principle: K/V compress better than Q)."},
    "ev.f.head_dim": {
        "zh": "<b>每个注意力头的维度</b> · 点积在这个维度上进行，决定单头的表达力与计算量。",
        "en": "<b>Per attention-head dim</b> · the dot-product runs over this dim, setting the capacity and compute of each head."},
    "ev.f.first_k_dense_replace": {
        "zh": "<b>前几层用 Dense FFN</b> · 靠前的层用普通「胖」FFN 更稳定，之后才换成 MoE；这个数就是分界层。",
        "en": "<b>First N layers stay Dense FFN</b> · early layers use a plain wide FFN for stability before switching to MoE; this is the cutover count."},
    "ev.f.n_routed_experts": {
        "zh": "<b>路由专家总数</b> · 显存绝对大头 —— 所有专家都要常驻显存，哪怕每 token 只用几个（原理：容量靠「多专家」，算力靠「稀疏激活」）。",
        "en": "<b>Total routed experts</b> · the memory hog — every expert stays resident in VRAM even though each token uses only a few (principle: capacity from many experts, compute stays sparse)."},
    "ev.f.num_experts_per_tok": {
        "zh": "<b>每 token 激活的专家数 (top-k)</b> · 只影响计算量、不影响显存；这正是 MoE「省算力不省显存」的原因。",
        "en": "<b>Experts activated per token (top-k)</b> · affects compute only, not memory — exactly why MoE saves flops but not VRAM."},
    "ev.f.moe_intermediate_size": {
        "zh": "<b>每个专家的中间层宽度</b> · 比 Dense FFN「瘦」很多；乘以专家数才是总量。",
        "en": "<b>Per-expert intermediate width</b> · much narrower than a Dense FFN; multiply by the expert count for the total."},
    "ev.f.n_shared_experts": {
        "zh": "<b>每层常驻的共享专家</b> · 所有 token 都会经过，负责通用知识，与被路由的专属专家互补。",
        "en": "<b>Always-on shared experts</b> · every token passes through them for common knowledge, complementing the routed specialists."},
    "ev.f.intermediate_size": {
        "zh": "<b>FFN 中间层宽度</b> · FFN 先升维到这里再降回，通常是 hidden 的 3–4 倍，是 Dense 模型的参数大头。",
        "en": "<b>FFN intermediate width</b> · the FFN expands to this (usually 3-4x hidden) then back down; the bulk of a dense model params."},
    "ev.f.index_n_heads": {
        "zh": "<b>DSA 稀疏索引头数</b> · 只用来挑「该关注哪些 token」，省的是算力不是显存；本身几乎不产生权重。",
        "en": "<b>DSA sparse-index heads</b> · used only to choose which tokens to attend to — saves compute, not memory, and adds almost no weights."},
    "ev.f.num_nextn_predict_layers": {
        "zh": "<b>MTP 额外预测层数</b> · 投机解码：一次前向多猜几个 token 来提速；不需要时可不加载、省显存。",
        "en": "<b>Extra MTP predict layers</b> · speculative decoding guesses several tokens per forward pass for speed; skip loading them to save memory."},
    "ev.f.quant": {
        "zh": "<b>有效精度 {bpp} 字节/参数</b> · 把「参数量」换算成「字节数」的乘数；fp8=1、bf16=2、fp4≈0.5，越低越省显存。",
        "en": "<b>Effective {bpp} bytes/param</b> · the multiplier turning param count into bytes; fp8=1, bf16=2, fp4=0.5 — lower saves memory."},
    "ev.mtp_composite": {"zh": "一整套额外的 attention + MoE 专家 + eh_proj",
                          "en": "a full extra attention + MoE experts + eh_proj"},
    "ev.vision_composite": {"zh": "标准 pre-norm ViT block（qkv/o 带 bias）+ patch conv + 位置嵌入 + patchmerger projector",
                             "en": "standard pre-norm ViT blocks (qkv/o with bias) + patch conv + pos-emb + patchmerger projector"},
    "ev.f.vt_hidden_size": {
        "zh": "<b>ViT 每 patch 的向量宽度</b> · vision tower 的 H；attention/MLP 权重都随它平方增长。",
        "en": "<b>ViT per-patch vector width</b> · the vision tower H; attention/MLP weights grow with its square."},
    "ev.f.vt_intermediate_size": {
        "zh": "<b>ViT MLP 中间层宽度</b> · 每个 ViT block 的 FFN 升维宽度，vision 权重的大头。",
        "en": "<b>ViT MLP intermediate width</b> · the FFN expansion width of each ViT block, the bulk of vision weights."},
    "ev.f.vt_num_hidden_layers": {
        "zh": "<b>ViT 层数</b> · vision tower 的深度，权重 ≈ 单 block × 层数。",
        "en": "<b>ViT layer count</b> · depth of the vision tower; weights are one block times this."},
    "ev.f.vt_patch_size": {
        "zh": "<b>图像切块边长（像素）</b> · 图像被切成 patch_size² 的小块，每块变成 1 个 ViT token；决定 patch conv 权重与每图 token 数。",
        "en": "<b>Image patch side (pixels)</b> · images are cut into patch_size² tiles, one ViT token each; sets the patch-conv size and tokens per image."},
    "ev.col_formula": {"zh": "① 公式", "en": "① formula"},
    "ev.col_apparent": {"zh": "② 实测", "en": "② safetensors"},
    "ev.verdict_match": {"zh": "吻合", "en": "match"},
    "ev.verdict_pack": {"zh": "打包", "en": "packed"},
    "ev.match_caption": {"zh": "↑ 两条等长 = 两个独立来源算出同一个数 = 可信",
                          "en": "↑ equal bars = two independent sources produced the same number = trustworthy"},
    "ev.pack_box_title": {"zh": "为什么实测只有一半？", "en": "Why is the measured bar half-length?"},
    "ev.pack_box_body": {
        "zh": "两个 {true}-bit 值被打包进 1 个 {bits}-bit 字节（safetensors 头因此显示成整数类型、宽度砍半）。存储省了 {pack}×，参数一个没少 —— 真实位宽 = {bits} ÷ {pack}：",
        "en": "Two {true}-bit values are packed into one {bits}-bit byte (so the safetensors header shows an integer type at half width). Storage shrinks {pack}×, no parameters are lost — true bit width = {bits} ÷ {pack}:"},
    "ev.sigma_ok": {"zh": "Σ 逐部件加总 = {sigma} GiB　✓ 与 index.json 声明的 total_size {idx} GiB 一致（偏差 {dev}）",
                     "en": "Σ over components = {sigma} GiB　✓ matches index.json's declared total_size {idx} GiB (dev {dev})"},
    "ev.sigma_line": {"zh": "Σ 逐部件加总 = {sigma} GiB", "en": "Σ over components = {sigma} GiB"},
    "ev.no_exact": {"zh": "本次未读取 safetensors（--no-exact 或网络失败）——仅有 config 公式估算，无法交叉验证。上面 A/B 的推导依然成立，但缺少独立第二来源的核对。",
                     "en": "safetensors was not read this run (--no-exact or network failure) — formula estimate only, no cross-check available. The A/B derivations above still hold, but lack the independent second source."},
    "ev.kv_foot_title": {"zh": "KV cache 每 token 公式（运行时）：", "en": "KV cache per-token formula (runtime):"},
    "ev.kv_foot_body": {"zh": "纯公式、无 safetensors 对账（KV 是运行时分配，不在权重文件里）；总量见「显存拆解」页。",
                         "en": "formula only, no safetensors reconciliation (KV is allocated at runtime, not in the weight files); see the Memory Breakdown tab for totals."},
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
    "pstruct.vision_block": {"zh": "Vision tower + projector（图像入口）", "en": "Vision tower + projector (image input)"},

    # ---- structured warnings ({key, params} in kv_structure()/main(); rendered
    # via render_warning() for terminal/API, and mirrored key-for-key in
    # template.js I18N so the in-page language switcher can re-render them.
    # Params arrive pre-formatted (strings/ints), so templates only substitute.
    "warn.hybrid_state": {
        "zh": "混合架构：{n_kv_layers}/{L} 层为 attention 存 KV，"
              "其余 {n_linear} 层为 linear/SSM 定长 state ≈ "
              "{state_mib} MiB/请求（随并发不随 context 增长，"
              "并行页已计入静态区）。按 槽位数=并发数 的最小需求计；"
              "SGLang 默认启发式可能预分配更多槽，部署时建议显式设 --max-mamba-cache-size。",
        "en": "Hybrid architecture: {n_kv_layers}/{L} layers are attention with paged KV; "
              "the other {n_linear} layers keep a fixed linear/SSM state ≈ "
              "{state_mib} MiB/request (grows with concurrency, not context; "
              "counted into the static region on the parallel tab). Sized at "
              "slots = concurrency, the minimum; SGLang's default heuristic may "
              "pre-allocate more slots — set --max-mamba-cache-size explicitly when deploying.",
    },
    "warn.hybrid_nostate": {
        "zh": "混合架构：{n_kv_layers}/{L} 层为 attention 存 KV，"
              "其余 {n_linear} 层为 linear/SSM 定长 state（不计入 KV 池）；"
              "config 缺 linear_* 维度字段，state 显存未建模。",
        "en": "Hybrid architecture: {n_kv_layers}/{L} layers are attention with paged KV; "
              "the other {n_linear} layers keep a fixed linear/SSM state (not part of "
              "the KV pool); the config lacks the linear_* dimension fields, so that "
              "state's VRAM is unmodeled.",
    },
    "warn.sliding_capped": {
        "zh": "滑窗注意力：{n_sliding} 层 KV 存储上限已按 min(context, {window}) 计。",
        "en": "Sliding-window attention: KV storage for {n_sliding} layers is capped "
              "at min(context, {window}).",
    },
    "warn.sliding_unidentified": {
        "zh": "检出 sliding_window={window} 但无法从 config 判定哪些层滑窗；"
              "KV 存储未封顶（保守按全 context 计，可能高估）。",
        "en": "sliding_window={window} detected but the config does not say which "
              "layers are sliding; KV storage is left uncapped (conservatively full "
              "context — likely an overestimate).",
    },
    "warn.block_sparse": {
        "zh": "块稀疏注意力：{n_sparse} 层 decode 读取封顶 min(context, {cap}) tokens；"
              "KV 存储仍为全量（块稀疏需保留全部块）。",
        "en": "Block-sparse attention: decode reads for {n_sparse} layers are capped at "
              "min(context, {cap}) tokens; KV storage stays full (all blocks must be retained).",
    },
    "warn.dsa_topk": {
        "zh": "DSA top-k 稀疏：decode 读取封顶 min(context, {topk})；"
              "逐层稀疏频率（index_topk_freq 等）未区分。",
        "en": "DSA top-k sparsity: decode reads are capped at min(context, {topk}); "
              "per-layer sparsity frequency (index_topk_freq etc.) is not differentiated.",
    },
    "warn.weights_sum_mismatch": {
        "zh": "权重总量存疑：safetensors 头求和 {got_gib} GiB "
              "vs index 声明 {declared_gib} GiB"
              "（差 {diff_pct}，可能有分片头未读到或含未加载张量）。",
        "en": "Weight total in doubt: safetensors headers sum to {got_gib} GiB vs "
              "{declared_gib} GiB declared by the index (off by {diff_pct}; some shard "
              "headers may be unread, or the index counts tensors that never load).",
    },
    "warn.headers_fetch_failed": {
        "zh": "未能读取 safetensors 头（{err}），回退到公式估算，权重为估算值。",
        "en": "Could not read the safetensors headers ({err}); falling back to the "
              "config formula — weight numbers are estimates.",
    },
    "warn.fp4_inflation": {
        "zh": "fp4 权重运行时会膨胀（B200 实测 +4.6%~+22%，随 kernel 路径而异）——"
              "safetensors 口径的 weights 偏乐观，贴边的 fit 判定请留余量",
        "en": "fp4 weights inflate at runtime (+4.6%–+22% measured on B200, varies by "
              "kernel path) — safetensors-based weight numbers are optimistic; leave "
              "headroom on fit verdicts near the boundary",
    },
    "warn.mla_absorb": {
        "zh": "MLA 权重吸收：SGLang 加载时将 kv_b_proj 反量化为 bf16 w_kc/w_vc（fp8 原件保留），"
              "全尺寸 ≈ {full_gib} GiB，随 attention-TP 切分（纯 TP÷tp；dp-attention 每卡整份）。"
              "未计入上方 safetensors 权重表；并行 tab 已计入。",
        "en": "MLA weight absorption: at load SGLang dequantizes kv_b_proj into bf16 "
              "w_kc/w_vc (the fp8 originals are kept), full size ≈ {full_gib} GiB, "
              "sharded by attention-TP (pure TP ÷tp; dp-attention keeps a full copy "
              "per GPU). Not included in the safetensors weight table above; the "
              "parallel tab does include it.",
    },
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


def warning(key: str, **params) -> dict:
    """A structured warning: {key, params} instead of a baked string.

    Producers (kv_structure(), main(), app.py) emit these; render_warning()
    turns one into prose for the terminal report and the JSON API, and
    template.js carries a mirror of the warn.* templates so the in-page
    language switcher can re-render them client-side. `key` is stored without
    the "warn." prefix; params must be pre-formatted (plain strings/numbers).
    """
    assert f"warn.{key}" in MSG, f"unknown warning key: {key}"
    return {"key": key, "params": params}


def render_warning(w, lang: str | None = None) -> str:
    """Render a structured warning to prose; plain strings pass through
    (belt-and-braces for any producer not yet migrated)."""
    if isinstance(w, str):
        return w
    entry = MSG[f"warn.{w['key']}"]
    s = entry.get(lang or _lang, entry["zh"])
    return s.format(**w["params"])
