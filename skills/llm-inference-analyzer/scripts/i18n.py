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

    # ---- stacked bars / totals
    "bar.weights_static_label": {"zh": "权重（静态）", "en": "Weights (static)"},
    "bar.other_label": {"zh": "其它", "en": "other"},
    "bar.pct_paren": {"zh": "（{pct:.0f}%）", "en": " ({pct:.0f}%)"},
    "bar.total_head": {"zh": "静态 + 动态 · 部署总占用 ≈ {gib} GiB（context {ctx} × {req} 并发）",
                        "en": "Static + dynamic · total deployment footprint ≈ {gib} GiB (context {ctx} × {req} concurrent)"},
    "bar.lin_part": {"zh": " + linear state {lin}", "en": " + linear state {lin}"},
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
    "html.lbl_inst": {"zh": "机型", "en": "instance"},
    "html.lbl_frac": {"zh": "mem-fraction-static", "en": "mem-fraction-static"},
    "html.lbl_custom_mem": {"zh": "单卡", "en": "per GPU"},
    "html.lbl_custom_gpn": {"zh": "每节点", "en": "per node"},
    "html.lbl_custom_cards": {"zh": "卡", "en": "GPUs"},
    "html.fnote": {"zh": "KV Cache 与总占用随选择实时重算；权重（静态）部分不变",
                   "en": "KV cache and totals recompute live on selection; weights (static) stay fixed"},
    "html.tab_evidence": {"zh": "推导依据", "en": "How We Know"},
    "html.tab_estimate": {"zh": "显存拆解", "en": "Memory Breakdown"},
    "html.tab_parallel": {"zh": "并行切分", "en": "Parallel Sharding"},
    "html.tab_roofline": {"zh": "性能 Roofline", "en": "Perf Roofline"},

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
