# API Reference — llm-inference-analyzer hosted service

```
BASE = https://llm-inference-analyzer.ybalbert.people.aws.dev
```

Three anonymous JSON endpoints (no auth, no token). Every response carries `engine_version` identifying the exact math revision.

| Endpoint | Purpose | Rate limit |
|---|---|---|
| `GET /api/v1/analyze` | Full analysis for one parameter combination — **the endpoint the skill uses** | 120/hour/IP |
| `GET /api/v1/catalog` | Hardware vocabulary (instances, GPUs, perf specs) + server defaults | none |
| `GET /api/v1/whatif` | Renderer-shaped feed for the HTML report page (raw bytes, i18n refs) | 1200/hour/IP |

---

## GET /api/v1/analyze

### Request parameters

All optional except `model`. The response's `assumptions` object echoes every
effective value and lists which came from defaults (`defaults_used`).

| Param | Type / range | Default | Meaning |
|---|---|---|---|
| `model` | `org/name` | **required** | HuggingFace model id |
| `context` | int 128 – 16777216 | 131072 | context length per request (drives KV) |
| `requests` | int 1 – 65536 | 16 | concurrent running requests |
| `kv_dtype` | `auto` `bf16` `fp16` `fp8` `fp4` | auto | auto mirrors SGLang: fp8 for DSA models, else model dtype; response echoes the resolved value |
| `tp` | int 1 – 128 | 8 | tensor parallel size |
| `pp` | int 1 – 64 | 1 | pipeline parallel size |
| `ep` | int 1 – 128 | =tp | expert parallel size (MoE) |
| `dp_attention` | bool | false | DP attention; only takes effect for MLA models or TP > kv-heads — `assumptions.dp_attention` echoes the *effective* value |
| `instance` | name | p5en.48xlarge | AWS instance (or `h800-8gpu`/`h20-8gpu`); **overrides** `gpu`/`gpu_mem_gib`. Unknown name → 400 listing supported ones |
| `gpu` | name | — | GPU by name (H100/H200/H800/H20/B200/B300/A100/L40S/A10G — see `/catalog`); memory size comes from the server catalog |
| `gpus_per_node` | int 1 – 72 | 8 | used with `gpu`/`gpu_mem_gib` |
| `gpu_mem_gib` | float > 0 | — | raw GiB-per-GPU fallback for uncataloged hardware (no roofline section then) |
| `mem_fraction_static` | 0.3 – 0.99 | 0.9 | SGLang semantics: fraction pre-allocated for weights + KV pool |
| `fixed_overhead_gib` | 0 – 16 | tp-scaled | per-GPU CUDA context/NCCL; default `0.65 + 0.265×(tp−1)` (measured 0.65 @TP1, 2.5–2.9 @TP8 on B200); explicit value overrides |
| `batch_tokens` | int 128 – 131072 | 8192 | decode-roofline batch and the default forward size F when `chunk_tokens` is unset |
| `chunk_tokens` | int 128 – 131072 | =batch_tokens | chunked-prefill size: drives the prefill roofline verdict AND the serving-transient (activation) estimate in the parallel/fit verdicts; set to the engine's chunked-prefill size under sustained load (SGLang DSv4 default: 16384) |
| `weight_dtype` | `bf16` `fp8` `fp4` | checkpoint's | roofline what-if: idealized dtype conversion (weights section still reports the real checkpoint) |

### Response schema

```jsonc
{
  "model": "org/name",
  "architecture": "Glm4MoeForCausalLM",      // config.json architectures[0]
  "engine_version": "…",

  "assumptions": {                            // every effective parameter, echoed
    "context": 131072, "concurrent_requests": 16,
    "kv_dtype": "fp8",                        // resolved value (auto → concrete)
    "kv_dtype_requested": "auto",
    "hardware": "p5en.48xlarge (8xH200 141 GiB) [default]",
    "gpu_mem_gib": 141.0, "gpus_per_node": 8,
    "tp": 8, "pp": 1, "ep": 8,
    "dp_attention": false,                    // EFFECTIVE value (may differ from requested)
    "dp_attention_requested": false,
    "mem_fraction_static": 0.9, "fixed_overhead_gib": 1.0,
    "batch_tokens": 8192, "chunk_tokens": 8192,
    "defaults_used": ["context", "requests", …]  // which params the caller didn't set
  },

  "weights": {
    "total_gib": 148.62, "total_params_b": 159.7,
    "dtype": "fp8 (mixed)",                   // human label of checkpoint precision
    "exact_from_safetensors": true,           // true = read from safetensors headers (ground truth)
    "active_params_b": 12.3,                  // MoE active params; null for dense
    "components": [                           // sorted by bytes desc
      {"name": "MoE routed experts", "params_b": 143.1, "gib": 133.9, "share": 0.9012}
    ],
    "warnings": []                            // e.g. formula fallback — quote to the user
  },

  "kv_cache": {
    "dtype": "fp8",
    "per_token_kib": 68.6,                    // across all KV layers
    "per_request_gib": 8.583,                 // at `context` — the unit for concurrency math
    "total_gib": 137.33,                      // per_request × requests (demand, not capacity)
    "mla_vs_mha_savings_x": 8.1,              // null for non-MLA
    "linear_state_total_gib": 0,              // SSM/linear-attention state pool (hybrid models)
    "notes": []                               // e.g. DSA indexer extra bytes — quote when relevant
  },

  "runtime": {
    "activation_gib": 8.4,                    // serving transient peak per forward over F tokens
                                              // (chunk_tokens, falling back to batch_tokens);
                                              // ≈ base + per-token × F, calibrated on B200; models with
                                              // MTP layers get ×1.15 (spec decoding assumed on) —
                                              // this lives OUTSIDE the mem-fraction static region
    "vision_encoder_activation_gib": 0,       // VLMs only
    "runtime_total_gib": 145.8,               // kv total + linear state + activations
    "grand_total_gib": 309.1,                 // weights + runtime + 5% fragmentation,
    "grand_total_note": "…single-copy (no TP replication); use `parallel` for per-GPU fit"
  },

  "parallel": {                               // fit verdict at the REQUESTED tp/pp/ep
    // or {"error": "pp(N) exceeds layer count L"}
    "tp": 8, "pp": 1, "ep": 8, "dp_attention": false,
    "world_gpus": 8, "gpu_mem_gib": 141.0,
    "mem_fraction_static": 0.9, "fixed_overhead_gib": 1.0,
    "fits": true,                             // the headline verdict: starts, KV fits, AND serving transient fits
    "weights_fit": true,                      // engine can start (weights + overhead < static budget)
    "kv_fits_demand": true,                   // requested context×requests fits the KV pool
    "oom_gpus": 0,
    "serving_oom_risk": false,                // true = STARTS but crashes on the first full-chunk
                                              // prefill (transient > non-static headroom); always
                                              // quote serving_note instead of a bare "not fits"
    "serving_note": null,                     // human explanation + levers when serving_oom_risk
    "per_gpu": [                              // one entry per pp stage (tp ranks of a stage are identical)
      {"pp_stage": 0, "layers": "L0-L91",
       "weights_gib": 20.1, "kv_pool_gib": 96.2,     // pool CAPACITY (SGLang pre-allocation)
       "kv_demand_gib": 137.3,                       // what context×requests actually needs
       "kv_utilization_pct": 142.7,                  // >100 = demand exceeds pool
       "activation_gib": 1.05, "linear_state_gib": 0,
       "used_gib": 128.9, "free_gib": 12.1,
       "can_start": true,
       "short_by_gib": null}                         // when can_start=false: exact deficit
    ],
    "max_used_gib": 128.9,
    "cluster_weights_gib": 176.4,             // sum over all GPUs (includes TP replication)
    "weight_replication_overhead_gib": 27.8,  // cluster_weights − single-copy weights
    "kv_note": "MLA latent has no head dim: KV fully replicated per GPU…",  // replication explainer
    "max_concurrency": 11,                    // bottleneck stage, at `context`; per-rank under dp-attention
    "max_concurrency_cluster": 11,            // × tp under dp-attention, else same
    "kv_pool_tokens": 1467000,                // matches SGLang's logged max_total_num_tokens
    "kv_pool_tokens_cluster": 1467000
  },

  "tp_sweep": {                               // "how should I shard this" — pp=1 sweep
    "note": "pp=1 on <hardware>; max_concurrency at context=…",
    "rows": [                                 // tp ∈ {1,2,4,8,…} ≤ gpus_per_node (plus requested tp);
                                              // MLA/over-sharded shapes get a dp_attention=true twin row
      {"tp": 8, "dp_attention": false,
       "per_gpu_weights_gib": 20.1, "kv_pool_gib": 96.2, "kv_demand_gib": 137.3,
       "used_gib": 128.9, "free_gib": 12.1,
       "fits": false,                         // can_start AND free≥0 AND demand≤pool
       "can_start": true,
       "max_concurrency": 11}                 // cluster-level (already × tp under dp-attention)
    ]
  },

  "roofline": {                               // requires a cataloged GPU; else {"note": "no compute/bandwidth spec…"}
    "gpu": "H200", "peak_tflops": 1979, "peak_dtype": "fp8",  // falls back to bf16 if dtype unsupported
    "hbm_tbs": 4.8, "knee_flops_per_byte": 412.3,
    "tp": 8, "dp_attention": false,
    "weight_dtype": "fp8",                    // "… (idealized what-if)" suffix when weight_dtype override active
    "note": "theoretical upper bounds…real systems typically reach 50-70%",
    "decode": {
      "bound": "memory",                      // memory | compute
      "aggregate_intensity": 121.4,
      "knee_ratio": 0.29,                     // aggregate intensity / knee; <1 = memory-bound
      "batch_tokens_per_step": 16,            // = `requests` (one token per running request)
      "step_ms": 8.31,
      "tokens_per_s_per_request": 120.3,      // 1000 / step_ms
      "tokens_per_s_tp_group": 1925,          // × batch — the throughput upper bound
      "guidance": "memory-bound: concurrency is nearly free until…",
      "kernels": [                            // per-kernel breakdown
        {"kernel": "MoE routed experts", "tflops": 0.39, "hbm_gib": 124.0,
         "intensity": 3.2, "bound": "memory", "time_share_pct": 84.1, "note": "…"}
      ]
    },
    "prefill": {
      "bound": "compute",
      "aggregate_intensity": 3805.1, "knee_ratio": 9.23,
      "chunk_tokens": 8192, "chunk_ms": 212.4,
      "ttft_s_full_context": 3.4,             // chunk time × (context / chunk) — TTFT lower bound
      "guidance": "prefill compute-bound (typical): …",
      "kernels": [ /* same shape as decode */ ]
    }
  },

  "report_url": "https://…/reports/org--name.html",   // report has an in-page zh/en switcher
  "report_note": "interactive 4-tab report (deep links: #evidence #estimate #parallel #roofline); requires HuggingFace login in a browser; generating in background if absent — may take ~1 min on first request"
}
```

---

## GET /api/v1/catalog

No parameters. The hardware vocabulary — **query this instead of recalling GPU specs from memory** when the user names hardware you're unsure the server knows.

```jsonc
{
  "engine_version": "…",
  "instances": {                              // valid `instance` values
    "p6-b300.48xlarge": {"gpu": "B300", "gpus": 8, "gpu_mem_gib": 268.6},
    "p6-b200.48xlarge": {"gpu": "B200", "gpus": 8, "gpu_mem_gib": 179.1},
    "p5en.48xlarge":    {"gpu": "H200", "gpus": 8, "gpu_mem_gib": 141.0},
    "p5.48xlarge":      {"gpu": "H100", "gpus": 8, "gpu_mem_gib": 80.0},
    // + p4de/p4d (A100), g6e (L40S), g5 (A10G), h800-8gpu, h20-8gpu
  },
  "gpus": {"H200": 141.0, "B300": 268.6, …},  // valid `gpu` values → GiB (largest variant)
  "gpu_perf": {                               // GPUs with roofline support
    "H200": {"bf16_tflops": 989, "fp8_tflops": 1979, "fp4_tflops": null, "hbm_tbs": 4.8}
  },
  "kv_dtypes": ["auto", "bf16", "fp16", "fp8", "fp4"],
  "defaults": {"context": 131072, "requests": 16, "kv_dtype": "auto", "tp": 8,
               "pp": 1, "instance": "p5en.48xlarge",
               "mem_fraction_static": 0.9, "fixed_overhead_gib": 1.0,
               "batch_tokens": 8192}
}
```

---

## GET /api/v1/whatif

Same request parameters as `/analyze`. The response is **renderer-shaped**: it feeds the HTML report page's dynamic tabs on every dropdown change. Differences from `/analyze`: all sizes are raw **bytes** (un-rounded floats, not GiB), times are **ms**, field names are camelCase matching template.js, and kernel notes are i18n key references (`noteRefs`) instead of prose. Prefer `/analyze` for answering users; `/whatif` is useful when you need un-rounded numbers, per-component weight bytes per GPU, or per-kernel times — or to cross-check the report page's rendering.

Schema below verified against the live service (dense GQA, MLA+DSA+MoE with dp-attention, hybrid SSM, `weight_dtype` override, `pp>layers`, uncataloged hardware).

```jsonc
{
  "echo": {                                   // effective parameters (like /analyze assumptions)
    "ctx": 131072, "req": 16, "kvDtype": "fp8",   // kvDtype resolved (auto → concrete)
    "tp": 8, "pp": 1, "ep": 8,
    "dpAttn": true,                           // EFFECTIVE value
    "dpAvailable": true,                      // whether dp-attention is even applicable (MLA or tp > kv-heads)
    "frac": 0.9, "memGib": 179.1, "gpn": 8, "fixedGib": 1.0,
    "chunk": 8192, "weightDtype": "fp8"
  },

  "estimate": {                               // single-copy totals, all BYTES
    "kvPerTok": 262144.0, "kvPerReq": 3.4e10, "kvTotal": 5.5e11,
    "linTotal": 0,                            // SSM/linear state pool (hybrid models)
    "runtime": 5.5e11,                        // kvTotal + linTotal + activations
    "total": 6.2e11,                          // weights + runtime
    "grand": 6.5e11,                          // total × 1.05 fragmentation
    "mhaTotal": 6.6e12                        // "if this MLA model stored full MHA KV"; null for non-MLA
  },

  "parallel": {                               // or {"error": "ppExceeds", "pp": 64, "L": 28}
    "stages": [                               // one per pp stage; all tp ranks of a stage identical
      {
        "w": {                                // per-component weight BYTES on one GPU —
                                              // the only API surface with this breakdown
          "embed": 1.9e9, "vision": 0, "lmHead": 2.4e8,
          "attention": 1.5e10,                // includes MLA absorb + replication under dp-attention
          "denseFfn": 8.6e7, "moeRouted": 9.1e10, "moeShared": 3.5e8,
          "mtp": 3.4e9, "others": 4.4e8      // norms + DSA indexer + MoE router
        },
        "weights": 1.16e11,                   // sum of w
        "kv": 1.37e11,                        // KV demand on this GPU at ctx×req
        "kvParts": [                          // per storage group (full vs sliding-window)
          {"layers": 64, "window": 0, "b": 1.37e11}   // window 0 = full context
        ],
        "kvCap": 5.9e10,                      // KV pool capacity (≥0); kvCapRaw can be negative = deficit
        "kvCapRaw": 5.9e10,
        "canStart": true,
        "act": 3.7e8, "linState": 0,
        "used": 7.7e10, "free": 8.2e9,
        "maxReq": 6,                          // max concurrency this stage's pool supports
        "maxTokens": 913293.3,                // = SGLang max_total_num_tokens (per rank under dp)
        "layers": [0, 64],                    // [lo, hi) layer range of this stage
        "nDense": 64, "nMoe": 0
      }
    ],
    "experts": [                              // per tp-rank expert placement; [null, …] for dense models
      {"cnt": 32, "lo": 0, "hi": 31, "sliceDenom": 1}   // rank holds experts lo..hi;
                                              // sliceDenom>1 = each expert is a 1/N slice (ep<tp)
    ],
    "world": 8,                               // tp × pp GPUs
    "clusterWeights": 9.3e11,                 // sum over all GPUs (incl. replication)
    "clusterKv": 5.5e11, "kvSingle": 5.5e11,  // clusterKv/kvSingle = replication factor
    "oomTotal": 0, "allStart": true, "maxUsed": 7.7e10,
    "minMaxReq": 6, "minMaxTok": 913293.3     // bottleneck stage — matches /analyze max_concurrency
  },

  // roofline is null when hardware has no perf spec (gpu_mem_gib-only)
  "roofline": {
    "perf": {"gpu": "B200", "peak": 4500, "bw": 8.0,   // TFLOPs at effective dtype, TB/s
             "dtype": "fp8",
             "fallback": null},               // set to the requested dtype when GPU lacks it → bf16 used
    "knee": 562.5,                            // peak/bw, FLOPs per byte
    "rows": [                                 // one per kernel; observed keys:
                                              // attention, indexer (DSA), dense_ffn, moe_gate,
                                              // moe_routed, moe_shared, lm_head, attn_core,
                                              // linear_state (hybrid SSM), embed (VLM vision)
      {"key": "attn_core", "label": "Attn core (KV/score)",
       "color": "var(--s2)",                  // report CSS var — ignore
       "dec": {"flops": 1.9e11, "bytes": 1.2e10, "intensity": 16.0,
               "timeMs": 3.6,                 // per-GPU kernel time, max(bytes/bw, flops/peak)
               "noteRefs": [["mlaLabel", 512, 64], ["_raw", " + "],
                            ["dsaDecodeNote", "128K", 2048]]},
                                              // i18n [key, args…]; renderable ones worth knowing:
                                              //   replNote N   = kernel replicated ×N per GPU (not TP-sliced)
                                              //   moeDecodeNote B pct = decode batch B touches pct% of experts
                                              //   gqaLabel q kv / mlaLabel rank rope = attention geometry
       "pre": { /* same shape */ }}
    ],
    "phases": {
      "dec": {"aggFlops": 5.4e12, "aggBytes": 6.1e11,
              "aggIntensity": 8.8, "isMem": true,      // the memory/compute-bound verdict
              "sumMs": 183.2,                 // Σ kernel times (one GPU does 1/tp of this)
              "phaseMs": 45.8,                // sumMs / tp = step time
              "kneeRatio": 33.4,              // knee / aggIntensity when isMem (how far from knee)
              "tpsPerReq": 21.8,              // 1000 / phaseMs
              "tpsGroup": 349.3},             // × B — matches /analyze tokens_per_s_tp_group
      "pre": { /* same core fields + */ "ttftS": 2.4}  // matches /analyze ttft_s_full_context
    },
    "B": 16, "T": 8192,                       // decode batch (=requests), prefill chunk tokens
    "wdtypeOverride": "fp4"                   // non-null when weight_dtype what-if active
  },

  "engineVersion": "ca44c4b7ecf6"
}
```

---

## Errors

| Status | Meaning | What to do |
|---|---|---|
| 400 | invalid param — bad model id format, unknown `instance`/`gpu` (message lists supported values), out-of-range number | fix the call; for unknown hardware fall back to `gpu_mem_gib` |
| 404 | model nonexistent on huggingface.co **or gated** (service holds no tokens, by design) | tell the user; gated models cannot be analyzed |
| 429 | rate limit (120 analyze/h, 1200 whatif/h, per IP) | wait or reduce sweep size |
| 502 | huggingface.co upstream error | retry later |
