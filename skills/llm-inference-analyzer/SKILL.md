---
name: llm-inference-analyzer
description: Analyze LLM inference deployment for any HuggingFace model — GPU memory (VRAM/显存) breakdown, TP/PP/EP parallelism partitioning across GPU nodes, and roofline performance bounds (memory- vs compute-bound, theoretical tokens/s) — via a hosted analysis API, plus a shareable interactive HTML report. Use this whenever the user asks how much GPU memory a model needs, whether a model fits on specific GPUs or instances ("能跑在 8×H100 上吗"), how many GPUs/nodes are needed, how to shard a model with tensor/pipeline/expert parallelism, whether a context × concurrency target fits, what throughput (tokens/s) or TTFT to expect, which instance type is the best choice, how to reduce VRAM usage, or wants a 显存拆解/并行切分/性能分析 for a model given its HuggingFace ID. Also use it to compare quantization variants (fp8/fp4/AWQ/GPTQ) or KV-cache dtype choices.
---

# LLM Inference Analyzer

Given a HuggingFace model ID, a hosted service computes the full
inference-deployment analysis. This skill contains **no compute logic** — every
number comes from the API, so all users always get the same, latest engine.

```
BASE=https://llm-inference-analyzer.ybalbert.people.aws.dev
GET $BASE/api/v1/analyze?model=<org/name>&...   # anonymous JSON — the workhorse
GET $BASE/api/v1/catalog                        # hardware vocabulary + defaults
```

Full request/response schemas for `/analyze`, `/catalog`, and `/whatif`:
**`references/api.md`**.

**Always ground your numbers in an API call — never estimate VRAM or throughput
from memory.** Hardware facts (GPU memory sizes, instance shapes, TFLOPs) live
in the server catalog — query `/api/v1/catalog` rather than recalling them.

## Every answer has two halves: the directional reply AND the report link

1. **Answer in chat, in words.** The API gives you a *directional* answer —
   lead with the bottom line ("不够，差 283 GiB" / "TP4 即可，每卡 38 GiB 剩
   42 GiB"), then the supporting numbers. Name the assumptions that came from
   defaults (`assumptions.defaults_used`). Don't dump raw JSON.
2. **Hand over `report_url` for the fine-grained interactive analysis.** The
   HTML report has dropdowns for context/concurrency/KV dtype/TP/instance
   what-ifs, four deep-linkable tabs (`#evidence` `#estimate` `#parallel`
   `#roofline`), and an in-page zh/en switcher. Point the user at the tab that
   matches their question. It
   **requires a HuggingFace login in the browser** — the login is the user's
   own action; no token ever passes through this skill. First-ever request for
   a model generates the report in the background (~1 min); the API answer
   itself is immediate.

## The core questions and how to answer each

These are the questions this skill exists for. Each row: what to call, which
response fields carry the verdict, and which report tab to hand off.

### "这个模型需要多少显存？能跑在 8×H100 上吗？"

Call with `gpu=H100&gpus_per_node=8` (or the matching `instance`). Total need is
`runtime.grand_total_gib` — but **the fit verdict is `parallel.fits`, never
grand_total ÷ GPU count** (TP replication, MLA KV replication, and the SGLang
memory budget all break naive division; `parallel.kv_note` explains what
replicates). If it doesn't fit, `per_gpu[].short_by_gib` and
`kv_utilization_pct` say by how much and why. Distinguish the two failure
modes: `weights_fit=false` = engine can't even start; `kv_fits_demand=false` =
starts, but not at the requested context × concurrency (then quote
`max_concurrency`). Hand off: `#estimate` (breakdown) and `#parallel` (fit).

### "用 H100 部署需要多少张卡？每张卡占多少、剩多少？"

Call with the target hardware; read `tp_sweep.rows` — the smallest TP with
`fits=true` is the answer, and its `used_gib`/`free_gib`/`per_gpu_weights_gib`
give the per-card picture. For the requested shape's exact per-card numbers use
`parallel.per_gpu[]`. If no single-node TP fits, escalate: multi-node TP
(`tp=16&gpus_per_node=8`) or `pp>1`. Hand off: `#parallel`.

### "H200 单机部署能支持 128K context × 64 并发吗？"

Call with `context=131072&requests=64&instance=p5en.48xlarge`. Verdict:
`parallel.kv_fits_demand`; if false, quote `parallel.max_concurrency` ("最多
N 路") — under dp-attention quote `max_concurrency_cluster`. Quote
`kv_cache.per_request_gib` so follow-up concurrency questions become mental
math. `parallel.kv_pool_tokens` matches SGLang's logged `max_total_num_tokens`
— useful when the user wants to cross-check a real deployment. Hand off:
`#parallel`.

### "在 B300 上的最优部署参数是多少？"

No single field answers this — synthesize: (1) `tp_sweep.rows` → smallest
fitting TP (more TP than needed wastes weight replication:
`weight_replication_overhead_gib`); (2) if the sweep shows a
`dp_attention=true` twin row with much higher `max_concurrency`, recommend it
(MLA/over-sharded models); (3) `assumptions.kv_dtype` — the auto choice is
usually right, but fp8 KV doubles capacity vs bf16 when KV is the bottleneck;
(4) roofline `decode.guidance` tells whether extra concurrency is free
(memory-bound) or harmful. State this is a static-analysis recommendation —
real tuning needs a benchmark. Hand off: `#parallel` + `#roofline`.

### "选择什么机型最佳？"

One call per candidate instance (get the list from `/api/v1/catalog`), then
compare: fits at all → smallest world size → `free_gib` headroom →
`roofline.decode.tokens_per_s_tp_group` per node as the throughput tiebreaker.
A few calls is fine (rate limit 120/h). Present a small table; recommendation
first.

### "B300 上理论最大吞吐（token/s）是多少？"

Call with `gpu=B300` (roofline needs a cataloged GPU — raw `gpu_mem_gib` gives
no roofline). Answer from `roofline.decode.tokens_per_s_tp_group` — **always
label it "theoretical upper bound, 实测通常 50–70%"**, never "expected
throughput". Throughput scales with concurrency while memory-bound: quote
`decode.bound` and `knee_ratio` (how far from the knee). Per-request speed is
`tokens_per_s_per_request`. Bigger `requests` → higher group throughput until
KV runs out (`max_concurrency`) or the knee flips it compute-bound. Hand off:
`#roofline`.

### "B300 上理论最快 TTFT 是多少？"

Same call; `roofline.prefill.ttft_s_full_context` at the requested `context`
(don't pro-rate it to other context lengths — attention work grows faster than
linear; re-call with the target `context` instead).
`prefill.guidance` explains the lever: prefill is normally compute-bound, so
weight quantization alone doesn't cut TTFT unless low-precision compute is used
— `weight_dtype=fp8/fp4` shows that what-if. Same upper-bound caveat. Hand
off: `#roofline`.

### "L40S 上有什么办法能降低显存占用？"

Run what-ifs and quantify each lever with real deltas — the API call is cheap:

- **KV dtype**: re-call with `kv_dtype=fp8` (or fp4 — flag fp4 as aggressive);
  compare `kv_cache.total_gib`. Note: DSA models carry a per-token indexer key
  that does not shrink with kv_dtype (engine accounts for it; see
  `kv_cache.notes`).
- **Shorter context / fewer requests**: KV is linear in both;
  `per_request_gib` makes this mental math.
- **Weight quantization**: if the org publishes an fp8/fp4/AWQ variant, analyze
  *that* model id — `weights.total_gib` of the real checkpoint beats any
  idealized estimate. (`weight_dtype` only affects the roofline what-if, not
  the memory sections.)
- **More partitioning**: higher TP / adding PP shrinks per-GPU weights
  (`tp_sweep.rows`); dp-attention removes KV replication for MLA models.
- **`mem_fraction_static`**: lowering it shrinks the KV pool, not real usage —
  raise it (≤0.95) only when activation headroom allows; mention the trade.

## Calling notes

```bash
# minimal — server defaults: 128K × 16 requests, kv auto, TP8, p5en.48xlarge (8×H200)
curl -s "$BASE/api/v1/analyze?model=zai-org/GLM-5.2-FP8"

# explicit shape
curl -s "$BASE/api/v1/analyze?model=Qwen/Qwen3-32B&context=32768&requests=64&tp=4&instance=p5.48xlarge&kv_dtype=fp8"

# GPU by name, or raw memory for uncataloged hardware (no roofline then)
curl -s "$BASE/api/v1/analyze?model=deepseek-ai/DeepSeek-V4-Flash&gpu=B300&gpus_per_node=8"
curl -s "$BASE/api/v1/analyze?model=Qwen/Qwen3-32B&gpu_mem_gib=141&gpus_per_node=4"
```

- Parameters established earlier in the conversation (hardware, context,
  concurrency) carry forward — re-send them on every follow-up call.
- Sweeps the API doesn't return in one response (compare models, quant
  variants, instances) are just multiple calls.
- `/api/v1/whatif` exists but is renderer-shaped for the report page (raw
  bytes, i18n refs) — prefer `/analyze`; see `references/api.md` if you need it.

## Interpreting results — the rules that prevent wrong answers

- **Weights vs KV vs activation are different beasts.** Weights are paid once;
  KV grows linearly with context × requests; activation is
  concurrency-independent.
- **`weights.exact_from_safetensors: true` means weight numbers are ground
  truth** (mixed precision, sub-byte packing included). If `weights.warnings`
  mentions a formula fallback, say the weight number is an estimate.
- **KV dtype is a deployment decision, not a model property.** Name
  `assumptions.kv_dtype` whenever a KV number matters.
- **Fit is per-GPU** (`parallel.fits` / `per_gpu`), never grand_total ÷ N.
- **`kv_pool_gib` is capacity, `kv_demand_gib` is need** — utilization >100%
  means the target doesn't fit even though the engine starts.
- **Roofline numbers are upper bounds** (perfect overlap, zero comm cost);
  the robust takeaways are the memory/compute-bound verdicts and `guidance`.
- **Not in the totals**: fixed overhead beyond `fixed_overhead_gib` and comm
  buffers vary — mention this when the fit is tight (free_gib small).
- Surface `weights.warnings` and `kv_cache.notes` to the user when non-empty.

## When something breaks or looks wrong

- **404 "gated"**: the service cannot analyze gated/nonexistent repos by design
  (it holds no user tokens). Say so. If the user still wants a number, a hand
  estimate from config.json is possible but must be clearly labeled as
  unvalidated — the service is the only validated source.
- **429**: rate limit (120 analyses/hour/IP) — trim the sweep or wait.
- **400 unknown instance/GPU**: the error lists supported values; fall back to
  `gpu_mem_gib=<GiB>` with a clearly labeled assumption.
- **Service unreachable**: say so and recommend retrying — don't substitute
  hand math for the validated engine.
- **A number looks off for an exotic model**: fetch
  `https://huggingface.co/<id>/raw/main/config.json`, sanity-check the inputs
  (layer count, hidden size, experts), and report suspected engine gaps to the
  service owner rather than hand-correcting silently.
- The engine is validated against GLM-5.2-FP8 (MLA+DSA+MTP), DeepSeek-V4-Flash
  (fp4/fp8 MQA), Qwen3-32B (dense GQA), Qwen2.5-7B-AWQ (int4 packing),
  Qwen3.5-4B (hybrid SSM), Kimi-K2.6 (VLM); live-server validated on g5 and
  8×B200. `engine_version` in every response identifies the exact math revision.
