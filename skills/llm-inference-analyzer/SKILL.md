---
name: llm-inference-analyzer
description: Analyze LLM inference deployment for any HuggingFace model — GPU memory (VRAM/显存) breakdown, TP/PP/EP parallelism partitioning across GPU nodes, and roofline performance bounds (memory- vs compute-bound, theoretical tokens/s) — via a hosted analysis API, plus a shareable interactive HTML report. Use this whenever the user asks how much GPU memory a model needs, whether a model fits on specific GPUs or instances ("能跑在 8×H100 上吗"), how to shard a model with tensor/pipeline/expert parallelism, how much KV cache grows with context or concurrency, what throughput or decode speed to expect, whether a deployment is memory-bound or compute-bound, or wants a 显存拆解/并行切分/性能分析 for a model given its HuggingFace ID. Also use it to compare quantization variants (fp8/fp4/AWQ/GPTQ) or KV-cache dtype choices.
---

# LLM Inference Analyzer

Given a HuggingFace model ID, a hosted service computes the full inference-deployment analysis. This skill contains **no compute logic** — all numbers come from one API call, so every user always gets the same, latest engine:

```
BASE=https://llm-inference-analyzer.ybalbert.people.aws.dev
GET $BASE/api/v1/analyze?model=<org/name>&...   # anonymous, JSON
GET $BASE/api/v1/catalog                        # hardware vocabulary + defaults
```

The response covers four areas (mirroring the report's four tabs): **weights** (per-component breakdown, exact from safetensors headers), **kv_cache + runtime** (KV scaling, linear/SSM state, activation), **parallel** (per-GPU fit verdict at the requested TP/PP/EP + a TP sweep table), and **roofline** (memory/compute-bound verdicts, theoretical tokens/s, TTFT). It also returns `report_url` — the interactive HTML report for humans.

## Every answer has two halves: the chat reply AND the report link

1. **Answer in chat, in words.** Lead with the bottom line ("不够，差 283 GiB" / "fp4 experts，共 148.6 GiB"), then the supporting numbers from the JSON. Don't dump raw JSON at the user.
2. **Hand over `report_url`.** Tell the user it opens an interactive report (dropdowns for context/concurrency/KV dtype/TP/instance what-ifs; tabs deep-linkable as `#evidence` `#estimate` `#parallel` `#roofline`) and that **it requires a HuggingFace login in the browser** — the login is the user's own action, no token ever passes through this skill. First-ever request for a model generates the report in the background (~1 min); the API answer itself is immediate.

**Always ground your numbers in an API call** — never estimate VRAM or throughput from memory. Re-calling with adjusted parameters costs one HTTP request.

## How to call

All parameters except `model` are optional; the response's `assumptions` object echoes every effective value and lists which came from defaults (`defaults_used`) — mention those assumptions when you answer.

```bash
# minimal — server defaults: 128K context × 16 requests, kv auto, TP8, p5en.48xlarge (8×H200)
curl -s "$BASE/api/v1/analyze?model=zai-org/GLM-5.2-FP8"

# explicit deployment shape
curl -s "$BASE/api/v1/analyze?model=Qwen/Qwen3-32B&context=32768&requests=64&tp=4&instance=p5.48xlarge&kv_dtype=fp8"

# GPU by name (server catalog knows the memory size) or raw memory fallback
curl -s "$BASE/api/v1/analyze?model=deepseek-ai/DeepSeek-V4-Flash&gpu=B300&gpus_per_node=8"
curl -s "$BASE/api/v1/analyze?model=Qwen/Qwen3-32B&gpu_mem_gib=141&gpus_per_node=4"
```

| Param | Default | Meaning |
|---|---|---|
| `model` | required | HuggingFace id, `org/name` |
| `context` | 131072 | context length per request (drives KV) |
| `requests` | 16 | concurrent running requests |
| `kv_dtype` | auto | `auto`/`bf16`/`fp16`/`fp8`/`fp4`; auto mirrors SGLang (fp8 for DSA models, else model dtype) |
| `tp` / `pp` / `ep` | 8 / 1 / =tp | parallelism sizes |
| `dp_attention` | false | DP attention (only effective for MLA or TP > kv-heads; response echoes the effective value) |
| `instance` | p5en.48xlarge | AWS instance or `h800-8gpu`/`h20-8gpu`; overrides `gpu`/`gpu_mem_gib` |
| `gpu` + `gpus_per_node` | — | GPU by name (H100/H200/B200/B300/A100/H800/H20/L40S/A10G), default 8/node |
| `gpu_mem_gib` | — | raw GiB-per-GPU fallback for hardware not in the catalog (no roofline then) |
| `mem_fraction_static` | 0.9 | SGLang semantics: weights + KV pool pre-allocation |
| `fixed_overhead_gib` | 1.0 | per-GPU CUDA context/NCCL (measured 2.5 at TP8 on B200) |
| `chunk_tokens` | =batch_tokens | chunked-prefill size for the roofline prefill verdict |
| `weight_dtype` | checkpoint's | roofline what-if: idealized bf16/fp8/fp4 conversion |
| `lang` | zh | language of the linked HTML report |

## Translating casual questions into calls

Users speak loosely; you translate the *semantics*, the server owns the *facts* (GPU memory sizes live in the server catalog — never recall them from memory):

- "能跑在 8×H100 上吗" → `gpu=H100&gpus_per_node=8`, answer from `parallel.fits` + `assumptions`
- "TP/PP/EP 怎么切？每卡占多少" → default hardware or whatever the conversation established; answer from `tp_sweep.rows` (per-TP fit + max concurrency) and `parallel.per_gpu`
- "128K × 64 并发 KV 多大" → `context=131072&requests=64`, answer from `kv_cache.total_gib`; quote `per_request_gib` so follow-up concurrency questions are mental math
- "B300 上能跑多少 tokens/s" → `gpu=B300`, answer from `roofline.decode.tokens_per_s_tp_group` — as a *theoretical upper bound*
- GPU not in the catalog (the error lists what is) → ask the user for GiB/GPU or use `gpu_mem_gib` with a clearly labeled assumption
- Parameters established earlier in the conversation (hardware, context, concurrency) carry forward — re-send them on follow-up calls

Sweeps the API doesn't return in one response (e.g. compare three models, or fp8 vs fp4 variants) are just multiple calls.

## Interpreting and reporting results

Lead with the bottom line — total GiB, and if the user named hardware, whether it fits and at what TP. Things to get right:

- **Weights vs runtime are different beasts.** Weights are paid once; KV grows linearly with `context × requests` (use `kv_cache.per_request_gib` for "how many users fit"); activation is concurrency-independent.
- **`exact_from_safetensors: true` means weight numbers are ground truth** (mixed precision, sub-byte packing included). If `weights.warnings` mentions a formula fallback, say the weight number is an estimate.
- **KV dtype is a deployment decision, not a model property.** Name the assumed dtype whenever the KV number matters (`assumptions.kv_dtype`). fp4 KV is aggressive — flag it. DSA models cache an extra per-token index-key that does NOT shrink with kv-dtype; the engine accounts for it.
- **Fit checks are per-GPU, not total ÷ N.** Use `parallel.fits` / `parallel.per_gpu`, never divide `grand_total_gib` by GPU count: MLA models replicate KV across TP ranks (see `parallel.kv_note`; `dp_attention=true` is the fix, at the cost of replicated attention weights — the engine models both sides). GQA KV stops sharding once TP exceeds kv-heads.
- **`parallel` KV is capacity vs demand.** `kv_pool_gib` is the SGLang-style pre-allocated pool; `kv_utilization_pct` > 100 means the requested context × concurrency doesn't fit — quote `max_concurrency` (and `max_concurrency_cluster` under dp-attention).
- **Roofline numbers are upper bounds.** Say "theoretical upper bound (实测通常 50–70%)", never "expected throughput". The robust takeaways are the memory/compute-bound verdict and the concurrency knee guidance in `roofline.decode.guidance`.
- **Not in the totals:** fixed overhead beyond the assumed `fixed_overhead_gib` and comm buffers vary — mention when the fit is tight.

## When something breaks or looks wrong

- **404 with "gated"**: the service cannot analyze gated/nonexistent repos by design (it holds no user tokens). Say so; offer a formula estimate from `references/methodology.md`, clearly labeled as such.
- **Service unreachable**: tell the user, and offer a rough formula-based estimate from `references/methodology.md` — label it clearly as unvalidated hand math, and recommend retrying the service for real numbers.
- **A number looks off for an exotic model**: fetch `https://huggingface.co/<id>/raw/main/config.json` and compare against the modeling rules in `references/methodology.md`; report suspected engine gaps to the service owner rather than hand-correcting silently.
- The engine is validated against GLM-5.2-FP8 (MLA+DSA+MTP), DeepSeek-V4-Flash (fp4/fp8 MQA), Qwen3-32B (dense GQA), Qwen2.5-7B-AWQ (int4 packing), Qwen3.5-4B (hybrid SSM), Kimi-K2.6 (VLM); live-server validated on g5 and 8×B200. `engine_version` in every response identifies the exact math revision.
