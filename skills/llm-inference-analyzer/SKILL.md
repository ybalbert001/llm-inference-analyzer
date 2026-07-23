---
name: llm-inference-analyzer
description: Analyze LLM inference deployment for any HuggingFace model — GPU memory (VRAM/显存) breakdown, TP/PP/EP parallelism partitioning across GPU nodes, and roofline performance bounds (memory- vs compute-bound, theoretical tokens/s) — and generate an interactive 4-tab HTML report. Use this whenever the user asks how much GPU memory a model needs, whether a model fits on specific GPUs or instances ("能跑在 8×H100 上吗"), how to shard a model with tensor/pipeline/expert parallelism, how much KV cache grows with context or concurrency, what throughput or decode speed to expect, whether a deployment is memory-bound or compute-bound, or wants a 显存拆解/并行切分/性能分析 diagram for a model given its HuggingFace ID. Also use it to compare quantization variants (fp8/fp4/AWQ/GPTQ) or KV-cache dtype choices.
---

# LLM Inference Analyzer

Given only a HuggingFace model ID, produces a full inference-deployment analysis:

1. **Evidence (推导依据)** — traces every weight number back to its source: config.json fields → structural formula → safetensors reconciliation bar chart (detects sub-byte packing), so a skeptical reader can audit the numbers.
2. **VRAM breakdown** — static weights per component (embed, attention, dense FFN, MoE experts, MTP, vision tower, quantization scales) + runtime memory (KV cache scaling with context × concurrency, linear/SSM state pools for hybrid models, activation workspace). Weight numbers come from real safetensors headers (ground truth, handles mixed precision like fp4 experts + bf16 attention).
3. **Parallelism partitioning** — how weights and KV shard across GPUs under TP/PP/EP on real instance types (AWS P5/P6 etc. or H800/H20 bare metal), per-GPU occupancy, KV-pool capacity vs demand (SGLang mem-fraction-static semantics), and MLA's DP-attention tradeoff including the weight-replication cost.
4. **Roofline performance** — decode/prefill arithmetic intensity, memory- vs compute-bound verdict, theoretical tokens/s upper bounds and TTFT estimates per component, with what-if controls for weight precision and chunked-prefill size; dp-attention aware.

All four views ship in one self-contained interactive HTML file (offline, dark-mode aware, live recompute on dropdown change).

## Every answer has two halves: the chat reply AND the HTML report

This skill is not "either a chat answer or a report" — each invocation should deliver both:

1. **Answer in chat, in words.** Lead with the bottom line ("不够，差 283 GiB" / "fp4 experts，共 148.6 GiB"), then the supporting numbers, extracted from the script's terminal report. The user is having a conversation — don't just point at a file, and don't dump the raw report.
2. **Always pass `--html <file>` and hand over the path.** The HTML is the reviewable/shareable artifact — dropdowns let the user explore what-ifs (context, concurrency, KV dtype, TP/PP/EP, instance type, weight precision, chunk size) that the chat answer didn't cover, and the tabs (`#evidence`, `#estimate`, `#parallel`, `#roofline`) go deeper than any single answer can. Generate it on the first run so follow-up questions can point back to it; name the file after the model (e.g. `qwen3-32b.html`) and mention which tab answers the user's question.

The conversation continues after the report exists: for follow-ups ("那并发降到 16 呢？" / "换 fp8 KV 呢？"), answer in chat — either read the value off the already-generated HTML's logic by re-running the script with adjusted flags, or point at the relevant dropdown in the existing report. Re-running costs seconds; never recompute by hand.

**Always ground your numbers in a script run** — never estimate VRAM or throughput from memory when the script is available.

## How to run

The bundled script does all the work — run it, don't re-derive the math by hand:

```bash
python3 scripts/main.py <org/model-id> [options]
```

Common invocations:

```bash
# Terminal report only, defaults (128K context × 16 requests, kv-dtype auto)
python3 scripts/main.py zai-org/GLM-5.2-FP8

# Full interactive 3-tab HTML report
python3 scripts/main.py deepseek-ai/DeepSeek-V4-Flash --html dsv4.html

# Custom deployment shape + parallelism initial values
python3 scripts/main.py Qwen/Qwen3-32B --context 32768 --requests 64 \
    --tp 4 --instance p5.48xlarge --kv-dtype fp8 --html qwen.html

# English-language output
python3 scripts/main.py Qwen/Qwen3-32B --lang en --html qwen.html
```

Key options (full list: `--help`):

| Flag | Default | Meaning |
|---|---|---|
| `--context` | 131072 | context length per request (drives KV cache) |
| `--requests` | 16 | concurrent running requests (KV scales linearly) |
| `--kv-dtype` | auto | `auto`/`bf16`/`fp16`/`fp8`/`fp4`; auto mirrors SGLang (fp8 for DSA/sparse-attention models, else model dtype) |
| `--batch-tokens` | 8192 | tokens per forward pass (drives activation estimate) |
| `--html FILE` | — | write the interactive 3-tab report |
| `--tp` / `--pp` / `--ep` | 8 / 1 / auto | initial parallelism sizes shown in the parallel tab |
| `--instance` | p5en.48xlarge | initial GPU node type (AWS types + `h800-8gpu`/`h20-8gpu`) |
| `--mem-fraction-static` | 0.9 | initial mem-fraction-static in the parallel tab (SGLang semantics: weights + KV pool pre-allocation) |
| `--fixed-overhead-gib` | 1 | per-GPU fixed overhead (CUDA context / NCCL buffers) |
| `--lang` | zh | `zh` or `en` for report and HTML |
| `--no-exact` | off | skip safetensors headers, formula-only estimate |
| `--overhead` | 0.05 | fragmentation allowance on the total |

Needs network access to huggingface.co. For gated repos, set `HF_TOKEN`. The script only range-requests safetensors JSON headers (a few hundred KB) — it never downloads weights. Instance specs are fetched via `aws ec2 describe-instance-types` when credentials exist, otherwise a built-in static table is used — both are fine.

The HTML supports deep links: `#evidence` (default tab, derivation audit trail), `#estimate` (VRAM breakdown), `#parallel` (sharding), `#roofline` (performance). If the user's question is about sharding or performance, mention the relevant tab (or hash URL) when handing over the file.

## Interpreting and reporting results

Lead with the bottom line — total GiB, and if the user named hardware, whether it fits and at what TP. Then the composition. Things to get right:

- **Weights vs runtime are different beasts.** Weights are paid once at load; KV cache grows linearly with `context × concurrent requests` (use the per-request increment to answer "how many concurrent users fit"). Activation is independent of concurrency.
- **"exact from safetensors" means the weight numbers are ground truth**, including mixed-precision and sub-byte-packed checkpoints. If the script fell back to formula mode (a `note:` on stderr), say the weight number is an estimate.
- **KV dtype is a deployment decision, not a model property.** `auto` mirrors SGLang defaults; other engines differ. Name the assumed dtype whenever the KV number matters. fp4 KV is aggressive/experimental — flag it. DSA models (GLM-5.x, DeepSeek-V4) additionally cache a per-token-per-layer fp8 index-key vector (~132 B) that does NOT shrink with kv-dtype — the script accounts for it automatically.
- **Hybrid models (linear/SSM + attention layers)** store paged KV only for the attention layers; linear layers keep a fixed per-request state pool that grows with concurrency, not context. Sliding-window layers cap storage at the window. The script classifies layers from `layer_types` and reports both pools.
- **Fit checks are per-GPU, not just total.** For "does it run on N × GPU?", use the parallel tab's per-GPU numbers rather than dividing the grand total by N: MLA models replicate KV across TP ranks (can OOM even when the total "fits" — the DP-attention toggle is the fix), and GQA KV stops shrinking once TP exceeds the KV-head count. DP-attention is not free: attention weights, embed, and MLA absorbed w_kc/w_vc replicate per rank (GLM-5.2 @TP8: ΔW ≈ +14.6 GiB/GPU, validated on 8×B200) — the parallel and roofline tabs both model this.
- **Roofline numbers are upper bounds.** Decode tokens/s assumes perfect overlap and no NCCL cost — real systems hit 50–70%. Say "theoretical upper bound", not "expected throughput". The memory-bound/compute-bound verdict and the "concurrency is free up to ~N" knee-point guidance are the robust takeaways. The tab has what-if dropdowns for weight precision (bf16/fp8/fp4 ideal conversion; default = checkpoint's real precision) and chunked-prefill size, and it honors the DP-attention toggle (replicated weights are read in full per GPU).
- **Not in the total**: per-GPU fixed overhead beyond the default 1 GiB assumption and multi-GPU comm buffers can vary. Mention this when the fit is tight.
- **Parallel tab KV is capacity, not demand.** The membar shows the SGLang-style pre-allocated KV pool (`mem-fraction-static × cap − fixed − weights`), matching what `nvidia-smi` reports on a live server. The per-GPU utilization bar (KV demand ÷ pool capacity) is the fit verdict: >100% means the requested context × concurrency doesn't fit — quote the "max concurrency ≈ N" readout.
- Models with built-in KV compression configs (`compress_ratios`, `sliding_window` on sparse-attention models) may use less KV than reported — the script computes the no-compression upper bound.

## When something breaks or looks wrong

The script is validated against GLM-5.2-FP8 (fp8, MLA+DSA+MTP, 703.7 GiB), DeepSeek-V4-Flash (fp4/fp8 mixed, MQA, 148.6 GiB), Qwen3-32B (dense GQA bf16), Qwen2.5-7B-AWQ (int4-in-int32 packing), Qwen3.5-4B (hybrid linear/SSM + GQA), and moonshotai/Kimi-K2.6 (VLM, vision tower byte-exact vs safetensors); live-server validation on g5 and 8×B200. If it crashes or a number looks off on a new model:

1. Fetch the config yourself and look for unusual fields: `curl -sL https://huggingface.co/<id>/raw/main/config.json`
2. Read `references/methodology.md` — it documents the full estimation approach (config formulas, safetensors reconciliation for sub-byte packing, MLA/GQA/MHA KV rules, TP/PP/EP sharding semantics, roofline component model, SGLang kv-dtype auto logic) and known limitations. Extend the script rather than hand-computing.
3. Cross-check totals against `model.safetensors.index.json` metadata `total_size` (fetch via `resolve/main/`, not `raw/main/` — the latter returns an LFS pointer for large files).

For multimodal models the script models the vision tower + projector (weights, ViT encoder transient activation, image-token count via `mm_tokens_per_image` or patches ÷ merge) — image tokens enter the ordinary text KV cache, so KV needs no separate pool. For models not on HuggingFace, or hypothetical configs, the script can't fetch — offer a formula-based estimate from `references/methodology.md` instead and label it clearly as such.
