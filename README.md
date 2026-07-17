# LLM Inference Analyzer

English | [中文](README_zh.md)

A Claude Code Skill: given nothing but a HuggingFace model ID, it produces a complete inference-deployment analysis — VRAM breakdown, parallelism partitioning, performance upper bounds — plus an interactive single-file HTML report.

## What problem does it solve

Before deploying a large model, engineers always face the same set of questions:

- **"How much GPU memory does this model need? Can it run on 8×H100?"**
- **"How should I shard it with TP/PP/EP? How much does each GPU hold, and how much is left?"**
- **"With 128K context × 64 concurrent requests, how big does the KV cache grow?"**
- **"Is decode memory-bound or compute-bound? What's the theoretical tokens/s?"**

## Core idea

The answers to all of these are already written in the model repository — just scattered, and easy to misread. The script computes from three data sources, without downloading any weights:

1. **`config.json`** — the model architecture (layer count, hidden size, number of MoE experts, attention type MLA/GQA/MQA, etc.), from which it derives the per-component "expected parameter count" following the Transformer structure, plus the per-token KV-cache storage shape.
2. **safetensors shard JSON headers** — via HTTP Range requests it reads only the first few hundred KB of each shard, obtaining the true dtype, shape, and byte size of every tensor. This is the ground truth for weight memory, covering cases the config can't express (e.g. DeepSeek-V4 declares fp8 but its experts are actually fp4).
3. **GPU instance spec table** — per-GPU memory, memory bandwidth, and compute (fetched via the AWS API, falling back to a built-in table), used for per-GPU fit checks in parallelism partitioning and for roofline performance bounds.

Why this works: the first two sources are independent and can be **reconciled** — the config formula gives the logical parameter count, the safetensors shapes give the apparent one, and their ratio exposes fp4/int4 sub-byte packing exactly (the apparent value is 1/2, 1/4, or 1/8 of the logical one), so the true bit width is recovered without relying on any vendor-specific fields. Beyond weights, runtime memory (KV cache, activations) and performance bounds are deterministic functions of the architecture parameters — given context length, concurrency, and parallelism, they can be derived precisely.

## How to use

**1. As a Claude Code Skill** — install `llm-inference-analyzer.skill` (or drop `skills/llm-inference-analyzer/` into your skills directory), then just ask in natural language.

**2. Run the script directly** (Python standard library only):

```bash
# Terminal report, defaults: 128K context × 16 concurrent requests
python3 scripts/main.py zai-org/GLM-5.2-FP8

# Full interactive 3-tab HTML report
python3 scripts/main.py deepseek-ai/DeepSeek-V4-Flash --html dsv4.html

# Custom deployment shape
python3 scripts/main.py Qwen/Qwen3-32B --context 32768 --requests 64 \
    --tp 4 --instance p5.48xlarge --kv-dtype fp8 --html qwen.html
```

## Validated models

> **Note:** the models below are currently validated **theoretically only** — the config-derived formulas are reconciled against the real safetensors headers (parameter counts, dtypes, total bytes), and internal conservation checks pass. They have **not yet been verified against real deployment measurements** (e.g. actual VRAM usage reported by SGLang/vLLM on GPUs). Empirical experiment data will be added to the [`validate_experiments/`](validate_experiments/) directory as it is collected.

| model_id | Architecture | Params (total / active) |
|---|---|---|
| deepseek-ai/DeepSeek-V4-Flash | MoE + MQA + DSA + MTP | 291B / 14B |
| deepseek-ai/DeepSeek-V4-Pro | MoE + MLA + DSA + MTP | 1.6T / 50B |
| zai-org/GLM-5.2-FP8 | MoE + MLA + DSA + MTP | 753B / 41B |
| nvidia/GLM-5.2-NVFP4 | MoE + MLA + DSA + MTP | 753B / 41B |
| Qwen/Qwen3-32B | Dense + GQA | 33B |
