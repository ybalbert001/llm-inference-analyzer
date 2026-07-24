# LLM Inference Analyzer

English | [中文](README_zh.md)

Given nothing but a HuggingFace model ID, it produces a complete inference-deployment analysis — VRAM breakdown, parallelism partitioning, performance upper bounds — plus an interactive single-file HTML report.

## What problem does it solve

Before deploying a large model, engineers always want to figure out questions like these:

- **"How much GPU memory does DeepSeek-V4-Flash need? Can it run on 8×H100?"**
- **"How many H100 GPUs does Qwen3.5-27B need? How much does each GPU hold, and how much is left?"**
- **"Can DeepSeek-V4-Pro on a single H200 node support 128K context × 64 concurrent requests?"**
- **"What are the optimal deployment parameters for GLM-5.2-FP8 on B300?"**
- **"Which instance type is the best choice for GLM-5.2-FP8?"**
- **"What's the theoretical max throughput (tokens/s) of GLM-5.2-FP8 on B300?"**
- **"What's the theoretical best TTFT of Kimi-2.6 on B300?"**
- **"How can I reduce the GPU memory footprint of Qwen3-32B on L40S?"**

## Core idea

The answers to all of these are already written in the model repository — just scattered, and easy to misread. The script computes from three data sources, without downloading any weights:

1. **`config.json`** — the model architecture (layer count, hidden size, number of MoE experts, attention type MLA/GQA/MQA, etc.), from which it derives the per-component "expected parameter count" following the Transformer structure, plus the per-token KV-cache storage shape.
2. **safetensors shard JSON headers** — via HTTP Range requests it reads only the first few hundred KB of each shard, obtaining the true dtype, shape, and byte size of every tensor. This is the ground truth for weight memory, covering cases the config can't express (e.g. DeepSeek-V4 declares fp8 but its experts are actually fp4).
3. **GPU instance spec table** — per-GPU memory, memory bandwidth, and compute (fetched via the AWS API, falling back to a built-in table), used for per-GPU fit checks in parallelism partitioning and for roofline performance bounds.

Why this works: the first two sources are independent and can be **reconciled** — the config formula gives the logical parameter count, the safetensors shapes give the apparent one, and their ratio exposes fp4/int4 sub-byte packing exactly (the apparent value is 1/2, 1/4, or 1/8 of the logical one), so the true bit width is recovered without relying on any vendor-specific fields. Beyond weights, runtime memory (KV cache, activations) and performance bounds are deterministic functions of the architecture parameters — given context length, concurrency, and parallelism, they can be derived precisely.

## How to use

**1. Use the deployed webapp** — live at <https://llm-inference-analyzer.ybalbert.people.aws.dev>

**2. Install as an Agent Skill**(./llm-inference-analyzer.zip), then just ask in natural language.

## Validated models

> **Note:** all models below pass **theoretical validation** — the config-derived formulas are reconciled against the real safetensors headers (parameter counts, dtypes, total bytes), and internal conservation checks pass. In addition, a subset has been **verified against real deployment measurements** (SGLang on real GPUs: per-GPU weight loading, KV cell size, memory-budget closure, dp-attention, decode/prefill roofline). See the [`validate_experiments/`](validate_experiments/) directory for the experiment sessions and conclusions.

| model_id | Architecture | Params (total / active) | Empirical validation |
|---|---|---|---|
| deepseek-ai/DeepSeek-V4-Flash | MoE + MQA + DSA + MTP | 291B / 14B | ✅ 8×B200 (S1, S1R) |
| deepseek-ai/DeepSeek-V4-Pro | MoE + MLA + DSA + MTP | 1.6T / 50B | — |
| zai-org/GLM-5.2-FP8 | MoE + MLA + DSA + MTP | 753B / 41B | ✅ 8×B200 (S1, S1R) |
| nvidia/GLM-5.2-NVFP4 | MoE + MLA + DSA + MTP | 753B / 41B | ✅ 8×B200 (S1) |
| Qwen/Qwen3-32B | Dense + GQA | 33B | ✅ 8×B200 (S1, S1R) |
| moonshotai/Kimi-K2.6 | MoE + MLA + VLM | 1.03T / 33B | ✅ 8×B200 (SV, vision tower) |
| MiniMaxAI/MiniMax-M3 | MoE + GQA + Sparse Attention + MTP + VLM | 427B / 27B | — |
| Qwen/Qwen3.5-4B | Hybrid (Linear Attention + GQA) + VLM | 5B | ✅ 1×A10G (S0) + 8×B200 (S1R) |
| google/gemma-3-4b-it | Dense + GQA (sliding window) + VLM | 5B | — |
