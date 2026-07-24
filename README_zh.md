# LLM Inference Analyzer

[English](README.md) | 中文

只给一个 HuggingFace 模型 ID,就能得到完整的推理部署分析——显存拆解、并行切分、性能上界,外加一份可交互的单文件 HTML 报告。

## 想解决什么问题

部署一个大模型之前,工程师总会想要搞清楚如下类似问题:

- **"DeepSeek-V4-Flash模型需要多少显存?能跑在 8×H100 上吗?"**
- **"Qwen3.5-27B模型用H100 部署，需要多少张卡？每张卡占多少、剩多少?"**
- **"DeepSeek-V4-Pro模型用H200 单机部署，能支持128K context × 64 并发吗?"**
- **"GLM-5.2-FP8模型在 B300 上的最优化参数是多少？"**
- **"GLM-5.2-FP8模型选择什么机型是最佳？"**
- **"GLM-5.2-FP8模型在B300 上理论上最大吞吐(token/s)是多少？"**
- **"Kimi-2.6模型在 B300 上理论上最快TTFT是多少？"**
- **"Qwen3-32B模型在 L40S 上有什么办法能降低显存占用"**


## 核心想法

这些问题的答案其实都已经写在模型仓库里,只是分散且需要正确解读。脚本基于三个数据源计算,无需下载权重:

1. **`config.json`** — 模型结构(层数、hidden size、MoE 专家数、注意力类型 MLA/GQA/MQA 等),据此按 Transformer 结构逐部件推导"应有参数量",以及 KV cache 每 token 的存储形态。
2. **safetensors 分片的 JSON 头部** — 通过 HTTP Range 请求只读每个分片开头的几百 KB,拿到所有张量的真实 dtype、shape、字节数。这是权重显存的 ground truth,能覆盖 config 说不清的情况(如 DeepSeek-V4 声明 fp8 但 experts 实为 fp4)。
3. **GPU 机型规格表** — 单卡显存、显存带宽、算力(AWS API 拉取,失败回退内置表),用于并行切分的逐卡 fit 检查和 roofline 性能上界。

为什么可行:前两个来源相互独立,可以**对账**——config 公式给出逻辑参数量,safetensors shape 给出表观参数量,两者的比值恰好暴露 fp4/int4 的 sub-byte 打包(表观值是逻辑值的 1/2、1/4 或 1/8),因此不依赖任何厂商特定字段就能还原真实位宽。权重之外的运行时内存(KV cache、activation)和性能上界都是结构参数的确定性函数,给定 context、并发、并行度即可精确推导。

## 使用方式

**1. 访问部署的 webapp**， 线上地址：<https://llm-inference-analyzer.ybalbert.people.aws.dev>

**2. 安装成 Agent Skill**(./llm-inference-analyzer.zip), 通过自然语言提问。


## 已验证的模型

> **说明：** 下列模型均通过**理论层面的验证**——即 config 公式推导与真实 safetensors 头信息（参数量、dtype、总字节数）相互对账一致，且内部守恒律自检全部通过。此外，其中一部分模型已经**与真实部署实测数据对照验证**（SGLang 真实 GPU 部署：每卡权重加载、KV cell size、显存预算闭合、dp-attention、decode/prefill roofline）。各实验会话与结论见 [`validate_experiments/`](validate_experiments/) 目录。

| model_id | 模型结构类型 | 参数规模(总 / 激活) | 实测验证 |
|---|---|---|---|
| deepseek-ai/DeepSeek-V4-Flash | MoE + MQA + DSA + MTP | 291B / 14B | ✅ 8×B200（S1、S1R） |
| deepseek-ai/DeepSeek-V4-Pro | MoE + MLA + DSA + MTP | 1.6T / 50B | — |
| zai-org/GLM-5.2-FP8 | MoE + MLA + DSA + MTP | 753B / 41B | ✅ 8×B200（S1、S1R） |
| nvidia/GLM-5.2-NVFP4 | MoE + MLA + DSA + MTP | 753B / 41B | ✅ 8×B200（S1） |
| Qwen/Qwen3-32B | Dense + GQA | 33B | ✅ 8×B200（S1、S1R） |
| moonshotai/Kimi-K2.6 | MoE + MLA + VLM | 1.03T / 33B | ✅ 8×B200（SV，vision tower） |
| MiniMaxAI/MiniMax-M3 | MoE + GQA + Sparse Attention + MTP + VLM | 427B / 27B | — |
| Qwen/Qwen3.5-4B | 混合架构（Linear Attention + GQA）+ VLM | 5B | ✅ 1×A10G（S0）+ 8×B200（S1R） |
| google/gemma-3-4b-it | Dense + GQA（sliding window）+ VLM | 5B | — |
