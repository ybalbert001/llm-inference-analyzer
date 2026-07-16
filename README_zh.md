# LLM Inference Analyzer

[English](README.md) | 中文

一个 Claude Code Skill:只给一个 HuggingFace 模型 ID,就能得到完整的推理部署分析——显存拆解、并行切分、性能上界,外加一份可交互的单文件 HTML 报告。

## 想解决什么问题

部署一个大模型之前,工程师总要回答同一组问题:

- **"这个模型需要多少显存?能跑在 8×H100 上吗?"**
- **"TP/PP/EP 该怎么切?每张卡占多少、剩多少?"**
- **"128K context × 64 并发,KV cache 会涨到多大?"**
- **"decode 是 memory-bound 还是 compute-bound?理论上能跑多少 tokens/s?"**

## 核心想法

这些问题的答案其实都已经写在模型仓库里,只是分散且需要正确解读。脚本基于三个数据源计算,无需下载权重:

1. **`config.json`** — 模型结构(层数、hidden size、MoE 专家数、注意力类型 MLA/GQA/MQA 等),据此按 Transformer 结构逐部件推导"应有参数量",以及 KV cache 每 token 的存储形态。
2. **safetensors 分片的 JSON 头部** — 通过 HTTP Range 请求只读每个分片开头的几百 KB,拿到所有张量的真实 dtype、shape、字节数。这是权重显存的 ground truth,能覆盖 config 说不清的情况(如 DeepSeek-V4 声明 fp8 但 experts 实为 fp4)。
3. **GPU 机型规格表** — 单卡显存、显存带宽、算力(AWS API 拉取,失败回退内置表),用于并行切分的逐卡 fit 检查和 roofline 性能上界。

为什么可行:前两个来源相互独立,可以**对账**——config 公式给出逻辑参数量,safetensors shape 给出表观参数量,两者的比值恰好暴露 fp4/int4 的 sub-byte 打包(表观值是逻辑值的 1/2、1/4 或 1/8),因此不依赖任何厂商特定字段就能还原真实位宽。权重之外的运行时内存(KV cache、activation)和性能上界都是结构参数的确定性函数,给定 context、并发、并行度即可精确推导。

## 使用方式

**1. 作为 Claude Code Skill** — 安装 `llm-inference-analyzer.skill`(或把 `skills/llm-inference-analyzer/` 放入 skills 目录),自然语言提问。

**2. 直接运行脚本**(仅依赖 Python 标准库):

```bash
# 终端报告,默认 128K context × 16 并发
python3 scripts/main.py zai-org/GLM-5.2-FP8

# 完整三 TAB HTML 报告
python3 scripts/main.py deepseek-ai/DeepSeek-V4-Flash --html dsv4.html

# 自定义部署形态
python3 scripts/main.py Qwen/Qwen3-32B --context 32768 --requests 64 \
    --tp 4 --instance p5.48xlarge --kv-dtype fp8 --html qwen.html
```

## 已验证的模型

| model_id | 模型结构类型 | 参数规模(总 / 激活) |
|---|---|---|
| deepseek-ai/DeepSeek-V4-Flash | MoE + MQA + DSA + MTP | 291B / 14B |
| deepseek-ai/DeepSeek-V4-Pro | MoE + MLA + DSA + MTP | 1.6T / 50B |
| zai-org/GLM-5.2-FP8 | MoE + MLA + DSA + MTP | 753B / 41B |
| nvidia/GLM-5.2-NVFP4 | MoE + MLA + DSA + MTP | 753B / 41B |
| Qwen/Qwen3-32B | Dense + GQA | 33B |
