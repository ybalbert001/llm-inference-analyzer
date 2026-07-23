# Methodology — llm-inference-analyzer

> 目录：一、技术思路（显存拆解：config 公式 + safetensors 精确值 / 混合精度与 sub-byte 打包识别 / KV cache / activation / vision tower）· 一B、并行切分口径（TP/PP/EP、dp-attention 复制）· 一C、性能 Roofline · 二、验证结果 · 三、注意事项与已知局限 · 四、关键细节备忘

输入一个 HuggingFace 模型 ID，自动拆解该模型部署所需的 GPU 显存，输出终端报告和一个四 TAB 的可交互单文件 HTML：

- **TAB 0「推导依据 / How We Know」（默认页，`#evidence`）**：config 字段 → 结构公式 → safetensors 对账的完整审计链，含双来源对账柱状图（sub-byte 打包检出显示半长条+⚡）与 index.json total_size 交叉核对
- **TAB 1「显存拆解」（design_1，`#estimate`）**：静态权重（逐部件，含 vision tower）+ 运行时内存（KV cache、linear/SSM state 池、activation）
- **TAB 2「并行切分」（design_2，`#parallel`）**：权重 + KV 按 TP/PP/EP 切分到 AWS GPU 节点，逐卡显示占用与剩余
- **TAB 3「性能 Roofline」（design_3，`#roofline`）**：Decode/Prefill 的 kernel 算术强度、理论瓶颈与性能上界，含权重精度 / chunked-prefill-size what-if 控件

```bash
python3 main.py zai-org/GLM-5.2-FP8 --html glm.html
python3 main.py deepseek-ai/DeepSeek-V4-Flash --context 131072 --requests 16 --html dsv4.html
python3 main.py Qwen/Qwen3-32B --kv-dtype fp8 --no-exact
python3 main.py Qwen/Qwen3-32B --tp 4 --instance p5.48xlarge --html qwen.html   # 并行 TAB 初始值
```

仅依赖 Python 标准库（AWS 机型规格用 aws cli 拉取，失败回退内置静态表）。gated 模型设置 `HF_TOKEN` 环境变量。

## 文件结构

| 文件 | 作用 |
|---|---|
| `main.py` | 主脚本：拉取配置、计算（显存拆解 + 并行切分）、渲染 |
| `template.html` | HTML/CSS 模板：页面骨架、样式和 `string.Template` 占位符 |
| `template.js` | 页面交互与显存、并行、Roofline 前端计算；生成时内联到 HTML |
| `*.html`（生成物） | 单文件、无外部依赖、离线可交互、自动适配深色模式；`#evidence`/`#estimate`/`#parallel`/`#roofline` hash 直达各 TAB |

四个 TAB 共用一份嵌入页面的 `viz_json` 数据；顶部筛选行的 context / 并发 / KV 精度为全局共享（推导依据页为纯静态，隐藏筛选行）。`template.js` 在生成时内联，因此输出仍是可离线打开的单文件。

---

## 一、技术思路

### 1. 两级数据来源：config 公式 + safetensors 精确值

**第一级（公式估算）**：拉取 `config.json`，按 Transformer 结构逐部件数参数：

- embed / lm_head：`vocab_size × hidden_size`（`tie_word_embeddings` 时只算一份）
- Attention：区分三种结构（见下）
- FFN：dense 层 `3 × hidden × intermediate_size`；MoE 层 `3 × hidden × moe_intermediate_size × n_routed_experts`，另加 shared experts 与 router/gate
- 特殊部件：DSA indexer（`index_n_heads` 存在时）、MTP 预测层（`num_nextn_predict_layers`）、norms
- 字节数 = 参数量 × dtype 字节（从 `quantization_config` / `torch_dtype` 判定）

**第二级（精确模式，默认开启）**：用 HTTP Range 请求只读取每个 safetensors 分片的 **JSON 头**（几百 KB，不下载权重），得到每个张量的真实 dtype、shape、字节数，按张量名正则归类到部件。这是混合精度 checkpoint 的唯一可靠来源。失败时自动回退公式估算（`--no-exact` 可强制跳过）。

> 注意：`model.safetensors.index.json` 必须用 `resolve/main/` URL 拉取——大文件在 `raw/main/` 下返回的是 git-lfs 指针而非 JSON（GLM-5.2 踩过此坑）。

### 2. 混合精度与 sub-byte 打包的自动识别（无需厂商提示）

**问题**：DeepSeek-V4-Flash 的 config 只声明 `quant_method: fp8`，但实际 MoE experts 是 **fp4**。若按「单一字节数 × 参数量」计算会高估 82%（271 vs 真实 148.6 GiB）。

**更隐蔽的问题**：fp4 权重在 safetensors 里不是以 4-bit dtype 存的，而是**两个 fp4 打包进一个 int8 字节**——header 里 dtype 显示 `I8`，shape 维度砍半（如 expert w1 存储 `I8 [2048, 2048]`，逻辑形状是 `2048 × 4096`）。直接读 dtype 会误标为 int8。

**解法——公式对账（reconciliation）**：手里有两个独立的参数量来源：

1. config 公式算出的「应有参数量」（与 dtype 无关的逻辑值）
2. safetensors shape 乘积的「表观参数量」

未打包时两者相等（偏差 <1%）；打包时表观值**恰好是公式值的 1/2、1/4 或 1/8**。检测规则：

```
部件含整数 dtype（I8/U8/I32...）张量，且 公式值/表观值 ≈ 2/4/8（±6% 容差）
→ 判定打包，参数量 × 倍数，位宽 = 存储位宽 ÷ 倍数
```

该方法不依赖任何厂商特定字段。实测：DeepSeek fp4-in-int8（2×）、AWQ int4-in-int32（8×，无任何 config 提示字段）均正确识别；GLM 纯 fp8 无误报。

4bit 的**名称**（fp4 vs int4）无法从字节判断，用 config 语义区分：`quant_method` 为 gptq/awq → int4；含 fp4/mxfp4/nvfp4 → fp4。只影响显示，不影响计算。

量化 scale 张量（`scales/scales_inv/qzeros/g_idx` 等）计入字节但不计入参数量，单独归为「量化scale」类（DeepSeek-V4-Flash 中占 6%）。

### 3. KV cache：结构判定 + 精度选择

**每 token 存储形态由 config 自动判定**（`kv_per_token_elems()`）：

| 判定 | 条件 | 每 token/层元素数 |
|---|---|---|
| MLA | 存在 `kv_lora_rank` | `kv_lora_rank + qk_rope_head_dim`（压缩 latent，全 head 共享） |
| GQA | `num_key_value_heads < num_attention_heads` | `2 × kv_heads × head_dim` |
| MHA | 两者相等 | `2 × heads × head_dim` |

实例：GLM-5.2 是 MLA（512+64=576）；Qwen3-32B 是 GQA 8 kv heads（2048）；DeepSeek-V4-Flash 是 GQA 1 kv head（=MQA，1024）——MQA 是 GQA 的极端形态，KV 节省效果与 MLA 同量级，但机制不同（结构上只留一份 vs 数学上压缩成 latent）。只有 MLA 分支才有「假如用 MHA 全存」的对比行。

**DSA index-key 附加项**：DSA 模型（有 `index_topk`，如 GLM-5.x、DeepSeek-V4）每 token 每层除 MLA latent 外还缓存一个 fp8 index-key 向量：`index_head_dim`（默认 128）字节 + 4 字节 scale = **132 B/token/层**。它随 KV 池分页但**不随 `--kv-dtype` 变**——GLM-5.x cell = layers × (576 × kv_bytes + 132)。漏算此项时 GLM 系 KV 偏差 fp8 轮 +23%、bf16 轮 +11.5%（8×B200 实测修正）。

**混合架构与滑窗（`kv_structure()` 按 `layer_types` 分类层）**：

- **full attention 层**：存全 context 分页 KV（block-sparse/DSA 变体也存全量——DSA 封顶的是*读取*不是*存储*）
- **sliding 层**：KV 存储封顶 `min(context, sliding_window)`；Gemma 式 `sliding_window_pattern`（每 swp 层 1 层 global）也能识别
- **linear/SSM 层**（Qwen3.5/GDN 式）：无分页 KV，改为每请求定长 state（conv state bf16 + ssm state），**随并发不随 context 增长**：`conv = (2·k_hd·n_k + v_hd·n_v) × (kernel−1) × 2 B`，`ssm = n_v·k_hd·v_hd × ssm_bytes`，对照 S0 g5 启动日志验证（conv 48 KiB + ssm 2 MiB/层/槽）。工具按 槽位=并发 计最小需求；SGLang 默认启发式可能预分配更多槽（建议部署时显式 `--max-mamba-cache-size`）
- `sliding_window` 存在但无法定位哪些层滑窗时**不封顶**（保守全 context，输出 warning）

KV 结果因此是「存储分组」列表 `[[层数, window], ...]` 而非单一乘法；roofline 侧的 decode 读取封顶（DSA top-k / 滑窗）是独立口径。

**KV 精度（`--kv-dtype`，默认 auto）**——KV 精度不是模型属性，config 里没有字段声明它，是推理引擎的运行时决策。auto 的解析逻辑对齐 SGLang 源码：

- 一般模型：auto → 模型 dtype（bf16）（`model_runner.py: configure_kv_cache_dtype`；除非权重 quant_config 带 `kv_cache_quant_algo: FP8`）
- DSA/稀疏注意力模型（有 `index_topk`，或架构为 DeepseekV4/V32）：auto → **fp8_e4m3**（`deepseek_v4_hook.py` 甚至 assert 只允许 fp8）
- SGLang 还支持 `fp4_e2m1`（mxfp4，CUDA 12.8+）：有效字节 = **0.5 + 1/16 ≈ 0.5625**/元素——`memory_pool.py` 中数据按 uint8 半宽存储，另配每 16 元素 1 字节的 scale buffer。因约束多（特定后端组合），永不进入 auto，需显式选择。

**KV 总量 = 每 token 字节 × context × 并发请求数**，随两者线性增长。

### 4. Activation 工作区

与 KV 不同，activation 与并发请求数无关，只与**单次 forward 的 token 数**（`--batch-tokens`，默认 8192，对应 vLLM chunked prefill 上限）有关——逐层执行时只有当前层的中间结果存活。

估算式（bf16）：`每 token ≈ 2B × (8 × hidden + 2 × inter_eff)`，其中 MoE 模型的 `inter_eff = (top_k + n_shared) × moe_intermediate_size`。这是工作区量级估算（与 vLLM profile 保留值同量级），非精确值。

### 4B. Vision tower（VLM）

`vision_tower_spec()` 从 `vision_config` 读取（字段别名覆盖 Kimi `vt_*` / Qwen `depth`/`num_heads` / MiniMax / Gemma）：

- **权重**：标准 pre-norm ViT block（qkv+o = 4H²+4H，MLP = 2HI+I+H，2 LN = 4H）× 层数，加 patch-embed conv、可选 pos-emb、final LN 与 projector（Kimi patchmerger：pre_norm + (H·merge)² + (H·merge)·text_H；通用回退：单 linear 到 text hidden）。对 moonshotai/Kimi-K2.6 safetensors **字节级精确**（471,143,920 params）。量化 checkpoint 中 vision tower 保持 bf16（quant config 不覆盖 vision_tower/mm_projector），故权重字节下限 2 B/参数。
- **运行时**：图像 token 就是普通 KV token（与文本同 cell，占 context 位置），KV 无需单独池；真正额外的是 ViT encoder 对 max_patches 的**瞬时** activation（编码期，非常驻）。每图 token 数优先取 `mm_tokens_per_image`（Gemma 池化 projector 固定 256/图），否则 `max_patches ÷ merge²`。
- **TP 切分（分数切分）**：只有 VisionAttention 的 qkv/o 按 attn-TP 切（`visionAttnFrac` 份额），ViT MLP 是普通 `nn.Linear` 每卡整份复制；vision tower 权重挂在 PP stage-0（与 embedding 同 stage），encoder activation 中 MLP 部分不随 TP 减少。

8×B200 Kimi-K2.6 实测：权重每卡 −0.04%、KV 池 +0.05%。

### 5. HTML 拆解图

- **布局**：左栏层结构示意（embed → dense 层 → MoE 层 → MTP → lm_head，颜色点对应右侧）；右栏分「静态·模型权重」和「动态·运行时内存」两组卡片；底部权重堆叠条 + 总占用堆叠条 + 可折叠表格视图
- **交互**：顶部筛选行三个下拉框——context（默认 32K~1M）、并发数（默认 1~1024）、KV 精度（auto/bf16/fp8/fp4）。Python 把不变量（每 token KV 元素数、权重字节、auto 解析结果等）以 JSON 嵌入页面，JS 在 change 事件里重算并更新所有带 id 的节点：KV 卡片推导链、标题、总占用条、表格行。静态权重部分不参与联动
- **精度标注**：每张卡片标题旁有统一样式的精度徽章（如 `fp4`、`fp8 90% + bf16 10%`），数据来自 safetensors 真实 dtype 按字节占比汇总（忽略 scale）；KV 卡片徽章动态显示当前选择（如 `fp8（auto）`）；表格视图有独立精度列
- 下拉框选项可用 `--ctx-options` / `--req-options` 自定义；CLI 当前值自动并入选项

---

## 一B、并行切分可视化（并行 TAB，design_2）

把模型权重与 GPU 节点硬件结合，可视化 TP/PP/EP 并行下每张卡分到哪些部件、占多少显存、剩多少给 KV。页面内所有并行参数（TP/PP/EP、机型、context、并发、KV 精度、DP attention、mem-fraction-static）实时切换重算；显存条色块 hover 显示各部件占用明细。CLI 的 `--tp/--pp/--ep/--instance/--fixed-overhead-gib/--mem-fraction-static/--tp-options/--pp-options` 设定初始值。

### 切分口径（对齐 SGLang/vLLM 语义）

**TP（每层内部切）**：
- attention 有 head 维的矩阵（q/o proj；MLA 的 q_b/kv_b/o_proj）每卡 1/TP；MLA 的 q_a/kv_a 产生全 head 共享 latent，**每卡整份复制**；GQA 的 k/v proj 按 kv head 切，最多切 `n_kv_heads` 份（TP 更大时开始复制）
- FFN / MoE 专家按 intermediate 维切 1/TP；embed/lm_head 按词表切 1/TP（vocab-parallel）；norms、MoE router 每卡复制

**PP（层间分段）**：层均分到 stage（不整除时余数给后面的 stage）；embed 在 stage-0，lm_head/MTP 在末 stage；`tie_word_embeddings` 且 PP>1 时末 stage 再持一份 embedding。activation 不随 PP 减少（每 stage 仍跑完整 microbatch）。

**EP（MoE 专家分组）**：约束 EP | TP。**EP 不改变每卡显存字节数**——EP=TP 每卡持 E/TP 个完整专家，EP=1 每卡持全部专家的 1/TP 切片，字节都是 routed 总量/TP。EP 改变的是切分形状（完整专家 vs 切片）与通信模式（all-to-all vs all-reduce），图中体现在专家小格与文案。

**KV cache 切分**（对每卡估算影响最大的一项）：
- GQA/MHA：按 kv head 切，每卡 1/min(TP, n_kv_heads)；TP 超过 kv_heads 后不再下降、开始复制（Qwen3-32B 8 kv heads 在 TP16 时每卡 KV = 总量/8）
- **MLA：latent 无 head 维，纯 TP 下每卡全量复制**（GLM-5.2 128K×16 并发 fp8 ≈ 137 GiB/卡，直接 OOM）。页面提供 **DP attention** 开关（对齐 SGLang `--enable-dp-attention`）：开启后 KV 按 TP 切 1/TP，代价是 attn_tp 组（=tp/dp=1）上的部件每卡整份复制——attention 小 KV 大，通常划算。这一对比是该图的核心演示场景。

**dp-attention 复制口径（经 SGLang 源码逐条确认，E4 实测 ΔW +14.63 GiB 验证）**——复制的不只是 attention 投影：

1. **embed 整份复制**（`VocabParallelEmbedding(use_attn_tp_group=...)`）；lm_head 不复制（`enable_dp_lm_head` 默认 False，仍 ÷tp）
2. **MLA 权重吸收物化**：加载时 kv_b_proj 被 dequant 成 bf16 w_kc/w_vc（fp8 原件保留、两份并存），不在 safetensors 口径内；形状含 num_local_heads 随 attn_tp 切——GLM-5.2 纯 TP8 每卡 0.27 GiB、dpAttn 每卡 2.13 GiB
3. **NextN draft embed**：draft 在 init 时按自己的分片规则分配一份完整 bf16 embed（加载后 alias 释放，但 KV 池 sizing 在释放前水位，故占池容量）；dpAttn 下整份复制。MTP 的 attention 份额同理转整份，expert FFN 仍 ÷tp

JS 侧统一用 `attnTpDiv = dpAttn ? 1 : tp` 切这些部件。验证（GLM-5.2 @8×B200）：纯 TP8 预测 89.94 vs 实测 91.43，DP8 预测 104.52 vs 实测 106.06，ΔW 14.58 vs 14.63（−0.3%）。dpAttn 下每 rank KV 池按完整 cell 报（与 SGLang 日志 `max_total_num_tokens` 直接对账），汇总行加 ×DP=集群。

**KV 容量口径（对齐 SGLang `--mem-fraction-static`）**：每卡显存按 mem-fraction-static（滑块，默认 0.9）分为静态区与非静态区。静态区 = 权重 + 每卡固定开销（默认 1 GiB，`--fixed-overhead-gib` 可调）+ **KV 池**——KV 池自动填满静态区剩余空间（`frac × cap − fixed − weights`），与 SGLang 启动时的预分配行为一致，显存条显示的即真实占用（可与 `nvidia-smi` 对账）。activation（近似按 1/TP）与 CUDA graph 落在非静态区。每卡另有 KV 利用率条：**KV 需求**（`ctx × 并发 × cell size` 按切分口径折算到每卡）÷ **KV 池容量**，>100% 标红表示该并发跑不满；并给出该容量支持的最大并发反推值。权重放不下静态区时显示"无法启动"。本 TAB 不再使用 design_1 的乘性碎片 5%——非静态区本身就是余量，再乘会重复扣减。

### GPU 机型

规格在生成时用 `aws ec2 describe-instance-types` 拉取（GPU 名/每节点数量/单卡显存），无 AWS 凭证时回退内置静态表（2026-07 快照）。内置机型：p6-b300 / p6-b200 / p5en / p5 / p4de / p4d / g6e.48xl / g6e.12xl / g5.48xl；页面另有"自定义"（单卡 GiB × 每节点卡数）。

### 实现要点

- **PP 不需要真 per-layer 数组**：模型只有 dense 层与 MoE 层两种原型，同型层字节相同。Python 侧把 analyze() 的聚合字节均摊成两个层原型（`per_layer_breakdown`），JS 按 stage 层区间做算术。生成时自检：层原型 × 层数重建总量，偏差 <0.5%（实测 0.000%）。
- attention 的 sliced/replicated 拆分按公式参数比例分摊 exact 字节（`attn_tp_partition`）。
- 守恒律自检（历史上用独立脚本复算过）：Σ(各卡权重) = 模型总量 + 闭式复制冗余；KV 复制倍数 = TP（MLA）或 TP/min(TP, n_kv)（GQA）。GLM-5.2 与 Qwen3-32B 均通过（dev < 1e-15）。另有 `validate_experiments/S1_b200_session/scripts/predict_from_html.py` / `predict_roofline.py` 从生成 HTML 提取 viz_json 复算 JS 逻辑，与实机对账。

---

## 一C、性能 Roofline（Roofline TAB，design_3）

回答"跑起来受带宽限制还是算力限制、上界是多少、优化投给谁"。图上只画 **Decode（当前并发）** 和 **Prefill（当前 chunk）** 两个聚合点（落在屋顶线上，强度 = 各部件 FLOPs 之和 ÷ 字节之和）；每个点下方一张卡片，内含**部件拆解表**（Dense GEMM / MoE experts / KV 读取 × TFLOPs / HBM 读写 / 强度 / 受限于 / 时间占比 / 时间 ms·单卡）+ 吞吐上界推导 + 处方文案。`#roofline` hash 直达。

**Tab 内 what-if 控件**：
- **权重精度**下拉（bf16/fp8/fp4）：默认选中 checkpoint 自身精度（safetensors 实测字节，逐位与原行为一致）；切换到其他精度走理想换算（params × bytes/param：bf16 2 / fp8 1 / mxfp4 0.5625 含 scale），峰值线同步切换
- **chunked-prefill-size** 下拉（1024~32768，默认 `--batch-tokens` 生成值）：只作用于 roofline，显存 TAB 的 activation 口径不变
- **DP attention 开关**（与并行 TAB 共享）也作用于 roofline，见下

**算术强度 = FLOPs ÷ HBM 搬运字节，全部由模型 config + 部署参数闭式算出**（字节复用显存拆解 TAB 的部件字节）。部件口径（decode，B=并发）：

| 部件 | FLOPs | HBM 字节 | 含义 |
|---|---|---|---|
| Dense GEMM | 2 × 非专家参数 × B | 非专家权重（每步读一遍） | 强度 ≈ B ÷ 字节/参数，加并发线性涨 |
| MoE experts | 2 × (topk/E) × routed参数 × B | routed权重 × min(1, B×topk/E) | 小并发就碰到很多不同专家——读得多算得少 |
| Attention core | 每 pair FLOPs × query-key pairs | 每 key KV 元素 × 实际读取 keys × dtype 字节 | 计算按 Q heads，缓存按 KV 结构；DSA 只读取 top-k selected KV |

Attention core 把**几何结构**和**访问模式**正交组合：

- MHA/GQA：每 pair FLOPs = `2 × q_heads × (qk_dim + v_dim)`；每 key KV 元素 = `kv_heads × (qk_dim + v_dim)`。GQA 的计算按 Q heads，缓存按较少的 KV heads。
- absorbed MLA：每 pair FLOPs = `2 × q_heads × (2 × kv_lora_rank + rope_dim)`；每 key 只存 `kv_lora_rank + rope_dim` 个压缩元素。
- dense pattern：decode 每请求访问全部 context；prefill causal pairs = `T × (T+1) / 2`。
- DSA pattern：decode 每请求访问 `min(context, index_topk)`；prefill 使用 top-k capped causal pairs。DSA indexer 扫描完整 context 的检索成本当前未建模，indexer 行只计算其投影 GEMM。

**混合架构（linear/SSM + attention）**：三个 attention pattern（dense/DSA/capped）全部只按 `kvLayers`（=存 KV 的层数）计 KV 读取和二次项 FLOPs，不再把全部 L 层当 full attention；linear 层另生成 `linear_state` kernel 行——decode 每步读+写定长 state（`2 × state × B` 字节），prefill 每 chunk 一遍（O(1) 不随 chunk 长度增长）。修正前 Qwen3.5-4B 65K TTFT 高估 +69%。

**dp-attention 与复制部件（roofline 侧）**：按「复制倍数」统一建模——全局字节 × factor 后共享 ÷TP，即每卡读整份：

- attention 权重：dpAttn 下 ×tp（每卡整份，GLM-5.2 decode 每步 12.9 GB 而非 ÷8 的 1.6 GB）
- indexer / moe_gate：纯 TP 下本就每卡复制（×tp，对齐内存模型）
- attn core KV 读：极性与权重**相反**——纯 TP 下 MLA latent 每卡读全量（×tp）、GQA 按 min(tp, kv_heads) 切；dpAttn 下各卡只读自己 B/tp 请求的 KV（×1）
- FLOPs 均不变：每卡只算自己的 token/head 份额

Prefill 的 KV HBM 字节按 KV 一次读取/写入计算，是不包含 FlashAttention tiling 和 DSA gather 重读的理想下界。部件时间 = max(字节/带宽, FLOPs/峰值)，**时间占比列直接指出优化对象**；明细表另有「时间 (ms，单卡)」列，chunk TTFT 公式按 ∑÷TP 推导展示。

**处方数字**（判定卡 vformula 行）：
- decode step ≈ Σ字节 ÷（带宽 × TP）→ 单请求/TP 组 tokens/s 理论上界（标注实测通常 50–70%；TP 组各卡并行读自己分片，忽略 NCCL 开销）
- 距拐点倍数 → **并发加到 ~N 之前吞吐近似白捡**（memory-bound 下 step 时间由搬运决定，几乎不随 batch 变）
- prefill：一个 chunk 毫秒数 → 处理整个 prompt 的秒数（TTFT 量级）

判定：聚合强度 < 拐点（峰值÷带宽）→ memory-bound。峰值按当前权重精度选（默认 checkpoint 主导 dtype；what-if 切换时跟随；GPU 不支持时回退 bf16 并注明）。纯 TP 下强度与 TP 无关、绝对时间随 TP 缩短；dpAttn/复制部件会改变每卡字节，强度随之变化。

GPU 算力/带宽为内置静态表 `GPU_PERF`（datasheet dense 口径近似，无 sparsity）：H100/H200/H800 989 bf16 / 1979 fp8（H800 为 H100 die，仅 NVLink 被砍），H20 148/296（算力大砍、HBM3 96GB），B200 2250/4500/9000(fp4)，B300 同 B200 但 fp4 13500，A100 312 bf16，L40S 362/733，A10G 125；带宽 H100/H800 3.35、H200 4.8、H20 4.0、B200/B300 8.0、A100 2.0、L40S 0.864、A10G 0.6 TB/s。H800/H20 为非 AWS 裸机节点（SXM，8×GPU/节点，`h800-8gpu`/`h20-8gpu`）。自定义机型无规格 → 提示选预设机型。

实例（GLM-5.2-FP8 @ H200 fp8 TP8，128K×16，MLA + DSA top-2048，拐点 ≈412）：Decode 聚合强度 4.3 → memory-bound，step 理论上界约 10.0 ms → 单请求约 100 tok/s、TP 组约 1605 tok/s；Prefill 强度约 1102 → compute-bound 2.7×，8K chunk 约 52 ms → 128K prompt 约 0.8 s。以上不含 DSA indexer 的 full-context 检索、TP 通信与 kernel 效率损失。

---

## 二、验证结果

| 模型 | 特点 | 权重结果 | 对照 |
|---|---|---|---|
| GLM-5.2-FP8 | 755B MoE、MLA、DSA、MTP、fp8 | 703.7 GiB | 与手工推导图一致（MoE 675 GiB 占 96%） |
| DeepSeek-V4-Flash | 291B MoE、MQA、fp4+fp8 混合精度 | 148.6 GiB | 与 index 声明的 148.65 GiB 完全一致 |
| Qwen3-32B | 稠密、GQA、bf16 | 61.0 GiB | 与官方参数量一致 |
| Qwen2.5-7B-AWQ | int4 AWQ（int32 打包） | 5.2 GiB | config-only 估算会低估至 3.5 GiB（embed/norm 仍 fp16 + scale 开销） |
| Qwen3.5-4B | 混合 linear/SSM + GQA | — | S0 g5 实机：conv/ssm state 对照启动日志逐项吻合 |
| Kimi-K2.6 | VLM（vision tower + patchmerger） | vision 471.1M params | safetensors 字节级精确；8×B200 实机权重每卡 −0.04% |

实机（SGLang live server）验证：S0（g5.2xlarge）、S1（8×B200，GLM-5.2 / DSv4 等，含 dp-attention ΔW、KV cell、mem 闭合、roofline 上界性），详见 `validate_experiments/`。

---

## 三、注意事项与已知局限

1. **KV cache 精度是部署决策不是模型属性**。auto 只是对齐 SGLang 当前默认；vLLM/TensorRT-LLM 规则不同。fp4 KV 较激进，长上下文精度影响未被广泛验证，生产前需自行评测。
2. **DeepSeek-V4-Flash 的 KV 是保守上界**。config 里的 `sliding_window: 128`、`compress_ratios`（4×/128× 交替）、`num_hash_layers` 表明有内建 KV 压缩机制，真实占用可能显著小于图中「全量 GQA」口径；等推理框架落地后可把 `compress_ratios` 语义编入。
3. **Activation 是量级估算**，真实值受 CUDA graph、attention kernel 实现影响。KV cache 部分是精确公式。
4. **MHA 对比口径**：MLA 卡片的「假如用 MHA」按 K+V 全存计算（`heads × (qk_head_dim + v_head_dim)`）；有些资料只算 K 或用对称口径，数值会差 2×。
5. **总占用 = 权重 + KV + linear/SSM state + activation + vision encoder activation + 碎片 5%**（`--overhead` 可调），未含 CUDA context（每卡 ~0.5-1 GiB）和多卡并行的通信 buffer（并行 TAB 用 `--fixed-overhead-gib` 单独建模，默认 1 GiB；S1 实测 TP8 约 2.5 GiB，紧张时注意）。
6. **精确模式的网络依赖**：需能访问 huggingface.co；每个分片一次小的 Range 请求（46 分片模型约几秒）。失败自动回退公式。
7. **多模态模型**：文本侧取 `text_config` 子树；vision tower + projector 已建模（权重、瞬时 activation、图像 token 数），但 ViT encoder 的 roofline kernel 成本未建模（图像编码耗时不在 TTFT 估算内）。
8. **公式覆盖范围**：MLA/GQA/MHA、MoE（含 shared/gate/first_k_dense_replace）、DSA indexer、MTP、q/o 低秩分解、混合 linear/SSM 层、滑窗层、vision tower。全新结构（如 V4 的 hyper-connection `hc_*` 参数）在精确模式下会被归入 norms & misc 兜底类——字节数不丢，只是分类粗。
9. **DSA indexer 检索成本未建模**：indexer 对完整 context 的扫描/打分（decode 与 prefill 两侧）不在 kernel 表内，indexer 行只含投影 GEMM——S1R 实测 GLM 系 prefill 有 +25% 量级额外耗时与此相关。

---

## 四、Session 中确认过的关键细节

- **GLM-5.2 图中 285.6 vs 本工具 571.3 GiB 的 MHA 对比差异**：口径问题（只算 K vs K+V 全存），MLA 实际占用 10 GiB 两边一致。
- **`intermediate_size` 可以缺失**（V4-Flash 全 MoE 无 dense 层），需容错。
- **`first_k_dense_replace` 为 0** 时左栏不画 Dense 层。
- **fp4 打包证据**：expert `w1.weight I8 [2048,2048]` vs 逻辑 `2048×4096` = 0.5 字节/参数；scale `F8_E8M0 [2048,128]` = 每 32 元素一个（MXFP4 block-32）；shared expert 对照组 `F8_E4M3 [2048,4096]` 全形状。
- **SGLang `--kv-cache-dtype` 默认 `auto`**；DSA 模型按 GPU 代际（SM≥10 → fp8）；V4 钩子强制 fp8；一般模型 → 模型 dtype。
- **SGLang fp4 KV 池布局**：`k/v_buffer` 半宽 uint8 + `k/v_scale_buffer` 每 16 元素 1 字节（`scale_block_size = 16`），故有效 0.5625 字节/元素，相对 fp8 实际省 1.78× 而非 2×。
