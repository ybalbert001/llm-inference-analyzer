# Validate Experiments / 实测验证数据

## 验证思路

初步打算针对不同的机型（B300 / B200 / H200 / H100），通过 sglang 部署进行实验。包括下面几个部分

 - 显存拆解部分
	 - 针对 weight，**只对齐总量**，暂不做分部件（attention / experts / embed）级别的对账。原因：
		 - 分部件字节数的 ground truth 本来就是 safetensors 头信息，工具已直接读取；加载到 GPU 不改变各部件大小，逐部件实测只是重复验证，且需要修改 sglang 代码。
		 - 总量对账已足以抓住真正可能出偏差的场景：运行时 dtype 转换（如 fp8 checkpoint 在不支持 fp8 的 GPU 上被上转为 bf16、在线量化）——这类转换会让实测总量偏离 safetensors 口径，正是需要发现的问题。
		 - 实测方法：取 SGLang 启动日志中权重加载前后的 `avail mem` 差值（`Load weight begin/end`），即为每卡权重实际显存。无需修改 sglang 代码。
		 - **对账前提**：纯 TP（不开 `--enable-dp-attention`、不开 EP），差值只取权重加载段（不含 CUDA graph 与常驻 buffer）；部署时**开启 MTP speculative decoding**——SGLang 只在开 MTP 时才加载 MTP 权重，开启后实测与工具静态总量（含 MTP）口径一致，无需扣减。
		 - **对账基准**：用工具并行 TAB 的每卡 weights 数，**不是**总量 ÷ TP——纯 TP 下也有每卡复制的部分（MLA 的 kv_a/rope 投影、norm、DSA indexer 等），工具公式为 `sliced/TP + replicated`。
		 - **判定标准**：允许 **1–2% 的误差**（allocator 保留粒度、TP 不整除 padding、fp8 scale 等零碎）；超过 2% 视为真实偏差（优先排查运行时 dtype 转换或组件漏算）。
	 - 针对 kv cache，分两层验证：先验证 **cell size（每 token 每 layer 的 KV 字节数 × 层数，即每 token 的 KV 占用）**，再验证**池子总量**。
		 - SGLang 的 KV pool 是供给侧驱动的——启动时把 `mem_fraction_static` 扣除权重后的剩余显存全部预分配给 KV pool，其大小与工作负载（context/并发）无关。工具的并行 TAB 同时建模两侧：需求侧 `context × requests × cell`（利用率条的分子），供给侧 `frac × 单卡显存 − 固定开销 − weights`（KV 池容量，即 `max_total_num_tokens × cell`）。供给侧池子总量是工具的直接预测值，其对账即 E3 的主验证目标。
		 - cell size 是需求/供给换算中唯一的原子量（池子字节 ↔ token 数的汇率），单独先验证；池子总量的偏差才能归因到 weights / 固定开销 / frac 三个成分上。
		 - 实测方法：从启动日志 / `/get_server_info` 读取 KV pool 字节数与 `max_total_num_tokens`，`实测 cell size = KV pool 字节数 ÷ max_total_num_tokens`，与工具理论值对账（例：DeepSeek-V4-Pro fp8 = (512+64) × 61 层 ≈ 61 KiB/token）。无需修改 sglang 代码。
		 - 注意口径：开启 MTP speculative decoding 时 cell size 会多一层；kv-cache-dtype 需两边对齐。
 - 并行切分部分 （可以基于 CC 进行自动化实验）
	 - 验证不同 kv-cache-dtype × mem-fraction-static 组合下，工具的整本显存账是否闭合。方法：**直接对账**——固定 `--mem-fraction-static` 启动（如默认 0.9），从日志读实测 `max_total_num_tokens`，与工具并行 TAB 同一 frac 下预测的池容量 tokens 对账（工具供给侧公式：`池 tokens = (frac × 单卡显存 − 固定开销 − weights) ÷ cell size`）。口径约定：
		 - **两个旋钮角色不同**：`--mem-fraction-static` 是工具滑块与启动参数两边显式对齐的自变量（可作为矩阵维度取多值，验证线性关系）；`kv-cache-dtype` 改变 cell size 本身，作为独立维度（fp8 / bf16 各验一轮），每轮两边口径一致。
		 - **对账用日志实际值**：`max_total_num_tokens` 与 KV pool 字节数均取日志/`/get_server_info` 实际值。注意若显式传了 `--max-total-tokens`，SGLang 取 min 钳制，会破坏"池子填满静态区"的前提——本实验不传该参数。
		 - **启动参数固定 `--disable-cuda-graph`**：池子在 graph 捕获前分配，`max_total_num_tokens` 理论上不受影响，但剩余显存对账（第二观测量）会被扰动，关掉更干净；同时固定 `max_running_requests`，控制 `req_to_token_pool` 等随并发上限增长的开销。
		 - **固定开销反推**：对账方程 `实测池 tokens = (frac × cap − fixed − weights) ÷ cell` 中，frac 是显式启动参数、weights 由 E1 钉死、cell 由 E2 钉死，**唯一未知数是 fixed**——每个矩阵格可独立解出一个 fixed 值，各格应一致。
	 - 验证开启 `--enable-dp-attention` 后的显存变化（MLA 模型）。原理：MLA 的 latent KV 无 head 维，纯 TP 下每卡整份复制；开 DP 后 attention 改为数据并行，各 rank 只存自己请求的 KV（每卡 1/TP），代价是 attention 权重每卡整份复制。同一模型同一 TP 开/关各跑一次，验证两个观测量并互相闭合：
		 - **观测量 1（代价）：每卡权重增量**。复用权重对账方法（`avail mem` 差值），开/关的每卡权重之差应等于工具并行 TAB 两种模式下每卡 weights 之差（差分对账，免疫共同的系统偏差）。
		 - **观测量 2（收益）：集群有效 KV 容量比值 ≈ TP**。注意语义变化：开 DP 后 `max_total_num_tokens` 是**每个 DP rank 自己池子**的容量，集群容量 = TP × 每 rank 值；纯 TP 下（KV 复制）集群容量 = max_total_num_tokens 本身。比值应略小于 TP，缺口正是权重复制侵占的 pool 预算。
		 - **闭合校验**：`纯TP容量 × TP − DP集群容量 ≈ TP × 每卡权重增量 ÷ cell size`，三个量均来自日志，账应对圆。
		 - 实验条件：显式 `--tp N --dp N`（SGLang 要求 dp=tp），两次启动用相同 `--mem-fraction-static`；若做打满请求的 sanity check 需用等长请求（调度按负载分发，长度悬殊时某 rank 会先满）。
 - 性能 Roofline
	 - 验证 decode 与带宽成正比（memory-bound 斜率）。并发 = 1，**不需要 PD 分离**：单请求下 prefill 只发生一次，用 `sglang.bench_serving` 直接测 **TPOT/ITL（逐 token 延迟）**即为纯 decode step 时间，天然扣除了 TTFT。
		 - **测比值而非绝对值**：实测吞吐通常只有理论上界的 50–70%（kernel 效率、通信开销），绝对值必然对不上；跨机型 TPOT 比值可将 kernel 效率损失作为公因子消掉。机型选 **B300 / B200 / H200 / H100**，理论带宽比值链：B200/H200 ≈ 1.67、H200/H100 ≈ 1.43、B200/H100 ≈ 2.39，实测 TPOT 比值应与之一致。
		 - **B300/B200 作为控制组**：两者带宽相同（8 TB/s）而 fp4 算力差 50%，memory-bound 预测 TPOT 比值 ≈ 1.0——若 B300 明显更快，说明 decode 并非 memory-bound，roofline 判定有误（证伪测试）。
		 - **实验条件**：四台机器同一模型、同一 checkpoint、同一 TP（比值法的公因子前提）。注意 8×H100 = 640 GiB 装不下 GLM-5.2-FP8（703.7 GiB），候选为 DeepSeek-V4-Flash 或 Qwen3-32B；不可 B 系列用 fp4、H 系列用 fp8（权重字节数不同，比值失真）。
	 - 验证 prefill 与算力成正比（compute-bound）。**单条请求（不搞并发）**，prompt 长度在 8192 的倍数上变化，在不同机型上测 TTFT，预期 **TTFT ∝ 1/峰值FLOPS**。
		 - **同样测跨机型比值**消掉 MFU 公因子：主验证比值 B200/H200 fp8 算力比 = 4500/1979 ≈ 2.27，实测 TTFT 比值应 ≈ 其倒数。
		 - **H200/H100 作为控制组**（与 decode 实验对偶）：两者算力相同（1979 fp8 / 989 bf16）仅带宽不同，compute-bound 预测 TTFT 比值 ≈ 1.0——若 H200 明显更快，说明 prefill 并非 compute-bound（证伪测试）。
		 - **TTFT vs prompt 长度并非严格线性**：GEMM 项线性，attention causal pairs = T(T+1)/2，长 prompt（64K+）二次项显现、略超线性；DSA 模型 top-k 封顶后形状不同。对账目标是**工具对每个长度的预测值序列**（形状 + 比值），不是假设纯线性。
		 - 实验条件：显式固定 `--chunked-prefill-size 8192`（对齐工具 8K chunk 口径）；TTFT 取 `bench_serving` 的 TTFT 指标。

**备注：MTP 开关在两组实验中配置相反**

- **显存拆解 / 并行切分实验：开启 MTP** speculative decoding——SGLang 只在开 MTP 时才加载 MTP 权重、KV cell size 才含 MTP 层，开启后实测与工具输出（含 MTP）口径一致。
- **Roofline 实验：关闭 MTP**（显式不开 speculative decoding）——spec decoding 一次 forward 出多个 token，会改变 TPOT/ITL 与 TTFT 的语义（实测 TPOT 显著低于单 token step 预测，且接受率不可控，跨机型比值法的公因子假设也不成立）。
- 因此两组实验需使用**不同的部署配置**，不可共用同一次部署。

## 具体验证-实验计划

### 0. 总览

共 6 个实验，按 MTP 配置分为两组，**组内可共用部署，组间不可**：

| 编号 | 实验 | 验证对象 | 组 | 机型 | 模型 |
|---|---|---|---|---|---|
| E1 | 权重总量对账 | 每卡 weights 字节 | A（开 MTP） | 8×B200 | 4 个（见覆盖矩阵） |
| E2 | KV cell size 对账 | 每 token KV 字节 | A（与 E1 共用启动） | 8×B200 | 3 个（见覆盖矩阵） |
| E3 | 显存账闭合 | KV 池容量（max_total_num_tokens）/ 固定开销 | A（开 MTP） | 8×B200 | GLM-5.2-FP8 全矩阵 + Qwen3-32B 抽查 |
| E4 | dp-attention 显存变化 | 权重增量 + KV 容量比 | A（开 MTP） | 8×B200 | GLM-5.2-FP8（MLA 专属） |
| E5 | Decode roofline | TPOT ∝ 1/带宽 | B（关 MTP） | B300 / B200 / H200 / H100 各一台 | DeepSeek-V4-Flash + Qwen3-32B |
| E6 | Prefill roofline | TTFT ∝ 1/算力 | B（与 E5 共用部署） | 同 E5 | 同 E5 |

**模型覆盖矩阵**——工具对不同结构走不同计算分支，每条分支至少被一个（模型 × 实验）组合覆盖：

| 工具计算分支 | 差异点 | 覆盖模型 | 覆盖实验 |
|---|---|---|---|
| KV：MLA | `kv_lora_rank + rope`，纯 TP 每卡全量复制 | GLM-5.2-FP8 | E2 / E3 / E4 |
| KV：MQA（GQA-1） | `2 × 1 × head_dim`，TP > 1 即复制 | DeepSeek-V4-Flash | E2 |
| KV：GQA-8 | `2 × kv_heads × head_dim`，TP 下按 kv head 切 1/min(TP, 8) | Qwen3-32B | E2 / E3 |
| 权重：纯 fp8 | 单一 dtype 直读 | GLM-5.2-FP8 | E1 |
| 权重：fp4 sub-byte 打包 | I8 存储 ÷2 还原 + scale 张量 | nvidia/GLM-5.2-NVFP4 | E1 |
| 权重：fp4 + fp8 混合精度 | 逐部件不同 dtype（experts fp4、attention fp8） | DeepSeek-V4-Flash | E1 |
| 权重：bf16 dense | 无量化路径、无 MoE | Qwen3-32B | E1 |
| Roofline：DSA top-k | decode 读取封顶 `min(context, index_topk)`、prefill capped pairs | DeepSeek-V4-Flash | E5 / E6 |
| Roofline：dense attention | decode 全 context、prefill causal pairs T(T+1)/2 | Qwen3-32B | E5 / E6 |
| MTP 有 / 无 | 权重含 MTP 层、cell size +1 层 vs 均无 | GLM-5.2 系 / Qwen3-32B | E1 / E2 |

模型选择理由：
- **GLM-5.2-FP8**（703.7 GiB，MoE + MLA + DSA + MTP）：组 A 主力。单节点 8×B200（1536 GiB）可容纳；是 MLA 模型，E4 才有意义；纯 fp8 无 sub-byte 干扰，适合作显存账基线。备选 DeepSeek-V4-Pro（需多节点，暂不作首选）。
- **DeepSeek-V4-Flash**（148.6 GiB，MoE + MQA + DSA + MTP）：组 B 主力兼 E1/E2 覆盖位。四种机型中最小的 8×H100（640 GiB）也装得下，满足"四台机器同一模型同一 TP"的比值法前提；混合精度权重与 MQA 结构补齐两条分支。
- **Qwen3-32B**（61 GiB，Dense + GQA，bf16，无 MTP）：覆盖工具最"普通"的路径——dense、GQA、无量化、无 MTP、无 DSA。体积小、启动快，适合 E3 抽查与 roofline 第二模型，成本几乎可忽略。
- **nvidia/GLM-5.2-NVFP4**：仅参与 E1，专门覆盖 fp4 sub-byte 打包还原这条最容易出错的权重路径（需 B 系列 GPU 支持 fp4 kernel）。

> 注意：组 A 的"开启 MTP"前提仅适用于**有 MTP 的模型**；Qwen3-32B 无 MTP，无此开关，工具口径同样不含 MTP，两边天然一致。

### 1. 公共约定

- **环境记录**：每次实验记录 sglang 版本（commit）、CUDA / driver 版本、机型、工具 commit。同一实验的所有对照组必须使用同一 sglang 版本。
- **工具基准值先行**：每个实验开始前，先用工具按实验参数（模型 / TP / context / 并发 / kv-dtype / dp-attention / **mem-fraction-static**）生成预测值并记入该实验的 `expected.md`，实测后不回改——避免"先看实测再对预测"。`--mem-fraction-static` 必须与实测启动参数严格一致，`runs.csv` 增加 frac 列。
- **工具口径**：并行 TAB 为供给侧口径——KV 显示池容量而非需求，**不含乘性碎片项**，固定开销 1 GiB 为独立显式项。expected.md 中记录工具 commit，确保口径可追溯。
- **目录规范**：每个实验一个子目录 `validate_experiments/E<N>_<slug>/`，内含：
	- `expected.md`——工具预测值 + 生成命令
	- `runs.csv`——每次启动/压测一行（参数 + 采集字段 + 计算结果）
	- `logs/`——原始启动日志与 bench_serving 输出
	- `conclusion.md`——判定结论与偏差分析
- **组 A 公共启动参数**：纯 TP（`--tp 8`，不开 `--enable-dp-attention`、不开 EP）、开启 MTP speculative decoding、`--disable-cuda-graph`、固定 `--max-running-requests`（全组统一一个值）。
- **组 B 公共启动参数**：关闭 speculative decoding、`--chunked-prefill-size 8192`、四台机器完全相同的启动命令（机型无关部分）。

### 2. 实验组 A：显存类（开 MTP）

#### E1 权重总量对账

- **目的**：验证工具并行 TAB 的每卡 weights 数（`sliced/TP + replicated` 口径）与实际加载一致。
- **配置**：组 A 公共参数，默认 `--mem-fraction-static`。
- **采集**：启动日志 `Load weight begin` 与 `Load weight end` 两行的 `avail mem`，差值 = 每卡实测权重。
- **模型矩阵**（每模型一次启动，四条权重路径各覆盖一条）：

| 模型 | 覆盖的权重路径 | 备注 |
|---|---|---|
| GLM-5.2-FP8 | 纯 fp8 | 基线；与 E2/E3 共用启动 |
| nvidia/GLM-5.2-NVFP4 | fp4 sub-byte 打包（÷2 还原 + scale） | 与 FP8 版同结构，两者每卡权重之差还可交叉验证量化压缩比 |
| DeepSeek-V4-Flash | fp4 + fp8 混合精度 | 逐部件不同 dtype，工具最复杂的识别路径 |
| Qwen3-32B | bf16 dense、无 MTP | 最简路径，兜底对照 |

- **判定**：每个模型独立判定，`|实测 − 工具每卡 weights| / 工具值 ≤ 2%`。1–2% 内视为通过；>2% 优先排查运行时 dtype 转换（fp8→bf16 上转、在线量化）与组件漏算。

#### E2 KV cell size 对账（与 E1 共用启动，零额外成本）

- **目的**：验证每 token KV 字节数（cell size），即 KV 公式中唯一有信息量的量。
- **采集**：同一份启动日志（或 `/get_server_info`）中的 KV pool 总字节数与 `max_total_num_tokens`。
- **计算**：`实测 cell size = KV pool 字节数 ÷ max_total_num_tokens`。
- **模型矩阵**（三种 KV 结构判定分支各覆盖一条，全部来自 E1 已有启动）：

| 模型 | KV 结构分支 | 理论 cell size 构成 |
|---|---|---|
| GLM-5.2-FP8 | MLA | `(kv_lora_rank 512 + rope 64) × 层数`，开 MTP +1 层 |
| DeepSeek-V4-Flash | MQA（GQA-1） | `2 × 1 × head_dim × 层数`，开 MTP +1 层 |
| Qwen3-32B | GQA-8 | `2 × 8 × head_dim × 层数`，无 MTP |

- **判定**：每模型独立与工具理论值对账，容差 1%。
- **dtype 矩阵**：`--kv-cache-dtype` ∈ {fp8, bf16} 各一次（GLM 的 bf16 轮可与 E3 共用启动；注意 DSA 模型 auto → fp8，需显式指定才能测 bf16——若 sglang 拒绝非 fp8 则记录该约束并跳过）。

#### E3 显存账闭合（KV 池容量直接对账）

- **目的**：验证给定 mem-fraction-static × kv-dtype 下，工具供给侧预测（KV 池容量 = `max_total_num_tokens`）与实际一致，即整本显存账（权重 + 固定开销 + KV 池 + 非静态区）闭合。
- **方法**（每个矩阵格**一次启动**）：
	1. 固定 `--mem-fraction-static` 启动（矩阵取值见下），不传 `--max-total-tokens`。
	2. 从日志/`/get_server_info` 读实测 `max_total_num_tokens` 与 KV pool 字节数。
	3. 工具并行 TAB 滑块调到同一 frac，读预测池容量 tokens（工具公式：`(frac × cap − fixed − weights) ÷ cell`）。
	4. 另记录初始化完成后的 `avail mem` 作为实测剩余显存（第二观测量，对账工具非静态区余量）。
- **判定**：
	1. **主判定**：`|实测 max_total_num_tokens − 工具预测| / 预测 ≤ 2%`。
	2. **固定开销反推**：每格用 `fixed = frac × cap − weights − 实测池 tokens × cell` 独立解出 fixed（weights 取 E1 实测、cell 取 E2 实测），各格结果应一致；若稳定偏离 1 GiB，在 conclusion 中给出 `--fixed-overhead-gib` 建议默认值。
	3. **旁证**：实测剩余显存 ≈ 工具非静态区余量（cap − used），容差 GiB 量级（≤2 GiB，受 workspace 等未建模项扰动，不作主判定）。
- **矩阵**：GLM-5.2-FP8 跑全矩阵（2×2 = 4 格，4 次启动）；Qwen3-32B 抽查 2 格（frac {0.85, 0.9} × bf16）——验证闭合逻辑对 GQA / dense / 无 MTP 路径同样成立，且其权重小、池子大，对"固定开销"的相对敏感度更高。frac 取两值也顺带验证了池容量对 frac 的线性关系（斜率应 = cap ÷ cell）。

| 维度 | 取值（GLM 全矩阵） |
|---|---|
| mem-fraction-static | 0.85、0.90 |
| kv-cache-dtype | fp8、bf16 |

> 无需 context × 并发的需求侧实测矩阵——需求只是 `ctx × req × cell` 的恒等式，cell 已由 E2 验证。

#### E4 dp-attention 显存变化

- **目的**：验证工具 dp-attention 模式下的两个预测——每卡权重增量（代价）与集群 KV 容量提升（收益），并做三量闭合。
- **配置**：同一模型同一节点两次启动，**相同 `--mem-fraction-static`**：
	- 启动 ①（纯 TP）：`--tp 8`
	- 启动 ②（DP attention）：`--tp 8 --dp 8 --enable-dp-attention`
- **采集**：两次启动各自的每卡权重（`avail mem` 差值法）W₁ / W₂，`max_total_num_tokens` K₁ / K₂（注意语义：K₁ 是全局池子，K₂ 是**每个 DP rank 自己的池子**）。
- **判定**（三条独立检查）：
	1. **权重增量**：`W₂ − W₁ ≈ 工具两种模式每卡 weights 之差`（差分对账，免疫共同系统偏差），容差取增量的 5% 或 0.5 GiB 取大者。
	2. **每 rank 池容量**：K₂ 直接与工具 DP 模式下同一 frac 的预测池容量对账（`(frac × cap − fixed − W₂预测) ÷ cell`），容差 2%；K₁ 同理与纯 TP 模式预测对账。旁证：`8 × K₂ ÷ K₁` 应略小于 8，缺口 = 权重复制侵占的 pool 预算。
	3. **闭合校验**：`K₁ × 8 − 8 × K₂ ≈ 8 × (W₂ − W₁) ÷ cell_size`，三个量全部来自日志，账应对圆（此条不依赖工具，纯实测自洽）。
- **可选 sanity check**：以等长请求打满两种部署，确认 DP 模式下实际可同时容纳的 token 数确实 ≈ 8 × K₂。

### 3. 实验组 B：Roofline 类（关 MTP）

四台机器（B300 / B200 / H200 / H100）各部署，`--tp 8`、同一 checkpoint、同一启动命令，E5 与 E6 复用同一部署、只跑不同 bench。**不可 B 系列用 fp4、H 系列用 fp8**。

**两个模型各跑一轮**，覆盖工具 roofline 的两条 attention 访问模式分支：

| 模型 | 分支 | 预期差异 |
|---|---|---|
| DeepSeek-V4-Flash（fp8） | DSA top-k：decode 读取封顶 `min(context, index_topk)`，prefill capped pairs | 长 prompt 下 TTFT 趋回线性 |
| Qwen3-32B（bf16） | dense attention：decode 全 context，prefill causal pairs T(T+1)/2 | 长 prompt 下 TTFT 超线性；比值用 bf16 算力（H100/H200 989、B200 2250） |

注意：Qwen3-32B 是 bf16 权重，其 prefill 比值预期按 **bf16 峰值算力**计算（B200/H200 = 2250/989 ≈ 2.27，恰与 fp8 比值相同——同代架构 fp8:bf16 均为 2:1）；decode 带宽比值与 dtype 无关，两模型预期一致，互为复核。

#### E5 Decode roofline（TPOT ∝ 1/带宽）

- **目的**：验证 decode 是 memory-bound、跨机型 TPOT 比值等于带宽反比。
- **压测**：`sglang.bench_serving`，并发 = 1，固定 input 8192 / output 512，重复 3 次取中位数 TPOT（ITL 作旁证）。
- **判定**（比值法，容差 ±10%）：

| 比值 | 预期（带宽反比） | 性质 |
|---|---|---|
| TPOT(H200) / TPOT(B200) | ≈ 1.67 | 主验证 |
| TPOT(H100) / TPOT(H200) | ≈ 1.43 | 主验证 |
| TPOT(H100) / TPOT(B200) | ≈ 2.39 | 链条一致性 |
| TPOT(B300) / TPOT(B200) | ≈ 1.0 | **控制组（证伪）**：带宽同、fp4 算力差 50%；若 B300 明显更快 ⇒ decode 非 memory-bound，判定推翻 |

- **旁证（不作判定）**：各机型 `实测 TPOT ÷ 工具理论 step 时间`，预期落在 1.4–2.0×（对应 50–70% 效率），四台机器该系数应接近——这正是比值法能消掉公因子的前提自检。

#### E6 Prefill roofline（TTFT ∝ 1/算力）

- **目的**：验证 prefill 是 compute-bound、跨机型 TTFT 比值等于峰值 FLOPS 反比。
- **压测**：单条请求（并发 = 1，逐条发送），prompt 长度 ∈ {8192, 16384, 32768, 65536, 131072}（8192 的整数倍，对齐 chunked-prefill 口径），每个长度重复 3 次取中位数 TTFT。
- **判定**（每个长度独立算比值，容差 ±10%）：

| 比值 | 预期（fp8 算力反比） | 性质 |
|---|---|---|
| TTFT(H200) / TTFT(B200) | ≈ 4500/1979 ≈ 2.27 | 主验证 |
| TTFT(H100) / TTFT(H200) | ≈ 1.0 | **控制组（证伪）**：算力同（1979 fp8）、带宽差 1.43×；若 H200 明显更快 ⇒ prefill 非 compute-bound，判定推翻 |

- **形状对账**：同一机型上 TTFT vs prompt 长度的曲线与工具对各长度的预测序列对形状（GEMM 线性项 + causal pairs 二次项；DSA top-k 封顶后趋回线性），不假设纯线性。

### 4. 执行顺序与成本估算

1. **E1 + E2**（4 个模型各一次启动，权重与 cell size 一并采集）→ 这两个量是后面所有账的基础，先钉死。
2. **E3**（GLM 全矩阵 4 次启动 + Qwen 抽查 2 次，每格一次启动，可基于 CC 自动化：改参数 → 重启 → 抓日志 → 填 runs.csv）。
3. **E4**（2 次启动 + 可选打满测试）。
4. **E5 + E6**（切换到组 B 部署：4 台机器 × 2 模型 × 1 次部署，每个部署跑 decode bench 3 次 + prefill 5 长度 × 3 次）。

失败处理约定：任一判定不通过时，先在 conclusion.md 记录偏差与排查结论，再决定是修工具（公式/参数）还是修实验口径；**不回改 expected.md**。