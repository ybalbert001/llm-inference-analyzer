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
	 - 针对 kv cache，验证 **cell size（每 token 每 layer 的 KV 字节数 × 层数，即每 token 的 KV 占用）**，而不是验证某个总 GiB 数。原因：
		 - SGLang 的 KV pool 是供给侧驱动的——启动时把 `mem_fraction_static` 扣除权重后的剩余显存全部预分配给 KV pool，其大小与工作负载（context/并发）无关；而工具算的 `context × requests × cell` 是需求侧的最坏情况上界。两者语义不同，直接对总数没有意义。
		 - cell size 是这条公式里唯一有信息量的待验证量，总 GiB 只是它乘以 token 数的线性函数（恒等式，无需验证）。
		 - 实测方法：从启动日志 / `/get_server_info` 读取 KV pool 字节数与 `max_total_num_tokens`，`实测 cell size = KV pool 字节数 ÷ max_total_num_tokens`，与工具理论值对账（例：DeepSeek-V4-Pro fp8 = (512+64) × 61 层 ≈ 61 KiB/token）。无需修改 sglang 代码。
		 - 注意口径：开启 MTP speculative decoding 时 cell size 会多一层；kv-cache-dtype 需两边对齐。
 - 并行切分部分 （可以基于 CC 进行自动化实验）
	 - 验证不同的 context window、并发池子大小、kv-cache-dtype 组合下，工具的整本显存账是否闭合。方法：**调节 `--mem-fraction-static`，使日志中的 `max_total_num_tokens ≈ context × 并发数**（即把 KV pool 钉在工具计算的 KV 需求上），此时验证 **剩余显存 ≈ 工具计算的剩余值**。注意 SGLang 对 `--max-total-tokens` 是 min 钳制而非照单分配，所以不通过触发 OOM 来验证，而是做显存账闭合对账。口径约定：
		 - **两个旋钮角色不同**：`--mem-fraction-static` 是细调旋钮（对 token 数单调线性，先跑一次按比例外推，两次启动即可落点）；`kv-cache-dtype` 改变 cell size 本身，作为实验矩阵的独立维度（fp8 / bf16 各验一轮），不作调节用，每轮两边口径一致。
		 - **"≈" 的容差**：`max_total_num_tokens` 落点允许 ±1–2%；对账时用日志中实际生效值计算，不用目标值。
		 - **启动参数固定 `--disable-cuda-graph`**：排除 CUDA graph 缓冲这个工具未建模的最大扰动项；同时固定 `max_running_requests`，控制 `req_to_token_pool` 等随并发上限增长的开销。剩余显存的偏差容忍度比权重对账松（GiB 量级），系统性偏差可反推工具"固定开销 1 GiB"参数的真实值。
	 - 验证开启 `--enable-dp-attention` 后的显存变化（MLA 模型）。原理：MLA 的 latent KV 无 head 维，纯 TP 下每卡整份复制；开 DP 后 attention 改为数据并行，各 rank 只存自己请求的 KV（每卡 1/TP），代价是 attention 权重每卡整份复制。同一模型同一 TP 开/关各跑一次，验证两个观测量并互相闭合：
		 - **观测量 1（代价）：每卡权重增量**。复用权重对账方法（`avail mem` 差值），开/关的每卡权重之差应等于工具并行 TAB 两种模式下每卡 weights 之差（差分对账，免疫共同的系统偏差）。
		 - **观测量 2（收益）：集群有效 KV 容量比值 ≈ TP**。注意语义变化：开 DP 后 `max_total_num_tokens` 是**每个 DP rank 自己池子**的容量，集群容量 = TP × 每 rank 值；纯 TP 下（KV 复制）集群容量 = max_total_num_tokens 本身。比值应略小于 TP，缺口正是权重复制侵占的 pool 预算。
		 - **闭合校验**：`纯TP容量 × TP − DP集群容量 ≈ TP × 每卡权重增量 ÷ cell size`，三个量均来自日志，账应对圆。
		 - 实验条件：显式 `--tp N --dp N`（SGLang 要求 dp=tp）；沿用"钉 pool"法时每 rank 目标为 `context × 并发 ÷ TP`；若做打满请求的 sanity check 需用等长请求（调度按负载分发，长度悬殊时某 rank 会先满）。
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

...