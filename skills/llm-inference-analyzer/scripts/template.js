var GIB = 1073741824;

function gib(b) {
  var v = b / GIB;
  return v >= 0.95
    ? v.toLocaleString('en-US', {minimumFractionDigits: 1, maximumFractionDigits: 1})
    : v.toFixed(2);
}
function el(id) { return document.getElementById(id); }
function setText(id, txt) { var e = el(id); if (e) e.textContent = txt; }
// fp4 = mxfp4: 0.5 B data + 1 uint8 scale per 16 elements
function kvBytesPer(dtype) { return dtype === 'fp8' ? 1 : dtype === 'fp4' ? 0.5625 : 2; }
// stored KV token-positions summed over layers, honoring sliding-window caps.
// kvGroups: [[layerCount, window], ...]; window=0 = full context. Falls back to
// nKvLayers×ctx (or L×ctx) when no group model is present.
function kvLayerTokens(ctx) {
  var g = D.kvGroups;
  if (g && g.length) {
    var s = 0;
    for (var i = 0; i < g.length; i++) s += g[i][0] * (g[i][1] ? Math.min(ctx, g[i][1]) : ctx);
    return s;
  }
  return (D.nKvLayers || D.L) * ctx;
}
function ctxLabel(v) { return v % 1048576 === 0 ? (v / 1048576) + 'M' : (v / 1024) + 'K'; }
// human token count: 1.94M / 317K / 512
function tokLabel(v) {
  if (v >= 1048576) return (v / 1048576).toFixed(2) + 'M';
  if (v >= 1024) return Math.round(v / 1024) + 'K';
  return String(Math.round(v));
}
function kvDtypeNow() {
  var sel = el('f-kv').value;
  return sel === 'auto' ? D.kvAuto : sel;
}

/* ============================ i18n ============================ */
// All user-facing strings assembled at runtime (not baked in by Python) live
// here, keyed by lang. Tr(key) returns the current-language function; call it
// with whatever arguments that entry needs, e.g. Tr('gpuLine')(w, kv, act, tail).
var I18N = {
  zh: {
    weightsStaticLabel: function () { return '权重（静态）'; },
    totalHead: function (gib, ctxStr, req) {
      return '静态 + 动态 · 部署总占用 ≈ ' + gib + ' GiB（context ' + ctxStr + ' × ' + req + ' 并发）';
    },
    totalLine: function (w, kv, act, ovPct, grand, kvPerReq) {
      return '权重 ' + w + '（静态） + KV ' + kv + ' + Activation ' + act + '（动态） + 碎片 ~' +
        ovPct + '% ≈ <b>' + grand + ' GiB</b>' +
        "<span class='pct'>　·　KV 随 context × 并发线性增长：每并发 +" + kvPerReq + ' GiB</span>';
    },
    tblKvLabel: function (ctxStr, req) { return 'KV cache（' + ctxStr + ' ctx × ' + req + ' req）'; },
    kvAutoSuffix: function () { return '（auto）'; },
    customInstLabel: function (gpn, memGib) { return '自定义 ' + gpn + '×' + memGib + ' GiB'; },
    instLabel: function (inst, count, gpu, memGib) {
      return inst + '（' + count + '×' + gpu + ' ' + memGib + ' GiB）';
    },
    fixedOverhead: function () { return '固定开销（CUDA context/NCCL）'; },
    kvCapLabel: function () { return 'KV cache 容量'; },
    fracLineTip: function (frac) { return 'mem-fraction-static = ' + frac + '：左侧为静态区（权重 + KV 池），启动时整体预分配'; },
    gpuPctTip: function (memGib, pct) { return '（占 ' + memGib + ' GiB 卡的 ' + pct + '%）'; },
    freeTipLabel: function () { return '非静态区余量（CUDA graph 等）'; },
    freePct: function (pct) { return '（' + pct + '%）'; },
    cannotStart: function (gib) { return '静态区放不下权重，差 ' + gib + ' GiB，无法启动'; },
    kvUtilBarLabel: function () { return 'KV 需求/容量'; },
    kvUtilTip: function (d, c, pct) {
      return '<b>KV 需求 ' + d + ' GiB ÷ 池容量 ' + c + ' GiB = ' + pct + '%</b><br>黑色刻度线 = 池容量位置；越过为红色超容';
    },
    kvUtilLine: function (d, c, pct, maxTok, ctxLbl, maxReq) {
      return 'KV 需求 <b>' + d + '</b> / 容量 <b>' + c + '</b> GiB（<b>' + pct + '%</b>）· 池容量 ≈ <b>' +
        maxTok + '</b> tokens（≈ ' + maxReq + ' 个 ' + ctxLbl + ' 满 context 请求）';
    },
    sumCapLine: function (maxTok, ctxLbl, maxReq, frac) {
      return 'mem-fraction-static ' + frac + ' 下 KV 池容量 ≈ <b>' + maxTok +
        '</b> tokens（max_total_num_tokens，按瓶颈 stage）≈ ' + maxReq + ' 个 ' + ctxLbl + ' 满 context 请求';
    },
    pctParen: function (pct) { return '（' + pct + '%）'; },
    layerChipLabel: function (lo, hi, count) { return 'L' + lo + '–L' + hi + ' ×' + count + ' 层'; },
    perLayerFrac: function (tp) { return '，每层 1⁄' + tp; },
    expertShapeFull: function () { return '完整专家'; },
    expertShapeSliced: function (denom) { return '专家 ×1⁄' + denom + ' 切片'; },
    expertLine: function (cnt, shape, ep) { return '：每 MoE 层 ' + cnt + ' 个' + shape + '（EP' + ep + '）'; },
    warnPpExceeds: function (pp, L) { return 'PP(' + pp + ') 超过层数 ' + L; },
    oomExceeds: function (gib) { return '超出 ' + gib + ' GiB'; },
    gpuLine: function (weights, kv, act, tail) {
      return '权重 <b>' + weights + '</b> + KV 池 <b>' + kv + '</b> + act ' + act + ' GiB' + tail;
    },
    freeTail: function (free) { return '，剩 <b>' + free + '</b> GiB'; },
    nodeWarn: function (count) { return '⚠ ' + count + ' 卡显存不足'; },
    sumHeadTail: function (world, nNodes, instLabel, oomPart) {
      return '　·　' + world + ' 卡 / ' + nNodes + ' 节点（' + instLabel + '）' + oomPart;
    },
    oomPartOom: function (count) { return '　·　⚠ ' + count + ' 卡不足'; },
    oomPartOk: function () { return '　·　全部放得下'; },
    kvCacheLegendLabel: function (kvDtype) { return 'KV cache（' + kvDtype + '）'; },
    kvNoteDpOn: function () { return 'DP attention 开启：KV 按 TP 切，无复制'; },
    kvNoteMlaRepl: function (kvRepl, clusterKvGib) {
      return '<b>MLA latent 无 head 维：KV 每卡全量复制，集群共 ' + kvRepl + '× 单份 KV（' + clusterKvGib +
        ' GiB）</b>——试试勾选 DP attention';
    },
    kvNoteTpExceeds: function (tp, nkv, replRatio) {
      return '<b>TP(' + tp + ') > kv heads(' + nkv + ')：KV 已切到底，多出的 ' + replRatio +
        '× 为复制</b>——试试勾选 DP attention';
    },
    kvNoteSliced: function (tp) { return 'KV 按 kv head 切 1/' + tp + '，无复制'; },
    sumLine: function (maxUsed, memGib, clusterWeights, modelGib, replGib) {
      return '每卡最大占用 <b>' + maxUsed + ' GiB</b>（上限 ' + memGib + '）　·　' +
        '集群权重合计 ' + clusterWeights + ' GiB（模型 ' + modelGib + ' GiB + 复制冗余 ' + replGib + '）';
    },
    tableHeader: function () {
      return "<tr><th>GPU</th><th>rank</th><th>层</th>" +
        "<th class='num'>权重 GiB</th><th class='num'>KV 容量 GiB</th><th class='num'>KV 需求 GiB</th><th class='num'>act GiB</th>" +
        "<th class='num'>合计 GiB</th><th class='num'>剩余 GiB</th></tr>";
    },
    gemmDecodeNote: function (B) { return '权重每步读一遍，' + B + ' token 共享 → 强度随并发线性涨'; },
    gemmPrefillNote: function (Tstr) { return Tstr + ' tokens 共享一遍权重读取 → 强度很高'; },
    moeDecodeNote: function (B, pct) { return 'decode batch=' + B + ' 约碰到 ' + pct + '% 专家：读得多算得少'; },
    moePrefillNote: function (Tstr) { return 'prefill ' + Tstr + ' tokens 几乎读满全部专家，FLOPs 也大'; },
    mhaLabel: function (qHeads) { return 'MHA（' + qHeads + ' heads）'; },
    gqaLabel: function (qHeads, kvHeads) { return 'GQA（' + qHeads + ' Q / ' + kvHeads + ' KV heads）'; },
    mlaLabel: function (kvLoraRank, ropeHeadDim) {
      return 'MLA absorbed（latent ' + kvLoraRank + ' + RoPE ' + ropeHeadDim + '）';
    },
    denseDecodeNote: function (ctxLbl) { return '每请求读取完整 ' + ctxLbl + ' context'; },
    densePrefillNote: function () { return 'chunk 内 dense causal attention；HBM 按 KV 一次读取/写入的理想下界'; },
    dsaLabel: function (topk) { return 'DSA top-' + topk; },
    dsaDecodeNote: function (ctxLbl, attended) {
      return '每请求从 ' + ctxLbl + ' context 读取 top-' + attended + ' selected KV；indexer full-context 检索当前未建模';
    },
    dsaPrefillNote: function (topk) { return 'chunk 内 top-' + topk + ' causal attention；HBM 为理想下界，不含 gather 重读'; },
    customInstNoSpec: function () { return '自定义机型无算力/带宽规格'; },
    customInstNoSpecNote: function () {
      return '请选择一个预设机型（H100/H200/H800/H20/B200/B300/A100/L40S/A10G）以绘制 roofline。';
    },
    roofTitle: function (gpu, dtype, peak, bw, fallbackPart) {
      return 'Roofline（单卡）：' + gpu + ' — ' + dtype + ' dense ≈ ' + peak + ' TFLOPs/s · HBM ' + bw + ' TB/s' + fallbackPart;
    },
    roofFallback: function (fallback) { return '（该 GPU 无 ' + fallback + ' 算力，按 bf16 口径）'; },
    attnCoreLabel: function () { return 'Attn core（KV/score）'; },
    roofNote: function (kneeStr) {
      return '每个<b>点是一个 kernel</b>：● = decode 阶段，▲ = prefill 阶段。落点强度 &lt; 拐点（峰值 ÷ 带宽 ≈ ' +
        kneeStr + ' FLOPs/B）→ 受显存带宽限制（memory-bound），GPU 算力用不满；反之受算力限制。' +
        '强度 = FLOPs ÷ HBM 搬运字节，由模型结构 + 当前 context/并发/KV 精度自动算出。' +
        '本图为<b>单卡</b> roofline：TP 会把每个 kernel 的 FLOPs 与 HBM 字节同比例切分，强度是分片不变量，故切换 TP 时点与屋顶都不动；TP 仅影响下方卡片的延迟/吞吐数字。' +
        '峰值为 datasheet dense 口径近似值，真实 kernel 达不到 100%。';
    },
    decodeHeadSuffix: function (B) { return '（每步 ' + B + ' tokens，' + B + ' 并发）'; },
    prefillHeadSuffix: function (tokens) { return '（chunk ' + tokens + ' tokens）'; },
    decodeFormula: function (gibBytes, bw, tp, stepMs, tpsPerReq, tpsGroup, B) {
      return "<div class='vformula'>step ≈ " + gibBytes + " GiB ÷（" + bw +
        " TB/s × TP" + tp + "）≈ " + stepMs + " ms → 单请求上界 ~" + tpsPerReq +
        " tok/s · TP 组上界 ~" + tpsGroup + " tok/s（×" + B + " 并发）</div>";
    },
    decodeFootPrefix: function () { return '理论上界（实测通常 50–70%）。'; },
    decodeFootMem: function (kneeRatio) {
      return '距拐点还有 ' + kneeRatio + '×。' + '时间占比大的 kernel 是优化对象：' +
        '权重项吃量化，KV 项吃 KV 量化/缩 context，并发只对随 batch 涨强度的 kernel 有效。';
    },
    decodeFootCompute: function () {
      return '已 compute-bound：加并发只会线性加长 step 时间，吞吐不再提升；关注算力利用率与更高算力精度（fp8/fp4）。';
    },
    prefillFormula: function (chunkMs, tp, ctxLbl, seconds) {
      return "<div class='vformula'>一个 chunk ≈ " + chunkMs + " ms（TP" + tp + "）→ 处理 " +
        ctxLbl + " prompt ≈ " + seconds + " s</div>";
    },
    prefillFootMem: function () { return 'memory-bound（少见，多为小 chunk / 大模型）。'; },
    prefillFootCompute: function () { return 'compute-bound：prefill 吃算力，量化权重省不了时间（除非用低精度算力）；'; },
    prefillFootTail: function () { return '缩短 TTFT 的手段：更高算力 GPU、prefix cache、chunked prefill 与 decode 重叠。'; },
    kernelDetailsSummary: function () { return '逐 kernel 明细（FLOPs / HBM / 强度 / 时间占比）'; },
    kbPhaseStep: function () { return '步'; },
    kbPhaseChunk: function () { return ' chunk'; },
    kbHeadFlops: function (suffix) { return 'FLOPs（每' + suffix + '）'; },
    compTblComponent: function () { return '部件'; },
    compTblHbm: function () { return 'HBM 读写'; },
    compTblIntensity: function () { return '强度 F/B'; },
    compTblBoundBy: function () { return '受限于'; },
    compTblTimeShare: function () { return '时间占比'; },
    compTblNote: function () { return '说明'; },
    compTblTotal: function () { return '合计'; },
    boundByMem: function () { return '带宽'; },
    boundByCompute: function () { return '算力'; },
    legendDecode: function (req) { return 'Decode（每步 ' + req + ' tokens）'; },
    legendPrefill: function (tokens) { return 'Prefill（chunk ' + tokens + ' tokens）'; },
    kneeLabel: function (kneeStr) { return '拐点 ≈ ' + kneeStr + ' FLOPs/B'; },
    peakLabel: function (peakStr, dtype) { return '峰值 ' + peakStr + ' TFLOPs/s（' + dtype + ' dense）'; },
    bwLabel: function (bw) { return 'HBM 带宽 ' + bw + ' TB/s'; },
    pointTooltip: function (phaseLabel, intensityStr, bound, attainStr, pctStr) {
      return '<b>' + phaseLabel + '</b> · 强度 ≈ ' + intensityStr + ' FLOPs/B → ' + bound +
        '<br>可达 ≈ ' + attainStr + ' TFLOPs/s（峰值的 ' + pctStr + '%）';
    },
  },
  en: {
    weightsStaticLabel: function () { return 'Weights (static)'; },
    totalHead: function (gib, ctxStr, req) {
      return 'Static + dynamic · total deployment footprint ≈ ' + gib + ' GiB (context ' + ctxStr + ' × ' + req + ' concurrent)';
    },
    totalLine: function (w, kv, act, ovPct, grand, kvPerReq) {
      return 'Weights ' + w + ' (static) + KV ' + kv + ' + Activation ' + act + ' (dynamic) + ~' +
        ovPct + '% fragmentation ≈ <b>' + grand + ' GiB</b>' +
        "<span class='pct'> · KV scales linearly with context × concurrency: +" + kvPerReq + ' GiB per concurrent request</span>';
    },
    tblKvLabel: function (ctxStr, req) { return 'KV cache (' + ctxStr + ' ctx × ' + req + ' req)'; },
    kvAutoSuffix: function () { return ' (auto)'; },
    customInstLabel: function (gpn, memGib) { return 'Custom ' + gpn + '×' + memGib + ' GiB'; },
    instLabel: function (inst, count, gpu, memGib) {
      return inst + ' (' + count + '×' + gpu + ' ' + memGib + ' GiB)';
    },
    fixedOverhead: function () { return 'fixed overhead (CUDA context/NCCL)'; },
    kvCapLabel: function () { return 'KV cache capacity'; },
    fracLineTip: function (frac) { return 'mem-fraction-static = ' + frac + ': left of this line is the static region (weights + KV pool), pre-allocated at startup'; },
    gpuPctTip: function (memGib, pct) { return ' (' + pct + '% of the ' + memGib + ' GiB GPU)'; },
    freeTipLabel: function () { return 'non-static headroom (CUDA graph etc.)'; },
    freePct: function (pct) { return ' (' + pct + '%)'; },
    cannotStart: function (gib) { return 'weights exceed static region by ' + gib + ' GiB — cannot start'; },
    kvUtilBarLabel: function () { return 'KV demand/cap'; },
    kvUtilTip: function (d, c, pct) {
      return '<b>KV demand ' + d + ' GiB ÷ pool capacity ' + c + ' GiB = ' + pct + '%</b><br>black tick = pool capacity; red = overflow past it';
    },
    kvUtilLine: function (d, c, pct, maxTok, ctxLbl, maxReq) {
      return 'KV demand <b>' + d + '</b> / capacity <b>' + c + '</b> GiB (<b>' + pct + '%</b>) · pool ≈ <b>' +
        maxTok + '</b> tokens (≈ ' + maxReq + ' full-' + ctxLbl + '-context requests)';
    },
    sumCapLine: function (maxTok, ctxLbl, maxReq, frac) {
      return 'At mem-fraction-static ' + frac + ', KV pool ≈ <b>' + maxTok +
        '</b> tokens (max_total_num_tokens, bottleneck stage) ≈ ' + maxReq + ' full-' + ctxLbl + '-context requests';
    },
    pctParen: function (pct) { return ' (' + pct + '%)'; },
    layerChipLabel: function (lo, hi, count) { return 'L' + lo + '–L' + hi + ' ×' + count + ' layers'; },
    perLayerFrac: function (tp) { return ', 1⁄' + tp + ' per layer'; },
    expertShapeFull: function () { return 'full experts'; },
    expertShapeSliced: function (denom) { return 'experts ×1⁄' + denom + ' sliced'; },
    expertLine: function (cnt, shape, ep) { return ': ' + cnt + ' per MoE layer, ' + shape + ' (EP' + ep + ')'; },
    warnPpExceeds: function (pp, L) { return 'PP(' + pp + ') exceeds layer count ' + L; },
    oomExceeds: function (gib) { return 'over by ' + gib + ' GiB'; },
    gpuLine: function (weights, kv, act, tail) {
      return 'Weights <b>' + weights + '</b> + KV pool <b>' + kv + '</b> + act ' + act + ' GiB' + tail;
    },
    freeTail: function (free) { return ', <b>' + free + '</b> GiB free'; },
    nodeWarn: function (count) { return '⚠ ' + count + ' GPU(s) out of memory'; },
    sumHeadTail: function (world, nNodes, instLabel, oomPart) {
      return ' · ' + world + ' GPUs / ' + nNodes + ' node(s) (' + instLabel + ')' + oomPart;
    },
    oomPartOom: function (count) { return ' · ⚠ ' + count + ' GPU(s) short'; },
    oomPartOk: function () { return ' · all fits'; },
    kvCacheLegendLabel: function (kvDtype) { return 'KV cache (' + kvDtype + ')'; },
    kvNoteDpOn: function () { return 'DP attention on: KV split by TP, no replication'; },
    kvNoteMlaRepl: function (kvRepl, clusterKvGib) {
      return '<b>MLA latent has no head dim: KV is fully replicated per GPU, cluster holds ' + kvRepl +
        '× a single KV copy (' + clusterKvGib + ' GiB)</b> — try enabling DP attention';
    },
    kvNoteTpExceeds: function (tp, nkv, replRatio) {
      return '<b>TP(' + tp + ') > kv heads(' + nkv + '): KV is fully sharded, the extra ' + replRatio +
        '× is replication</b> — try enabling DP attention';
    },
    kvNoteSliced: function (tp) { return 'KV sharded by kv head 1/' + tp + ', no replication'; },
    sumLine: function (maxUsed, memGib, clusterWeights, modelGib, replGib) {
      return 'Max per-GPU usage <b>' + maxUsed + ' GiB</b> (cap ' + memGib + ')  ·  ' +
        'Cluster weights total ' + clusterWeights + ' GiB (model ' + modelGib + ' GiB + replication overhead ' + replGib + ')';
    },
    tableHeader: function () {
      return "<tr><th>GPU</th><th>rank</th><th>layers</th>" +
        "<th class='num'>weights GiB</th><th class='num'>KV cap GiB</th><th class='num'>KV need GiB</th><th class='num'>act GiB</th>" +
        "<th class='num'>total GiB</th><th class='num'>free GiB</th></tr>";
    },
    gemmDecodeNote: function (B) { return 'weights read once per step, shared by ' + B + ' tokens → intensity scales linearly with concurrency'; },
    gemmPrefillNote: function (Tstr) { return Tstr + ' tokens share one weight read → very high intensity'; },
    moeDecodeNote: function (B, pct) { return 'decode batch=' + B + ' touches ~' + pct + '% of experts: reads more than it computes'; },
    moePrefillNote: function (Tstr) { return 'prefill ' + Tstr + ' tokens touches nearly all experts, FLOPs is large too'; },
    mhaLabel: function (qHeads) { return 'MHA (' + qHeads + ' heads)'; },
    gqaLabel: function (qHeads, kvHeads) { return 'GQA (' + qHeads + ' Q / ' + kvHeads + ' KV heads)'; },
    mlaLabel: function (kvLoraRank, ropeHeadDim) {
      return 'MLA absorbed (latent ' + kvLoraRank + ' + RoPE ' + ropeHeadDim + ')';
    },
    denseDecodeNote: function (ctxLbl) { return 'each request reads the full ' + ctxLbl + ' context'; },
    densePrefillNote: function () { return 'dense causal attention within the chunk; HBM is the ideal one-pass KV read/write lower bound'; },
    dsaLabel: function (topk) { return 'DSA top-' + topk; },
    dsaDecodeNote: function (ctxLbl, attended) {
      return 'each request reads top-' + attended + ' selected KV from ' + ctxLbl + ' context; indexer full-context retrieval is not modeled';
    },
    dsaPrefillNote: function (topk) { return 'top-' + topk + ' causal attention within the chunk; HBM is the ideal lower bound, excluding gather reloads'; },
    customInstNoSpec: function () { return 'Custom instance has no compute/bandwidth spec'; },
    customInstNoSpecNote: function () {
      return 'Select a preset instance (H100/H200/H800/H20/B200/B300/A100/L40S/A10G) to draw the roofline.';
    },
    roofTitle: function (gpu, dtype, peak, bw, fallbackPart) {
      return 'Roofline (per-GPU): ' + gpu + ' — ' + dtype + ' dense ≈ ' + peak + ' TFLOPs/s · HBM ' + bw + ' TB/s' + fallbackPart;
    },
    roofFallback: function (fallback) { return ' (this GPU has no ' + fallback + ' compute; falling back to bf16)'; },
    attnCoreLabel: function () { return 'Attn core (KV/score)'; },
    roofNote: function (kneeStr) {
      return 'Each <b>point is one kernel</b>: ● = decode phase, ▲ = prefill phase. Intensity below the knee (peak ÷ bandwidth ≈ ' +
        kneeStr + ' FLOPs/B) → memory-bound, GPU compute goes underused; otherwise compute-bound. ' +
        'Intensity = FLOPs ÷ HBM bytes moved, computed from model structure + current context/concurrency/KV dtype. ' +
        'This is a <b>per-GPU</b> roofline: TP splits each kernel\'s FLOPs and HBM bytes by the same factor, so intensity is a sharding invariant — the points and the roof stay put when you change TP; TP only moves the latency/throughput figures in the cards below. ' +
        'Peak is a datasheet dense-throughput approximation; real kernels never reach 100%.';
    },
    decodeHeadSuffix: function (B) { return ' (per step ' + B + ' tokens, ' + B + ' concurrent)'; },
    prefillHeadSuffix: function (tokens) { return ' (chunk ' + tokens + ' tokens)'; },
    decodeFormula: function (gibBytes, bw, tp, stepMs, tpsPerReq, tpsGroup, B) {
      return "<div class='vformula'>step ≈ " + gibBytes + " GiB ÷ (" + bw +
        " TB/s × TP" + tp + ") ≈ " + stepMs + " ms → per-request bound ~" + tpsPerReq +
        " tok/s · TP-group bound ~" + tpsGroup + " tok/s (×" + B + " concurrent)</div>";
    },
    decodeFootPrefix: function () { return 'Theoretical bound (real-world is typically 50–70%).'; },
    decodeFootMem: function (kneeRatio) {
      return kneeRatio + '× from the knee. ' + 'The kernel with the largest time share is the optimization target: ' +
        'quantize the weights term, quantize KV or shrink context for the KV term; concurrency only helps kernels whose intensity scales with batch.';
    },
    decodeFootCompute: function () {
      return 'Already compute-bound: adding concurrency only lengthens step time linearly with no throughput gain; focus on compute utilization and higher-throughput dtypes (fp8/fp4).';
    },
    prefillFormula: function (chunkMs, tp, ctxLbl, seconds) {
      return "<div class='vformula'>one chunk ≈ " + chunkMs + " ms (TP" + tp + ") → processing a " +
        ctxLbl + " prompt ≈ " + seconds + " s</div>";
    },
    prefillFootMem: function () { return 'Memory-bound (uncommon — usually small chunks / large models).'; },
    prefillFootCompute: function () { return 'Compute-bound: prefill is compute-hungry; quantizing weights alone won’t save time (unless using lower-precision compute); '; },
    prefillFootTail: function () { return 'Ways to shorten TTFT: a higher-compute GPU, prefix caching, overlapping chunked prefill with decode.'; },
    kernelDetailsSummary: function () { return 'Per-kernel detail (FLOPs / HBM / intensity / time share)'; },
    kbPhaseStep: function () { return 'step'; },
    kbPhaseChunk: function () { return 'chunk'; },
    kbHeadFlops: function (suffix) { return 'FLOPs (per ' + suffix + ')'; },
    compTblComponent: function () { return 'component'; },
    compTblHbm: function () { return 'HBM access'; },
    compTblIntensity: function () { return 'intensity F/B'; },
    compTblBoundBy: function () { return 'bound by'; },
    compTblTimeShare: function () { return 'time share'; },
    compTblNote: function () { return 'note'; },
    compTblTotal: function () { return 'total'; },
    boundByMem: function () { return 'bandwidth'; },
    boundByCompute: function () { return 'compute'; },
    legendDecode: function (req) { return 'Decode (per step ' + req + ' tokens)'; },
    legendPrefill: function (tokens) { return 'Prefill (chunk ' + tokens + ' tokens)'; },
    kneeLabel: function (kneeStr) { return 'knee ≈ ' + kneeStr + ' FLOPs/B'; },
    peakLabel: function (peakStr, dtype) { return 'peak ' + peakStr + ' TFLOPs/s (' + dtype + ' dense)'; },
    bwLabel: function (bw) { return 'HBM bandwidth ' + bw + ' TB/s'; },
    pointTooltip: function (phaseLabel, intensityStr, bound, attainStr, pctStr) {
      return '<b>' + phaseLabel + '</b> · intensity ≈ ' + intensityStr + ' FLOPs/B → ' + bound +
        '<br>attainable ≈ ' + attainStr + ' TFLOPs/s (' + pctStr + '% of peak)';
    },
  }
};
function Tr(key) { return (I18N[D.lang] || I18N.zh)[key]; }

/* ============================ tab switching ============================ */
function setTab(name) {
  // evidence is the default (no body class); estimate/parallel/roofline are class-gated
  document.body.classList.toggle('tab-estimate', name === 'estimate');
  document.body.classList.toggle('tab-parallel', name === 'parallel');
  document.body.classList.toggle('tab-roofline', name === 'roofline');
  el('tab-btn-evidence').classList.toggle('active', name === 'evidence');
  el('tab-btn-estimate').classList.toggle('active', name === 'estimate');
  el('tab-btn-parallel').classList.toggle('active', name === 'parallel');
  el('tab-btn-roofline').classList.toggle('active', name === 'roofline');
  el('fnote').style.display = name === 'estimate' ? '' : 'none';
}
el('tab-btn-evidence').addEventListener('click', function(){ setTab('evidence'); });
el('tab-btn-estimate').addEventListener('click', function(){ setTab('estimate'); });
el('tab-btn-parallel').addEventListener('click', function(){ setTab('parallel'); });
el('tab-btn-roofline').addEventListener('click', function(){ setTab('roofline'); });
if (typeof location !== 'undefined') {
  if (location.hash === '#evidence') setTab('evidence');
  if (location.hash === '#estimate') setTab('estimate');
  if (location.hash === '#parallel') setTab('parallel');
  if (location.hash === '#roofline') setTab('roofline');
}

/* ====================== TAB 1: estimate (design_1) ====================== */
function barHtml(segs, total) {
  var h = '';
  for (var i = 0; i < segs.length; i++) {
    var s = segs[i], share = s.bytes / total, pct = share * 100, inner = '';
    if (pct >= 30) {
      inner = "<span class='seg-label'>" + s.label + " " + gib(s.bytes) +
              " GiB" + Tr('pctParen')(Math.round(pct)) + "</span>";
    }
    h += "<div class='seg' style='flex:" + share.toFixed(6) +
         ";background:var(--s" + (s.slot + 1) + ")' title='" + s.label + ": " +
         gib(s.bytes) + " GiB'>" + inner + "</div>";
  }
  return h;
}
function legendHtml(segs) {
  return segs.map(function (s) {
    return "<span class='lg'><i style='background:var(--s" + (s.slot + 1) +
           ")'></i>" + s.label + "&ensp;<b>" + gib(s.bytes) + "</b></span>";
  }).join('');
}

function updateEstimate() {
  var ctx = +el('f-ctx').value;
  var req = +el('f-req').value;
  var kvSel = el('f-kv').value;
  var kvDtype = kvDtypeNow();
  var kvPerTok = D.kvElemsPerLayer * (D.nKvLayers || D.L) * kvBytesPer(kvDtype);
  var kvPerReq = D.kvElemsPerLayer * kvLayerTokens(ctx) * kvBytesPer(kvDtype);
  var kvTotal = kvPerReq * req;
  var runtime = kvTotal + D.actBytes;
  var total = D.weightsBytes + runtime;
  var grand = total * (1 + D.overhead);
  var ctxStr = ctx.toLocaleString('en-US');

  setText('d-runtime', gib(runtime));
  setText('d-ctx', ctxStr);
  setText('d-req', req);
  setText('d-kv-total', gib(kvTotal));
  setText('d-ctx2', ctxStr);
  setText('d-kv-per-req', gib(kvPerReq));
  setText('d-req2', req);
  setText('d-kv-total2', gib(kvTotal));
  setText('d-kv-dtype', kvDtype + (kvSel === 'auto' ? Tr('kvAutoSuffix')() : ''));
  setText('d-kv-per-tok', (kvPerTok / 1024).toLocaleString('en-US',
    {minimumFractionDigits: 1, maximumFractionDigits: 1}));
  if (D.mhaRatio) setText('d-mha', gib(kvTotal * D.mhaRatio));

  var segs = [
    {label: Tr('weightsStaticLabel')(), bytes: D.weightsBytes, slot: D.weightsSlot},
    {label: 'KV Cache', bytes: kvTotal, slot: 5},
    {label: 'Activation', bytes: D.actBytes, slot: 6}
  ];
  setText('d-tot-head', Tr('totalHead')(gib(total), ctxStr, req));
  el('d-tot-bar').innerHTML = barHtml(segs, total);
  el('d-tot-legend').innerHTML = legendHtml(segs);
  el('d-total-line').innerHTML = Tr('totalLine')(
    gib(D.weightsBytes), gib(kvTotal), gib(D.actBytes),
    Math.round(D.overhead * 100), gib(grand), gib(kvPerReq));

  setText('d-tbl-kv-label', Tr('tblKvLabel')(ctxStr, req));
  setText('d-tbl-kv-dtype', kvDtype);
  setText('d-tbl-kv-val', gib(kvTotal));
}

/* ====================== TAB 2: parallel (design_2) ====================== */
// component colors match the parallel structure column (design_2):
// yellow embed / green attention / blue FFN / pink MTP / gray lm_head
var C = { embed:'var(--s3)', lmHead:'var(--lmh)', attn:'var(--s2)', ffn:'var(--s1)',
          mtp:'var(--s7)', others:'var(--s8)', kv:'var(--s5)', act:'var(--s6)' };
var COMPS = [
  {k:'embed',     label:'embed',        color:C.embed},
  {k:'lmHead',    label:'lm_head',      color:C.lmHead},
  {k:'attention', label:'attention',    color:C.attn},
  {k:'denseFfn',  label:'dense FFN',    color:C.ffn},
  {k:'moeRouted', label:'MoE experts',  color:C.ffn},
  {k:'moeShared', label:'MoE shared',   color:C.ffn},
  {k:'mtp',       label:'MTP',          color:C.mtp},
  {k:'others',    label:'norms/router', color:C.others}
];

function readParams(){
  var inst = el('f-inst').value, memGib, gpn, instLabel;
  if (inst === 'custom'){
    memGib = +el('f-cmem').value || 1; gpn = +el('f-cgpn').value || 1;
    instLabel = Tr('customInstLabel')(gpn, memGib);
  } else {
    var s = D.instances[inst];
    memGib = s.memGib; gpn = s.count;
    instLabel = Tr('instLabel')(inst, s.count, s.gpu, s.memGib);
  }
  var tp = +el('f-tp').value;
  return {
    tp:tp, pp:+el('f-pp').value, ep:+(el('f-ep').value||1),
    dpAttn: el('f-dp').checked && dpAvailable(tp),
    memGib:memGib, gpn:gpn, instLabel:instLabel,
    ctx:+el('f-ctx').value, req:+el('f-req').value,
    kvDtype: kvDtypeNow(), frac:+el('f-frac').value
  };
}

// DP attention only changes the picture when KV would otherwise be
// replicated across TP ranks: MLA (latent has no head dim, replicated at
// any TP) or GQA/MQA once TP exceeds the kv-head count.
function dpAvailable(tp){
  return D.kvIsMla || tp > (D.kvNKvHeads || tp);
}

function rebuildEpOptions(tp){
  if (!D.nMoe){ el('ep-box').style.display='none'; return; }
  var sel = el('f-ep'), prev = +sel.value || EP_INIT;
  var divisors = [];
  for (var d=1; d<=tp; d++) if (tp%d===0) divisors.push(d);
  var pick = divisors.indexOf(prev)>=0 ? prev : tp;
  sel.innerHTML = divisors.map(function(d){
    return "<option value='"+d+"'"+(d===pick?' selected':'')+">"+d+"</option>";
  }).join('');
}

function stageRange(s, pp){                       // layers [lo, hi); later stages get +1
  var base = Math.floor(D.L/pp), rem = D.L%pp;
  var lo = 0;
  for (var i=0;i<s;i++) lo += base + (i >= pp-rem ? 1 : 0);
  return [lo, lo + base + (s >= pp-rem ? 1 : 0)];
}

// bytes on one GPU of pipeline stage s (all tp ranks of a stage are identical)
function gpuMemory(s, P){
  var r = stageRange(s, P.pp), lo=r[0], hi=r[1], n=hi-lo;
  var nd = Math.max(0, Math.min(hi, D.nDense) - Math.min(lo, D.nDense));
  var nm = n - nd;
  var L = D.layer, w = {};
  w.embed  = (s===0) ? D.embed/P.tp : 0;
  w.lmHead = 0;
  if (s===P.pp-1) w.lmHead = D.tied ? (P.pp>1 ? D.embed/P.tp : 0) : D.lmHead/P.tp;
  w.attention = n * (P.dpAttn
    ? L.attnQo + L.attnKvProj + L.attnRepl
    : L.attnQo/P.tp + L.attnKvProj/Math.min(P.tp, D.kvNKvHeads||P.tp) + L.attnRepl);
  w.denseFfn  = nd * L.denseFfn/P.tp;
  w.moeRouted = nm * L.moeRouted/P.tp;
  w.moeShared = nm * L.moeShared/P.tp;
  w.mtp = (s===P.pp-1 && L.mtpTotal)
    ? L.mtpTotal*(L.mtpSlicedFrac/P.tp + (1-L.mtpSlicedFrac)) : 0;
  w.others = n*(L.norms + L.indexer) + nm*L.moeGate;

  var weights = 0; for (var k in w) weights += w[k];
  // KV demand: bytes this GPU must hold to serve ctx × req at the chosen sharding.
  // Stored token-positions honor sliding-window caps (kvLayerTokens); scale to
  // this pipeline stage's share of layers (exact when pp=1: n===L).
  var stageLayerTokens = kvLayerTokens(P.ctx) * n / D.L;
  var kvStage = D.kvElemsPerLayer * kvBytesPer(P.kvDtype) * stageLayerTokens * P.req;
  var kv = P.dpAttn  ? kvStage/P.tp
         : D.kvIsMla ? kvStage
                     : kvStage/Math.min(P.tp, D.kvNKvHeads);
  var act = D.actBytes/P.tp;
  var cap = P.memGib*GIB, fixed = D.fixedGib*GIB;
  // SGLang-style allocation: the static region (frac × cap) is pre-allocated at
  // startup as weights + KV pool; whatever the weights don't take becomes KV
  // capacity. Activation / CUDA graph live in the (1-frac) non-static region.
  var staticBudget = P.frac*cap;
  var kvCap = staticBudget - fixed - weights;
  var canStart = kvCap > 0;
  var used = canStart ? staticBudget + act : weights + fixed + act;
  var maxReq = (canStart && kv > 0) ? Math.floor(kvCap*P.req/kv) : 0;
  // pool capacity in tokens (= SGLang max_total_num_tokens for this GPU):
  // demand kv covers ctx*req tokens, so tokens/byte = ctx*req/kv
  var maxTokens = (canStart && kv > 0) ? kvCap*P.ctx*P.req/kv : 0;
  return { w:w, weights:weights, kv:kv, kvCap:Math.max(kvCap,0), kvCapRaw:kvCap,
           canStart:canStart, act:act, used:used, maxReq:maxReq, maxTokens:maxTokens,
           layers:[lo,hi], nDense:nd, nMoe:nm };
}

// experts held by tp rank t (EP grouping)
function expertInfo(t, P){
  if (!D.nMoe || !D.nExperts) return null;
  var grp = P.tp/P.ep, epRank = Math.floor(t/grp);
  var base = Math.floor(D.nExperts/P.ep), rem = D.nExperts%P.ep;
  var cnt = base + (epRank < rem ? 1 : 0);
  var lo = epRank*base + Math.min(epRank, rem);
  return { cnt:cnt, lo:lo, hi:lo+cnt-1, sliceDenom:grp };
}

function memBar(m, P){
  var cap = P.memGib*GIB, h = '';
  // static region fills left-to-right; non-static blocks (activation) anchor at
  // the RIGHT edge with the free headroom in between, so they never overlap the
  // frac boundary line.
  var parts = [];
  COMPS.forEach(function(c){ if (m.w[c.k] > 0) parts.push({label:c.label, b:m.w[c.k], color:c.color}); });
  parts.push({label:Tr('fixedOverhead')(), b:D.fixedGib*GIB, color:'var(--fixed)'});
  if (m.canStart) parts.push({label:Tr('kvCapLabel')(), b:m.kvCap, color:C.kv});
  function seg(s0){
    // clamp tiny-but-real segments to ~0.6% so they stay visible (activation
    // per GPU can be <0.2% of the card); flex-grow normalizes the slight excess
    var frac = Math.min(Math.max(s0.b/cap, 0.006), 1);
    return "<div class='seg' style='flex:"+frac.toFixed(6)+" 0 0;background:"+s0.color+
         "' data-tip='<b>"+s0.label+"</b>&emsp;"+gib(s0.b)+" GiB "+
         "<span class=\"tpct\">"+Tr('gpuPctTip')(P.memGib, (s0.b/cap*100).toFixed(1))+"</span>'></div>";
  }
  parts.forEach(function(s0){ h += seg(s0); });
  var free = cap - m.used;
  if (free > 0){
    var ffrac = free/cap;
    var flabel = ffrac > 0.14 ? 'CUDA graph… ' + Math.round(ffrac*100) + '%' : '';
    h += "<div class='free' style='flex:"+ffrac.toFixed(6)+" 0 0' data-tip='<b>"+Tr('freeTipLabel')()+"</b>&emsp;"+
         gib(free)+" GiB <span class=\"tpct\">"+Tr('freePct')((ffrac*100).toFixed(1))+"</span>'>"+flabel+"</div>";
  }
  h += seg({label:'activation', b:m.act, color:C.act});
  // dashed marker at the mem-fraction-static boundary
  h += "<div class='fracline' style='left:"+(P.frac*100).toFixed(2)+"%' data-tip='"+
       Tr('fracLineTip')(P.frac.toFixed(2))+"'></div>";
  return "<div class='membar'>"+h+"</div>";
}

// KV demand vs KV pool capacity as a bullet chart: the scale is
// max(demand, capacity); the capacity mark is a thick tick on a light track,
// demand is the dark fill. Overflow past the tick turns red, so 216% and 500%
// look different (the tick sits at a different position).
function kvUtilBar(m, P){
  if (!m.canStart || m.kvCap <= 0) return '';
  var ratio = m.kv/m.kvCap, pct = (ratio*100).toFixed(0), over = ratio > 1;
  var scale = Math.max(m.kv, m.kvCap);
  var capPos = m.kvCap/scale*100, fillW = m.kv/scale*100;
  var okW = Math.min(fillW, capPos);
  var tip = Tr('kvUtilTip')(gib(m.kv), gib(m.kvCap), pct);
  var h = "<div class='kvutil' data-tip='"+tip+"'>"+
    "<span class='kvutil-lb'>"+Tr('kvUtilBarLabel')()+"</span>"+
    "<div class='kvutil-track'>"+
      "<div class='kvutil-capzone' style='width:"+capPos.toFixed(2)+"%'></div>"+
      "<div class='kvutil-fill' style='width:"+okW.toFixed(2)+"%'></div>"+
      (over ? "<div class='kvutil-fill over' style='left:"+capPos.toFixed(2)+"%;width:"+
              (fillW-capPos).toFixed(2)+"%'></div>" : '')+
      "<div class='kvutil-capline' style='left:"+capPos.toFixed(2)+"%'></div>"+
    "</div><span class='kvutil-pct"+(over?' over':'')+"'>"+pct+"%</span></div>";
  return h;
}

function chips(m, s, t, P){
  var h = '';
  if (m.w.embed>0) h += "<span class='chip-io' style='background:"+C.embed+"'>embed ⁄"+P.tp+"</span>";
  var lo=m.layers[0], hi=m.layers[1];
  h += "<span class='chip'><i style='background:"+C.attn+"'></i><i style='background:"+C.ffn+
       "'></i><span class='lb'>"+Tr('layerChipLabel')(lo, hi-1, hi-lo)+(P.dpAttn?'':Tr('perLayerFrac')(P.tp))+"</span></span>";
  if (m.w.mtp>0) h += "<span class='chip-io' style='background:"+C.mtp+"'>MTP</span>";
  if (m.w.lmHead>0) h += "<span class='chip-io' style='background:"+C.lmHead+"'>lm_head ⁄"+P.tp+"</span>";
  var eh = '';
  var ei = expertInfo(t, P);
  if (ei && m.nMoe>0){
    var nsq = Math.min(ei.cnt, 12), sq='';
    for (var i=0;i<nsq;i++) sq += "<span class='eg"+(ei.sliceDenom>1?' cut':'')+"'></span>";
    var shape = ei.sliceDenom===1 ? Tr('expertShapeFull')() : Tr('expertShapeSliced')(ei.sliceDenom);
    eh = "<div class='experts'>"+sq+(ei.cnt>nsq?'…':'')+" e"+ei.lo+"–e"+ei.hi+
         Tr('expertLine')(ei.cnt, shape, P.ep)+"</div>";
  }
  return "<div class='chips'>"+h+"</div>"+eh;
}

function updateParallel(){
  var P = readParams();
  rebuildEpOptions(P.tp); P.ep = +el('f-ep').value || 1;
  el('dp-box').style.display = dpAvailable(P.tp) ? '' : 'none';
  el('custom-box').style.display = el('f-inst').value==='custom' ? 'inline-flex' : 'none';

  var warn = [];
  if (P.pp > D.L) { warn.push(Tr('warnPpExceeds')(P.pp, D.L)); }
  el('warnmsg').style.display = warn.length ? '' : 'none';
  el('warnmsg').textContent = warn.join(D.lang === 'en' ? '; ' : '；');
  if (warn.length) { el('cluster').innerHTML=''; return; }

  // per-stage memory (identical across tp ranks)
  var stages = [];
  for (var s=0;s<P.pp;s++) stages.push(gpuMemory(s, P));

  var world = P.tp*P.pp, nNodes = Math.ceil(world/P.gpn);
  var cap = P.memGib*GIB;
  var html = '', oomTotal = 0, maxUsed = 0;
  var clusterWeights = 0, clusterKv = 0;

  for (var nd=0; nd<nNodes; nd++){
    var cards = '', nodeOom = 0;
    for (var g=nd*P.gpn; g<Math.min((nd+1)*P.gpn, world); g++){
      var s0 = Math.floor(g/P.tp), t = g%P.tp;
      var m = stages[s0];
      clusterWeights += m.weights; clusterKv += m.kv;
      var free = cap - m.used, oom = !m.canStart || free < 0;
      if (oom){ nodeOom++; oomTotal++; }
      maxUsed = Math.max(maxUsed, m.used);
      var memTxt = !m.canStart
        ? "<span class='oombadge'>"+Tr('cannotStart')(gib(-m.kvCapRaw))+"</span>"
        : oom
          ? "<span class='oombadge'>"+Tr('oomExceeds')(gib(-free))+"</span>"
          : "<span class='gpu-mem'><b>"+gib(m.used)+"</b> / "+P.memGib+" GiB</span>";
      var utilLine = m.canStart
        ? "<div class='gpu-line'>"+Tr('kvUtilLine')(gib(m.kv), gib(m.kvCap),
            (m.kv/m.kvCap*100).toFixed(0), tokLabel(m.maxTokens), ctxLabel(P.ctx), m.maxReq)+"</div>"
        : '';
      cards += "<div class='gpu"+(oom?' oom':'')+"'>"+
        "<div class='gpu-head'><span class='gpu-name'>GPU-"+g+"</span>"+
        "<span class='gpu-rank'>pp"+s0+"·tp"+t+"</span>"+memTxt+"</div>"+
        memBar(m, P)+
        kvUtilBar(m, P)+
        "<div class='gpu-line'>"+Tr('gpuLine')(gib(m.weights), gib(m.canStart?m.kvCap:0), gib(m.act), oom?'':Tr('freeTail')(gib(free)))+"</div>"+
        utilLine+
        chips(m, s0, t, P)+"</div>";
    }
    html += "<div class='node'><div class='node-head'>"+
      "<span class='node-title'>Node-"+(nd+1)+"</span>"+
      "<span class='node-sub'>"+P.instLabel+"</span>"+
      (nodeOom ? "<span class='node-warn'>"+Tr('nodeWarn')(nodeOom)+"</span>" : '')+
      "</div><div class='gpugrid'>"+cards+"</div></div>";
  }
  el('cluster').innerHTML = html;

  // summary + legend
  var repl = clusterWeights - D.weightsBytes;
  var kvSingle = D.kvElemsPerLayer*kvBytesPer(P.kvDtype)*kvLayerTokens(P.ctx)*P.req;
  var kvRepl = clusterKv/kvSingle;
  el('sum-head').textContent = 'TP'+P.tp+' × PP'+P.pp+(D.nMoe?' × EP'+P.ep:'')+
    Tr('sumHeadTail')(world, nNodes, P.instLabel,
      oomTotal ? Tr('oomPartOom')(oomTotal) : Tr('oomPartOk')());

  var lg = '';
  var seen = {};
  COMPS.forEach(function(c){
    var b = 0; stages.forEach(function(m){ b = Math.max(b, m.w[c.k]); });
    if (b>0 && !seen[c.color]){
      seen[c.color] = 1;
      var label = c.color===C.ffn ? (D.nMoe ? 'FFN / MoE experts' : 'FFN') : c.label;
      lg += "<span class='lg'><i style='background:"+c.color+"'></i>"+label+"</span>";
    }
  });
  lg += "<span class='lg'><i style='background:"+C.kv+"'></i>"+Tr('kvCapLabel')()+" ("+P.kvDtype+")</span>";
  lg += "<span class='lg'><i style='background:"+C.act+"'></i>activation</span>";
  lg += "<span class='lg'><i style='background:var(--fixed)'></i>"+Tr('fixedOverhead')()+"</span>";
  el('legend').innerHTML = lg;

  var kvNote = P.dpAttn
    ? Tr('kvNoteDpOn')()
    : D.kvIsMla
      ? Tr('kvNoteMlaRepl')(kvRepl.toFixed(1), gib(clusterKv))
      : (P.tp > D.kvNKvHeads
          ? Tr('kvNoteTpExceeds')(P.tp, D.kvNKvHeads, (P.tp/Math.min(P.tp,D.kvNKvHeads)).toFixed(1))
          : Tr('kvNoteSliced')(P.tp));
  var minMaxReq = Infinity, minMaxTok = Infinity, allStart = true;
  stages.forEach(function(m){
    if (!m.canStart) allStart = false;
    else { minMaxReq = Math.min(minMaxReq, m.maxReq); minMaxTok = Math.min(minMaxTok, m.maxTokens); }
  });
  var capLine = allStart && isFinite(minMaxTok)
    ? '<br>'+Tr('sumCapLine')(tokLabel(minMaxTok), ctxLabel(P.ctx), minMaxReq, P.frac.toFixed(2)) : '';
  el('sum-line').innerHTML =
    Tr('sumLine')(gib(maxUsed), P.memGib, gib(clusterWeights), gib(D.weightsBytes), gib(Math.max(repl,0)))+
    '<br>'+kvNote+capLine;

  // table view
  var th = Tr('tableHeader')();
  var rows = '';
  for (var g2=0; g2<world; g2++){
    var s2 = Math.floor(g2/P.tp), t2 = g2%P.tp, m2 = stages[s2];
    var f2 = cap - m2.used;
    rows += "<tr><td>GPU-"+g2+"</td><td>pp"+s2+"·tp"+t2+"</td>"+
      "<td>L"+m2.layers[0]+"–L"+(m2.layers[1]-1)+"</td>"+
      "<td class='num'>"+gib(m2.weights)+"</td>"+
      "<td class='num"+(m2.canStart?"":" oomcell")+"'>"+gib(m2.kvCap)+"</td>"+
      "<td class='num"+(m2.canStart && m2.kv>m2.kvCap?" oomcell":"")+"'>"+gib(m2.kv)+"</td>"+
      "<td class='num'>"+gib(m2.act)+"</td><td class='num'>"+gib(m2.used)+"</td>"+
      "<td class='num"+(f2<0?" oomcell":"")+"'>"+gib(f2)+"</td></tr>";
  }
  el('dtable').innerHTML = th+rows;
}

/* ====================== TAB 3: roofline (design_3) ====================== */
function fmtNum(x) {
  if (x >= 1000) return Math.round(x).toLocaleString('en-US');
  if (x >= 10) return x.toFixed(0);
  if (x >= 1) return x.toFixed(1);
  return x.toFixed(2);
}

/**
 * Snapshot the user-controlled inputs needed by every roofline calculator.
 *
 * Output fields:
 *   decodeTokens   - B, one generated token per active request in a decode step
 *   prefillTokens  - T, tokens processed by one prefill chunk
 *   contextTokens  - S, cached tokens read by each decode request
 *   kvBytes        - storage bytes per KV element at the selected precision
 *   layers         - number of transformer layers
 *   attentionGeometry/pattern - model structure serialized by Python
 *   topk/nExperts  - routed experts selected per token / total routed experts
 */
function currentRooflineWorkload() {
  return {
    decodeTokens: +el('f-req').value,
    prefillTokens: D.batchTokens,
    contextTokens: +el('f-ctx').value,
    kvBytes: kvBytesPer(kvDtypeNow()),
    layers: D.L,
    attentionGeometry: D.attentionCore.geometry,
    attentionPattern: D.attentionCore.pattern,
    topk: D.topk,
    nExperts: D.nExperts
  };
}

/**
 * Build the common output of a single phase calculation.
 *
 * Inputs are total FLOPs and HBM bytes for the phase. Output is always
 * {flops, bytes, intensity, note}, where intensity = FLOPs / HBM bytes.
 */
function phaseWork(flops, bytes, note) {
  return { flops: flops, bytes: bytes, intensity: flops / bytes, note: note };
}

/**
 * Shared arithmetic for a weight-bearing GEMM.
 *
 * Input kernel: {params, bytes}; input workload: decode/prefill token counts.
 * Formula per phase:
 *   FLOPs = 2 * logical weight parameters * tokens
 *   HBM bytes = stored weight bytes, streamed once and shared by all tokens
 * Output: {dec: phaseWork, pre: phaseWork}.
 */
function calculateWeightGemmWork(kernel, workload) {
  var B = workload.decodeTokens;
  var T = workload.prefillTokens;
  return {
    dec: phaseWork(
      2 * kernel.params * B,
      kernel.bytes,
      Tr('gemmDecodeNote')(B)
    ),
    pre: phaseWork(
      2 * kernel.params * T,
      kernel.bytes,
      Tr('gemmPrefillNote')(T.toLocaleString('en-US'))
    )
  };
}

/**
 * Attention Q/K/V/O projection kernel.
 *
 * Inputs: aggregate projection {params, bytes} and current workload.
 * Formula: FLOPs = 2 * params * tokens; HBM = projection weight bytes.
 * Output: decode and prefill phase work from calculateWeightGemmWork().
 */
function calculateAttentionProjectionWork(kernel, workload) {
  return calculateWeightGemmWork(kernel, workload);
}

/**
 * DSA indexer projection kernel.
 *
 * Inputs: aggregate indexer {params, bytes} and current workload.
 * Formula: FLOPs = 2 * params * tokens; HBM = indexer weight bytes.
 * This covers projection GEMMs only; full-context scoring/top-k is not modeled.
 * Output: decode and prefill phase work from calculateWeightGemmWork().
 */
function calculateDsaIndexerWork(kernel, workload) {
  return calculateWeightGemmWork(kernel, workload);
}

/**
 * Dense FFN kernel.
 *
 * Inputs: aggregate FFN {params, bytes} and current workload.
 * Formula: FLOPs = 2 * params * tokens; HBM = FFN weight bytes.
 * Output: decode and prefill phase work from calculateWeightGemmWork().
 */
function calculateDenseFfnWork(kernel, workload) {
  return calculateWeightGemmWork(kernel, workload);
}

/**
 * MoE router kernel.
 *
 * Inputs: aggregate router {params, bytes} and current workload.
 * Formula: FLOPs = 2 * params * tokens; HBM = router weight bytes.
 * Output: decode and prefill phase work from calculateWeightGemmWork().
 */
function calculateMoeRouterWork(kernel, workload) {
  return calculateWeightGemmWork(kernel, workload);
}

/**
 * Routed MoE expert kernels.
 *
 * Inputs: routed-expert {params, bytes}; workload supplies B, T, topk, nExperts.
 * Decode formulas:
 *   active parameter ratio = topk / nExperts
 *   touched expert ratio = min(1, B * topk / nExperts)
 *   FLOPs = 2 * params * active ratio * B
 *   HBM bytes = stored expert bytes * touched expert ratio
 * Prefill assumes a large chunk touches all experts:
 *   FLOPs = 2 * params * active ratio * T; HBM bytes = all expert bytes
 * Output: {dec: phaseWork, pre: phaseWork}.
 */
function calculateMoeExpertsWork(kernel, workload) {
  var B = workload.decodeTokens;
  var T = workload.prefillTokens;
  var activeRatio = workload.topk / workload.nExperts;
  var touchedRatio = Math.min(1, B * activeRatio);
  return {
    dec: phaseWork(
      2 * kernel.params * activeRatio * B,
      kernel.bytes * touchedRatio,
      Tr('moeDecodeNote')(B, Math.round(touchedRatio * 100))
    ),
    pre: phaseWork(
      2 * kernel.params * activeRatio * T,
      kernel.bytes,
      Tr('moePrefillNote')(T.toLocaleString('en-US'))
    )
  };
}

/**
 * Shared MoE expert kernel.
 *
 * Inputs: aggregate shared-expert {params, bytes} and current workload.
 * Formula: FLOPs = 2 * params * tokens; HBM = shared-expert weight bytes.
 * Output: decode and prefill phase work from calculateWeightGemmWork().
 */
function calculateMoeSharedExpertsWork(kernel, workload) {
  return calculateWeightGemmWork(kernel, workload);
}

/**
 * LM-head projection kernel.
 *
 * Inputs: lm_head {params, bytes} and current workload.
 * Formula: FLOPs = 2 * params * tokens; HBM = lm_head weight bytes.
 * Output: decode and prefill phase work from calculateWeightGemmWork().
 */
function calculateLmHeadWork(kernel, workload) {
  return calculateWeightGemmWork(kernel, workload);
}

/**
 * Standard multi-head attention geometry.
 *
 * Input spec: {qHeads, kvHeads, qkHeadDim, valueHeadDim}.
 * Per attended query/key pair:
 *   QK FLOPs = 2 * qHeads * qkHeadDim
 *   AV FLOPs = 2 * qHeads * valueHeadDim
 * KV elements per key = kvHeads * (qkHeadDim + valueHeadDim)
 * For MHA qHeads == kvHeads.
 */
function calculateMhaAttentionGeometry(spec) {
  return {
    label: Tr('mhaLabel')(spec.qHeads),
    flopsPerPair: 2 * spec.qHeads * (spec.qkHeadDim + spec.valueHeadDim),
    kvElementsPerKey: spec.kvHeads * (spec.qkHeadDim + spec.valueHeadDim)
  };
}

/**
 * Grouped-query (including multi-query) attention geometry.
 *
 * Input spec: {qHeads, kvHeads, qkHeadDim, valueHeadDim}.
 * FLOPs still scale with qHeads because every query head computes QK and AV.
 * KV storage scales with kvHeads because each KV head is shared by a group:
 *   FLOPs/pair = 2 * qHeads * (qkHeadDim + valueHeadDim)
 *   KV elements/key = kvHeads * (qkHeadDim + valueHeadDim)
 */
function calculateGqaAttentionGeometry(spec) {
  return {
    label: Tr('gqaLabel')(spec.qHeads, spec.kvHeads),
    flopsPerPair: 2 * spec.qHeads * (spec.qkHeadDim + spec.valueHeadDim),
    kvElementsPerKey: spec.kvHeads * (spec.qkHeadDim + spec.valueHeadDim)
  };
}

/**
 * Absorbed MLA attention geometry used with a compressed latent KV cache.
 *
 * Input spec: {qHeads, kvLoraRank, ropeHeadDim}.
 * Let Dc = kvLoraRank and Dr = ropeHeadDim. For each query/key pair:
 *   latent QK = 2 * qHeads * Dc
 *   RoPE QK   = 2 * qHeads * Dr
 *   latent AV = 2 * qHeads * Dc
 * Therefore FLOPs/pair = 2 * qHeads * (2*Dc + Dr), while the shared cache
 * stores only Dc + Dr elements per key position.
 */
function calculateMlaAttentionGeometry(spec) {
  return {
    label: Tr('mlaLabel')(spec.kvLoraRank, spec.ropeHeadDim),
    flopsPerPair: 2 * spec.qHeads * (2 * spec.kvLoraRank + spec.ropeHeadDim),
    kvElementsPerKey: spec.kvLoraRank + spec.ropeHeadDim
  };
}

/**
 * Select the geometry calculator declared by the model config.
 *
 * Input: the serialized attention geometry. Output:
 * {label, flopsPerPair, kvElementsPerKey}.
 */
function calculateAttentionGeometry(spec) {
  if (spec.kind === 'mha') return calculateMhaAttentionGeometry(spec);
  if (spec.kind === 'gqa') return calculateGqaAttentionGeometry(spec);
  if (spec.kind === 'mla') return calculateMlaAttentionGeometry(spec);
  throw new Error('Unsupported attention geometry: ' + spec.kind);
}

/**
 * Number of causal query/key pairs in a dense T-token chunk.
 *
 * Query i attends i+1 keys, so pairs = 1 + ... + T = T*(T+1)/2.
 */
function denseCausalPairs(tokens) {
  return tokens * (tokens + 1) / 2;
}

/**
 * Number of causal pairs when every query attends at most `limit` keys.
 *
 * The first min(T, limit) queries form a dense causal triangle. Every later
 * query contributes exactly `limit` selected keys.
 */
function cappedCausalPairs(tokens, limit) {
  var denseTokens = Math.min(tokens, limit);
  return denseCausalPairs(denseTokens) + Math.max(0, tokens - limit) * limit;
}

/**
 * Dense attention access pattern across all transformer layers.
 *
 * Inputs: B decode queries, context S, prefill chunk T, and layer count L.
 * Decode pairs and KV reads = B*S*L.
 * Prefill pairs = T*(T+1)/2*L. Prefill KV reads = T*L, the ideal one-pass
 * HBM lower bound; FlashAttention tiling/reloads are intentionally excluded.
 */
function calculateDenseAttentionPattern(workload) {
  var decodePairs = workload.decodeTokens * workload.contextTokens * workload.layers;
  return {
    label: 'dense',
    decodePairs: decodePairs,
    prefillPairs: denseCausalPairs(workload.prefillTokens) * workload.layers,
    decodeKvReads: decodePairs,
    prefillKvReads: workload.prefillTokens * workload.layers,
    decodeNote: Tr('denseDecodeNote')(ctxLabel(workload.contextTokens)),
    prefillNote: Tr('densePrefillNote')()
  };
}

/**
 * DSA sparse attention access pattern across all transformer layers.
 *
 * Input pattern: {kind:'dsa', topk}. DSA is composed with MHA/GQA/MLA; it is
 * not a KV geometry. Each decode query attends min(S, topk) selected keys.
 * Prefill uses the exact top-k-capped causal pair count. KV HBM for prefill
 * remains the ideal one-pass lower bound and excludes irregular gather reloads.
 * Full-context index scoring/top-k is not modeled; the DSA indexer row currently
 * covers its projection GEMMs only.
 */
function calculateDsaAttentionPattern(workload, pattern) {
  var attended = Math.min(workload.contextTokens, pattern.topk);
  return {
    label: Tr('dsaLabel')(pattern.topk),
    decodePairs: workload.decodeTokens * attended * workload.layers,
    prefillPairs: cappedCausalPairs(workload.prefillTokens, pattern.topk) * workload.layers,
    decodeKvReads: workload.decodeTokens * attended * workload.layers,
    prefillKvReads: workload.prefillTokens * workload.layers,
    decodeNote: Tr('dsaDecodeNote')(ctxLabel(workload.contextTokens), attended),
    prefillNote: Tr('dsaPrefillNote')(pattern.topk)
  };
}

/**
 * Capped attention (sliding window or block-sparse) across a subset of layers.
 *
 * `capLayers` of the total layers cap each query at min(context, cap) keys; the
 * remaining layers are dense (full context). This splits the per-layer read
 * accordingly, so MiniMax-style 57/60-sparse or Gemma-style 5:1-sliding models
 * are not treated as fully dense.
 */
function calculateCappedAttentionPattern(workload, pattern) {
  var S = workload.contextTokens, B = workload.decodeTokens, T = workload.prefillTokens;
  var capL = Math.min(pattern.capLayers, workload.layers);
  var denseL = workload.layers - capL;
  var attended = Math.min(S, pattern.cap);
  var decodeKvReads = B * (attended * capL + S * denseL);
  var prefillPairsCapped = cappedCausalPairs(T, pattern.cap) * capL
                         + denseCausalPairs(T) * denseL;
  return {
    label: Tr('dsaLabel') ? Tr('dsaLabel')(pattern.cap) : 'capped',
    decodePairs: decodeKvReads,
    prefillPairs: prefillPairsCapped,
    decodeKvReads: decodeKvReads,
    prefillKvReads: T * workload.layers,
    decodeNote: Tr('dsaDecodeNote')(ctxLabel(S), attended) + ' × ' + capL + 'L',
    prefillNote: Tr('dsaPrefillNote')(pattern.cap)
  };
}

/**
 * Select the dense / DSA / capped access-pattern calculator.
 *
 * Output: pair counts for FLOPs and KV-key read counts for ideal HBM traffic.
 */
function calculateAttentionPattern(workload) {
  var pattern = workload.attentionPattern;
  if (pattern.kind === 'dense') return calculateDenseAttentionPattern(workload);
  if (pattern.kind === 'dsa') return calculateDsaAttentionPattern(workload, pattern);
  if (pattern.kind === 'capped') return calculateCappedAttentionPattern(workload, pattern);
  throw new Error('Unsupported attention pattern: ' + pattern.kind);
}

/**
 * Weight-free attention core: QK score plus score-weighted value aggregation.
 *
 * Geometry determines FLOPs per attended pair and KV elements per key.
 * Pattern determines how many pairs are computed and how many KV keys reach
 * HBM. Output is the common {dec, pre} phase work consumed by the roofline.
 *
 * Formulas per phase:
 *   FLOPs = geometry.flopsPerPair * pattern.queryKeyPairs
 *   HBM bytes = geometry.kvElementsPerKey * pattern.kvKeyReads * kvBytes
 */
function calculateAttentionCoreWork(workload) {
  var geometry = calculateAttentionGeometry(workload.attentionGeometry);
  var pattern = calculateAttentionPattern(workload);
  var prefix = geometry.label + ' + ' + pattern.label + ': ';
  return {
    dec: phaseWork(
      geometry.flopsPerPair * pattern.decodePairs,
      geometry.kvElementsPerKey * pattern.decodeKvReads * workload.kvBytes,
      prefix + pattern.decodeNote
    ),
    pre: phaseWork(
      geometry.flopsPerPair * pattern.prefillPairs,
      geometry.kvElementsPerKey * pattern.prefillKvReads * workload.kvBytes,
      prefix + pattern.prefillNote
    )
  };
}

// Every emitted weight kernel must opt into one explicit calculator. Adding a
// Python-side kernel without updating this table fails visibly instead of
// silently applying an unrelated formula.
var ROOFLINE_KERNEL_CALCULATORS = {
  attention: calculateAttentionProjectionWork,
  indexer: calculateDsaIndexerWork,
  dense_ffn: calculateDenseFfnWork,
  moe_gate: calculateMoeRouterWork,
  moe_routed: calculateMoeExpertsWork,
  moe_shared: calculateMoeSharedExpertsWork,
  lm_head: calculateLmHeadWork
};

/**
 * Orchestrate all per-kernel calculators without owning any kernel formula.
 *
 * Output: {rows, B, T}. Each row keeps display metadata and the common
 * {dec, pre} phase results consumed by the chart, table, and verdict cards.
 */
function rooflineKernels() {
  var workload = currentRooflineWorkload();
  var rows = (D.kernels || []).map(function (kernel) {
    var calculate = ROOFLINE_KERNEL_CALCULATORS[kernel.key];
    if (!calculate) throw new Error('No roofline calculator for kernel: ' + kernel.key);
    var work = calculate(kernel, workload);
    return {
      key: kernel.key,
      label: kernel.label,
      color: kernel.color,
      kind: kernel.kind,
      dec: work.dec,
      pre: work.pre
    };
  });

  var attentionCore = calculateAttentionCoreWork(workload);
  rows.push({
    key: 'attn_core',
    label: Tr('attnCoreLabel')(),
    color: C.kv,
    kind: 'attn',
    dec: attentionCore.dec,
    pre: attentionCore.pre
  });

  return { rows: rows, B: workload.decodeTokens, T: workload.prefillTokens };
}

// phases we render; PH[i].pick(row) returns that phase's {flops,bytes,intensity,note}
var PH = [
  { key: 'dec', label: 'Decode', pick: function (r) { return r.dec; } },
  { key: 'pre', label: 'Prefill', pick: function (r) { return r.pre; } }
];

function aggregate(rows, phase, label, color) {
  var f = 0, b = 0;
  rows.forEach(function (r) { var c = phase.pick(r); f += c.flops; b += c.bytes; });
  return { label: label, color: color, flops: f, bytes: b, intensity: f / b };
}

// roofline execution model: kernel time = max(bytes/BW, FLOPs/peak)
function compTime(c, perf) {
  return Math.max(c.bytes / (perf.bw * 1e12), c.flops / (perf.peak * 1e12));
}

function currentGpuPerf() {
  var inst = el('f-inst').value;
  if (inst === 'custom' || !D.instances[inst]) return null;
  var gpu = D.instances[inst].gpu;
  var perf = D.gpuPerf[gpu];
  if (!perf) return null;
  var dt = D.weightDtype;
  var peak = perf[dt], usedDt = dt;
  if (!peak) { peak = perf.bf16; usedDt = 'bf16'; }
  return { gpu: gpu, peak: peak, bw: perf.bw, dtype: usedDt,
           fallback: usedDt !== dt ? dt : null };
}

function drawRoofline(perf, pts) {
  var W = 860, H = 420, ml = 64, mr = 24, mt = 16, mb = 46;
  var pw = W - ml - mr, ph = H - mt - mb;
  var knee = perf.peak / perf.bw;  // TFLOPs / (TB/s) = FLOPs/byte
  // log-log: x 0.1..10^4 FLOPs/B, y from ~peak/2e4 up to peak*2
  var x0 = Math.log10(0.1), x1 = Math.log10(10000);
  var y1 = Math.log10(perf.peak * 2), y0 = y1 - 4.6;
  function X(v) { return ml + (Math.log10(v) - x0) / (x1 - x0) * pw; }
  function Y(v) { return mt + (y1 - Math.log10(v)) / (y1 - y0) * ph; }
  function perfAt(i) { return Math.min(i * perf.bw, perf.peak); }

  var s = "<svg viewBox='0 0 " + W + " " + H + "' xmlns='http://www.w3.org/2000/svg'>";

  // shaded regions under the roof: memory-bound (left of knee) / compute-bound
  var kx = X(knee), py = Y(perf.peak), by = mt + ph;
  var slopePath = 'M' + X(0.1) + ',' + Y(perfAt(0.1));
  for (var lx = -1; lx <= Math.log10(knee) + 0.001; lx += 0.05) {
    var iv = Math.pow(10, Math.min(lx, Math.log10(knee)));
    slopePath += ' L' + X(iv) + ',' + Y(perfAt(iv));
  }
  s += "<path d='" + slopePath + " L" + kx + "," + by + " L" + X(0.1) + "," + by +
       " Z' fill='var(--s6)' opacity='0.07'/>";
  s += "<rect x='" + kx + "' y='" + py + "' width='" + (ml + pw - kx) +
       "' height='" + (by - py) + "' fill='var(--s2)' opacity='0.08'/>";

  // gridlines + tick labels (decades), plus minor 2×/5× lines between them
  for (var d = Math.ceil(x0); d <= x1; d++) {
    var gx = X(Math.pow(10, d));
    s += "<line x1='" + gx + "' y1='" + mt + "' x2='" + gx + "' y2='" + by +
         "' stroke='var(--grid)' stroke-width='1'/>";
    s += "<text x='" + gx + "' y='" + (by + 18) + "' text-anchor='middle' " +
         "fill='var(--muted)' font-size='11'>" +
         (d >= 0 ? Math.pow(10, d).toLocaleString('en-US') : Math.pow(10, d)) + "</text>";
  }
  for (var md = Math.floor(x0); md <= x1; md++) {
    [2, 5].forEach(function (m) {
      var lv = md + Math.log10(m);
      if (lv <= x0 || lv >= x1) return;
      var mx = X(Math.pow(10, lv));
      s += "<line x1='" + mx + "' y1='" + mt + "' x2='" + mx + "' y2='" + by +
           "' stroke='var(--grid)' stroke-width='1' opacity='0.4'/>";
    });
  }
  for (var dy = Math.ceil(y0); dy <= y1; dy++) {
    var gy = Y(Math.pow(10, dy));
    s += "<line x1='" + ml + "' y1='" + gy + "' x2='" + (ml + pw) + "' y2='" + gy +
         "' stroke='var(--grid)' stroke-width='1'/>";
    s += "<text x='" + (ml - 8) + "' y='" + (gy + 4) + "' text-anchor='end' " +
         "fill='var(--muted)' font-size='11'>" +
         (dy >= 0 ? Math.pow(10, dy).toLocaleString('en-US') : Math.pow(10, dy)) + "</text>";
  }
  for (var mdy = Math.floor(y0); mdy <= Math.ceil(y1); mdy++) {
    [2, 5].forEach(function (m) {
      var lv = mdy + Math.log10(m);
      if (lv <= y0 || lv >= y1) return;
      var my = Y(Math.pow(10, lv));
      s += "<line x1='" + ml + "' y1='" + my + "' x2='" + (ml + pw) + "' y2='" + my +
           "' stroke='var(--grid)' stroke-width='1' opacity='0.4'/>";
      s += "<text x='" + (ml - 8) + "' y='" + (my + 3) + "' text-anchor='end' " +
           "fill='var(--muted)' font-size='9' opacity='0.8'>" + fmtNum(m * Math.pow(10, mdy)) + "</text>";
    });
  }
  // axis titles
  s += "<text x='" + (ml + pw / 2) + "' y='" + (H - 8) + "' text-anchor='middle' " +
       "fill='var(--ink2)' font-size='12'>Arithmetic Intensity (FLOPs / Byte, log)</text>";
  s += "<text x='14' y='" + (mt + ph / 2) + "' text-anchor='middle' fill='var(--ink2)' " +
       "font-size='12' transform='rotate(-90 14 " + (mt + ph / 2) + ")'>TFLOPs/s (log)</text>";

  // roof: slope + flat
  s += "<path d='" + slopePath + "' fill='none' stroke='var(--ink)' stroke-width='2.5'/>";
  s += "<line x1='" + kx + "' y1='" + py + "' x2='" + (ml + pw) + "' y2='" + py +
       "' stroke='var(--ink)' stroke-width='2.5'/>";
  // knee marker
  s += "<line x1='" + kx + "' y1='" + py + "' x2='" + kx + "' y2='" + by +
       "' stroke='var(--s1)' stroke-width='1.5' stroke-dasharray='5 4'/>";
  s += "<text x='" + (kx + 6) + "' y='" + (by - 8) + "' fill='var(--s1)' font-size='11.5'>" +
       Tr('kneeLabel')(fmtNum(knee)) + "</text>";
  // region labels
  s += "<text x='" + X(Math.sqrt(0.1 * knee)) + "' y='" + (mt + ph * 0.62) +
       "' text-anchor='middle' fill='var(--s6)' font-size='13' opacity='0.85'>Memory-bound</text>";
  s += "<text x='" + X(Math.sqrt(knee * 10000) * 0.8) + "' y='" + (mt + ph * 0.4) +
       "' text-anchor='middle' fill='var(--s4)' font-size='13' opacity='0.85'>Compute-bound</text>";
  // roof annotations
  s += "<text x='" + (ml + pw - 8) + "' y='" + (py - 8) + "' text-anchor='end' " +
       "fill='var(--ink2)' font-size='11.5'>" +
       Tr('peakLabel')(perf.peak.toLocaleString('en-US'), perf.dtype) + "</text>";
  var midSlope = Math.pow(10, (Math.log10(0.1) + Math.log10(knee)) / 2);
  s += "<text x='" + (X(midSlope) - 6) + "' y='" + (Y(perfAt(midSlope)) - 12) +
       "' fill='var(--ink2)' font-size='11.5' " +
       "transform='rotate(-33 " + (X(midSlope) - 6) + " " + (Y(perfAt(midSlope)) - 12) + ")'>" +
       Tr('bwLabel')(perf.bw) + "</text>";

  // one point per kernel × phase (attainable = on/below the roof at its
  // intensity). Decode = circle, prefill = triangle. A point's x AND y are both
  // fixed by its intensity, so kernels with equal intensity land on the SAME
  // spot. Rather than merge them away, we keep every kernel as its own small
  // dot: coincident same-phase dots are fanned out HORIZONTALLY around their
  // true intensity, and each dot connects to its own label by a thin leader.
  var clusters = [];
  pts.forEach(function (p) {
    var iv = Math.max(0.1, Math.min(p.intensity, 10000));
    var px = X(iv), pyv = Y(perfAt(iv));
    var c = null;
    for (var i = 0; i < clusters.length; i++) {
      if (clusters[i].phase === p.phase && Math.abs(clusters[i].px - px) < 18) { c = clusters[i]; break; }
    }
    if (!c) { c = { phase: p.phase, px: px, py: pyv, members: [] }; clusters.push(c); }
    c.members.push(p);
  });
  clusters.sort(function (a, b) { return a.px - b.px; });

  // draw each cluster as ONE marker at its true intensity, its fill divided
  // among member colors (decode circle → pie sectors, prefill triangle →
  // vertical stripes). Labels stack VERTICALLY next to the marker, one row per
  // kernel, each row tinted the member's color; a single leader connects the
  // marker to its label column. Neighbouring clusters alternate above/below.
  var ROWH = 13, GAP = 16, R = 7;
  var labelRects = [];   // placed label-column bounding boxes, for dodging
  clusters.forEach(function (c, ci) {
    c.members.sort(function (a, b) { return b.flops - a.flops; });
    var n = c.members.length, cx = c.px, cy = c.py, isPre = c.phase === 'pre';

    // marker: single color when alone, color-split when merged
    var marker = "";
    if (isPre) {
      var tx0 = cx - 7.5, tx1 = cx + 7.5, tyT = cy - 8, tyB = cy + 6;
      var tri = "M" + cx + "," + tyT + " L" + tx1 + "," + tyB + " L" + tx0 + "," + tyB + " Z";
      if (n === 1) {
        marker = "<path d='" + tri + "' fill='" + c.members[0].color +
                 "' stroke='var(--surface)' stroke-width='1'/>";
      } else {
        var clipId = 'tclip' + ci;
        marker = "<clipPath id='" + clipId + "'><path d='" + tri + "'/></clipPath>";
        var w = (tx1 - tx0) / n;
        c.members.forEach(function (m, i) {
          marker += "<rect x='" + (tx0 + i * w) + "' y='" + tyT + "' width='" + w +
                    "' height='" + (tyB - tyT) + "' fill='" + m.color +
                    "' clip-path='url(#" + clipId + ")'/>";
        });
        marker += "<path d='" + tri + "' fill='none' stroke='var(--surface)' stroke-width='1'/>";
      }
    } else {
      if (n === 1) {
        marker = "<circle cx='" + cx + "' cy='" + cy + "' r='" + R + "' fill='" +
                 c.members[0].color + "' stroke='var(--surface)' stroke-width='1'/>";
      } else {
        // pie: n equal sectors starting at 12 o'clock
        for (var i = 0; i < n; i++) {
          var a0 = -Math.PI / 2 + i / n * 2 * Math.PI;
          var a1 = -Math.PI / 2 + (i + 1) / n * 2 * Math.PI;
          marker += "<path d='M" + cx + "," + cy +
            " L" + (cx + R * Math.cos(a0)) + "," + (cy + R * Math.sin(a0)) +
            " A" + R + " " + R + " 0 0 1 " +
            (cx + R * Math.cos(a1)) + "," + (cy + R * Math.sin(a1)) +
            " Z' fill='" + c.members[i].color + "'/>";
        }
        marker += "<circle cx='" + cx + "' cy='" + cy + "' r='" + R +
                  "' fill='none' stroke='var(--surface)' stroke-width='1'/>";
      }
    }

    // tooltip: phase header + one line per member
    var top = c.members[0];
    var bound = top.intensity < knee ? 'memory-bound' : 'compute-bound';
    var attain = perfAt(top.intensity);
    var tip = Tr('pointTooltip')(isPre ? 'Prefill ▲' : 'Decode ●', fmtNum(top.intensity), bound,
      fmtNum(attain), (attain / perf.peak * 100).toFixed(1));
    c.members.forEach(function (m) {
      tip += "<br>· " + m.label + " — FLOPs " + (m.flops / 1e12).toFixed(3) +
        " T · HBM " + gib(m.bytes) + " GiB · " + fmtNum(m.intensity) + " F/B";
    });

    // vertical label column: alternate up/down between neighbours, clamp into
    // the plot; near either edge switch anchor so text stays inside
    var up = (ci % 2) === 0;
    var stackH = GAP + (n - 1) * ROWH + 6;
    if (up && cy - stackH < mt + 4) up = false;
    if (!up && cy + stackH > by - 4) up = true;
    var anchor = 'middle', lx = cx;
    if (cx > ml + pw - 90) { anchor = 'end'; lx = cx + R + 2; }
    else if (cx < ml + 90) { anchor = 'start'; lx = cx - R - 2; }

    // global label-column collision avoidance: estimate this column's rect and
    // push it further along its direction (row by row) until it clears every
    // previously placed column; flip direction if it would leave the plot
    var colW = 0;
    c.members.forEach(function (m) {
      var w = 0;
      for (var k = 0; k < m.label.length; k++) w += m.label.charCodeAt(k) > 255 ? 10 : 5.5;
      if (w > colW) colW = w;
    });
    var colL = anchor === 'end' ? lx - colW : anchor === 'middle' ? lx - colW / 2 : lx;
    function colRect(dir, shift) {
      var yTop = dir ? cy - GAP - shift - (n - 1) * ROWH - 8 : cy + GAP + shift - 8;
      return { l: colL - 2, r: colL + colW + 2, t: yTop, b: yTop + (n - 1) * ROWH + 16 };
    }
    function collides(rc) {
      if (rc.t < mt || rc.b > by) return true;
      return labelRects.some(function (q) {
        return rc.l < q.r && q.l < rc.r && rc.t < q.b && q.t < rc.b;
      });
    }
    var shift = 0, dir = up, rc = colRect(dir, 0);
    for (var t2 = 1; t2 <= 30 && collides(rc); t2++) {
      // alternate: try the opposite direction at the same shift, then grow
      dir = (t2 % 2) ? !dir : dir;
      if (t2 % 2 === 0) shift += ROWH;
      rc = colRect(dir, shift);
    }
    up = dir;
    labelRects.push(rc);

    s += "<g style='cursor:default' data-tip='" + tip + "'>" + marker;
    // leader: marker edge → nearest end of the label column
    var colNearY = up ? cy - GAP - shift + 9 : cy + GAP + shift - 9;
    s += "<line x1='" + cx + "' y1='" + (up ? cy - R : cy + R) + "' x2='" + lx +
         "' y2='" + colNearY + "' stroke='var(--grid)' stroke-width='1'/>";
    c.members.forEach(function (m, i) {
      // largest-FLOPs row sits closest to the marker
      var labelY = up ? cy - GAP - shift - i * ROWH : cy + GAP + shift + i * ROWH;
      s += "<text x='" + lx + "' y='" + (labelY + 3) + "' text-anchor='" + anchor +
           "' fill='" + m.color + "' font-size='10'>" + m.label + "</text>";
    });
    s += "</g>";
  });

  // legend: marker-shape key for the two phases
  s += "<g transform='translate(" + (ml + 10) + "," + (mt + 12) + ")'>";
  s += "<circle cx='6' cy='0' r='6' fill='var(--ink2)'/>";
  s += "<text x='18' y='4' fill='var(--ink2)' font-size='11.5'>" +
       Tr('legendDecode')(+el('f-req').value) + "</text>";
  s += "<path d='M6,20 L12.5,32 L-0.5,32 Z' fill='var(--ink2)'/>";
  s += "<text x='18' y='31' fill='var(--ink2)' font-size='11.5'>" +
       Tr('legendPrefill')(fmtNum(D.batchTokens)) + "</text>";
  s += "</g>";

  s += "</svg>";
  return s;
}

// ---- per-kernel FLOPs & HBM Access bar charts (one phase). Two grouped bars
// per kernel: FLOPs (TFLOPs) and HBM bytes (GiB), each normalized to the max
// across kernels so the tallest fills the track. Sorted by FLOPs descending.
function kernelBars(rows, phase) {
  var items = rows.map(function (r) {
    var c = phase.pick(r);
    return { label: r.label, color: r.color, flops: c.flops, bytes: c.bytes,
             intensity: c.intensity };
  }).filter(function (it) { return it.flops > 0 || it.bytes > 0; })
    .sort(function (a, b) { return b.flops - a.flops; });
  var maxF = Math.max.apply(null, items.map(function (i) { return i.flops; }).concat([1]));
  var maxB = Math.max.apply(null, items.map(function (i) { return i.bytes; }).concat([1]));
  var knee = null;
  var rowsHtml = items.map(function (it) {
    var fW = Math.max(1.5, it.flops / maxF * 100);
    var bW = Math.max(1.5, it.bytes / maxB * 100);
    return "<div class='kbrow'>" +
      "<div class='kblabel'><i class='dot' style='background:" + it.color + "'></i>" + it.label + "</div>" +
      "<div class='kbbars'>" +
        "<div class='kbtrack'><div class='kbbar' style='width:" + fW + "%;background:" + it.color +
          "'></div><span class='kbval'>" + (it.flops / 1e12).toFixed(3) + " TFLOP</span></div>" +
        "<div class='kbtrack'><div class='kbbar kbbar-hbm' style='width:" + bW + "%;background:" + it.color +
          "'></div><span class='kbval'>" + gib(it.bytes) + " GiB</span></div>" +
      "</div></div>";
  }).join('');
  return "<div class='kbhead'><span></span><span class='kbh-f'>" +
    Tr('kbHeadFlops')(phase.key === 'dec' ? Tr('kbPhaseStep')() : Tr('kbPhaseChunk')()) +
    "</span><span class='kbh-b'>HBM Access</span></div>" + rowsHtml;
}

function compTableHtml(comps, perf, total) {
  var totalTime = 0;
  comps.forEach(function (c) { totalTime += compTime(c, perf); });
  var rows = comps.map(function (c) {
    var t = compTime(c, perf);
    var iv = c.flops / c.bytes;
    var isMem = iv < perf.peak / perf.bw;
    return "<tr><td><i class='dot' style='background:" + c.color + "'></i>" + c.label + "</td>" +
      "<td class='num'>" + (c.flops / 1e12).toFixed(2) + "</td>" +
      "<td class='num'>" + gib(c.bytes) + "</td>" +
      "<td class='num'>" + fmtNum(iv) + "</td>" +
      "<td class='num'>" + (isMem ? Tr('boundByMem')() : Tr('boundByCompute')()) + "</td>" +
      "<td class='num'><b>" + Math.round(t / totalTime * 100) + "%</b></td>" +
      "<td>" + c.note + "</td></tr>";
  }).join('');
  rows += "<tr class='sep'><td><b>" + Tr('compTblTotal')() + "</b></td>" +
    "<td class='num'><b>" + (total.flops / 1e12).toFixed(2) + "</b></td>" +
    "<td class='num'><b>" + gib(total.bytes) + "</b></td>" +
    "<td class='num'><b>" + fmtNum(total.intensity) + "</b></td><td></td>" +
    "<td class='num'>100%</td><td></td></tr>";
  return "<div style='overflow-x:auto'><table style='width:100%'>" +
    "<thead><tr><th>" + Tr('compTblComponent')() + "</th><th class='num'>TFLOPs</th><th class='num'>" + Tr('compTblHbm')() + "</th>" +
    "<th class='num'>" + Tr('compTblIntensity')() + "</th><th class='num'>" + Tr('compTblBoundBy')() +
    "</th><th class='num'>" + Tr('compTblTimeShare')() + "</th><th>" + Tr('compTblNote')() + "</th></tr></thead>" +
    "<tbody>" + rows + "</tbody></table></div>";
}

// map a phase's kernel rows into the {label,color,flops,bytes,note} shape the
// table + roofline points want, dropping zero-work kernels
function phaseComps(rows, phase) {
  return rows.map(function (r) {
    var c = phase.pick(r);
    return { label: r.label, color: r.color, flops: c.flops, bytes: c.bytes, note: c.note };
  }).filter(function (c) { return c.flops > 0 || c.bytes > 0; });
}

function updateRoofline() {
  var perf = currentGpuPerf();
  if (!perf) {
    el('roof-title').textContent = Tr('customInstNoSpec')();
    el('roof-svg').innerHTML = '';
    el('roof-note').textContent = Tr('customInstNoSpecNote')();
    el('roof-verdicts').innerHTML = '';
    return;
  }
  var w = rooflineKernels();
  var knee = perf.peak / perf.bw;

  // one roofline point per kernel × phase (decode ●, prefill ▲)
  var pts = [];
  w.rows.forEach(function (r) {
    PH.forEach(function (ph) {
      var c = ph.pick(r);
      if (c.flops <= 0 && c.bytes <= 0) return;
      pts.push({ label: r.label, color: r.color, phase: ph.key,
                 flops: c.flops, bytes: c.bytes, intensity: c.intensity });
    });
  });

  el('roof-title').textContent = Tr('roofTitle')(perf.gpu, perf.dtype,
    perf.peak.toLocaleString('en-US'), perf.bw, perf.fallback ? Tr('roofFallback')(perf.fallback) : '');
  el('roof-svg').innerHTML = drawRoofline(perf, pts);
  el('roof-note').innerHTML = Tr('roofNote')(fmtNum(knee));

  var tpNow = +el('f-tp').value || 1;
  var vh = "";

  // ---- per-phase cards: kernel bar chart (FLOPs & HBM) + table + prescription
  PH.forEach(function (ph) {
    var comps = phaseComps(w.rows, ph);
    var agg = aggregate(w.rows, ph, ph.label, ph.key === 'dec' ? C.kv : C.others);
    var isMem = agg.intensity < knee;
    var phTime = 0;
    comps.forEach(function (c) { phTime += compTime(c, perf); });
    phTime /= tpNow;

    var badge = "<span class='vbadge' style='background:" +
      (isMem ? 'var(--s6)' : 'var(--s2)') + "'>" +
      (isMem ? 'memory-bound' : 'compute-bound') + " · " +
      (isMem ? knee / agg.intensity : agg.intensity / knee).toFixed(1) + "×</span>";

    var head, foot;
    if (ph.key === 'dec') {
      var stepMs = phTime * 1000;
      var tpsPerReq = 1000 / stepMs;
      head = ph.label + Tr('decodeHeadSuffix')(w.B);
      foot = Tr('decodeFormula')(gib(agg.bytes), perf.bw, tpNow, stepMs.toFixed(1),
        fmtNum(tpsPerReq), fmtNum(tpsPerReq * w.B), w.B) +
        "<div class='cl'>" + Tr('decodeFootPrefix')() +
        (isMem
          ? Tr('decodeFootMem')((knee / agg.intensity).toFixed(1))
          : Tr('decodeFootCompute')()) +
        "</div>";
    } else {
      head = ph.label + Tr('prefillHeadSuffix')(fmtNum(D.batchTokens));
      foot = Tr('prefillFormula')((phTime * 1000).toFixed(0), tpNow, ctxLabel(+el('f-ctx').value),
        (phTime * (+el('f-ctx').value) / D.batchTokens).toFixed(1)) +
        "<div class='cl'>" +
        (isMem ? Tr('prefillFootMem')() : Tr('prefillFootCompute')()) +
        Tr('prefillFootTail')() + "</div>";
    }

    vh += "<div class='card' style='border-left-color:" + agg.color + "'>" +
      "<div class='ch'><i class='dot' style='background:" + agg.color + "'></i>" +
      "<span class='ct'>" + head + "</span>" + badge + "</div>" +
      "<div class='kbchart'>" + kernelBars(w.rows, ph) + "</div>" +
      "<details class='ktbl'><summary>" + Tr('kernelDetailsSummary')() + "</summary>" +
      compTableHtml(comps, perf, agg) + "</details>" + foot + "</div>";
  });

  el('roof-verdicts').innerHTML = vh;
}

/* ========================= shared wiring ========================= */
// hover tooltip: delegated, survives re-render
(function(){
  var tip = el('tip');
  document.addEventListener('mousemove', function(e){
    var t = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (!t){ tip.style.display='none'; return; }
    tip.innerHTML = t.getAttribute('data-tip');
    tip.style.display = 'block';
    var x = e.clientX + 12, y = e.clientY + 14;
    var r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 10;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 10;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  });
})();

function updateAll(){ updateEstimate(); updateParallel(); updateRoofline(); }

function renderKvWarnings(){
  var box = el('kv-warnings');
  if (!box) return;
  var ws = D.kvWarnings || [];
  box.innerHTML = ws.map(function(w){ return "<div class='kvw'>⚠ "+w+"</div>"; }).join('');
}
renderKvWarnings();
// shared filters drive all tabs; parallel/instance filters drive their tabs
['f-ctx','f-req','f-kv'].forEach(function(id){
  el(id).addEventListener('change', updateAll);
});
['f-pp','f-ep'].forEach(function(id){
  el(id).addEventListener('change', updateParallel);
});
el('f-tp').addEventListener('change', function(){ updateParallel(); updateRoofline(); });
el('f-inst').addEventListener('change', function(){ updateParallel(); updateRoofline(); });
el('f-dp').addEventListener('change', updateParallel);
el('f-cmem').addEventListener('input', updateParallel);
el('f-cgpn').addEventListener('input', updateParallel);
el('f-frac').addEventListener('input', function(){ syncFracLabel(); updateParallel(); });
function syncFracLabel(){ setText('f-frac-val', (+el('f-frac').value).toFixed(2)); }
syncFracLabel();
rebuildEpOptions(+el('f-tp').value);
if (D.nMoe){ el('f-ep').value = String(EP_INIT); }

/* ========================= language switcher ========================= */
// Static, Python-rendered fragments (cards, structure diagrams, table rows,
// bar labels, chrome text) are swapped wholesale from D.frags[lang]; dynamic
// numbers (KV/parallel/roofline) are re-derived by updateAll() below, which
// already reads every string through Tr() and so picks up D.lang for free.
var FRAG_IDS = {
  'tab-evidence': 'evidence',
  'h-title': 'title', 'h-subtitle': 'subtitle', 'h-meta': 'meta',
  'h-struct-title': 'struct_title', 'h-struct': 'struct',
  'h-grp-static': 'grp_static', 'h-static-cards': 'static_cards',
  'h-grp-runtime': 'grp_runtime', 'h-runtime-cards': 'runtime_cards',
  'h-weights-bar-head': 'weights_bar_head', 'h-weights-segs': 'weights_segs',
  'h-weights-legend': 'weights_legend',
  'h-details-table-all': 'details_table_all',
  'h-th-component': 'th_component', 'h-th-dtype': 'th_dtype',
  'h-th-params': 'th_params', 'h-th-share': 'th_share',
  'h-table-rows': 'table_rows',
  'h-pstruct-title': 'pstruct_title', 'h-pstruct': 'pstruct',
  'h-parallel-arrow': 'parallel_arrow',
  'h-details-table-gpu': 'details_table_gpu',
  'lbl-ctx': 'lbl_ctx', 'lbl-req': 'lbl_req', 'lbl-kv': 'lbl_kv',
  'lbl-dp': 'lbl_dp', 'lbl-inst': 'lbl_inst', 'lbl-frac': 'lbl_frac',
  'lbl-custom-mem': 'lbl_custom_mem', 'lbl-custom-gpn': 'lbl_custom_gpn',
  'lbl-custom-cards': 'lbl_custom_cards',
  'fnote': 'fnote'
};
// selects whose <option> list itself is a per-language fragment (labels
// differ, e.g. 'auto（bf16）' vs 'auto (bf16)'); value must survive the swap
var FRAG_SELECTS = { 'f-kv': 'kv_options', 'f-inst': 'instance_options' };

function applyLangFragments(lang){
  var fr = D.frags[lang];
  if (!fr) return;
  Object.keys(FRAG_IDS).forEach(function(id){
    var e = el(id), key = FRAG_IDS[id];
    if (e && fr[key] !== undefined) e.innerHTML = fr[key];
  });
  Object.keys(FRAG_SELECTS).forEach(function(id){
    var e = el(id), key = FRAG_SELECTS[id];
    if (!e || fr[key] === undefined) return;
    var prev = e.value;
    e.innerHTML = fr[key];
    if (prev && e.querySelector("option[value='" + prev + "']")) e.value = prev;
  });
  document.title = fr.page_title;
  el('tab-btn-evidence').textContent = fr.tab_evidence;
  el('tab-btn-estimate').textContent = fr.tab_estimate;
  el('tab-btn-parallel').textContent = fr.tab_parallel;
  el('tab-btn-roofline').textContent = fr.tab_roofline;
}

function setLang(lang){
  if (!D.frags[lang] || lang === D.lang) { syncLangButtons(lang); return; }
  D.lang = lang;
  document.documentElement.setAttribute('lang', lang);
  applyLangFragments(lang);
  syncLangButtons(lang);
  try { localStorage.setItem('vram-estimate-lang', lang); } catch (e) {}
  updateAll();
}
function syncLangButtons(lang){
  ['zh','en'].forEach(function(l){
    var b = el('lang-btn-' + l);
    if (b) b.classList.toggle('active', l === lang);
  });
}
el('lang-btn-zh').addEventListener('click', function(){ setLang('zh'); });
el('lang-btn-en').addEventListener('click', function(){ setLang('en'); });
syncLangButtons(D.lang);
if (typeof location !== 'undefined') {
  var hashLang = location.hash === '#en' ? 'en' : location.hash === '#zh' ? 'zh' : null;
  var storedLang = null;
  try { storedLang = localStorage.getItem('vram-estimate-lang'); } catch (e) {}
  var initLang = hashLang || storedLang;
  if (initLang && initLang !== D.lang && D.frags[initLang]) {
    D.lang = initLang;
    document.documentElement.setAttribute('lang', initLang);
    applyLangFragments(initLang);
  }
  syncLangButtons(D.lang);
}

updateAll();
