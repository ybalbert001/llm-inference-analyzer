#!/usr/bin/env python3
"""Replicate the tool's roofline-tab predictions (decode step / prefill TTFT).

Mirrors template.js exactly: per-kernel time = max(bytes/BW, FLOPs/peak),
GEMM FLOPs = 2*params*tokens with weight bytes streamed once, MoE decode
touched-ratio, attention-core geometry (MHA/GQA/MLA) × pattern (dense/DSA),
chunked prefill = per-chunk sum with causal pairs accumulated over the prefix
(done*c + c*(c+1)/2), DSA prefill pairs capped at topk per query.

TP model: ideal aggregation — a TP-N deployment is one virtual GPU with N×BW
and N×peak (weights/FLOPs shard evenly; comm overhead is part of the common
efficiency factor the ratio method cancels).

Usage: predict_roofline.py <report.html> <tp> <decode_ctx> "<prefill_lens>"
"""
import json, re, sys

def viz(html_path):
    text = open(html_path, encoding="utf-8").read()
    m = re.search(r'var\s+D\s*=\s*(\{.*?\});\s*\n', text, re.S)
    return json.loads(m.group(1))

def kv_bytes_per(dt):
    return 1 if dt == "fp8" else 0.5625 if dt == "fp4" else 2

def geometry(spec):
    if spec["kind"] == "mla":
        return (2 * spec["qHeads"] * (2 * spec["kvLoraRank"] + spec["ropeHeadDim"]),
                spec["kvLoraRank"] + spec["ropeHeadDim"])
    return (2 * spec["qHeads"] * (spec["qkHeadDim"] + spec["valueHeadDim"]),
            spec["kvHeads"] * (spec["qkHeadDim"] + spec["valueHeadDim"]))

def main():
    D = viz(sys.argv[1])
    tp = int(sys.argv[2])
    decode_ctx = int(sys.argv[3])
    prefill_lens = [int(x) for x in sys.argv[4].split()]

    perf = D["gpuPerf"]["B200"]
    dt = D["weightDtype"]
    peak = (perf.get(dt) or perf["bf16"]) * 1e12 * tp
    bw = perf["bw"] * 1e12 * tp
    # attention-core pairs/KV reads are charged to KV-bearing layers only;
    # hybrid models' linear/SSM layers pay a fixed O(1) state read+write instead
    L = D.get("nKvLayers") or D["L"]
    lin_state = D.get("linStateBytes") or 0
    kv_b = kv_bytes_per(D["kvAuto"] if D["kvChoice"] == "auto" else D["kvChoice"])
    flops_pair, kv_elems = geometry(D["attentionCore"]["geometry"])
    pat = D["attentionCore"]["pattern"]
    topk_experts, n_experts = D["topk"], D["nExperts"]

    def t(flops, bytes_):
        return max(bytes_ / bw, flops / peak)

    def kernel_times(tokens, is_decode, B=1):
        total = 0.0
        for k in D["kernels"]:
            if k["key"] == "moe_routed" and n_experts:
                ar = topk_experts / n_experts
                if is_decode:
                    touched = min(1.0, B * ar)
                    total += t(2 * k["params"] * ar * B, k["bytes"] * touched)
                else:
                    total += t(2 * k["params"] * ar * tokens, k["bytes"])
            else:
                total += t(2 * k["params"] * tokens, k["bytes"])
        return total

    # ---- decode step (TPOT), B=1, context S -------------------------------
    S = decode_ctx
    if pat["kind"] == "dsa":
        att = min(S, pat["topk"])
        cap_l = pat.get("capLayers", L)
        dec_pairs = att * cap_l + S * (L - cap_l)
    else:
        dec_pairs = S * L
    dec_core = t(flops_pair * dec_pairs, kv_elems * dec_pairs * kv_b)
    # hybrid linear/SSM state: read+written once per request per step
    dec_lin = t(lin_state, 2 * lin_state) if lin_state else 0.0
    tpot = kernel_times(1, True) + dec_core + dec_lin

    # ---- prefill TTFT: chunked, per-chunk sum -----------------------------
    chunk = D["batchTokens"]
    def dsa_pairs(done, c, cap):
        # sum over absolute positions p in [done, done+c): min(p+1, cap)
        lo, hi = done, done + c
        if hi <= cap:
            return (lo + 1 + hi) * c // 2
        if lo >= cap:
            return cap * c
        head = cap - lo               # positions lo..cap-1 attend p+1
        return (lo + 1 + cap) * head // 2 + (c - head) * cap

    results = []
    for T in prefill_lens:
        done, ttft = 0, 0.0
        while done < T:
            c = min(chunk, T - done)
            if pat["kind"] == "dsa":
                cap_l = pat.get("capLayers", L)
                pairs = dsa_pairs(done, c, pat["topk"]) * cap_l \
                      + (done * c + c * (c + 1) // 2) * (L - cap_l)
            else:
                pairs = (done * c + c * (c + 1) // 2) * L
            core = t(flops_pair * pairs, kv_elems * (done + c) * L * kv_b)
            lin = t(lin_state * c, 2 * lin_state) if lin_state else 0.0
            ttft += kernel_times(c, False) + core + lin
            done += c
        results.append({"prompt": T, "ttft_s": round(ttft, 4),
                        "ratio_vs_first": round(ttft, 6)})
    base = results[0]["ttft_s"]
    for r in results:
        r["ratio_vs_first"] = round(r["ttft_s"] / base, 3)

    print(json.dumps({
        "gpu": "B200", "tp": tp, "weight_dtype": dt,
        "peak_tflops_aggregate": peak / 1e12, "bw_tbs_aggregate": bw / 1e12,
        "kv_dtype": D["kvAuto"] if D["kvChoice"] == "auto" else D["kvChoice"],
        "pattern": pat, "decode_ctx": S,
        "pred_tpot_ms": round(tpot * 1000, 3),
        "prefill": results,
    }, indent=1))

if __name__ == "__main__":
    main()
