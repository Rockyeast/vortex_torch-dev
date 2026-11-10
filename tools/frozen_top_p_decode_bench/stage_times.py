"""Per-layer decode stages of the fast Top-P path (vxq) at the student's shape, timed in CUDA graphs.

Batches 1/16/64 at 4K/8K/16K context. The selected count is set to what Top-P 0.9 picks on AIME texts
(about 92/155/194 blocks). Also times the attention kernel reading every block, as a same-kernel dense reference.
"""
import math
import torch

from vortex_torch.engine.sgl import frozen_dynamic_top_p as F
from vortex_torch.frozen_dynamic_top_p.centroid_scoring_triton import centroid_per_q_logmass
from vortex_torch.frozen_dynamic_top_p.isolated_paged_attention_triton import paged_gqa_decode_attention_with_tail_splitk
from vortex_torch.frozen_dynamic_top_p.top_p_selection_triton import normalized_gqa_select_bucketed

HQ, HKV, D, T, MAXB = 12, 2, 128, 1.1, 2048
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), "FAST_MIN_MAJOR", F.FAST_MIN_MAJOR, flush=True)
torch.manual_seed(1)


def timed(fn, reps=10):
    fn(); fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); a.record()
    for _ in range(20):
        g.replay()
    b.record(); b.synchronize()
    return a.elapsed_time(b) / 20 / reps * 1000  # microseconds


for blocks, target in ((256, 92), (512, 155), (1024, 194)):
    for batch in (1, 16, 64):
        slots = batch * blocks + 1
        k = torch.randn(slots, HKV, 16, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cen = (torch.randn(slots, HKV, 4, D, device=dev) * 0.3).bfloat16()
        alpha = -math.log(0.1) * T / target
        rank = torch.arange(blocks, device=dev, dtype=torch.float32)
        for b_ in range(batch):
            cen[b_ * blocks:(b_ + 1) * blocks, :, :, 0] += (-alpha * rank[torch.randperm(blocks, device=dev)] * math.sqrt(D))[:, None, None].bfloat16()
        cnt = torch.full((slots, HKV, 4), 4, device=dev, dtype=torch.uint8)
        q = (torch.randn(batch, HQ, D, device=dev) * 0.02).bfloat16()
        q[:, :, 0] = 1
        ws = F.allocate_decode_workspace(max_batch=batch, max_blocks=MAXB, q_heads=HQ, kv_heads=HKV, head_dim=D, device=dev, dtype=torch.bfloat16)
        mapping = torch.zeros(batch, MAXB, device=dev, dtype=torch.int32)
        mapping[:, :blocks] = torch.arange(batch * blocks, device=dev, dtype=torch.int32).view(batch, blocks)
        done = torch.full((batch,), blocks, device=dev, dtype=torch.int32)
        tail_p = torch.full((batch,), slots - 1, device=dev, dtype=torch.int32)
        tail_c = torch.full((batch,), 7, device=dev, dtype=torch.int32)
        P, S = ws.per_q_scores[:batch], ws.shared_scores[:batch]
        L, Y, C = ws.logical_indices[:batch], ws.physical_indices[:batch], ws.selected_counts[:batch]
        M, E = ws.merged_output[:batch], ws.selected_lse[:batch]

        def score():
            centroid_per_q_logmass(q, cen, cnt, mapping, done, per_q_out=P)

        def select():
            normalized_gqa_select_bucketed(P, mapping, done, num_kv_heads=HKV, temperature=T, top_p=0.9, cap_fraction=0.75, score_out=S, logical_out=L, physical_out=Y, count_out=C)

        def attn():
            paged_gqa_decode_attention_with_tail_splitk(q, k, v, Y, C, tail_p, tail_c, partial_output=ws.split_partial_output[:batch], partial_lse=ws.split_partial_lse[:batch],
                                                        output=M, lse=E, num_splits=F.BLACKWELL_ATTENTION_SPLITS)

        score(); select(); attn(); torch.cuda.synchronize()
        sel = C.float().mean().item()
        t = {n: timed(f) for n, f in (("score", score), ("select", select), ("attn", attn))}
        # same attention kernel over every block: what dense would cost in this kernel
        cap = Y.shape[-1]
        full = min(blocks, cap)
        Yd = Y.clone(); Yd[:, :, :full] = mapping[:, None, :full]; Cd = torch.full_like(C, full)

        def attn_all():
            paged_gqa_decode_attention_with_tail_splitk(q, k, v, Yd, Cd, tail_p, tail_c, partial_output=ws.split_partial_output[:batch], partial_lse=ws.split_partial_lse[:batch],
                                                        output=M, lse=E, num_splits=F.BLACKWELL_ATTENTION_SPLITS)
        t["attn_all"] = timed(attn_all)
        tot = t["score"] + t["select"] + t["attn"]
        print(f"ctx {blocks*16//1024:>2}K batch {batch:>2} sel {sel:6.1f} | score {t['score']:7.1f} select {t['select']:7.1f} attn {t['attn']:7.1f} = {tot:7.1f} us/layer ({tot*28/1000:5.2f} ms/step)"
              f" | attn over all {full} blocks {t['attn_all']:7.1f} us ({t['attn_all']*28/1000:5.2f} ms/step)", flush=True)
        del k, v, cen, ws
        torch.cuda.empty_cache()
