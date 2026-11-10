"""vxr: scoring with empty-tile skip / wider tiles, and tensor-core attention, against the vxq behaviour. Parity and time."""
import math
import os
import torch

from vortex_torch.engine.sgl import frozen_dynamic_top_p as F
from vortex_torch.frozen_dynamic_top_p.centroid_scoring_triton import centroid_per_q_logmass
from vortex_torch.frozen_dynamic_top_p.isolated_paged_attention_triton import (
    paged_gqa_decode_attention_with_tail_splitk as attn_q, paged_gqa_decode_attention_with_tail_splitk_tc as attn_tc)
from vortex_torch.frozen_dynamic_top_p.top_p_selection_triton import normalized_gqa_select_bucketed

HQ, HKV, D, T, MAXB = 12, 2, 128, 1.1, 2048
dev = torch.device("cuda")
torch.manual_seed(1)
PART = os.environ.get("PART", "score,attn").split(",")


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
    return a.elapsed_time(b) / 20 / reps * 1000


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
        qsel = (torch.randn(batch, HQ, D, device=dev) * 0.02).bfloat16(); qsel[:, :, 0] = 1
        q = (torch.randn(batch, HQ, D, device=dev) * 0.5).bfloat16()
        ws = F.allocate_decode_workspace(max_batch=batch, max_blocks=MAXB, q_heads=HQ, kv_heads=HKV, head_dim=D, device=dev, dtype=torch.bfloat16)
        mapping = torch.zeros(batch, MAXB, device=dev, dtype=torch.int32)
        mapping[:, :blocks] = torch.arange(batch * blocks, device=dev, dtype=torch.int32).view(batch, blocks)
        done = torch.full((batch,), blocks, device=dev, dtype=torch.int32)
        tail_p = torch.full((batch,), slots - 1, device=dev, dtype=torch.int32)
        tail_c = torch.full((batch,), 7, device=dev, dtype=torch.int32)
        P, S = ws.per_q_scores[:batch], ws.shared_scores[:batch]
        L, Y, C = ws.logical_indices[:batch], ws.physical_indices[:batch], ws.selected_counts[:batch]
        M, E = ws.merged_output[:batch], ws.selected_lse[:batch]
        head = f"ctx {blocks*16//1024:>2}K batch {batch:>2}"

        def score():
            centroid_per_q_logmass(qsel, cen, cnt, mapping, done, per_q_out=P)

        if "score" in PART:
            os.environ.update(VORTEX_SCORE_SKIP_EMPTY="0", VORTEX_SCORE_BLOCK_N="16", VORTEX_SCORE_WARPS="4")
            P.fill_(7.0); score(); torch.cuda.synchronize(); ref = P[:, :, :blocks].clone(); t0 = timed(score)
            line = f"{head} | score before {t0:6.1f} us |"
            for bn, nw in ((16, 4), (32, 4), (64, 4), (64, 8)):
                os.environ.update(VORTEX_SCORE_SKIP_EMPTY="1", VORTEX_SCORE_BLOCK_N=str(bn), VORTEX_SCORE_WARPS=str(nw))
                P.fill_(7.0); score(); torch.cuda.synchronize()
                same = torch.equal(P[:, :, :blocks], ref); diff = (P[:, :, :blocks] - ref).abs().max().item()
                rest = bool(torch.isinf(P[:, :, blocks:]).all())
                line += f" skip n{bn} w{nw} {timed(score):6.1f} (equal {same} maxdiff {diff:.1e} rest-inf {rest}) |"
            print(line, flush=True)
        os.environ.update(VORTEX_SCORE_SKIP_EMPTY="1", VORTEX_SCORE_BLOCK_N="16", VORTEX_SCORE_WARPS="4")
        score()
        normalized_gqa_select_bucketed(P, mapping, done, num_kv_heads=HKV, temperature=T, top_p=0.9, cap_fraction=0.75, score_out=S, logical_out=L, physical_out=Y, count_out=C)
        torch.cuda.synchronize()
        if "attn" in PART:
            kw = dict(partial_output=ws.split_partial_output[:batch], partial_lse=ws.split_partial_lse[:batch], output=M, lse=E, num_splits=F.BLACKWELL_ATTENTION_SPLITS)

            def a_q():
                attn_q(q, k, v, Y, C, tail_p, tail_c, **kw)

            a_q(); torch.cuda.synchronize()
            ref, ref_lse = M.float().clone(), E.clone()
            line = f"{head} sel {C.float().mean().item():6.1f} | attn per-Q {timed(a_q):6.1f} us |"
            for tb, nw in ((1, 4), (2, 4), (4, 4), (4, 8), (8, 4), (8, 8)):
                def a_t():
                    attn_tc(q, k, v, Y, C, tail_p, tail_c, tile_blocks=tb, num_warps=nw, **kw)
                a_t(); torch.cuda.synchronize()
                err = ((M.float() - ref).norm() / ref.norm()).item()
                line += f" tc b{tb} w{nw} {timed(a_t):6.1f} (err {err:.1e}, lse {(E - ref_lse).abs().max().item():.1e}) |"
            print(line, flush=True)
        del k, v, cen, ws
        torch.cuda.empty_cache()
