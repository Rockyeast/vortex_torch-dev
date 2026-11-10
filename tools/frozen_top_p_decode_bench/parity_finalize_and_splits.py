"""vxr: (1) centroid update from token locations vs the mask form: same centroids; (2) attention split count at small batch."""
import math, os
import torch
from vortex_torch.engine.sgl import frozen_dynamic_top_p as F
from vortex_torch.frozen_dynamic_top_p.centroid_scoring_triton import centroid_per_q_logmass
from vortex_torch.frozen_dynamic_top_p.isolated_paged_attention_triton import paged_gqa_decode_attention_with_tail_splitk_tc as attn_tc
from vortex_torch.frozen_dynamic_top_p.top_p_selection_triton import normalized_gqa_select_bucketed
HQ, HKV, D, T, MAXB = 12, 2, 128, 1.1, 2048
dev = torch.device("cuda"); torch.manual_seed(5)

# (1) finalize parity through the engine function, both forms
slots = 4096
for n in (1, 64, 5000):
    k = torch.randn(slots, HKV, 16, D, device=dev, dtype=torch.bfloat16)
    loc = torch.randperm(slots * 16, device=dev)[:n].to(torch.int64).contiguous()
    res = []
    for fold in (False, True):
        F.FOLD_FINALIZE = fold
        cache = {"k": k.view(slots * 16 * HKV // (16 * HKV), HKV, 16, D), "centroids": torch.zeros(slots, HKV, 4, D, device=dev, dtype=torch.bfloat16),
                 "centroid_counts": torch.zeros(slots, HKV, 4, device=dev, dtype=torch.uint8)}
        F.finalize_newly_completed_blocks(cache, loc, num_kv_heads=HKV, head_dim=D)
        torch.cuda.synchronize(); res.append((cache["centroids"].clone(), cache["centroid_counts"].clone()))
    done = int((loc % 16 == 15).sum())
    print(f"finalize n={n}: blocks completed {done}, centroids equal {torch.equal(res[0][0], res[1][0])}, counts equal {torch.equal(res[0][1], res[1][1])}, nonzero rows {int((res[1][1].sum((1, 2)) > 0).sum())}", flush=True)
F.FOLD_FINALIZE = True


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


# (2) attention: splits x tile at small batch
for blocks, target in ((256, 92), (1024, 194)):
    for batch in (1, 4, 16, 64):
        slots = batch * blocks + 1
        k = torch.randn(slots, HKV, 16, D, device=dev, dtype=torch.bfloat16); v = torch.randn_like(k)
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
        tail_p = torch.full((batch,), slots - 1, device=dev, dtype=torch.int32); tail_c = torch.full((batch,), 7, device=dev, dtype=torch.int32)
        P, S = ws.per_q_scores[:batch], ws.shared_scores[:batch]
        L, Y, C = ws.logical_indices[:batch], ws.physical_indices[:batch], ws.selected_counts[:batch]
        M, E = ws.merged_output[:batch], ws.selected_lse[:batch]
        centroid_per_q_logmass(qsel, cen, cnt, mapping, done, per_q_out=P)
        normalized_gqa_select_bucketed(P, mapping, done, num_kv_heads=HKV, temperature=T, top_p=0.9, cap_fraction=0.75, score_out=S, logical_out=L, physical_out=Y, count_out=C)
        torch.cuda.synchronize()
        line = f"ctx {blocks*16//1024:>2}K batch {batch:>2} sel {C.float().mean().item():5.0f} |"; ref = None
        for ns in (8, 16, 32):
            po = torch.empty(batch, HQ, ns, D, device=dev); pl = torch.empty(batch, HQ, ns, device=dev)
            for tb in (2, 4, 8):
                def a():
                    attn_tc(q, k, v, Y, C, tail_p, tail_c, partial_output=po, partial_lse=pl, output=M, lse=E, num_splits=ns, tile_blocks=tb)
                a(); torch.cuda.synchronize()
                if ref is None:
                    ref = M.float().clone()
                err = ((M.float() - ref).norm() / ref.norm()).item()
                line += f" s{ns} b{tb} {timed(a):5.1f} ({err:.0e}) |"
        print(line, flush=True)
        del k, v, cen, ws; torch.cuda.empty_cache()
