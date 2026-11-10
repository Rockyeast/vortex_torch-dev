"""Both attention kernels against an FP64 reference of the same definition: which one is closer to the exact answer?"""
import math
import torch

from vortex_torch.engine.sgl import frozen_dynamic_top_p as F
from vortex_torch.frozen_dynamic_top_p.centroid_scoring_triton import centroid_per_q_logmass
from vortex_torch.frozen_dynamic_top_p.isolated_paged_attention_triton import (
    paged_gqa_decode_attention_with_tail_splitk as attn_q, paged_gqa_decode_attention_with_tail_splitk_tc as attn_tc)
from vortex_torch.frozen_dynamic_top_p.top_p_selection_triton import normalized_gqa_select_bucketed

HQ, HKV, D, T, MAXB = 12, 2, 128, 1.1, 2048
dev = torch.device("cuda"); torch.manual_seed(3)
SCALE = 1.0 / math.sqrt(D)
for blocks, target, tailn, qs in ((12, 6, 3, 0.5), (64, 30, 16, 0.5), (512, 155, 7, 0.5), (1024, 194, 11, 2.0)):
    batch = 8
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
    q = (torch.randn(batch, HQ, D, device=dev) * qs).bfloat16()
    ws = F.allocate_decode_workspace(max_batch=batch, max_blocks=MAXB, q_heads=HQ, kv_heads=HKV, head_dim=D, device=dev, dtype=torch.bfloat16)
    mapping = torch.zeros(batch, MAXB, device=dev, dtype=torch.int32)
    mapping[:, :blocks] = torch.arange(batch * blocks, device=dev, dtype=torch.int32).view(batch, blocks)
    done = torch.full((batch,), blocks, device=dev, dtype=torch.int32)
    tail_p = torch.full((batch,), slots - 1, device=dev, dtype=torch.int32)
    tail_c = torch.full((batch,), tailn, device=dev, dtype=torch.int32)
    P, S = ws.per_q_scores[:batch], ws.shared_scores[:batch]
    L, Y, C = ws.logical_indices[:batch], ws.physical_indices[:batch], ws.selected_counts[:batch]
    M, E = ws.merged_output[:batch], ws.selected_lse[:batch]
    centroid_per_q_logmass(qsel, cen, cnt, mapping, done, per_q_out=P)
    normalized_gqa_select_bucketed(P, mapping, done, num_kv_heads=HKV, temperature=T, top_p=0.9, cap_fraction=0.75, score_out=S, logical_out=L, physical_out=Y, count_out=C)
    torch.cuda.synchronize()
    # FP64 reference, same definition: q*SCALE rounded to BF16 for selected blocks, unscaled q then *SCALE for the tail
    exact = torch.zeros(batch, HQ, D, dtype=torch.float64, device=dev)
    q_scaled = (q.float() * SCALE).bfloat16().double()
    for b_ in range(batch):
        for h in range(HQ):
            kv = h // (HQ // HKV)
            sl = Y[b_, kv, :int(C[b_, kv])].long()
            kk = k[sl, kv].double().reshape(-1, D); vv = v[sl, kv].double().reshape(-1, D)
            lg = [kk @ q_scaled[b_, h]]
            kt = k[slots - 1, kv, :tailn].double(); vt = v[slots - 1, kv, :tailn].double()
            lg.append((kt @ q[b_, h].double()) * SCALE)
            p = torch.softmax(torch.cat(lg), 0)
            exact[b_, h] = p @ torch.cat([vv, vt])
    exact_bf16 = exact.float().bfloat16().double()   # the best any BF16 output can do
    kw = dict(partial_output=ws.split_partial_output[:batch], partial_lse=ws.split_partial_lse[:batch], output=M, lse=E, num_splits=F.BLACKWELL_ATTENTION_SPLITS)
    res = {}
    attn_q(q, k, v, Y, C, tail_p, tail_c, **kw); torch.cuda.synchronize(); res["per-Q"] = M.double().clone()
    for tb in (2, 8):
        attn_tc(q, k, v, Y, C, tail_p, tail_c, tile_blocks=tb, **kw); torch.cuda.synchronize(); res[f"tc b{tb}"] = M.double().clone()
    n = exact.norm()
    line = f"blocks {blocks:>4} sel {C.float().mean().item():6.1f} tail {tailn:>2} | rounding floor {((exact_bf16 - exact).norm() / n).item():.2e} |"
    for name, o in res.items():
        line += f" {name}: err vs exact {((o - exact).norm() / n).item():.2e}, elements off the best BF16 value {(o != exact_bf16).double().mean().item():.2%} |"
    print(line, flush=True)
