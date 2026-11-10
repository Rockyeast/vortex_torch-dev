"""Selection kernel time by real block count and warps (vxr), CUDA graphs. Checks the selected sets do not change."""
import math, os
import torch
from vortex_torch.engine.sgl import frozen_dynamic_top_p as F
from vortex_torch.frozen_dynamic_top_p.centroid_scoring_triton import centroid_per_q_logmass
from vortex_torch.frozen_dynamic_top_p.top_p_selection_triton import normalized_gqa_select_bucketed
HQ, HKV, D, T, MAXB = 12, 2, 128, 1.1, 2048
dev = torch.device("cuda"); torch.manual_seed(1)


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


for blocks in (256, 512, 1000, 1040, 1536, 2000):
    for batch in (1, 64):
        target = max(8, int(blocks * 0.19))
        slots = batch * blocks + 1
        cen = (torch.randn(slots, HKV, 4, D, device=dev) * 0.3).bfloat16()
        alpha = -math.log(0.1) * T / target
        rank = torch.arange(blocks, device=dev, dtype=torch.float32)
        for b_ in range(batch):
            cen[b_ * blocks:(b_ + 1) * blocks, :, :, 0] += (-alpha * rank[torch.randperm(blocks, device=dev)] * math.sqrt(D))[:, None, None].bfloat16()
        cnt = torch.full((slots, HKV, 4), 4, device=dev, dtype=torch.uint8)
        q = (torch.randn(batch, HQ, D, device=dev) * 0.02).bfloat16(); q[:, :, 0] = 1
        ws = F.allocate_decode_workspace(max_batch=batch, max_blocks=MAXB, q_heads=HQ, kv_heads=HKV, head_dim=D, device=dev, dtype=torch.bfloat16)
        mapping = torch.zeros(batch, MAXB, device=dev, dtype=torch.int32)
        mapping[:, :blocks] = torch.arange(batch * blocks, device=dev, dtype=torch.int32).view(batch, blocks)
        done = torch.full((batch,), blocks, device=dev, dtype=torch.int32)
        P, S = ws.per_q_scores[:batch], ws.shared_scores[:batch]
        L, Y, C = ws.logical_indices[:batch], ws.physical_indices[:batch], ws.selected_counts[:batch]
        centroid_per_q_logmass(q, cen, cnt, mapping, done, per_q_out=P)

        def sel():
            normalized_gqa_select_bucketed(P, mapping, done, num_kv_heads=HKV, temperature=T, top_p=0.9, cap_fraction=0.75, score_out=S, logical_out=L, physical_out=Y, count_out=C)

        line = f"blocks {blocks:>4} batch {batch:>2} |"; ref = None
        for w in (4, 8, 16, 32):
            os.environ["VORTEX_SELECT_WARPS"] = str(w)
            try:
                sel(); torch.cuda.synchronize()
                cur = (L.clone(), Y.clone(), C.clone())
                same = True if ref is None else all(torch.equal(a, b) for a, b in zip(ref, cur))
                ref = ref or cur
                line += f" w{w} {timed(sel):6.1f} us (sel {C.float().mean().item():.0f}, same {same}) |"
            except Exception as e:
                line += f" w{w} failed {type(e).__name__} |"
        print(line, flush=True)
        del cen, ws; torch.cuda.empty_cache()
