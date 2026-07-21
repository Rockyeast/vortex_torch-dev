"""JIT loader + Python wrapper for the hand-written CUDA block-table MLA decode.

Mirrors the closure/cache pattern of ``vortex_torch/indexer/utils_sglang.py``:
``get_cuda_mla_module()`` JIT-compiles ``csrc/mla_cuda.cu`` once (process-cached)
and returns the extension module; ``decode_blocktable_mla_cuda`` is a drop-in for
``triton_mla_kernel.decode_blocktable_mla`` (identical signature) so the backend
swap is a one-line change.

The kernel (``csrc/mla_ldm.cuh``) is a from-scratch CUDA flash-decode: ldmatrix +
``mma.sync.m16n8k16`` GEMMs, bf16-packed register-O, register-resident online
softmax, a 576->584 smem bank-conflict pad, ``__launch_bounds__`` occupancy
forcing, and split-KV. See ``cuda_mla/REPORT.md`` for the design + benchmarks.
"""
# 中文读法：CUDA MLA kernel 封装层。这里负责加载 C++/CUDA 扩展、计算 launch geometry、分配临时 buffer、调用 decode kernel。
import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
_MODULE = None
_SM_COUNT: dict = {}
# Optional override of the split-KV count (>=1). In a full model the MLA decode is
# a small fraction of the forward, so the split path's extra stage-2 + MidO traffic
# can cost more than its parallelism buys; setting this to 1 forces single-pass.
_SPLITS_ENV = os.environ.get("VORTEX_CUDA_MLA_SPLITS")


# 加载 CUDA 扩展：第一次调用时编译/加载手写 MLA decode kernel，后续复用缓存模块。
def get_cuda_mla_module():
    """JIT-compile (once, process-cached) and return the CUDA MLA extension."""
    global _MODULE
    if _MODULE is None:
        build_dir = os.path.join(_CSRC, "build")
        os.makedirs(build_dir, exist_ok=True)
        _MODULE = load(
            name="vortex_cuda_mla",
            sources=[os.path.join(_CSRC, "mla_cuda.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math",
                               "--expt-relaxed-constexpr", "-lineinfo"],
            extra_include_paths=[_CSRC],
            build_directory=build_dir,
            verbose=False,
        )
    return _MODULE


def _sm_count(device) -> int:
    idx = device.index if hasattr(device, "index") else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT[idx]


# kv_lora_rank — the latent width the kernel reduces over (ldm::CKV). Constant for
# MLA; mid_o is [target_ctas, M, CKV]. The decode path asserts Lk==576, Lv==512.
CKV = 512


# launch 几何计算：根据 batch、head、block 数决定 CUDA grid/block 和 split 策略。
def mla_decoder_geometry(bs: int, H: int, block_size: int, max_blocks: int,
                         max_split_cap: int = -1, chunk_min: int = -1, minb: int = -1):
    """Host-only launch geometry for a decode batch size — NO allocation. Returns a
    dict with ``target_ctas`` (run_wq grid / work-queue length) and ``M`` (=ceil(H/16)*16,
    the mid-buffer head-pad). Used to size the shared scratch at the max decode bs."""
    tc, M, minb_, target, cap, cmin = get_cuda_mla_module().mla_decoder_geometry(
        bs, H, block_size, max_blocks, max_split_cap, chunk_min, minb)
    return {"target_ctas": tc, "M": M, "minb": minb_,
            "target": target, "max_split_cap": cap, "chunk_min": cmin}


# 临时 buffer 分配：decode kernel 需要的中间 logits/归约空间在这里一次性准备。
def allocate_mla_buffers(max_bs: int, H: int, block_size: int, max_blocks: int,
                         device, max_split_cap: int = -1, chunk_min: int = -1,
                         minb: int = -1) -> dict:
    """Allocate the work-queue + split-reduction scratch ONCE, sized for the maximum
    decode batch size. ``target_ctas`` grows monotonically with bs, so these buffers
    cover every smaller bs; all per-bs ``MLADecoder``s slice into them. This replaces
    the old one-scratch-allocation-per-cuda-graph-bs behaviour, which duplicated the
    (largest) ``target_ctas*M*512`` fp32 ``mid_o`` for every captured batch size."""
    g = mla_decoder_geometry(max_bs, H, block_size, max_blocks,
                             max_split_cap, chunk_min, minb)
    tc, M = g["target_ctas"], g["M"]
    i32 = {"dtype": torch.int32, "device": device}
    f32 = {"dtype": torch.float32, "device": device}
    # Uninitialised (torch.empty) is correct: the contract is plan -> run. Every decode
    # step, plan() (schedule_wq) repopulates the whole work queue (work_*), and run()'s
    # stage1 writes the mid_* scratch for exactly the active work items
    # [0, sum(nsplits)) that stage2_wq then reduces. The schedule keeps
    # sum(nsplits) <= target_ctas (the +bs margin), and every slot a reduction reads,
    # [WorkOffset[b], WorkOffset[b+1]), is an active item stage1 wrote — so nothing here
    # is ever read before being written, even with one scratch shared across batch
    # sizes / steps. (Verified: full-scratch poison does not leak into the output.)
    return {
        "work_batch": torch.empty(tc, **i32),
        "work_kv_start": torch.empty(tc, **i32),
        "work_kv_end": torch.empty(tc, **i32),
        "work_offset": torch.empty(max_bs + 1, **i32),
        "mid_o": torch.empty((tc, M, CKV), **f32),
        "mid_m": torch.empty((tc, M), **f32),
        "mid_l": torch.empty((tc, M), **f32),
    }


# decoder 闭包：把固定参数和 buffer 绑定起来，返回后端每次 decode 可直接调用的函数。
def make_mla_decoder(bs: int, H: int, block_size: int, max_blocks: int, buffers: dict,
                     max_split_cap: int = -1, chunk_min: int = -1, minb: int = -1):
    """flashinfer-style plan/run decoder. The launch geometry is fixed per (bs), but
    the work-queue / split scratch is NOT allocated here — pass ``buffers`` from
    ``allocate_mla_buffers(max_bs, ...)`` (sized for the max decode bs) and the decoder
    slices them to this bs's ``target_ctas``. ``plan(seqlens)`` builds the load-balanced
    work queue (once per decode step, shared by all layers); ``run(q, latent,
    block_table, o, sm_scale)`` does the per-layer decode. Both are fixed-grid =>
    cuda-graph-capturable."""
    return get_cuda_mla_module().MLADecoder(
        bs, H, block_size, max_blocks,
        buffers["work_batch"], buffers["work_kv_start"], buffers["work_kv_end"],
        buffers["work_offset"], buffers["mid_o"], buffers["mid_m"], buffers["mid_l"],
        max_split_cap, chunk_min, minb)


# CUDA decode 调用入口：把 block_table/latent cache/query 交给扩展 kernel，输出当前 token attention。
def decode_blocktable_mla_cuda(
    q: torch.Tensor,            # [bs, H, Lk]   fused [q_nope_out | q_pe], Lk=576
    latent: torch.Tensor,       # [num_slots, Lk]  fused [kv_c | k_pe]
    block_table: torch.Tensor,  # [bs, max_blocks] selected page ids (int32)
    seqlens: torch.Tensor,      # [bs] sparse token count (int32)
    sm_scale: float,
    block_size: int,
    kv_lora_rank: int,          # Lv = 512
    o: torch.Tensor = None,     # [bs, H, Lv] (allocated if None)
    splits: int = None,
) -> torch.Tensor:
    """Drop-in replacement for ``decode_blocktable_mla`` backed by the CUDA kernel."""
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    assert Lk == 576 and Lv == 512, f"MLA latent must be 512+64=576 (got Lk={Lk}, Lv={Lv})"
    if o is None:
        o = q.new_empty((bs, H, Lv))
    if splits is None:
        if _SPLITS_ENV is not None:
            splits = max(1, int(_SPLITS_ENV))
        else:
            # split-KV count. In a FULL model the MLA decode is a small fraction of
            # the forward, so the split path's extra stage-2 + MidO traffic (which
            # scales with `splits`) easily outweighs its parallelism — measured on
            # DeepSeek-V2-Lite RULER: splits=2 matched Triton (378 vs 376 tok/s),
            # while the standalone-microbench-optimal ~7 splits was ~30% slower.
            # So target a MODEST fill (~1 CTA/SM, total CTAs ~ bs*splits ~ SM),
            # capped at 4. Pure host arithmetic (bs + SM count) => fixed launch
            # geometry => cuda-graph-capture safe. Override with VORTEX_CUDA_MLA_SPLITS.
            splits = max(1, min(4, _sm_count(q.device) // max(1, bs)))
    get_cuda_mla_module().decode(
        q, latent, block_table.to(torch.int32), seqlens.to(torch.int32), o,
        float(sm_scale), int(block_size), int(splits),
    )
    return o
