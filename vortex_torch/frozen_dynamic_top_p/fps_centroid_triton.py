"""Isolated Triton block-finalization kernel for FPS R4 + one Lloyd update.

Input layout is ``[..., 16, head_dim]`` and output centroid cache layout is
``[..., 4, head_dim]`` with a separate ``[..., 4]`` uint8 count cache.  The
leading dimensions normally mean ``[completed_block, kv_head]``.  This module
is intentionally not wired into NSA, Vortex, or the training selector.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 16
CLUSTERS = 4


@triton.jit
def _fps_r4_lloyd1_kernel(
    k_ptr,
    centroid_ptr,
    count_ptr,
    slot_ptr,
    valid_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HAS_SLOT_MAPPING: tl.constexpr,
    INPUT_USES_SLOT_MAPPING: tl.constexpr,
    HAS_VALID_MASK: tl.constexpr,
):
    stream = tl.program_id(0)
    logical_block = stream // NUM_KV_HEADS
    kv_head = stream % NUM_KV_HEADS
    if HAS_VALID_MASK:
        valid = tl.load(valid_ptr + logical_block) != 0
        if not valid:
            return
    input_stream = stream
    if INPUT_USES_SLOT_MAPPING:
        input_slot = tl.load(slot_ptr + logical_block)
        input_stream = input_slot * NUM_KV_HEADS + kv_head
    token = tl.arange(0, 16)[:, None]
    dim = tl.arange(0, BLOCK_D)[None, :]
    dim_mask = dim < HEAD_DIM
    base = input_stream * 16 * HEAD_DIM
    points = tl.load(
        k_ptr + base + token * HEAD_DIM + dim,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    # Deterministic FPS, matching the evaluation reference.  The first seed is
    # furthest from the block mean; subsequent seeds maximize distance to the
    # nearest already selected seed.  Strict comparisons retain the lower
    # cluster index on ties in the Lloyd assignment below.
    mean = tl.sum(points, axis=0) * (1.0 / 16.0)
    delta = points - mean[None, :]
    distance_to_mean = tl.sum(delta * delta, axis=1)
    seed0 = tl.argmax(distance_to_mean, axis=0)
    center0 = tl.load(
        k_ptr + base + seed0 * HEAD_DIM + tl.arange(0, BLOCK_D),
        mask=tl.arange(0, BLOCK_D) < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    delta0 = points - center0[None, :]
    nearest = tl.sum(delta0 * delta0, axis=1)

    seed1 = tl.argmax(nearest, axis=0)
    center1 = tl.load(
        k_ptr + base + seed1 * HEAD_DIM + tl.arange(0, BLOCK_D),
        mask=tl.arange(0, BLOCK_D) < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    delta1 = points - center1[None, :]
    nearest = tl.minimum(nearest, tl.sum(delta1 * delta1, axis=1))

    seed2 = tl.argmax(nearest, axis=0)
    center2 = tl.load(
        k_ptr + base + seed2 * HEAD_DIM + tl.arange(0, BLOCK_D),
        mask=tl.arange(0, BLOCK_D) < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    delta2 = points - center2[None, :]
    nearest = tl.minimum(nearest, tl.sum(delta2 * delta2, axis=1))

    seed3 = tl.argmax(nearest, axis=0)
    center3 = tl.load(
        k_ptr + base + seed3 * HEAD_DIM + tl.arange(0, BLOCK_D),
        mask=tl.arange(0, BLOCK_D) < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)

    # One Lloyd assignment.  Counts and means are computed in FP32 before the
    # centroids are cast to the output cache dtype by tl.store.
    delta0 = points - center0[None, :]
    delta1 = points - center1[None, :]
    delta2 = points - center2[None, :]
    delta3 = points - center3[None, :]
    distance0 = tl.sum(delta0 * delta0, axis=1)
    distance1 = tl.sum(delta1 * delta1, axis=1)
    distance2 = tl.sum(delta2 * delta2, axis=1)
    distance3 = tl.sum(delta3 * delta3, axis=1)
    best = distance0
    assignment = tl.zeros((16,), tl.int32)
    take1 = distance1 < best
    best = tl.where(take1, distance1, best)
    assignment = tl.where(take1, 1, assignment)
    take2 = distance2 < best
    best = tl.where(take2, distance2, best)
    assignment = tl.where(take2, 2, assignment)
    take3 = distance3 < best
    assignment = tl.where(take3, 3, assignment)

    member0 = assignment == 0
    member1 = assignment == 1
    member2 = assignment == 2
    member3 = assignment == 3
    count0 = tl.sum(member0.to(tl.int32), axis=0)
    count1 = tl.sum(member1.to(tl.int32), axis=0)
    count2 = tl.sum(member2.to(tl.int32), axis=0)
    count3 = tl.sum(member3.to(tl.int32), axis=0)
    updated0 = tl.sum(tl.where(member0[:, None], points, 0.0), axis=0) / count0
    updated1 = tl.sum(tl.where(member1[:, None], points, 0.0), axis=0) / count1
    updated2 = tl.sum(tl.where(member2[:, None], points, 0.0), axis=0) / count2
    updated3 = tl.sum(tl.where(member3[:, None], points, 0.0), axis=0) / count3

    out_stream = stream
    if HAS_SLOT_MAPPING:
        slot = tl.load(slot_ptr + logical_block)
        out_stream = slot * NUM_KV_HEADS + kv_head
    out_dim = tl.arange(0, BLOCK_D)
    out_base = out_stream * 4 * HEAD_DIM
    out_mask = out_dim < HEAD_DIM
    tl.store(centroid_ptr + out_base + 0 * HEAD_DIM + out_dim, updated0, mask=out_mask)
    tl.store(centroid_ptr + out_base + 1 * HEAD_DIM + out_dim, updated1, mask=out_mask)
    tl.store(centroid_ptr + out_base + 2 * HEAD_DIM + out_dim, updated2, mask=out_mask)
    tl.store(centroid_ptr + out_base + 3 * HEAD_DIM + out_dim, updated3, mask=out_mask)
    count_base = out_stream * 4
    tl.store(count_ptr + count_base + 0, count0)
    tl.store(count_ptr + count_base + 1, count1)
    tl.store(count_ptr + count_base + 2, count2)
    tl.store(count_ptr + count_base + 3, count3)


def fps_r4_lloyd1(
    k_blocks: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
    centroid_out: torch.Tensor | None = None,
    count_out: torch.Tensor | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build centroid/count caches for completed 16-token blocks.

    Args:
        k_blocks: CUDA BF16/FP16/FP32/FP8 tensor shaped ``[..., 16, D]``.
        output_dtype: centroid cache dtype; defaults to the input dtype.  It is
            ignored when ``centroid_out`` is supplied.
        centroid_out/count_out: optional preallocated contiguous cache views.
            Counts use uint8 because a completed block contains only 16 tokens.
        num_warps: Triton launch configuration used by the isolated benchmark.
    """

    if not k_blocks.is_cuda:
        raise ValueError("k_blocks must be a CUDA tensor")
    if not k_blocks.is_contiguous():
        raise ValueError("k_blocks must be contiguous")
    if k_blocks.ndim < 3 or k_blocks.shape[-2] != BLOCK_SIZE:
        raise ValueError("k_blocks must have shape [..., 16, head_dim]")
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if k_blocks.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        *fp8_dtypes,
    ):
        raise ValueError("k_blocks dtype must be BF16, FP16, FP32, or CUDA FP8")
    head_dim = k_blocks.shape[-1]
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in [1, 256]")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if centroid_out is not None:
        output_dtype = centroid_out.dtype
    elif output_dtype is None:
        output_dtype = torch.bfloat16 if k_blocks.dtype in fp8_dtypes else k_blocks.dtype
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("output_dtype must be BF16, FP16, or FP32")

    leading = k_blocks.shape[:-2]
    streams = k_blocks.numel() // (BLOCK_SIZE * head_dim)
    centroid_shape = (*leading, CLUSTERS, head_dim)
    count_shape = (*leading, CLUSTERS)
    if centroid_out is None:
        centroids = torch.empty(
            centroid_shape, device=k_blocks.device, dtype=output_dtype
        )
    else:
        if (
            centroid_out.shape != centroid_shape
            or centroid_out.device != k_blocks.device
            or not centroid_out.is_contiguous()
        ):
            raise ValueError("centroid_out has incompatible shape/device/layout")
        centroids = centroid_out
    if count_out is None:
        counts = torch.empty(
            count_shape, device=k_blocks.device, dtype=torch.uint8
        )
    else:
        if (
            count_out.shape != count_shape
            or count_out.device != k_blocks.device
            or count_out.dtype not in (torch.uint8, torch.int32)
            or not count_out.is_contiguous()
        ):
            raise ValueError("count_out has incompatible shape/device/dtype/layout")
        counts = count_out
    block_d = triton.next_power_of_2(head_dim)
    _fps_r4_lloyd1_kernel[(streams,)](
        k_blocks,
        centroids,
        counts,
        k_blocks,
        k_blocks,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        NUM_KV_HEADS=1,
        HAS_SLOT_MAPPING=False,
        INPUT_USES_SLOT_MAPPING=False,
        HAS_VALID_MASK=False,
        num_warps=num_warps,
    )
    return centroids, counts


def fps_r4_lloyd1_scatter(
    k_blocks: torch.Tensor,
    slot_ids: torch.Tensor,
    centroid_cache: torch.Tensor,
    count_cache: torch.Tensor,
    *,
    num_warps: int = 4,
) -> None:
    """Build completed blocks and scatter summaries into preallocated slots.

    ``k_blocks`` is ``[M, Hkv, 16, D]`` and ``slot_ids[M]`` maps every packed
    logical block to the first dimension of caches shaped
    ``[S, Hkv, 4, D]`` and ``[S, Hkv, 4]``.
    """

    if not k_blocks.is_cuda or not slot_ids.is_cuda:
        raise ValueError("k_blocks and slot_ids must be CUDA tensors")
    if k_blocks.ndim != 4 or k_blocks.shape[-2] != BLOCK_SIZE:
        raise ValueError("k_blocks must have shape [M, Hkv, 16, D]")
    if not k_blocks.is_contiguous() or not slot_ids.is_contiguous():
        raise ValueError("k_blocks and slot_ids must be contiguous")
    logical_blocks, kv_heads, _, head_dim = k_blocks.shape
    if slot_ids.shape != (logical_blocks,) or slot_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("slot_ids must be contiguous int32/int64 [M]")
    expected_centroid_tail = (kv_heads, CLUSTERS, head_dim)
    expected_count_tail = (kv_heads, CLUSTERS)
    if (
        centroid_cache.ndim != 4
        or centroid_cache.shape[1:] != expected_centroid_tail
        or centroid_cache.device != k_blocks.device
        or centroid_cache.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not centroid_cache.is_contiguous()
    ):
        raise ValueError("centroid_cache has incompatible shape/device/dtype/layout")
    if (
        count_cache.ndim != 3
        or count_cache.shape[1:] != expected_count_tail
        or count_cache.shape[0] != centroid_cache.shape[0]
        or count_cache.device != k_blocks.device
        or count_cache.dtype not in (torch.uint8, torch.int32)
        or not count_cache.is_contiguous()
    ):
        raise ValueError("count_cache has incompatible shape/device/dtype/layout")
    if logical_blocks == 0:
        return
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")

    streams = logical_blocks * kv_heads
    _fps_r4_lloyd1_kernel[(streams,)](
        k_blocks,
        centroid_cache,
        count_cache,
        slot_ids,
        slot_ids,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        NUM_KV_HEADS=kv_heads,
        HAS_SLOT_MAPPING=True,
        INPUT_USES_SLOT_MAPPING=False,
        HAS_VALID_MASK=False,
        num_warps=num_warps,
    )


def fps_r4_lloyd1_scatter_from_cache(
    k_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    centroid_cache: torch.Tensor,
    count_cache: torch.Tensor,
    *,
    num_warps: int = 4,
) -> None:
    """Capture-safe conditional construction directly from physical K pages.

    The launch shape is fixed by ``slot_ids``. Streams whose corresponding
    ``valid_mask`` entry is false return before loading K or writing metadata.
    This avoids dynamic boolean compaction while preserving boundary-only
    construction semantics under CUDA Graph replay.
    """

    if k_cache.ndim != 4 or k_cache.shape[-2] != BLOCK_SIZE:
        raise ValueError("k_cache must have shape [S, Hkv, 16, D]")
    if not k_cache.is_cuda or not k_cache.is_contiguous():
        raise ValueError("k_cache must be a contiguous CUDA tensor")
    slots, kv_heads, _, head_dim = k_cache.shape
    if slot_ids.ndim != 1 or slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_ids must be contiguous int32/int64 [M]")
    if not slot_ids.is_cuda or not slot_ids.is_contiguous():
        raise ValueError("slot_ids must be a contiguous CUDA tensor")
    if valid_mask.shape != slot_ids.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with the slot_ids shape")
    if not valid_mask.is_cuda or not valid_mask.is_contiguous():
        raise ValueError("valid_mask must be a contiguous CUDA tensor")
    if centroid_cache.shape != (slots, kv_heads, CLUSTERS, head_dim):
        raise ValueError("centroid_cache has incompatible shape")
    if count_cache.shape != (slots, kv_heads, CLUSTERS):
        raise ValueError("count_cache has incompatible shape")
    if not centroid_cache.is_contiguous() or not count_cache.is_contiguous():
        raise ValueError("summary caches must be contiguous")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")

    streams = slot_ids.numel() * kv_heads
    if streams == 0:
        return
    _fps_r4_lloyd1_kernel[(streams,)](
        k_cache,
        centroid_cache,
        count_cache,
        slot_ids,
        valid_mask,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        NUM_KV_HEADS=kv_heads,
        HAS_SLOT_MAPPING=True,
        INPUT_USES_SLOT_MAPPING=True,
        HAS_VALID_MASK=True,
        num_warps=num_warps,
    )


def centroid_cache_bytes(
    *,
    num_blocks: int,
    kv_heads: int,
    head_dim: int,
    centroid_element_size: int = 2,
    count_element_size: int = 1,
) -> dict[str, int | float]:
    """Return explicit centroid/count and original block-K cache sizes."""

    centroid_bytes = num_blocks * kv_heads * CLUSTERS * head_dim * centroid_element_size
    count_bytes = num_blocks * kv_heads * CLUSTERS * count_element_size
    original_k_bytes = num_blocks * kv_heads * BLOCK_SIZE * head_dim * centroid_element_size
    return {
        "centroid_bytes": centroid_bytes,
        "count_bytes": count_bytes,
        "total_bytes": centroid_bytes + count_bytes,
        "original_block_k_bytes": original_k_bytes,
        "fraction_of_original_k": (centroid_bytes + count_bytes) / original_k_bytes,
    }

