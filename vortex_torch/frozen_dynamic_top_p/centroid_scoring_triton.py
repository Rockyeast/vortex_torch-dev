"""Isolated decode scoring kernels for the frozen centroid selector.

The output is ``log(mean_g softmax_blocks(logsumexp_centroids(...)))`` for
every batch item, KV head, and completed historical block.  Temperature and
Top-P selection intentionally remain downstream.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _centroid_per_q_logmass_kernel(
    q_ptr,
    centroid_ptr,
    count_ptr,
    slot_ptr,
    block_count_ptr,
    per_q_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    K_SCALE: tl.constexpr,
):
    shared_row = tl.program_id(0)
    block_tile = tl.program_id(1)
    batch = shared_row // NUM_KV_HEADS
    kv_head = shared_row % NUM_KV_HEADS
    block = block_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    block_count = tl.load(block_count_ptr + batch)
    valid_block = (block < NUM_BLOCKS) & (block < block_count)
    slot = tl.load(
        slot_ptr + batch * NUM_BLOCKS + block,
        mask=valid_block,
        other=0,
    ).to(tl.int64)

    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    cache_stream = slot * NUM_KV_HEADS + kv_head
    centroid_base = cache_stream[:, None] * 4 * HEAD_DIM + dim[None, :]
    count_base = cache_stream * 4

    centroid0 = tl.load(
        centroid_ptr + centroid_base + 0 * HEAD_DIM,
        mask=valid_block[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
    centroid1 = tl.load(
        centroid_ptr + centroid_base + 1 * HEAD_DIM,
        mask=valid_block[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
    centroid2 = tl.load(
        centroid_ptr + centroid_base + 2 * HEAD_DIM,
        mask=valid_block[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
    centroid3 = tl.load(
        centroid_ptr + centroid_base + 3 * HEAD_DIM,
        mask=valid_block[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32)
    count0 = tl.load(count_ptr + count_base + 0, mask=valid_block, other=1).to(tl.float32)
    count1 = tl.load(count_ptr + count_base + 1, mask=valid_block, other=1).to(tl.float32)
    count2 = tl.load(count_ptr + count_base + 2, mask=valid_block, other=1).to(tl.float32)
    count3 = tl.load(count_ptr + count_base + 3, mask=valid_block, other=1).to(tl.float32)

    for group in tl.static_range(0, GROUP_SIZE):
        q_head = kv_head * GROUP_SIZE + group
        query_row = batch * NUM_Q_HEADS + q_head
        query = tl.load(
            q_ptr + query_row * HEAD_DIM + dim,
            mask=valid_dim,
            other=0.0,
        ).to(tl.float32)
        logit0 = tl.sum(centroid0 * query[None, :], axis=1) * SCALE * K_SCALE + tl.log(count0)
        logit1 = tl.sum(centroid1 * query[None, :], axis=1) * SCALE * K_SCALE + tl.log(count1)
        logit2 = tl.sum(centroid2 * query[None, :], axis=1) * SCALE * K_SCALE + tl.log(count2)
        logit3 = tl.sum(centroid3 * query[None, :], axis=1) * SCALE * K_SCALE + tl.log(count3)
        maximum = tl.maximum(
            tl.maximum(logit0, logit1), tl.maximum(logit2, logit3)
        )
        mass = (
            tl.exp(logit0 - maximum)
            + tl.exp(logit1 - maximum)
            + tl.exp(logit2 - maximum)
            + tl.exp(logit3 - maximum)
        )
        logmass = maximum + tl.log(mass)
        logmass = tl.where(valid_block, logmass, -float("inf"))
        tl.store(
            per_q_ptr + query_row * NUM_BLOCKS + block,
            logmass,
            mask=block < NUM_BLOCKS,
        )


@triton.jit
def _centroid_per_q_logmass_tensorcore_kernel(
    q_ptr,
    centroid_ptr,
    count_ptr,
    slot_ptr,
    block_count_ptr,
    per_q_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SCALE: tl.constexpr,
    K_SCALE: tl.constexpr,
):
    """Tensor-core equivalent of four centroid dot reductions per block."""

    shared_row = tl.program_id(0)
    block_tile = tl.program_id(1)
    batch = shared_row // NUM_KV_HEADS
    kv_head = shared_row % NUM_KV_HEADS
    block = block_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    block_count = tl.load(block_count_ptr + batch)
    valid_block = (block < NUM_BLOCKS) & (block < block_count)
    slot = tl.load(
        slot_ptr + batch * NUM_BLOCKS + block,
        mask=valid_block,
        other=0,
    ).to(tl.int64)

    query_row_in_group = tl.arange(0, BLOCK_M)
    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    query_head = kv_head * GROUP_SIZE + query_row_in_group
    query = tl.load(
        q_ptr
        + batch * NUM_Q_HEADS * HEAD_DIM
        + query_head[:, None] * HEAD_DIM
        + dim[None, :],
        mask=(query_row_in_group[:, None] < GROUP_SIZE) & valid_dim[None, :],
        other=0.0,
    )

    centroid_column = tl.arange(0, BLOCK_N * 4)
    centroid_block = centroid_column // 4
    centroid_cluster = centroid_column % 4
    centroid_slot = tl.gather(slot, centroid_block, axis=0)
    centroid_valid = tl.gather(valid_block, centroid_block, axis=0)
    cache_stream = centroid_slot * NUM_KV_HEADS + kv_head
    centroid_offset = (
        (cache_stream * 4 + centroid_cluster)[None, :] * HEAD_DIM
        + dim[:, None]
    )
    centroid = tl.load(
        centroid_ptr + centroid_offset,
        mask=valid_dim[:, None] & centroid_valid[None, :],
        other=0.0,
    )
    dot = tl.dot(query, centroid, out_dtype=tl.float32)
    count = tl.load(
        count_ptr + cache_stream * 4 + centroid_cluster,
        mask=centroid_valid,
        other=1,
    ).to(tl.float32)
    logits = dot * SCALE * K_SCALE + tl.log(count)[None, :]
    logits = tl.reshape(logits, (BLOCK_M, BLOCK_N, 4))
    maximum = tl.max(logits, axis=2)
    mass = tl.sum(tl.exp(logits - maximum[:, :, None]), axis=2)
    logmass = maximum + tl.log(mass)
    logmass = tl.where(valid_block[None, :], logmass, -float("inf"))
    tl.store(
        per_q_ptr
        + (batch * NUM_Q_HEADS + query_head)[:, None] * NUM_BLOCKS
        + block[None, :],
        logmass,
        mask=(query_row_in_group[:, None] < GROUP_SIZE)
        & (block[None, :] < NUM_BLOCKS),
    )


@triton.jit
def _centroid_per_q_logmass_tensorcore_packed_kv_kernel(
    q_ptr,
    centroid_ptr,
    count_ptr,
    slot_ptr,
    block_count_ptr,
    per_q_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    KV_PACK: tl.constexpr,
    KV_PACK_TILES: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Pack several small GQA groups into one tensor-core tile.

    The dot tile still performs exactly the same total multiply count as
    ``KV_PACK`` independent 16-row tiles: otherwise-unused query rows are
    filled by adjacent KV groups and the desired block diagonal is gathered
    from the result.  Cross-group dot products are computed but discarded.
    """

    packed_row = tl.program_id(0)
    block_tile = tl.program_id(1)
    batch = packed_row // KV_PACK_TILES
    kv_base = (packed_row % KV_PACK_TILES) * KV_PACK
    block = block_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    block_count = tl.load(block_count_ptr + batch)
    valid_block = (block < NUM_BLOCKS) & (block < block_count)
    slot = tl.load(
        slot_ptr + batch * NUM_BLOCKS + block,
        mask=valid_block,
        other=0,
    ).to(tl.int64)

    query_row = tl.arange(0, BLOCK_M)
    query_pack = query_row // GROUP_SIZE
    query_group = query_row % GROUP_SIZE
    query_kv_head = kv_base + query_pack
    valid_query = (query_pack < KV_PACK) & (query_kv_head < NUM_KV_HEADS)
    query_head = query_kv_head * GROUP_SIZE + query_group
    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    query = tl.load(
        q_ptr
        + batch * NUM_Q_HEADS * HEAD_DIM
        + query_head[:, None] * HEAD_DIM
        + dim[None, :],
        mask=valid_query[:, None] & valid_dim[None, :],
        other=0.0,
    )

    packed_column = tl.arange(0, KV_PACK * BLOCK_N * 4)
    centroid_pack = packed_column // (BLOCK_N * 4)
    centroid_within_pack = packed_column % (BLOCK_N * 4)
    centroid_block = centroid_within_pack // 4
    centroid_cluster = centroid_within_pack % 4
    centroid_kv_head = kv_base + centroid_pack
    centroid_slot = tl.gather(slot, centroid_block, axis=0)
    centroid_valid = tl.gather(valid_block, centroid_block, axis=0)
    centroid_valid &= centroid_kv_head < NUM_KV_HEADS
    cache_stream = centroid_slot * NUM_KV_HEADS + centroid_kv_head
    centroid_offset = (
        (cache_stream * 4 + centroid_cluster)[None, :] * HEAD_DIM
        + dim[:, None]
    )
    centroid = tl.load(
        centroid_ptr + centroid_offset,
        mask=valid_dim[:, None] & centroid_valid[None, :],
        other=0.0,
    )
    packed_dot = tl.dot(query, centroid, out_dtype=tl.float32)

    desired_column = tl.arange(0, BLOCK_N * 4)
    desired_packed_column = (
        query_pack[:, None] * (BLOCK_N * 4) + desired_column[None, :]
    )
    dot = tl.gather(packed_dot, desired_packed_column, axis=1)
    desired_block = desired_column // 4
    desired_cluster = desired_column % 4
    desired_slot = tl.gather(slot, desired_block, axis=0)
    desired_cache_stream = (
        desired_slot[None, :] * NUM_KV_HEADS + query_kv_head[:, None]
    )
    count = tl.load(
        count_ptr + desired_cache_stream * 4 + desired_cluster[None, :],
        mask=valid_query[:, None]
        & tl.gather(valid_block, desired_block, axis=0)[None, :],
        other=1,
    ).to(tl.float32)
    logits = dot * SCALE + tl.log(count)
    logits = tl.reshape(logits, (BLOCK_M, BLOCK_N, 4))
    maximum = tl.max(logits, axis=2)
    mass = tl.sum(tl.exp(logits - maximum[:, :, None]), axis=2)
    logmass = maximum + tl.log(mass)
    logmass = tl.where(valid_block[None, :], logmass, -float("inf"))
    tl.store(
        per_q_ptr
        + (batch * NUM_Q_HEADS + query_head)[:, None] * NUM_BLOCKS
        + block[None, :],
        logmass,
        mask=valid_query[:, None] & (block[None, :] < NUM_BLOCKS),
    )


@triton.jit
def _per_q_normalized_gqa_kernel(
    per_q_ptr,
    block_count_ptr,
    score_ptr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    shared_row = tl.program_id(0)
    batch = shared_row // NUM_KV_HEADS
    kv_head = shared_row % NUM_KV_HEADS
    block_count = tl.load(block_count_ptr + batch)
    block = tl.arange(0, BLOCK_N)
    valid = (block < NUM_BLOCKS) & (block < block_count)
    shared = tl.zeros((BLOCK_N,), tl.float32)
    for group in tl.static_range(0, GROUP_SIZE):
        q_head = kv_head * GROUP_SIZE + group
        q_row = batch * NUM_KV_HEADS * GROUP_SIZE + q_head
        logmass = tl.load(
            per_q_ptr + q_row * NUM_BLOCKS + block,
            mask=valid,
            other=-float("inf"),
        )
        maximum = tl.max(logmass, axis=0)
        probability = tl.exp(logmass - maximum)
        denominator = tl.sum(probability, axis=0)
        shared += probability / denominator
    shared *= 1.0 / GROUP_SIZE
    score = tl.log(shared)
    tl.store(score_ptr + shared_row * NUM_BLOCKS + block, score, mask=valid)
    tl.store(
        score_ptr + shared_row * NUM_BLOCKS + block,
        -float("inf"),
        mask=(block < NUM_BLOCKS) & ~valid,
    )


def centroid_per_q_logmass(
    q: torch.Tensor,
    centroid_cache: torch.Tensor,
    count_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    k_scale: float = 1.0,
    per_q_out: torch.Tensor | None = None,
    num_warps_score: int = 4,
) -> torch.Tensor:
    """Return count-weighted centroid logmass for every Q head and block.

    Shapes are ``q[B,Hq,D]``, centroid cache ``[S,Hkv,4,D]``, count cache
    ``[S,Hkv,4]``, logical-to-physical slots ``[B,N]``, block counts ``[B]``,
    and output ``[B,Hq,N]``.  Per-Q normalization and GQA aggregation are a
    separate stage so the Blackwell adapter can fuse them with Top-P without
    changing the frozen score or selection semantics.
    """

    tensors = (q, centroid_cache, count_cache, slot_ids, block_counts)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("all inputs must be CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous")
    if q.ndim != 3 or centroid_cache.ndim != 4 or count_cache.ndim != 3:
        raise ValueError("invalid q/centroid/count rank")
    batch, q_heads, head_dim = q.shape
    slots, kv_heads, clusters, cache_dim = centroid_cache.shape
    if clusters != 4 or cache_dim != head_dim:
        raise ValueError("centroid cache must be [S,Hkv,4,D] matching q")
    if count_cache.shape != (slots, kv_heads, 4):
        raise ValueError("count cache must be [S,Hkv,4]")
    if count_cache.dtype not in (torch.uint8, torch.int32):
        raise ValueError("count cache must use uint8 or int32")
    if q_heads % kv_heads != 0:
        raise ValueError("Q heads must be divisible by KV heads")
    if slot_ids.ndim != 2 or slot_ids.shape[0] != batch:
        raise ValueError("slot_ids must be [B,N]")
    if slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_ids must use int32 or int64")
    if block_counts.shape != (batch,) or block_counts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("block_counts must be int32/int64 [B]")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("q must use BF16, FP16, or FP32")
    if centroid_cache.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("centroid cache must use BF16, FP16, or FP32")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in [1, 256]")
    if not math.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError("k_scale must be positive and finite")
    num_blocks = slot_ids.shape[1]
    if num_blocks <= 0 or num_blocks > 4096:
        raise ValueError("this decode harness supports 1..4096 logical blocks")

    per_q_shape = (batch, q_heads, num_blocks)
    if per_q_out is None:
        per_q_out = torch.empty(per_q_shape, device=q.device, dtype=torch.float32)
    elif (
        per_q_out.shape != per_q_shape
        or per_q_out.device != q.device
        or per_q_out.dtype != torch.float32
        or not per_q_out.is_contiguous()
    ):
        raise ValueError("per_q_out has incompatible shape/device/dtype/layout")
    group_size = q_heads // kv_heads
    block_d = triton.next_power_of_2(head_dim)
    device_major, _ = torch.cuda.get_device_capability(q.device)
    use_tensorcore = (
        device_major >= 10
        and q.dtype in (torch.bfloat16, torch.float16)
        and centroid_cache.dtype == q.dtype
        and group_size <= 16
    )
    if use_tensorcore:
        block_n_score = 16
        # On Blackwell, wider GQA groups already provide enough independent
        # tensor-core work per program; a single software-pipeline stage avoids
        # extra register pressure.  This changes launch scheduling only.
        tensorcore_stages = 1 if group_size >= 4 else 2
        _centroid_per_q_logmass_tensorcore_kernel[
            (batch * kv_heads, triton.cdiv(num_blocks, block_n_score))
        ](
            q,
            centroid_cache,
            count_cache,
            slot_ids,
            block_counts,
            per_q_out,
            NUM_Q_HEADS=q_heads,
            NUM_KV_HEADS=kv_heads,
            GROUP_SIZE=group_size,
            NUM_BLOCKS=num_blocks,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_N=block_n_score,
            BLOCK_M=16,
            SCALE=1.0 / math.sqrt(head_dim),
            K_SCALE=k_scale,
            num_warps=4,
            num_stages=tensorcore_stages,
        )
    else:
        block_n_score = 8
        _centroid_per_q_logmass_kernel[
            (batch * kv_heads, triton.cdiv(num_blocks, block_n_score))
        ](
            q,
            centroid_cache,
            count_cache,
            slot_ids,
            block_counts,
            per_q_out,
            NUM_Q_HEADS=q_heads,
            NUM_KV_HEADS=kv_heads,
            GROUP_SIZE=group_size,
            NUM_BLOCKS=num_blocks,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_N=block_n_score,
            SCALE=1.0 / math.sqrt(head_dim),
            K_SCALE=k_scale,
            num_warps=num_warps_score,
        )
    return per_q_out


def centroid_normalized_scores(
    q: torch.Tensor,
    centroid_cache: torch.Tensor,
    count_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    k_scale: float = 1.0,
    per_q_out: torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
    num_warps_score: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-Q logmass scratch and normalized Shared-KV log scores."""

    per_q_out = centroid_per_q_logmass(
        q,
        centroid_cache,
        count_cache,
        slot_ids,
        block_counts,
        k_scale=k_scale,
        per_q_out=per_q_out,
        num_warps_score=num_warps_score,
    )
    batch, q_heads, num_blocks = per_q_out.shape
    kv_heads = centroid_cache.shape[1]
    score_shape = (batch, kv_heads, num_blocks)
    if score_out is None:
        score_out = torch.empty(score_shape, device=q.device, dtype=torch.float32)
    elif (
        score_out.shape != score_shape
        or score_out.device != q.device
        or score_out.dtype != torch.float32
        or not score_out.is_contiguous()
    ):
        raise ValueError("score_out has incompatible shape/device/dtype/layout")

    group_size = q_heads // kv_heads
    block_n_normalize = triton.next_power_of_2(num_blocks)
    _per_q_normalized_gqa_kernel[(batch * kv_heads,)](
        per_q_out,
        block_counts,
        score_out,
        NUM_KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        NUM_BLOCKS=num_blocks,
        BLOCK_N=block_n_normalize,
        num_warps=8 if block_n_normalize >= 512 else 4,
    )
    return per_q_out, score_out

