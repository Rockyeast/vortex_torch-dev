"""Isolated paged GQA decode attention used only for systems cost closure."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _attention_launch_config(device: torch.device) -> tuple[bool, int]:
    """Use the measured lower-register-pressure launch on Blackwell."""

    major, _ = torch.cuda.get_device_capability(device)
    return (True, 1) if major >= 10 else (False, 4)


@triton.jit
def _paged_gqa_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    slot_index_ptr,
    selected_count_ptr,
    tail_slot_ptr,
    tail_count_ptr,
    output_ptr,
    lse_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDEX_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    K_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    INCLUDE_TAIL: tl.constexpr,
    PER_Q_HEAD: tl.constexpr,
):
    row = tl.program_id(0)
    if PER_Q_HEAD:
        batch = row // NUM_Q_HEADS
        q_head_base = row % NUM_Q_HEADS
        kv_head = q_head_base // GROUP_SIZE
        active_group_size: tl.constexpr = 1
    else:
        batch = row // NUM_KV_HEADS
        kv_head = row % NUM_KV_HEADS
        q_head_base = kv_head * GROUP_SIZE
        active_group_size: tl.constexpr = GROUP_SIZE
    selected_row = batch * NUM_KV_HEADS + kv_head
    selected_count = tl.load(selected_count_ptr + selected_row).to(tl.int32)

    group = tl.arange(0, active_group_size)
    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    q_head = q_head_base + group
    query_unscaled = tl.load(
        q_ptr
        + batch * NUM_Q_HEADS * HEAD_DIM
        + q_head[:, None] * HEAD_DIM
        + dim[None, :],
        mask=valid_dim[None, :],
        other=0.0,
    )
    query = query_unscaled * SCALE

    maximum = tl.full((active_group_size,), -float("inf"), tl.float32)
    denominator = tl.zeros((active_group_size,), tl.float32)
    accumulator = tl.zeros((active_group_size, BLOCK_D), tl.float32)
    token = tl.arange(0, BLOCK_SIZE)

    for selected_rank in range(selected_count):
        slot = tl.load(
            slot_index_ptr + selected_row * INDEX_CAPACITY + selected_rank
        ).to(tl.int64)
        stream = slot * NUM_KV_HEADS + kv_head
        cache_offset = (
            stream * BLOCK_SIZE * HEAD_DIM
            + token[:, None] * HEAD_DIM
            + dim[None, :]
        )
        key = tl.load(
            k_ptr + cache_offset,
            mask=valid_dim[None, :],
            other=0.0,
        ).to(tl.float32) * K_SCALE
        value = tl.load(
            v_ptr + cache_offset,
            mask=valid_dim[None, :],
            other=0.0,
        ).to(tl.float32) * V_SCALE
        logits = tl.sum(
            query[:, None, :].to(tl.float32)
            * key[None, :, :],
            axis=2,
        )
        new_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        correction = tl.exp(maximum - new_maximum)
        probability = tl.exp(logits - new_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, axis=1)
        weighted_value = tl.sum(
            probability[:, :, None]
            * value[None, :, :],
            axis=1,
        )
        accumulator = accumulator * correction[:, None] + weighted_value
        maximum = new_maximum

    if INCLUDE_TAIL:
        tail_slot = tl.load(tail_slot_ptr + batch).to(tl.int64)
        tail_count = tl.load(tail_count_ptr + batch).to(tl.int32)
        stream = tail_slot * NUM_KV_HEADS + kv_head
        cache_offset = (
            stream * BLOCK_SIZE * HEAD_DIM
            + token[:, None] * HEAD_DIM
            + dim[None, :]
        )
        valid_token = token < tail_count
        tail_key = tl.load(
            k_ptr + cache_offset,
            mask=valid_token[:, None] & valid_dim[None, :],
            other=0.0,
        ).to(tl.float32) * K_SCALE
        tail_value = tl.load(
            v_ptr + cache_offset,
            mask=valid_token[:, None] & valid_dim[None, :],
            other=0.0,
        ).to(tl.float32) * V_SCALE
        tail_logits = tl.sum(
            query_unscaled[:, None, :].to(tl.float32)
            * tail_key[None, :, :],
            axis=2,
        ) * SCALE
        tail_logits = tl.where(
            valid_token[None, :], tail_logits, -float("inf")
        )
        new_maximum = tl.maximum(maximum, tl.max(tail_logits, axis=1))
        correction = tl.exp(maximum - new_maximum)
        probability = tl.exp(tail_logits - new_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, axis=1)
        weighted_value = tl.sum(
            probability[:, :, None]
            * tail_value[None, :, :],
            axis=1,
        )
        accumulator = accumulator * correction[:, None] + weighted_value
        maximum = new_maximum

    has_selected = selected_count > 0
    if INCLUDE_TAIL:
        has_selected = has_selected | (tail_count > 0)
    safe_denominator = tl.where(has_selected, denominator, 1.0)
    result = tl.where(
        has_selected,
        accumulator / safe_denominator[:, None],
        0.0,
    )
    lse = tl.where(
        has_selected,
        maximum + tl.log(safe_denominator),
        -float("inf"),
    )
    tl.store(
        output_ptr
        + batch * NUM_Q_HEADS * HEAD_DIM
        + q_head[:, None] * HEAD_DIM
        + dim[None, :],
        result,
        mask=valid_dim[None, :],
    )
    tl.store(lse_ptr + batch * NUM_Q_HEADS + q_head, lse)


@triton.jit
def _paged_gqa_decode_split_state_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    slot_index_ptr,
    selected_count_ptr,
    tail_slot_ptr,
    tail_count_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INDEX_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    K_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
):
    """One exact online-softmax state for a contiguous selected-rank slice."""

    q_row = tl.program_id(0)
    split = tl.program_id(1)
    batch = q_row // NUM_Q_HEADS
    q_head = q_row % NUM_Q_HEADS
    kv_head = q_head // GROUP_SIZE
    selected_row = batch * NUM_KV_HEADS + kv_head
    selected_count = tl.load(selected_count_ptr + selected_row).to(tl.int32)
    ranks_per_split = tl.cdiv(selected_count, NUM_SPLITS)
    first_rank = split * ranks_per_split
    last_rank = tl.minimum(first_rank + ranks_per_split, selected_count)

    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    query_unscaled = tl.load(
        q_ptr + q_row * HEAD_DIM + dim,
        mask=valid_dim,
        other=0.0,
    )
    query = query_unscaled * SCALE
    maximum = -float("inf")
    denominator = 0.0
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    token = tl.arange(0, BLOCK_SIZE)

    for selected_rank in range(first_rank, last_rank):
        slot = tl.load(
            slot_index_ptr + selected_row * INDEX_CAPACITY + selected_rank
        ).to(tl.int64)
        stream = slot * NUM_KV_HEADS + kv_head
        cache_offset = (
            stream * BLOCK_SIZE * HEAD_DIM
            + token[:, None] * HEAD_DIM
            + dim[None, :]
        )
        key = tl.load(
            k_ptr + cache_offset,
            mask=valid_dim[None, :],
            other=0.0,
        ).to(tl.float32) * K_SCALE
        value = tl.load(
            v_ptr + cache_offset,
            mask=valid_dim[None, :],
            other=0.0,
        ).to(tl.float32) * V_SCALE
        logits = tl.sum(
            query[None, :].to(tl.float32) * key, axis=1
        )
        new_maximum = tl.maximum(maximum, tl.max(logits, axis=0))
        correction = tl.exp(maximum - new_maximum)
        probability = tl.exp(logits - new_maximum)
        denominator = denominator * correction + tl.sum(probability, axis=0)
        accumulator = accumulator * correction + tl.sum(
            probability[:, None] * value, axis=0
        )
        maximum = new_maximum

    # The current partial block remains mandatory and exact.  Put it in split
    # zero only, then merge all independent states below.
    tail_count = tl.load(tail_count_ptr + batch).to(tl.int32)
    include_tail = (split == 0) & (tail_count > 0)
    tail_slot = tl.load(tail_slot_ptr + batch).to(tl.int64)
    stream = tail_slot * NUM_KV_HEADS + kv_head
    cache_offset = (
        stream * BLOCK_SIZE * HEAD_DIM
        + token[:, None] * HEAD_DIM
        + dim[None, :]
    )
    valid_tail = include_tail & (token < tail_count)
    key = tl.load(
        k_ptr + cache_offset,
        mask=valid_tail[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32) * K_SCALE
    value = tl.load(
        v_ptr + cache_offset,
        mask=valid_tail[:, None] & valid_dim[None, :],
        other=0.0,
    ).to(tl.float32) * V_SCALE
    logits = tl.sum(
        query_unscaled[None, :].to(tl.float32) * key, axis=1
    ) * SCALE
    logits = tl.where(valid_tail, logits, -float("inf"))
    new_maximum = tl.maximum(maximum, tl.max(logits, axis=0))
    correction = tl.exp(maximum - new_maximum)
    probability = tl.exp(logits - new_maximum)
    denominator = denominator * correction + tl.sum(probability, axis=0)
    accumulator = accumulator * correction + tl.sum(
        probability[:, None] * value, axis=0
    )
    maximum = new_maximum

    has_tokens = (first_rank < last_rank) | include_tail
    safe_denominator = tl.where(has_tokens, denominator, 1.0)
    normalized = tl.where(has_tokens, accumulator / safe_denominator, 0.0)
    partial_lse = tl.where(
        has_tokens,
        maximum + tl.log(safe_denominator),
        -float("inf"),
    )
    partial_base = (q_row * NUM_SPLITS + split) * HEAD_DIM
    tl.store(
        partial_output_ptr + partial_base + dim,
        normalized,
        mask=valid_dim,
    )
    tl.store(partial_lse_ptr + q_row * NUM_SPLITS + split, partial_lse)


@triton.jit
def _merge_paged_gqa_decode_split_states_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    lse_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    q_row = tl.program_id(0)
    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    splits = tl.arange(0, NUM_SPLITS)
    partial_lse = tl.load(partial_lse_ptr + q_row * NUM_SPLITS + splits)
    maximum = tl.max(partial_lse, axis=0)
    valid = partial_lse != -float("inf")
    weight = tl.where(valid, tl.exp(partial_lse - maximum), 0.0)
    denominator = tl.sum(weight, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + (q_row * NUM_SPLITS + splits[:, None]) * HEAD_DIM
        + dim[None, :],
        mask=valid_dim[None, :],
        other=0.0,
    )
    output = tl.sum(weight[:, None] * partial_output, axis=0) / denominator
    tl.store(output_ptr + q_row * HEAD_DIM + dim, output, mask=valid_dim)
    tl.store(lse_ptr + q_row, maximum + tl.log(denominator))


def paged_gqa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    physical_block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one query token against selected completed physical blocks.

    Shapes: q ``[B,Hq,D]``, K/V ``[S,Hkv,16,D]``, selected physical slots
    ``[B,Hkv,C]``, counts ``[B,Hkv]``, output ``[B,Hq,D]``.
    """

    tensors = (q, k_cache, v_cache, physical_block_indices, block_counts)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("invalid q or K/V cache shape")
    batch, q_heads, head_dim = q.shape
    slots, kv_heads, block_size, cache_dim = k_cache.shape
    if block_size != 16 or cache_dim != head_dim:
        raise ValueError("K/V cache must be [S,Hkv,16,D] matching q")
    if q_heads % kv_heads != 0:
        raise ValueError("Q heads must be divisible by KV heads")
    if physical_block_indices.ndim != 3 or physical_block_indices.shape[:2] != (
        batch,
        kv_heads,
    ):
        raise ValueError("physical_block_indices must be [B,Hkv,C]")
    if physical_block_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("physical_block_indices must use int32/int64")
    if block_counts.shape != (batch, kv_heads) or block_counts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("block_counts must be int32/int64 [B,Hkv]")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in [1,256]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("q must use BF16 or FP16")
    supported_cache_dtypes = (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if k_cache.dtype != v_cache.dtype or k_cache.dtype not in supported_cache_dtypes:
        raise ValueError("K/V cache must use matching BF16, FP16, or CUDA FP8 dtype")
    if not math.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError("k_scale must be positive and finite")
    if not math.isfinite(v_scale) or v_scale <= 0.0:
        raise ValueError("v_scale must be positive and finite")

    if output is None:
        output = torch.empty_like(q)
    if lse is None:
        lse = torch.empty((batch, q_heads), device=q.device, dtype=torch.float32)
    if output.shape != q.shape or output.dtype != q.dtype or not output.is_contiguous():
        raise ValueError("output has incompatible shape/dtype/layout")
    if lse.shape != (batch, q_heads) or lse.dtype != torch.float32 or not lse.is_contiguous():
        raise ValueError("lse has incompatible shape/dtype/layout")

    group_size = q_heads // kv_heads
    block_d = triton.next_power_of_2(head_dim)
    per_q_head, num_warps = _attention_launch_config(q.device)
    _paged_gqa_decode_kernel[(batch * (q_heads if per_q_head else kv_heads),)](
        q,
        k_cache,
        v_cache,
        physical_block_indices,
        block_counts,
        q,
        block_counts,
        output,
        lse,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        INDEX_CAPACITY=physical_block_indices.shape[2],
        BLOCK_D=block_d,
        SCALE=1.0 / math.sqrt(head_dim),
        K_SCALE=k_scale,
        V_SCALE=v_scale,
        INCLUDE_TAIL=False,
        PER_Q_HEAD=per_q_head,
        num_warps=num_warps,
        num_stages=2,
    )
    return output, lse


def paged_gqa_decode_attention_with_tail(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    physical_block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    tail_physical: torch.Tensor,
    tail_counts: torch.Tensor,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend selected completed blocks plus the mandatory partial tail.

    This is mathematically identical to computing the two attention states
    separately and combining them with their log-sum-exp values, while keeping
    the online-softmax accumulator in one Triton program.
    """

    tensors = (
        q,
        k_cache,
        v_cache,
        physical_block_indices,
        block_counts,
        tail_physical,
        tail_counts,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("invalid q or K/V cache shape")
    batch, q_heads, head_dim = q.shape
    _, kv_heads, block_size, cache_dim = k_cache.shape
    if block_size != 16 or cache_dim != head_dim or q_heads % kv_heads:
        raise ValueError("incompatible Q/K/V geometry")
    if physical_block_indices.ndim != 3 or physical_block_indices.shape[:2] != (
        batch,
        kv_heads,
    ):
        raise ValueError("physical_block_indices must be [B,Hkv,C]")
    if block_counts.shape != (batch, kv_heads):
        raise ValueError("block_counts must be [B,Hkv]")
    if tail_physical.shape != (batch,) or tail_counts.shape != (batch,):
        raise ValueError("tail metadata must be [B]")
    integer_types = (torch.int32, torch.int64)
    if (
        physical_block_indices.dtype not in integer_types
        or block_counts.dtype not in integer_types
        or tail_physical.dtype not in integer_types
        or tail_counts.dtype not in integer_types
    ):
        raise ValueError("selection and tail metadata must use int32/int64")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("q must use BF16 or FP16")
    supported_cache_dtypes = (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if k_cache.dtype != v_cache.dtype or k_cache.dtype not in supported_cache_dtypes:
        raise ValueError("K/V cache must use matching BF16, FP16, or CUDA FP8 dtype")
    if not math.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError("k_scale must be positive and finite")
    if not math.isfinite(v_scale) or v_scale <= 0.0:
        raise ValueError("v_scale must be positive and finite")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError("head_dim must be in [1,256]")

    if output is None:
        output = torch.empty_like(q)
    if lse is None:
        lse = torch.empty((batch, q_heads), device=q.device, dtype=torch.float32)
    if output.shape != q.shape or output.dtype != q.dtype or not output.is_contiguous():
        raise ValueError("output has incompatible shape/dtype/layout")
    if lse.shape != (batch, q_heads) or lse.dtype != torch.float32 or not lse.is_contiguous():
        raise ValueError("lse has incompatible shape/dtype/layout")

    group_size = q_heads // kv_heads
    per_q_head, num_warps = _attention_launch_config(q.device)
    _paged_gqa_decode_kernel[(batch * (q_heads if per_q_head else kv_heads),)](
        q,
        k_cache,
        v_cache,
        physical_block_indices,
        block_counts,
        tail_physical,
        tail_counts,
        output,
        lse,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        INDEX_CAPACITY=physical_block_indices.shape[2],
        BLOCK_D=triton.next_power_of_2(head_dim),
        SCALE=1.0 / math.sqrt(head_dim),
        K_SCALE=k_scale,
        V_SCALE=v_scale,
        INCLUDE_TAIL=True,
        PER_Q_HEAD=per_q_head,
        num_warps=num_warps,
        num_stages=2,
    )
    return output, lse


def paged_gqa_decode_attention_with_tail_splitk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    physical_block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    tail_physical: torch.Tensor,
    tail_counts: torch.Tensor,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    num_splits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent split-selected-rank execution for Blackwell decode.

    Scratch is explicit so the formal caller can allocate it before CUDA Graph
    capture.  This function changes only the reduction schedule; indices,
    counts, exact-tail inclusion, and online-softmax semantics are unchanged.
    """

    tensors = (
        q,
        k_cache,
        v_cache,
        physical_block_indices,
        block_counts,
        tail_physical,
        tail_counts,
        partial_output,
        partial_lse,
        output,
        lse,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all split-K inputs and outputs must be contiguous CUDA tensors")
    batch, q_heads, head_dim = q.shape
    _, kv_heads, block_size, cache_dim = k_cache.shape
    if (
        v_cache.shape != k_cache.shape
        or block_size != 16
        or cache_dim != head_dim
        or q_heads % kv_heads
    ):
        raise ValueError("incompatible split-K Q/K/V geometry")
    if physical_block_indices.ndim != 3 or physical_block_indices.shape[:2] != (
        batch,
        kv_heads,
    ):
        raise ValueError("physical_block_indices must be [B,Hkv,C]")
    if block_counts.shape != (batch, kv_heads):
        raise ValueError("block_counts must be [B,Hkv]")
    if tail_physical.shape != (batch,) or tail_counts.shape != (batch,):
        raise ValueError("tail metadata must be [B]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("q must use BF16 or FP16")
    supported_cache_dtypes = (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if k_cache.dtype != v_cache.dtype or k_cache.dtype not in supported_cache_dtypes:
        raise ValueError("K/V cache must use matching BF16, FP16, or CUDA FP8 dtype")
    if not math.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError("k_scale must be positive and finite")
    if not math.isfinite(v_scale) or v_scale <= 0.0:
        raise ValueError("v_scale must be positive and finite")
    integer_types = (torch.int32, torch.int64)
    if any(
        tensor.dtype not in integer_types
        for tensor in (
            physical_block_indices,
            block_counts,
            tail_physical,
            tail_counts,
        )
    ):
        raise ValueError("selection and tail metadata must use int32/int64")
    if num_splits not in (2, 4, 8):
        raise ValueError("split-K attention supports 2, 4, or 8 splits")
    if partial_output.shape != (batch, q_heads, num_splits, head_dim):
        raise ValueError("partial_output must be [B,Hq,S,D]")
    if partial_output.dtype != torch.float32:
        raise ValueError("partial_output must use FP32")
    if partial_lse.shape != (batch, q_heads, num_splits):
        raise ValueError("partial_lse must be [B,Hq,S]")
    if partial_lse.dtype != torch.float32:
        raise ValueError("partial_lse must use FP32")
    if output.shape != q.shape or output.dtype != q.dtype:
        raise ValueError("output has incompatible shape/dtype")
    if lse.shape != (batch, q_heads) or lse.dtype != torch.float32:
        raise ValueError("lse has incompatible shape/dtype")

    block_d = triton.next_power_of_2(head_dim)
    group_size = q_heads // kv_heads
    _paged_gqa_decode_split_state_kernel[(batch * q_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        physical_block_indices,
        block_counts,
        tail_physical,
        tail_counts,
        partial_output,
        partial_lse,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=kv_heads,
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        INDEX_CAPACITY=physical_block_indices.shape[2],
        BLOCK_D=block_d,
        NUM_SPLITS=num_splits,
        SCALE=1.0 / math.sqrt(head_dim),
        K_SCALE=k_scale,
        V_SCALE=v_scale,
        num_warps=1,
        num_stages=2,
    )
    _merge_paged_gqa_decode_split_states_kernel[(batch * q_heads,)](
        partial_output,
        partial_lse,
        output,
        lse,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        NUM_SPLITS=num_splits,
        num_warps=1,
    )
    return output, lse

