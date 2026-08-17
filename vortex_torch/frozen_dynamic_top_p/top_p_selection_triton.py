"""Production-shaped GPU Top-P selection for the frozen centroid selector.

The kernel consumes Shared-KV log scores and emits the two tensors needed by
block-sparse attention: selected block indices and per-row counts.  It also
emits the corresponding physical cache slots so the isolated paged-attention
benchmark does not need a host-side gather.

Selection semantics intentionally match ``dynamic_top_p_select``:

* FP32 softmax after dividing scores by temperature;
* stable descending order (lower logical index wins exact ties);
* logical blocks 0 and ``block_count - 1`` are always kept;
* keep the smallest candidate prefix whose cumulative mass reaches Top-P;
* cap total output at ``ceil(cap_fraction * padded_num_blocks)``.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_float_key(value, index):
    """Pack float32 value and inverse index into a stable descending key."""

    bits = value.to(tl.uint32, bitcast=True)
    sign = bits >> 31
    ordered = tl.where(sign != 0, ~bits, bits ^ 0x80000000).to(tl.uint64)
    inverse_index = (0xFFFFFFFF - index.to(tl.uint32)).to(tl.uint64)
    return (ordered << 32) | inverse_index


@triton.jit
def _decode_stable_index(key):
    return (0xFFFFFFFF - (key & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)


@triton.jit
def _dynamic_top_p_kernel(
    score_ptr,
    slot_ptr,
    block_count_ptr,
    logical_out_ptr,
    physical_out_ptr,
    count_out_ptr,
    NUM_HEADS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INV_TEMPERATURE: tl.constexpr,
    TOP_P: tl.constexpr,
    CAP_FRACTION: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NUM_HEADS
    offset = tl.arange(0, BLOCK_N)
    block_count = tl.load(block_count_ptr + batch).to(tl.int32)
    has_any = block_count > 0
    valid = (offset < NUM_BLOCKS) & (offset < block_count)

    score = tl.load(
        score_ptr + row * NUM_BLOCKS + offset,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    scaled = score * INV_TEMPERATURE
    maximum = tl.max(scaled, axis=0)
    unnormalized = tl.exp(scaled - maximum)
    unnormalized = tl.where(valid, unnormalized, 0.0)
    denominator = tl.maximum(tl.sum(unnormalized, axis=0), 1e-20)
    probability = unnormalized / denominator

    # A captured decode graph must also be replayable before the first full
    # historical block exists.  Keep all gathers in-bounds for that row and
    # suppress the mandatory stores/count below; the exact current tail then
    # becomes the complete attention result.
    last = tl.maximum(block_count - 1, 0)
    mandatory = valid & ((offset == 0) | (offset == last))
    forced_count = tl.sum(mandatory.to(tl.int32), axis=0)
    forced_mass = tl.sum(tl.where(mandatory, probability, 0.0), axis=0)

    candidate_probability = tl.where(valid & ~mandatory, probability, -1.0)
    candidate_score = tl.where(valid & ~mandatory, scaled, -float("inf"))
    candidate_key = _ordered_float_key(candidate_score, offset)
    sorted_candidate_key = tl.sort(candidate_key, dim=0, descending=True)
    sorted_candidate_index = _decode_stable_index(sorted_candidate_key)
    sorted_candidate_probability = tl.gather(
        candidate_probability, sorted_candidate_index, axis=0
    )
    sorted_candidate_probability = tl.maximum(sorted_candidate_probability, 0.0)
    cumulative_before = (
        tl.cumsum(sorted_candidate_probability, axis=0)
        - sorted_candidate_probability
    )
    remaining_mass = tl.maximum(TOP_P - forced_mass, 0.0)
    nucleus = (
        (cumulative_before < remaining_mass)
        & (sorted_candidate_probability > 0.0)
        & (offset < NUM_BLOCKS)
    )
    requested_candidates = tl.sum(nucleus.to(tl.int32), axis=0)
    candidate_count = tl.sum((valid & ~mandatory).to(tl.int32), axis=0)
    requested_candidates = tl.minimum(requested_candidates, candidate_count)
    # CAPACITY is the padded output width.  The semantic safety cap is
    # per-request, so shorter rows in a mixed-length batch must not inherit
    # the longest row's budget.
    row_capacity = tl.ceil(block_count.to(tl.float32) * CAP_FRACTION).to(tl.int32)
    row_capacity = tl.maximum(2, tl.minimum(row_capacity, CAPACITY))
    requested_candidates = tl.minimum(
        requested_candidates, row_capacity - forced_count
    )
    requested_count = forced_count + requested_candidates

    # The candidates are already in stable score order.  Merge the at-most-two
    # mandatory blocks into that prefix instead of paying for a second full
    # bitonic sort.
    output_rank = offset
    within_capacity = output_rank < CAPACITY
    tl.store(
        logical_out_ptr + row * CAPACITY + output_rank,
        -1,
        mask=within_capacity,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + output_rank,
        -1,
        mask=within_capacity,
    )

    first_index = tl.arange(0, 1)
    last_index = last + tl.arange(0, 1)
    first_score = tl.gather(scaled, first_index, axis=0)
    last_score = tl.gather(scaled, last_index, axis=0)
    first_key = _ordered_float_key(first_score, first_index)
    last_key = _ordered_float_key(last_score, last_index)
    has_distinct_last = has_any & (last != 0)

    candidate_output_valid = offset < requested_candidates
    mandatory_before_candidate = (first_key > sorted_candidate_key).to(tl.int32)
    mandatory_before_candidate += (
        has_distinct_last & (last_key > sorted_candidate_key)
    ).to(tl.int32)
    candidate_output_rank = offset + mandatory_before_candidate
    candidate_physical = tl.load(
        slot_ptr + batch * NUM_BLOCKS + sorted_candidate_index,
        mask=candidate_output_valid,
        other=-1,
    ).to(tl.int32)
    tl.store(
        logical_out_ptr + row * CAPACITY + candidate_output_rank,
        sorted_candidate_index,
        mask=candidate_output_valid,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + candidate_output_rank,
        candidate_physical,
        mask=candidate_output_valid,
    )

    first_output_rank = tl.sum(
        (candidate_output_valid & (sorted_candidate_key > first_key)).to(tl.int32),
        axis=0,
    ) + (has_distinct_last & (last_key > first_key)).to(tl.int32)
    last_output_rank = tl.sum(
        (candidate_output_valid & (sorted_candidate_key > last_key)).to(tl.int32),
        axis=0,
    ) + (first_key > last_key).to(tl.int32)
    first_physical = tl.load(slot_ptr + batch * NUM_BLOCKS).to(tl.int32)
    last_physical = tl.load(slot_ptr + batch * NUM_BLOCKS + last).to(tl.int32)
    tl.store(
        logical_out_ptr + row * CAPACITY + first_output_rank,
        0,
        mask=has_any,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + first_output_rank,
        first_physical,
        mask=has_any,
    )
    tl.store(
        logical_out_ptr + row * CAPACITY + last_output_rank,
        last,
        mask=has_distinct_last,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + last_output_rank,
        last_physical,
        mask=has_distinct_last,
    )
    tl.store(count_out_ptr + row, requested_count)


@triton.jit
def _normalized_gqa_dynamic_top_p_kernel(
    per_q_ptr,
    slot_ptr,
    block_count_ptr,
    score_out_ptr,
    logical_out_ptr,
    physical_out_ptr,
    count_out_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INV_TEMPERATURE: tl.constexpr,
    TOP_P: tl.constexpr,
    CAP_FRACTION: tl.constexpr,
):
    """Fuse frozen per-Q normalization/GQA mean with stable Top-P."""

    row = tl.program_id(0)
    batch = row // NUM_KV_HEADS
    kv_head = row % NUM_KV_HEADS
    offset = tl.arange(0, BLOCK_N)
    block_count = tl.load(block_count_ptr + batch).to(tl.int32)
    has_any = block_count > 0
    valid = (offset < NUM_BLOCKS) & (offset < block_count)

    shared_probability = tl.zeros((BLOCK_N,), tl.float32)
    for group in tl.static_range(0, GROUP_SIZE):
        q_head = kv_head * GROUP_SIZE + group
        q_row = batch * NUM_Q_HEADS + q_head
        logmass = tl.load(
            per_q_ptr + q_row * NUM_BLOCKS + offset,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(logmass, axis=0)
        unnormalized = tl.where(valid, tl.exp(logmass - maximum), 0.0)
        denominator = tl.maximum(tl.sum(unnormalized, axis=0), 1e-20)
        shared_probability += unnormalized / denominator
    shared_probability *= 1.0 / GROUP_SIZE
    score = tl.log(shared_probability)
    score = tl.where(valid, score, -float("inf"))
    tl.store(
        score_out_ptr + row * NUM_BLOCKS + offset,
        score,
        mask=offset < NUM_BLOCKS,
    )

    scaled = score * INV_TEMPERATURE
    maximum = tl.max(scaled, axis=0)
    unnormalized = tl.where(valid, tl.exp(scaled - maximum), 0.0)
    denominator = tl.maximum(tl.sum(unnormalized, axis=0), 1e-20)
    probability = unnormalized / denominator

    last = tl.maximum(block_count - 1, 0)
    mandatory = valid & ((offset == 0) | (offset == last))
    forced_count = tl.sum(mandatory.to(tl.int32), axis=0)
    forced_mass = tl.sum(tl.where(mandatory, probability, 0.0), axis=0)

    candidate_probability = tl.where(valid & ~mandatory, probability, -1.0)
    candidate_score = tl.where(valid & ~mandatory, scaled, -float("inf"))
    candidate_key = _ordered_float_key(candidate_score, offset)
    sorted_candidate_key = tl.sort(candidate_key, dim=0, descending=True)
    sorted_candidate_index = _decode_stable_index(sorted_candidate_key)
    sorted_candidate_probability = tl.gather(
        candidate_probability, sorted_candidate_index, axis=0
    )
    sorted_candidate_probability = tl.maximum(sorted_candidate_probability, 0.0)
    cumulative_before = (
        tl.cumsum(sorted_candidate_probability, axis=0)
        - sorted_candidate_probability
    )
    remaining_mass = tl.maximum(TOP_P - forced_mass, 0.0)
    nucleus = (
        (cumulative_before < remaining_mass)
        & (sorted_candidate_probability > 0.0)
        & (offset < NUM_BLOCKS)
    )
    requested_candidates = tl.sum(nucleus.to(tl.int32), axis=0)
    candidate_count = tl.sum((valid & ~mandatory).to(tl.int32), axis=0)
    requested_candidates = tl.minimum(requested_candidates, candidate_count)
    row_capacity = tl.ceil(block_count.to(tl.float32) * CAP_FRACTION).to(tl.int32)
    row_capacity = tl.maximum(2, tl.minimum(row_capacity, CAPACITY))
    requested_candidates = tl.minimum(
        requested_candidates, row_capacity - forced_count
    )
    requested_count = forced_count + requested_candidates

    output_rank = offset
    within_capacity = output_rank < CAPACITY
    tl.store(
        logical_out_ptr + row * CAPACITY + output_rank,
        -1,
        mask=within_capacity,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + output_rank,
        -1,
        mask=within_capacity,
    )

    first_index = tl.arange(0, 1)
    last_index = last + tl.arange(0, 1)
    first_score = tl.gather(scaled, first_index, axis=0)
    last_score = tl.gather(scaled, last_index, axis=0)
    first_key = _ordered_float_key(first_score, first_index)
    last_key = _ordered_float_key(last_score, last_index)
    has_distinct_last = has_any & (last != 0)

    candidate_output_valid = offset < requested_candidates
    mandatory_before_candidate = (first_key > sorted_candidate_key).to(tl.int32)
    mandatory_before_candidate += (
        has_distinct_last & (last_key > sorted_candidate_key)
    ).to(tl.int32)
    candidate_output_rank = offset + mandatory_before_candidate
    candidate_physical = tl.load(
        slot_ptr + batch * NUM_BLOCKS + sorted_candidate_index,
        mask=candidate_output_valid,
        other=-1,
    ).to(tl.int32)
    tl.store(
        logical_out_ptr + row * CAPACITY + candidate_output_rank,
        sorted_candidate_index,
        mask=candidate_output_valid,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + candidate_output_rank,
        candidate_physical,
        mask=candidate_output_valid,
    )

    first_output_rank = tl.sum(
        (candidate_output_valid & (sorted_candidate_key > first_key)).to(tl.int32),
        axis=0,
    ) + (has_distinct_last & (last_key > first_key)).to(tl.int32)
    last_output_rank = tl.sum(
        (candidate_output_valid & (sorted_candidate_key > last_key)).to(tl.int32),
        axis=0,
    ) + (first_key > last_key).to(tl.int32)
    first_physical = tl.load(slot_ptr + batch * NUM_BLOCKS).to(tl.int32)
    last_physical = tl.load(slot_ptr + batch * NUM_BLOCKS + last).to(tl.int32)
    tl.store(
        logical_out_ptr + row * CAPACITY + first_output_rank,
        0,
        mask=has_any,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + first_output_rank,
        first_physical,
        mask=has_any,
    )
    tl.store(
        logical_out_ptr + row * CAPACITY + last_output_rank,
        last,
        mask=has_distinct_last,
    )
    tl.store(
        physical_out_ptr + row * CAPACITY + last_output_rank,
        last_physical,
        mask=has_distinct_last,
    )
    tl.store(count_out_ptr + row, requested_count)


def top_p_capacity(num_blocks: int, cap_fraction: float = 0.75) -> int:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if not 0.0 < cap_fraction <= 1.0:
        raise ValueError("cap_fraction must be in (0, 1]")
    return min(num_blocks, max(2, math.ceil(num_blocks * cap_fraction)))


def dynamic_top_p_select_gpu(
    scores: torch.Tensor,
    logical_to_slot: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    temperature: float = 1.1,
    top_p: float = 0.9,
    cap_fraction: float = 0.75,
    logical_out: torch.Tensor | None = None,
    physical_out: torch.Tensor | None = None,
    count_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logical indices, physical slots, and counts on CUDA.

    ``scores`` is ``[B,Hkv,N]`` FP32 and ``logical_to_slot`` is ``[B,N]``.
    Index outputs are ``[B,Hkv,ceil(0.75*N)]`` int32, padded with ``-1``;
    counts are ``[B,Hkv]`` int32.
    """

    if not (scores.is_cuda and logical_to_slot.is_cuda and block_counts.is_cuda):
        raise ValueError("all inputs must be CUDA tensors")
    if not (
        scores.is_contiguous()
        and logical_to_slot.is_contiguous()
        and block_counts.is_contiguous()
    ):
        raise ValueError("all inputs must be contiguous")
    if scores.ndim != 3 or scores.dtype != torch.float32:
        raise ValueError("scores must be contiguous FP32 [B,Hkv,N]")
    batch, heads, num_blocks = scores.shape
    if logical_to_slot.shape != (batch, num_blocks):
        raise ValueError("logical_to_slot must be [B,N]")
    if logical_to_slot.dtype not in (torch.int32, torch.int64):
        raise ValueError("logical_to_slot must use int32 or int64")
    if block_counts.shape != (batch,) or block_counts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("block_counts must be int32/int64 [B]")
    if num_blocks > 4096:
        raise ValueError("this isolated selector supports at most 4096 blocks")
    if not 0.0 < temperature:
        raise ValueError("temperature must be positive")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    capacity = top_p_capacity(num_blocks, cap_fraction)
    output_shape = (batch, heads, capacity)
    if logical_out is None:
        logical_out = torch.empty(
            output_shape, device=scores.device, dtype=torch.int32
        )
    if physical_out is None:
        physical_out = torch.empty(
            output_shape, device=scores.device, dtype=torch.int32
        )
    if count_out is None:
        count_out = torch.empty(
            (batch, heads), device=scores.device, dtype=torch.int32
        )
    for name, tensor in (
        ("logical_out", logical_out),
        ("physical_out", physical_out),
    ):
        if (
            tensor.shape != output_shape
            or tensor.device != scores.device
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} has incompatible shape/device/dtype/layout")
    if (
        count_out.shape != (batch, heads)
        or count_out.device != scores.device
        or count_out.dtype != torch.int32
        or not count_out.is_contiguous()
    ):
        raise ValueError("count_out has incompatible shape/device/dtype/layout")

    block_n = triton.next_power_of_2(num_blocks)
    _dynamic_top_p_kernel[(batch * heads,)](
        scores,
        logical_to_slot,
        block_counts,
        logical_out,
        physical_out,
        count_out,
        NUM_HEADS=heads,
        NUM_BLOCKS=num_blocks,
        CAPACITY=capacity,
        BLOCK_N=block_n,
        INV_TEMPERATURE=1.0 / temperature,
        TOP_P=top_p,
        CAP_FRACTION=cap_fraction,
        num_warps=8 if block_n >= 1024 else 4,
    )
    return logical_out, physical_out, count_out


def normalized_gqa_dynamic_top_p_select_gpu(
    per_q_logmass: torch.Tensor,
    logical_to_slot: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    num_kv_heads: int,
    temperature: float = 1.1,
    top_p: float = 0.9,
    cap_fraction: float = 0.75,
    score_out: torch.Tensor,
    logical_out: torch.Tensor,
    physical_out: torch.Tensor,
    count_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse frozen per-Q normalization/GQA aggregation and Top-P selection."""

    tensors = (
        per_q_logmass,
        logical_to_slot,
        block_counts,
        score_out,
        logical_out,
        physical_out,
        count_out,
    )
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all fused normalization/Top-P tensors must be contiguous CUDA")
    if per_q_logmass.ndim != 3 or per_q_logmass.dtype != torch.float32:
        raise ValueError("per_q_logmass must be FP32 [B,Hq,N]")
    batch, q_heads, num_blocks = per_q_logmass.shape
    if num_kv_heads <= 0 or q_heads % num_kv_heads:
        raise ValueError("Q heads must be divisible by num_kv_heads")
    if logical_to_slot.shape != (batch, num_blocks):
        raise ValueError("logical_to_slot must be [B,N]")
    if block_counts.shape != (batch,):
        raise ValueError("block_counts must be [B]")
    integer_types = (torch.int32, torch.int64)
    if logical_to_slot.dtype not in integer_types or block_counts.dtype not in integer_types:
        raise ValueError("mapping/counts must use int32/int64")
    if not 1 <= num_blocks <= 4096:
        raise ValueError("fused normalization/Top-P supports 1..4096 blocks")
    if not 0.0 < temperature or not 0.0 < top_p <= 1.0:
        raise ValueError("invalid frozen temperature/Top-P")

    capacity = top_p_capacity(num_blocks, cap_fraction)
    if score_out.shape != (batch, num_kv_heads, num_blocks) or score_out.dtype != torch.float32:
        raise ValueError("score_out must be FP32 [B,Hkv,N]")
    output_shape = (batch, num_kv_heads, capacity)
    for name, tensor in (
        ("logical_out", logical_out),
        ("physical_out", physical_out),
    ):
        if tensor.shape != output_shape or tensor.dtype != torch.int32:
            raise ValueError(f"{name} has incompatible shape/dtype")
    if count_out.shape != (batch, num_kv_heads) or count_out.dtype != torch.int32:
        raise ValueError("count_out must be int32 [B,Hkv]")

    block_n = triton.next_power_of_2(num_blocks)
    _normalized_gqa_dynamic_top_p_kernel[(batch * num_kv_heads,)](
        per_q_logmass,
        logical_to_slot,
        block_counts,
        score_out,
        logical_out,
        physical_out,
        count_out,
        NUM_Q_HEADS=q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=q_heads // num_kv_heads,
        NUM_BLOCKS=num_blocks,
        CAPACITY=capacity,
        BLOCK_N=block_n,
        INV_TEMPERATURE=1.0 / temperature,
        TOP_P=top_p,
        CAP_FRACTION=cap_fraction,
        num_warps=8 if block_n >= 1024 else 4,
    )
    return score_out, logical_out, physical_out, count_out

