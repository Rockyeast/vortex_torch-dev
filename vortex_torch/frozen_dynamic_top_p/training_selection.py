"""Shared frozen Dynamic Top-P selection for sparse training consumers.

VERL and Slime use different NSA attention wrappers, but rollout/training must
produce the same discrete block pattern. Keep the frozen R4 selector here so
both training stacks call one implementation.
"""

from __future__ import annotations

import math

import torch


BLOCK_SIZE = 16
TOP_P = 0.9
CAP_FRACTION = 0.75


def validate_temperature(model_name: str, temperature: float | None) -> float:
    normalized = model_name.lower()
    if "qwen3-4b" in normalized or "qwen3-1.7b" in normalized:
        expected = 1.1
    elif (
        "deepseek-r1-distill-qwen-1.5b" in normalized
        or "r1distill-qwen-1.5b" in normalized
    ):
        expected = 1.1
    elif "phi-4-mini" in normalized:
        expected = 1.2
    else:
        raise ValueError(f"no frozen Dynamic Top-P policy for {model_name!r}")
    if temperature is None or not math.isclose(temperature, expected, abs_tol=0.0):
        raise ValueError(
            f"{model_name!r} requires frozen temperature {expected}, got {temperature!r}"
        )
    return expected


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


@torch.no_grad()
def frozen_nsa_selection(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal logical block indices/counts for an NSA consumer."""

    from vortex_torch.frozen_dynamic_top_p.centroid_scoring_triton import (
        centroid_normalized_scores,
    )
    from vortex_torch.frozen_dynamic_top_p.fps_centroid_triton import fps_r4_lloyd1
    from vortex_torch.frozen_dynamic_top_p.top_p_selection_triton import (
        dynamic_top_p_select_gpu,
    )

    if q.ndim != 4 or k.ndim != 4 or q.shape[:2] != k.shape[:2]:
        raise ValueError("q/k must be [B,T,H,D] with matching B,T")
    batch, tokens, q_heads, head_dim = q.shape
    if batch != 1:
        raise ValueError("formal packed NSA integration currently requires B=1")
    kv_heads = k.shape[2]
    if q_heads % kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    if head_dim != k.shape[-1]:
        raise ValueError("invalid frozen selector geometry")
    if cu_seqlens.ndim != 1 or int(cu_seqlens[-1]) != tokens:
        raise ValueError("cu_seqlens must cover the packed token axis")

    sequence_centroids = []
    sequence_counts = []
    sequence_full_blocks = []
    ranges: list[tuple[int, int]] = []
    for seq in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[seq].item())
        end = int(cu_seqlens[seq + 1].item())
        length = end - start
        if length < 1:
            raise ValueError("packed sequences must contain at least one token")
        full_blocks = length // BLOCK_SIZE
        # Packed micro-batches can contain a short final sequence.  Represent
        # it as one zero-padded logical block for selector bookkeeping.  The
        # NSA consumer still receives the original cu_seqlens, so its causal
        # mask prevents both padded and future tokens from contributing.
        selector_blocks = max(1, full_blocks)
        block_tokens = k[0, start : start + full_blocks * BLOCK_SIZE]
        if full_blocks == 0:
            block_tokens = torch.nn.functional.pad(
                k[0, start:end],
                (0, 0, 0, 0, 0, BLOCK_SIZE - length),
            )
        blocks = (
            block_tokens
            .view(selector_blocks, BLOCK_SIZE, kv_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        centroids, counts = fps_r4_lloyd1(blocks)
        sequence_centroids.append(centroids)
        sequence_counts.append(counts)
        sequence_full_blocks.append(selector_blocks)
        ranges.append((start, end))

    centroid_cache = torch.cat(sequence_centroids, dim=0)
    count_cache = torch.cat(sequence_counts, dim=0)
    max_completed = max(sequence_full_blocks)
    logical_to_slot = torch.zeros(
        (tokens, max_completed), device=q.device, dtype=torch.int32
    )
    completed_counts = torch.empty((tokens,), device=q.device, dtype=torch.int32)
    current_blocks = torch.empty_like(completed_counts)
    slot_base = 0
    logical = torch.arange(max_completed, device=q.device, dtype=torch.int32)
    for (start, end), full_blocks in zip(ranges, sequence_full_blocks):
        length = end - start
        positions = torch.arange(length, device=q.device, dtype=torch.int32)
        current = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        # Position zero has no historical probability domain. Treat block zero
        # as its sole candidate so NSA's causal mask computes the exact block.
        completed_counts[start:end] = current.clamp_min(1)
        current_blocks[start:end] = current
        valid_slots = slot_base + logical[:full_blocks]
        logical_to_slot[start:end, :full_blocks] = valid_slots[None, :]
        slot_base += full_blocks

    q_rows = q[0].contiguous()
    _, shared_scores = centroid_normalized_scores(
        q_rows,
        centroid_cache,
        count_cache,
        logical_to_slot,
        completed_counts,
    )
    logical_indices, _, selected_counts = dynamic_top_p_select_gpu(
        shared_scores,
        logical_to_slot,
        completed_counts,
        temperature=temperature,
        top_p=TOP_P,
        cap_fraction=CAP_FRACTION,
    )

    # The current causal block is mandatory but excluded from probability
    # normalization. Append it after selection; NSA masks future tokens.
    rows, heads, capacity = logical_indices.shape
    needs_current = current_blocks > 0
    final_counts = selected_counts + needs_current[:, None].to(torch.int32)
    output_capacity = _next_power_of_two(capacity + 1)
    block_indices = torch.zeros(
        (rows, heads, output_capacity), device=q.device, dtype=torch.int32
    )
    block_indices[:, :, :capacity] = logical_indices.clamp_min(0)
    rank = selected_counts.to(torch.int64)
    block_indices.scatter_(
        2,
        rank[:, :, None],
        current_blocks[:, None, None].expand(-1, heads, 1),
    )
    return (
        block_indices.view(batch, tokens, kv_heads, output_capacity),
        final_counts.view(batch, tokens, kv_heads),
    )
