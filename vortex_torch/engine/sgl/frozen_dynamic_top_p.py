"""Formal Vortex adapter for the frozen Dynamic Top-P decode formulation.

This module is opt-in through the registered ``frozen_dynamic_top_p`` vFlow.
It deliberately contains no tunable formulation defaults: model temperature
is checked against the pre-calibrated policy and every other algorithm field
is a constant.  The current implementation reuses the already parity-tested
Triton kernels while the runtime owns paging, cache lifetime, and dense
fallback decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


BLOCK_SIZE = 16
CLUSTERS = 4
TOP_P = 0.9
CAP_FRACTION = 0.75
BLACKWELL_ATTENTION_SPLITS = 8
MAX_LOGICAL_BLOCKS = 4096

_TEMPERATURE_POLICIES = {
    "qwen3-1.7b": 1.1,
    "qwen3-4b": 1.1,
    "phi-4-mini": 1.2,
}


@dataclass(frozen=True)
class FrozenDecodeMetadata:
    logical_to_physical: torch.Tensor
    completed_counts: torch.Tensor
    tail_physical: torch.Tensor
    tail_counts: torch.Tensor


@dataclass
class FrozenDecodeWorkspace:
    """Stable-address selector/attention buffers shared by CUDA graphs."""

    logical_block_starts: torch.Tensor
    logical_to_physical: torch.Tensor
    completed_counts: torch.Tensor
    tail_physical: torch.Tensor
    tail_counts: torch.Tensor
    per_q_scores: torch.Tensor
    shared_scores: torch.Tensor
    logical_indices: torch.Tensor
    physical_indices: torch.Tensor
    selected_counts: torch.Tensor
    selected_output: torch.Tensor
    selected_lse: torch.Tensor
    tail_output: torch.Tensor
    tail_lse: torch.Tensor
    split_partial_output: torch.Tensor
    split_partial_lse: torch.Tensor
    merged_output: torch.Tensor

    @property
    def max_batch(self) -> int:
        return self.logical_to_physical.shape[0]

    @property
    def max_blocks(self) -> int:
        return self.logical_to_physical.shape[1]


def allocate_decode_workspace(
    *,
    max_batch: int,
    max_blocks: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> FrozenDecodeWorkspace:
    """Allocate every shape-bearing decode tensor before graph capture."""

    if max_batch < 1 or not 1 <= max_blocks <= MAX_LOGICAL_BLOCKS:
        raise ValueError("invalid frozen decode workspace capacity")
    if q_heads % kv_heads or dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("invalid frozen decode workspace geometry")
    selection_capacity = min(
        max_blocks, max(2, math.ceil(max_blocks * CAP_FRACTION))
    )
    return FrozenDecodeWorkspace(
        logical_block_starts=(
            torch.arange(max_blocks, device=device, dtype=torch.int64) * BLOCK_SIZE
        ),
        logical_to_physical=torch.zeros(
            (max_batch, max_blocks), device=device, dtype=torch.int32
        ),
        completed_counts=torch.zeros(max_batch, device=device, dtype=torch.int32),
        tail_physical=torch.zeros(max_batch, device=device, dtype=torch.int32),
        tail_counts=torch.ones(max_batch, device=device, dtype=torch.int32),
        per_q_scores=torch.empty(
            (max_batch, q_heads, max_blocks), device=device, dtype=torch.float32
        ),
        shared_scores=torch.empty(
            (max_batch, kv_heads, max_blocks), device=device, dtype=torch.float32
        ),
        logical_indices=torch.empty(
            (max_batch, kv_heads, selection_capacity),
            device=device,
            dtype=torch.int32,
        ),
        physical_indices=torch.empty(
            (max_batch, kv_heads, selection_capacity),
            device=device,
            dtype=torch.int32,
        ),
        selected_counts=torch.empty(
            (max_batch, kv_heads), device=device, dtype=torch.int32
        ),
        selected_output=torch.empty(
            (max_batch, q_heads, head_dim), device=device, dtype=dtype
        ),
        selected_lse=torch.empty(
            (max_batch, q_heads), device=device, dtype=torch.float32
        ),
        tail_output=torch.empty(
            (max_batch, q_heads, head_dim), device=device, dtype=dtype
        ),
        tail_lse=torch.empty(
            (max_batch, q_heads), device=device, dtype=torch.float32
        ),
        split_partial_output=torch.empty(
            (
                max_batch,
                q_heads,
                BLACKWELL_ATTENTION_SPLITS,
                head_dim,
            ),
            device=device,
            dtype=torch.float32,
        ),
        split_partial_lse=torch.empty(
            (max_batch, q_heads, BLACKWELL_ATTENTION_SPLITS),
            device=device,
            dtype=torch.float32,
        ),
        merged_output=torch.empty(
            (max_batch, q_heads, head_dim), device=device, dtype=dtype
        ),
    )


def _head_packed_view(tensor: torch.Tensor, num_kv_heads: int, *tail: int):
    """Drop Vortex's final allocator sentinel before page/head reshaping."""

    usable_streams = (tensor.shape[0] // num_kv_heads) * num_kv_heads
    return tensor[:usable_streams].view(-1, num_kv_heads, *tail)


def validate_frozen_runtime(
    *,
    model_path: str,
    temperature: float | None,
    page_size: int,
    block_size: int,
    kv_dtype: torch.dtype,
) -> float:
    """Fail closed if runtime flags would change the frozen formulation."""

    normalized = model_path.lower()
    expected = next(
        (value for key, value in _TEMPERATURE_POLICIES.items() if key in normalized),
        None,
    )
    if expected is None:
        raise ValueError(
            "frozen_dynamic_top_p has no pre-calibrated temperature for "
            f"model {model_path!r}"
        )
    if temperature is None or not math.isclose(temperature, expected, abs_tol=0.0):
        raise ValueError(
            f"{model_path!r} requires the frozen temperature {expected}; "
            f"received {temperature!r}"
        )
    if block_size != BLOCK_SIZE or page_size != BLOCK_SIZE:
        raise ValueError(
            "formal frozen_dynamic_top_p currently requires "
            "page_size == block_size == 16"
        )
    supported_kv_dtypes = (
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if kv_dtype not in supported_kv_dtypes:
        raise ValueError(
            "formal frozen_dynamic_top_p requires BF16, FP8 E4M3, or FP8 E5M2 KV cache"
        )
    return expected


def resolve_layer_kv_scales(layer) -> tuple[float, float]:
    """Return stable Python scales for FP8 writes and dequantized attention.

    SGLang materializes ``k_scale_float``/``v_scale_float`` when a checkpoint
    supplies per-tensor KV scales.  Plain ``--kv-cache-dtype fp8_*`` models do
    not have scale parameters and use the runtime's standard unit scale.  Cache
    a scalar-tensor fallback once so decode and CUDA Graph replay never call
    ``Tensor.item()`` in the steady-state path.
    """

    resolved = []
    for name in ("k_scale", "v_scale"):
        cached_name = f"_frozen_{name}_float"
        cached = getattr(layer, cached_name, None)
        if cached is not None:
            resolved.append(cached)
            continue
        value = getattr(layer, f"{name}_float", None)
        if value is None:
            value = getattr(layer, name, None)
        if value is None:
            value = 1.0
        elif torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"frozen Dynamic Top-P requires scalar {name}")
            value = float(value.detach().cpu().item())
        else:
            value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"frozen Dynamic Top-P requires positive finite {name}")
        setattr(layer, cached_name, value)
        resolved.append(value)
    return resolved[0], resolved[1]


def build_decode_metadata(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    workspace: FrozenDecodeWorkspace | None = None,
) -> FrozenDecodeMetadata | None:
    """Map completed logical blocks and the current exact tail to cache pages.

    ``seq_lens`` includes the decode token already written to KV cache.  For a
    total length ``L``, blocks before ``(L-1)//16`` are completed probability
    candidates and block ``(L-1)//16`` is the mandatory exact tail.
    """

    if workspace is not None:
        batch = seq_lens.numel()
        if batch > workspace.max_batch:
            raise ValueError("decode batch exceeds frozen workspace capacity")
        if (workspace.max_blocks - 1) * BLOCK_SIZE >= req_to_token.shape[1]:
            raise ValueError("frozen workspace exceeds req_to_token width")
        from vortex_torch.frozen_dynamic_top_p.decode_metadata_triton import (
            build_decode_metadata_into,
        )

        logical_to_physical = workspace.logical_to_physical[:batch]
        completed_out = workspace.completed_counts[:batch]
        tail_physical = workspace.tail_physical[:batch]
        tail_counts = workspace.tail_counts[:batch]
        build_decode_metadata_into(
            req_to_token,
            req_pool_indices,
            seq_lens,
            logical_to_physical,
            completed_out,
            tail_physical,
            tail_counts,
        )
        return FrozenDecodeMetadata(
            logical_to_physical=logical_to_physical,
            completed_counts=completed_out,
            tail_physical=tail_physical,
            tail_counts=tail_counts,
        )

    total = seq_lens.to(torch.int64)
    # Device-value validation is safe on the eager dynamic-shape path.  The
    # preallocated path is entered during CUDA graph capture, where a Python
    # truth-value conversion would force a forbidden device synchronization;
    # SGLang has already validated positive decode lengths upstream.
    if workspace is None and (total < 1).any():
        raise ValueError("decode sequence lengths must be positive")
    completed = torch.div(total - 1, BLOCK_SIZE, rounding_mode="floor")
    max_completed = int(completed.max().item())
    if max_completed == 0:
        return None
    block_start = torch.arange(
        max_completed, device=total.device, dtype=torch.int64
    ) * BLOCK_SIZE
    token_locations = req_to_token[
        req_pool_indices.to(torch.int64)[:, None], block_start[None, :]
    ]
    logical_to_physical = torch.div(
        token_locations, BLOCK_SIZE, rounding_mode="floor"
    ).to(torch.int32).contiguous()
    completed_out = completed.to(torch.int32).contiguous()

    tail_start = completed * BLOCK_SIZE
    tail_token_location = req_to_token[
        req_pool_indices.to(torch.int64), tail_start
    ]
    computed_tail_physical = torch.div(
        tail_token_location, BLOCK_SIZE, rounding_mode="floor"
    ).to(torch.int32)
    computed_tail_counts = ((total - 1) % BLOCK_SIZE + 1).to(torch.int32)
    tail_physical = computed_tail_physical.contiguous()
    tail_counts = computed_tail_counts.contiguous()
    return FrozenDecodeMetadata(
        logical_to_physical=logical_to_physical,
        completed_counts=completed_out,
        tail_physical=tail_physical,
        tail_counts=tail_counts,
    )


def finalize_newly_completed_blocks(
    cache: dict[str, torch.Tensor],
    token_locations: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Build FPS-R4 + one-Lloyd summaries exactly at block boundaries."""

    completed_mask = token_locations.remainder(BLOCK_SIZE) == BLOCK_SIZE - 1
    completed_slots = torch.div(
        token_locations, BLOCK_SIZE, rounding_mode="floor"
    ).to(torch.int32).contiguous()
    if completed_slots.numel() == 0:
        return

    from vortex_torch.frozen_dynamic_top_p.fps_centroid_triton import (
        fps_r4_lloyd1_scatter_from_cache,
    )

    k_cache = _head_packed_view(
        cache["k"], num_kv_heads, BLOCK_SIZE, head_dim
    )
    centroid_cache = _head_packed_view(
        cache["centroids"], num_kv_heads, CLUSTERS, head_dim
    )
    count_cache = _head_packed_view(
        cache["centroid_counts"], num_kv_heads, CLUSTERS
    )
    fps_r4_lloyd1_scatter_from_cache(
        k_cache,
        completed_slots,
        completed_mask.contiguous(),
        centroid_cache,
        count_cache,
    )


def _formal_cache_views(
    cache: dict[str, torch.Tensor],
    *,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _head_packed_view(cache["k"], num_kv_heads, BLOCK_SIZE, head_dim),
        _head_packed_view(cache["v"], num_kv_heads, BLOCK_SIZE, head_dim),
        _head_packed_view(cache["centroids"], num_kv_heads, CLUSTERS, head_dim),
        _head_packed_view(cache["centroid_counts"], num_kv_heads, CLUSTERS),
    )


def rebuild_page_summaries(
    cache: dict[str, torch.Tensor],
    physical_pages: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Recompute frozen summaries after page relocation or host restore."""

    if physical_pages.numel() == 0:
        return
    from vortex_torch.frozen_dynamic_top_p.fps_centroid_triton import (
        fps_r4_lloyd1_scatter,
    )

    k_cache, _, centroid_cache, count_cache = _formal_cache_views(
        cache, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    pages = torch.unique(physical_pages.to(torch.int64)).to(torch.int32).contiguous()
    rebuilt_k = k_cache.index_select(0, pages.long()).contiguous()
    fps_r4_lloyd1_scatter(
        rebuilt_k,
        pages,
        centroid_cache,
        count_cache,
    )


def relocate_layer_cache(
    cache: dict[str, torch.Tensor],
    target_locations: torch.Tensor,
    source_locations: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Move token K/V in-place and rebuild metadata for touched target pages."""

    if target_locations.shape != source_locations.shape or target_locations.ndim != 1:
        raise ValueError("target/source locations must be matching vectors")
    if target_locations.dtype != torch.int64 or source_locations.dtype != torch.int64:
        raise TypeError("target/source locations must use int64")
    if not (target_locations.is_cuda and source_locations.is_cuda):
        raise ValueError("target/source locations must be CUDA tensors")
    if target_locations.numel() == 0:
        return
    if torch.unique(target_locations).numel() != target_locations.numel():
        raise ValueError("duplicate relocation targets are ambiguous")

    k_cache, v_cache, _, _ = _formal_cache_views(
        cache, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    source_page = torch.div(
        source_locations, BLOCK_SIZE, rounding_mode="floor"
    ).long()
    target_page = torch.div(
        target_locations, BLOCK_SIZE, rounding_mode="floor"
    ).long()
    source_offset = source_locations.remainder(BLOCK_SIZE).long()
    target_offset = target_locations.remainder(BLOCK_SIZE).long()
    # Clone every source before any destination write so overlapping moves and
    # page swaps have simultaneous-copy semantics.
    source_k = k_cache[source_page, :, source_offset, :].clone()
    source_v = v_cache[source_page, :, source_offset, :].clone()
    k_cache[target_page, :, target_offset, :] = source_k
    v_cache[target_page, :, target_offset, :] = source_v
    rebuild_page_summaries(
        cache,
        target_page,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def snapshot_layer_tokens(
    cache: dict[str, torch.Tensor],
    locations: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather logical token K/V; summaries are deterministic and rebuilt later."""

    k_cache, v_cache, _, _ = _formal_cache_views(
        cache, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    page = torch.div(locations, BLOCK_SIZE, rounding_mode="floor").long()
    offset = locations.remainder(BLOCK_SIZE).long()
    return k_cache[page, :, offset, :].clone(), v_cache[page, :, offset, :].clone()


def restore_layer_tokens(
    cache: dict[str, torch.Tensor],
    locations: torch.Tensor,
    key_tokens: torch.Tensor,
    value_tokens: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Restore logical token K/V and rebuild centroid/count metadata."""

    if key_tokens.shape != value_tokens.shape or key_tokens.shape != (
        locations.numel(),
        num_kv_heads,
        head_dim,
    ):
        raise ValueError("invalid offloaded K/V token shape")
    k_cache, v_cache, _, _ = _formal_cache_views(
        cache, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    page = torch.div(locations, BLOCK_SIZE, rounding_mode="floor").long()
    offset = locations.remainder(BLOCK_SIZE).long()
    k_cache[page, :, offset, :] = key_tokens.to(
        device=k_cache.device, dtype=k_cache.dtype, non_blocking=True
    )
    v_cache[page, :, offset, :] = value_tokens.to(
        device=v_cache.device, dtype=v_cache.dtype, non_blocking=True
    )
    rebuild_page_summaries(
        cache,
        page,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def _tail_attention_state(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    tail_physical: torch.Tensor,
    tail_counts: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, q_heads, head_dim = q.shape
    kv_heads = k_cache.shape[1]
    group_size = q_heads // kv_heads
    key = k_cache.index_select(0, tail_physical.long()).float()
    value = v_cache.index_select(0, tail_physical.long()).float()
    grouped_q = q.float().view(batch, kv_heads, group_size, head_dim)
    logits = torch.einsum("bhgd,bhtd->bhgt", grouped_q, key) / math.sqrt(head_dim)
    token = torch.arange(BLOCK_SIZE, device=q.device)
    valid = token[None, None, None, :] < tail_counts[:, None, None, None]
    logits = logits.masked_fill(~valid, -torch.inf)
    computed_lse = torch.logsumexp(logits, dim=-1).reshape(batch, q_heads)
    probability = torch.softmax(logits, dim=-1)
    computed_output = torch.einsum(
        "bhgt,bhtd->bhgd", probability, value
    ).reshape_as(q).to(q.dtype)
    if output is None:
        output = computed_output
    else:
        output.copy_(computed_output)
    if lse is None:
        lse = computed_lse
    else:
        lse.copy_(computed_lse)
    return output, lse


def frozen_select_and_attend(
    q: torch.Tensor,
    cache: dict[str, torch.Tensor],
    metadata: FrozenDecodeMetadata,
    *,
    num_kv_heads: int,
    temperature: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    workspace: FrozenDecodeWorkspace | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run frozen scoring, GPU Top-P, selected attention, and exact tail merge."""

    from vortex_torch.frozen_dynamic_top_p.centroid_scoring_triton import (
        centroid_per_q_logmass,
        centroid_normalized_scores,
    )
    from vortex_torch.frozen_dynamic_top_p.isolated_paged_attention_triton import (
        paged_gqa_decode_attention_with_tail,
        paged_gqa_decode_attention_with_tail_splitk,
    )
    from vortex_torch.frozen_dynamic_top_p.top_p_selection_triton import (
        dynamic_top_p_select_gpu,
        normalized_gqa_dynamic_top_p_select_gpu,
    )

    batch, q_heads, head_dim = q.shape
    if q_heads % num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    k_cache = _head_packed_view(
        cache["k"], num_kv_heads, BLOCK_SIZE, head_dim
    )
    v_cache = _head_packed_view(
        cache["v"], num_kv_heads, BLOCK_SIZE, head_dim
    )
    centroids = _head_packed_view(
        cache["centroids"], num_kv_heads, CLUSTERS, head_dim
    )
    centroid_counts = _head_packed_view(
        cache["centroid_counts"], num_kv_heads, CLUSTERS
    )
    if workspace is not None:
        if batch > workspace.max_batch:
            raise ValueError("decode batch exceeds frozen workspace capacity")
        if metadata.logical_to_physical.shape[1] != workspace.max_blocks:
            raise ValueError("metadata/workspace logical capacity mismatch")
        per_q_out = workspace.per_q_scores[:batch]
        score_out = workspace.shared_scores[:batch]
        logical_out = workspace.logical_indices[:batch]
        physical_out = workspace.physical_indices[:batch]
        count_out = workspace.selected_counts[:batch]
        selected_output_out = workspace.selected_output[:batch]
        selected_lse_out = workspace.selected_lse[:batch]
        tail_output_out = workspace.tail_output[:batch]
        tail_lse_out = workspace.tail_lse[:batch]
        split_partial_output = workspace.split_partial_output[:batch]
        split_partial_lse = workspace.split_partial_lse[:batch]
        merged_out = workspace.merged_output[:batch]
    else:
        per_q_out = score_out = None
        logical_out = physical_out = count_out = None
        selected_output_out = selected_lse_out = None
        tail_output_out = tail_lse_out = merged_out = None
        split_partial_output = split_partial_lse = None

    device_major, _ = torch.cuda.get_device_capability(q.device)
    if device_major >= 10 and workspace is not None:
        per_q_scores = centroid_per_q_logmass(
            q,
            centroids,
            centroid_counts,
            metadata.logical_to_physical,
            metadata.completed_counts,
            k_scale=k_scale,
            per_q_out=per_q_out,
        )
        (
            shared_scores,
            logical_indices,
            physical_indices,
            selected_counts,
        ) = normalized_gqa_dynamic_top_p_select_gpu(
            per_q_scores,
            metadata.logical_to_physical,
            metadata.completed_counts,
            num_kv_heads=num_kv_heads,
            temperature=temperature,
            top_p=TOP_P,
            cap_fraction=CAP_FRACTION,
            score_out=score_out,
            logical_out=logical_out,
            physical_out=physical_out,
            count_out=count_out,
        )
    else:
        _, shared_scores = centroid_normalized_scores(
            q,
            centroids,
            centroid_counts,
            metadata.logical_to_physical,
            metadata.completed_counts,
            k_scale=k_scale,
            per_q_out=per_q_out,
            score_out=score_out,
        )
        logical_indices, physical_indices, selected_counts = dynamic_top_p_select_gpu(
            shared_scores,
            metadata.logical_to_physical,
            metadata.completed_counts,
            temperature=temperature,
            top_p=TOP_P,
            cap_fraction=CAP_FRACTION,
            logical_out=logical_out,
            physical_out=physical_out,
            count_out=count_out,
        )
    if device_major >= 10 and workspace is not None:
        merged, _ = paged_gqa_decode_attention_with_tail_splitk(
            q,
            k_cache,
            v_cache,
            physical_indices,
            selected_counts,
            metadata.tail_physical,
            metadata.tail_counts,
            k_scale=k_scale,
            v_scale=v_scale,
            partial_output=split_partial_output,
            partial_lse=split_partial_lse,
            output=merged_out,
            lse=selected_lse_out,
            num_splits=BLACKWELL_ATTENTION_SPLITS,
        )
    else:
        merged, _ = paged_gqa_decode_attention_with_tail(
            q,
            k_cache,
            v_cache,
            physical_indices,
            selected_counts,
            metadata.tail_physical,
            metadata.tail_counts,
            k_scale=k_scale,
            v_scale=v_scale,
            output=merged_out,
            lse=selected_lse_out,
        )
    diagnostics = {
        "shared_scores": shared_scores,
        "logical_indices": logical_indices,
        "physical_indices": physical_indices,
        "selected_counts": selected_counts,
    }
    return merged, diagnostics

