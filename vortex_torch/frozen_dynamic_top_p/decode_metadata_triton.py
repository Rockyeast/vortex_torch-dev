"""Capture-safe logical-page metadata construction for frozen decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _build_decode_metadata_kernel(
    req_to_token_ptr,
    req_pool_index_ptr,
    seq_len_ptr,
    logical_to_physical_ptr,
    completed_count_ptr,
    tail_physical_ptr,
    tail_count_ptr,
    REQUEST_STRIDE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    logical = tl.arange(0, BLOCK_N)
    request = tl.load(req_pool_index_ptr + batch).to(tl.int64)
    total = tl.load(seq_len_ptr + batch).to(tl.int64)
    completed = (total - 1) // 16
    valid_logical = (logical < MAX_BLOCKS) & (logical < completed)
    token_location = tl.load(
        req_to_token_ptr + request * REQUEST_STRIDE + logical * 16,
        mask=valid_logical,
        other=0,
    ).to(tl.int64)
    physical = token_location // 16
    tl.store(
        logical_to_physical_ptr + batch * MAX_BLOCKS + logical,
        physical,
        mask=logical < MAX_BLOCKS,
    )

    scalar = tl.arange(0, 1)
    tail_token_location = tl.load(
        req_to_token_ptr + request * REQUEST_STRIDE + completed * 16 + scalar
    ).to(tl.int64)
    tl.store(completed_count_ptr + batch + scalar, completed)
    tl.store(tail_physical_ptr + batch + scalar, tail_token_location // 16)
    tl.store(tail_count_ptr + batch + scalar, (total - 1) % 16 + 1)


def build_decode_metadata_into(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    logical_to_physical: torch.Tensor,
    completed_counts: torch.Tensor,
    tail_physical: torch.Tensor,
    tail_counts: torch.Tensor,
) -> None:
    """Fill preallocated frozen metadata with one CUDA kernel launch."""

    inputs = (req_to_token, req_pool_indices, seq_lens)
    outputs = (logical_to_physical, completed_counts, tail_physical, tail_counts)
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in inputs + outputs):
        raise ValueError("decode metadata tensors must be contiguous CUDA tensors")
    if req_to_token.ndim != 2 or req_pool_indices.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("invalid request-table metadata ranks")
    batch, max_blocks = logical_to_physical.shape
    if req_pool_indices.shape != (batch,) or seq_lens.shape != (batch,):
        raise ValueError("request indices and sequence lengths must match batch")
    if any(tensor.shape != (batch,) for tensor in outputs[1:]):
        raise ValueError("metadata vector outputs must match batch")
    integer_types = (torch.int32, torch.int64)
    if any(tensor.dtype not in integer_types for tensor in inputs):
        raise TypeError("request table/indices/lengths must use int32 or int64")
    if logical_to_physical.dtype != torch.int32 or any(
        tensor.dtype != torch.int32 for tensor in outputs[1:]
    ):
        raise TypeError("frozen metadata outputs must use int32")
    if not 1 <= max_blocks <= 4096:
        raise ValueError("max_blocks must be in [1, 4096]")

    block_n = triton.next_power_of_2(max_blocks)
    _build_decode_metadata_kernel[(batch,)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        logical_to_physical,
        completed_counts,
        tail_physical,
        tail_counts,
        REQUEST_STRIDE=req_to_token.stride(0),
        MAX_BLOCKS=max_blocks,
        BLOCK_N=block_n,
        num_warps=8 if block_n >= 1024 else 4,
    )

