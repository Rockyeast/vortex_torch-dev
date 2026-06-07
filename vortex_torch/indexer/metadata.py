"""indexer 每个 forward batch 的运行时 metadata。

:class:`MetaData` 对象持有所有依赖 *当前* forward batch 的 tensor / 标量：
planner 在 ``init_forward_metadata`` 期间写入这些字段，编译后的 indexer kernel
在热路径上读取这些字段。它和 :class:`Context` 是分开的；:class:`Context`
只保存静态配置，例如 page/block 大小、head 数、分配预算、graph metadata、
codegen 草稿纸等。

预分配通过 :func:`MetaData.preallocate` 在 attention-backend ``__init__`` 阶段
执行一次。之后每个 forward batch 都复用同一批 buffer 对象
（CUDA graph 友好：指针地址不变，只改内容）。

布局摘要：

  * ``winfo_*``：workload scheduler 的输出；per-workload，长度为
    ``ctx.max_num_workloads``。
  * ``dense_kv_indptr`` / ``sparse_kv_indptr``：CSR 前缀和，长度为
    ``max_bs * num_kv_heads + 1``。trtllm 模式下为 ``None``。
  * ``dense_kv_indices`` / ``sparse_kv_indices``：扁平 CSR block id，长度为
    ``max_bs * num_kv_heads * max_blocks_per_seq``。trtllm 模式下为 ``None``。
  * ``dense_seqlens`` / ``sparse_seqlens``：每行 token 数，长度为
    ``max_bs * num_kv_heads``。flashinfer 模式下为 ``None``。
  * ``dense_block_tables`` / ``sparse_block_tables``：
    ``[max_bs * num_kv_heads, max_blocks_per_seq]`` 形状的 block id 表。
    flashinfer 模式下为 ``None``。
  * ``kv_last_page_len``：每行最后一个 page 的 token 数，长度为
    ``max_bs * num_kv_heads``。
  * ``batch_size``：当前请求数量，由每次 planner 调用设置。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .context import Context


class MetaData:
    """indexer 的每个 forward batch 状态。详见模块 docstring。"""

    __slots__ = (
        # workload scheduler 输出
        "winfo_q_indices",
        "winfo_is_first_workload_per_batch",
        "winfo_kv_offsets",
        "winfo_kv_lens",
        "winfo_num_workloads",
        "winfo_chunk_size",
        # CSR（flashinfer）buffer；backend == "trtllm" 时为 None
        "dense_kv_indptr",
        "sparse_kv_indptr",
        "dense_kv_indices",
        "sparse_kv_indices",
        # block-table（trtllm）buffer；backend == "flashinfer" 时为 None
        "dense_seqlens",
        "sparse_seqlens",
        "dense_block_tables",
        "sparse_block_tables",
        # 两种 backend 都会使用
        "kv_last_page_len",
        "batch_size",
    )

    # 类型标注（只提供信息；具体值由 ``preallocate`` 填充）。
    winfo_q_indices: torch.Tensor
    winfo_is_first_workload_per_batch: torch.Tensor
    winfo_kv_offsets: torch.Tensor
    winfo_kv_lens: torch.Tensor
    winfo_num_workloads: torch.Tensor
    winfo_chunk_size: torch.Tensor

    dense_kv_indptr: Optional[torch.Tensor]
    sparse_kv_indptr: Optional[torch.Tensor]
    dense_kv_indices: Optional[torch.Tensor]
    sparse_kv_indices: Optional[torch.Tensor]

    dense_seqlens: Optional[torch.Tensor]
    sparse_seqlens: Optional[torch.Tensor]
    dense_block_tables: Optional[torch.Tensor]
    sparse_block_tables: Optional[torch.Tensor]

    kv_last_page_len: torch.Tensor
    batch_size: int

    def __init__(self) -> None:
        for name in self.__slots__:
            object.__setattr__(
                self, name, 0 if name == "batch_size" else None
            )

    # ------------------------------------------------------------------
    # 预分配入口（按 attention backend 选择具体 buffer 集合）
    # ------------------------------------------------------------------
    @classmethod
    def preallocate(
        cls,
        ctx: "Context",
        *,
        device: torch.device | str,
    ) -> "MetaData":
        """根据已经填好的 :class:`Context` 构建适配当前 backend 的 ``MetaData``。

        根据 ``ctx.vortex_attention_backend`` 选择 buffer 集合：
          * ``"flashinfer"`` -> 分配 CSR buffer
            （``dense/sparse_kv_indptr``、``dense/sparse_kv_indices``）；
            block-table buffer 保持为 ``None``。
          * ``"trtllm"`` -> 分配 2D block_tables 和 per-row seqlens；
            CSR buffer 保持为 ``None``。

        ``winfo_*`` 和 ``kv_last_page_len`` 总是会分配。
        """
        md = cls()
        md._alloc_common(ctx, device=device)

        backend = (ctx.vortex_attention_backend or "flashinfer").lower()
        if backend == "trtllm":
            md._alloc_trtllm(ctx, device=device)
        elif backend == "flashinfer":
            md._alloc_flashinfer(ctx, device=device)
        else:
            raise ValueError(
                f"MetaData.preallocate: unknown vortex_attention_backend "
                f"{ctx.vortex_attention_backend!r}; expected 'flashinfer' or 'trtllm'"
            )
        return md

    # ------------------------------------------------------------------
    def _alloc_common(self, ctx: "Context", *, device) -> None:
        eff_bs = ctx.max_bs * ctx.num_kv_heads
        i32 = torch.int32
        u8 = torch.uint8

        self.winfo_q_indices = torch.zeros(
            (ctx.max_num_workloads,), dtype=i32, device=device,
        )
        self.winfo_is_first_workload_per_batch = torch.zeros(
            (ctx.max_num_workloads,), dtype=u8, device=device,
        )
        self.winfo_kv_offsets = torch.zeros(
            (ctx.max_num_workloads,), dtype=i32, device=device,
        )
        self.winfo_kv_lens = torch.zeros(
            (ctx.max_num_workloads,), dtype=i32, device=device,
        )
        self.winfo_num_workloads = torch.zeros((1,), dtype=i32, device=device)
        self.winfo_chunk_size = torch.zeros((1,), dtype=i32, device=device)

        # 每种 backend 的 planner 都会写每行最后一个 page/block 的 token 长度。
        self.kv_last_page_len = torch.ones((eff_bs,), dtype=i32, device=device)

    def _alloc_flashinfer(self, ctx: "Context", *, device) -> None:
        eff_bs = ctx.max_bs * ctx.num_kv_heads
        # CSR ``kv_indices`` 长度：完整请求预算内，每个 cached block 对应一个
        # int32。这里使用 ``max_num_blocks``，它已经是 planner 的分配预算，
        # 也和之前的 backend wiring 保持一致。
        indices_len = ctx.max_num_blocks
        i32 = torch.int32

        self.dense_kv_indptr = torch.zeros((eff_bs + 1,), dtype=i32, device=device)
        self.sparse_kv_indptr = torch.zeros((eff_bs + 1,), dtype=i32, device=device)
        self.dense_kv_indices = torch.zeros((indices_len,), dtype=i32, device=device)
        self.sparse_kv_indices = torch.zeros((indices_len,), dtype=i32, device=device)
        # flashinfer 模式下，block-table buffer 保持为 ``None``。

    def _alloc_trtllm(self, ctx: "Context", *, device) -> None:
        eff_bs = ctx.max_bs * ctx.num_kv_heads
        max_blocks_per_seq = ctx.max_num_blocks_per_request
        i32 = torch.int32

        self.dense_seqlens = torch.zeros((eff_bs,), dtype=i32, device=device)
        self.sparse_seqlens = torch.zeros((eff_bs,), dtype=i32, device=device)
        self.dense_block_tables = torch.zeros(
            (eff_bs, max_blocks_per_seq), dtype=i32, device=device,
        )
        self.sparse_block_tables = torch.zeros(
            (eff_bs, max_blocks_per_seq), dtype=i32, device=device,
        )
        # trtllm 模式下，CSR buffer 保持为 ``None``。

    # ------------------------------------------------------------------
    def set_batch_size(self, n: int) -> None:
        self.batch_size = n


__all__ = ["MetaData"]
