from __future__ import annotations

from typing import Any, Final, Optional, Union
import uuid

import torch

from ..abs import ContextBase
from ..utils import UNSET, Mode, resolve_dtype
from .metadata import MetaData


class Context(ContextBase):
    """静态、单实例的 indexer 上下文。

    这里只保存编译后 indexer 生命周期内固定不变的配置：形状、page/block 大小、
    head 数、分配预算、codegen 草稿纸（op/tensor 列表）、backend 身份等。

    每个 forward batch 会变化的 buffer（winfo_*、dense/sparse_kv_indptr+indices、
    dense/sparse_seqlens、dense/sparse_block_tables、kv_last_page_len）以及
    ``batch_size`` 放在单独的 :class:`MetaData` 对象里。这个对象会在
    attention-backend 的 ``__init__`` 阶段预分配，并通过 ``ctx.metadata`` 暴露。
    codegen 遇到每个 batch 都会变化的值时，会生成 ``ctx.metadata.<field>``。

    构建模式（来自各 attention backend 的 ``_compile``）:
        ctx = Context()
        ctx.create(self, model_runner)              # 静态字段
        ctx.metadata = MetaData.preallocate(ctx, device=...)
    """

    __slots__ = ContextBase.__slots__ + (
        # ---- 每个 forward batch 的状态（预分配 MetaData）----
        "metadata",
        # ---- batch 预算 ----
        "max_bs",
        # ---- 当前 attention backend ----
        # "flashinfer"（CSR indices，默认）或 "trtllm"（2D block_tables）。
        # 由 Context.create() 从 server_args.vortex_attention_backend 设置；
        # indexer compiler 的 IndexerBackend traits 会使用它。
        "vortex_attention_backend",
        # ---- workload scheduler 形状 ----
        "max_num_workloads",
        "workload_chunk_size",
        # ---- head / shape ----
        "group_size", "num_kv_heads", "num_qo_heads", "head_dim",
        # ---- 硬件 / paging ----
        "num_sms", "page_size", "max_num_pages", "max_num_pages_per_request",
        "block_size", "max_num_blocks", "max_num_blocks_per_request",
        "num_blocks_per_page", "num_pages_per_workload",
        # ---- topk / 预留槽策略 ----
        "topk_val", "topk_ratio", "block_reserved_bos", "block_reserved_eos",
        "max_topk_val",
        # ---- 辅助资源统计 ----
        "_aux_total_bytes", "_aux_total_flops",
        # ---- codegen / graph 草稿纸 ----
        "tensor_list", "op_list", "output_tensor_to_op_list",
        "op_to_input_tensor_list", "op_to_output_tensor_list",
        "side_effect_op_ids",
        "sparse_attention_name", "impl_backend", "tensor_id_to_tensor_name_map",
        "query_arg_names",
        "compilation_header_lines", "auxilary_func_def_lines",
        "compilation_cache_dir",
        # ---- tensor-core（bf16 compute）codegen 开关 ----
        # 为 True 时，triton W-kernel 会把计算块保持为 bf16
        # （load 转 bf16，累加提升到 fp32），并为适合 MMA 的 GeMM 生成 ``tl.dot``。
        # 只支持 Triton 实现。
        "use_tensor_core",
    )

    # ---- 类型标注（这里只声明，不赋值）----
    metadata: Optional[MetaData]
    max_bs: int
    vortex_attention_backend: str
    max_num_workloads: int
    workload_chunk_size: int
    group_size: int
    num_kv_heads: int
    num_qo_heads: int
    head_dim: int
    num_sms: int
    page_size: int
    max_num_pages: int
    max_num_pages_per_request: int
    block_size: int
    max_num_blocks: int
    max_num_blocks_per_request: int
    num_blocks_per_page: int
    num_pages_per_workload: int
    topk_val: int
    topk_ratio: float
    block_reserved_bos: int
    block_reserved_eos: int
    max_topk_val: Union[int, None]
    _aux_total_bytes: int
    _aux_total_flops: int
    tensor_list: list
    op_list: list
    output_tensor_to_op_list: list
    op_to_input_tensor_list: list
    op_to_output_tensor_list: list
    side_effect_op_ids: list
    sparse_attention_name: str
    impl_backend: str
    tensor_id_to_tensor_name_map: dict
    compilation_header_lines: list
    auxilary_func_def_lines: list
    compilation_cache_dir: str
    use_tensor_core: bool

    def __init__(self) -> None:
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Indexer")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)
            elif name == "_aux_total_flops":
                object.__setattr__(self, name, 0)
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile)
            elif name == "metadata":
                object.__setattr__(self, name, None)
            else:
                object.__setattr__(self, name, UNSET)

    # ------------------------------------------------------------------
    # 每个 batch 的便捷访问器：转发到 ``self.metadata``。
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        """当前 batch size，从 ``self.metadata`` 代理读取。

        这里故意保持为 property，而不是 slot，让唯一可写副本位于
        ``self.metadata``；``ctx.batch_size`` 现在是只读入口。
        """
        return 0 if self.metadata is None else self.metadata.batch_size

    def set_batch_size(self, n: int) -> None:
        """兼容旧调用的薄封装：转发到 ``self.metadata.set_batch_size``。"""
        if self.metadata is None:
            raise RuntimeError(
                "Context.set_batch_size called before MetaData was preallocated; "
                "did you forget `ctx.metadata = MetaData.preallocate(ctx, device=...)`?"
            )
        self.metadata.set_batch_size(n)

    # ------------------------------------------------------------------
    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False) -> "Context":
        """填充静态字段。每个 batch 的 ``MetaData`` 由调用方通过
        ``MetaData.preallocate(ctx, device=...)`` 单独分配；见本类 docstring。
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        sa = model_runner.server_args
        max_pages_per_req = (
            (model_runner.model_config.context_len + sa.page_size - 1) // sa.page_size
            if sa.vortex_max_seq_lens < 0
            else (sa.vortex_max_seq_lens + sa.page_size - 1) // sa.page_size
        )
        max_bs = int(model_runner.req_to_token_pool.size)
        self.max_bs = max_bs

        self.workload_chunk_size = sa.vortex_workload_chunk_size

        self.group_size = parent.group_size
        self.num_kv_heads = parent.num_kv_heads
        self.num_qo_heads = parent.num_qo_heads
        self.head_dim = parent.head_dim

        self.num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.page_size = sa.page_size
        self.block_size = sa.vortex_block_size
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
        assert self.workload_chunk_size % self.num_blocks_per_page == 0, "Workload chunk size must be a multiple of blocks per page."
        # 容量模型（如有需要可以调整）
        self.max_num_pages = max_pages_per_req * max_bs * self.num_kv_heads
        self.max_num_pages_per_request = max_pages_per_req
        self.max_num_blocks = self.max_num_pages * self.num_blocks_per_page
        self.max_num_blocks_per_request = self.max_num_pages_per_request * self.num_blocks_per_page
        self.num_pages_per_workload = self.workload_chunk_size // self.num_blocks_per_page

        # 把 trtllm 的 dense_block_tables 行跨度向上取整到 4 个 int32
        # （也就是 128 bits）的倍数。``dense_block_tables`` 每一行宽度是
        # ``max_num_blocks_per_request`` 个 int32；trtllm indexer kernel 会从
        # ``pid * max_blocks_per_seq`` 开始读取
        # ``indices[pid * max_blocks_per_seq + p * nbp]``。让这个乘积对每个
        # ``pid`` 都 16-byte 对齐，可以让 int32 load 合并成 128-bit 的
        # LDG.E.128 cache-line-aligned transaction。对 flashinfer 无害，因为
        # flashinfer 的 CSR indices 数组没有 per-row stride。
        # ``p < _page`` 已经会按真实 sequence length 限制 kernel 的 index 读取，
        # 因此外加的 padding entry 永远不会被读到（planner 也不会写它们）。
        # 注意：``self.vortex_attention_backend`` 在本方法更靠后才赋值，所以这里
        # 直接从 ``sa`` 读取。
        _attn_backend = (
            getattr(sa, "vortex_attention_backend", "flashinfer") or "flashinfer"
        )
        if _attn_backend == "trtllm" and self.max_num_blocks_per_request % 4 != 0:
            self.max_num_blocks_per_request = (
                (self.max_num_blocks_per_request + 3) // 4 * 4
            )
        self.topk_val = sa.vortex_topk_val
        self.max_topk_val = sa.vortex_max_topk_val
        self.topk_ratio = sa.vortex_topk_ratio
        self.vortex_dtype = resolve_dtype(
            getattr(sa, "vortex_dtype", "bfloat16"), default=torch.bfloat16
        )

        self.block_reserved_bos = sa.vortex_block_reserved_bos
        self.block_reserved_eos = sa.vortex_block_reserved_eos

        self.max_num_workloads = (
            (self.max_num_blocks // max(1, self.workload_chunk_size)) + max_bs * self.num_kv_heads
        )

        self.tensor_list = []
        self.op_list = []
        self.output_tensor_to_op_list = []
        self.op_to_input_tensor_list = []
        self.op_to_output_tensor_list = []
        self.side_effect_op_ids = []
        self.tensor_id_to_tensor_name_map = {}
        # 生成的 ``forward(...)`` 入口使用的 query 参数名。
        # MHA flow 只传一个 ``q``；MLA flow 会传吸收后的
        # ``q_nope_out`` / ``q_pe``，并把这里设置成 ["q_nope_out", "q_pe"]。
        self.query_arg_names = ["q"]
        self.compilation_header_lines = []
        self.auxilary_func_def_lines = []
        self.compilation_cache_dir = sa.vortex_compilation_cache_dir
        self.sparse_attention_name = parent.sparse_attention.__class__.__name__.lower() + f"_{uuid.uuid4().hex[:8]}"  # 当前 attention 实例的唯一名字
        self.impl_backend = getattr(sa, "vortex_impl_backend", "triton") or "triton"
        self.vortex_attention_backend = getattr(
            sa, "vortex_attention_backend", "flashinfer"
        ) or "flashinfer"
        self.use_tensor_core = bool(getattr(sa, "vortex_use_tensor_core", False))
        # Tensor-core（bf16-compute + tl.dot）codegen 目前只在 triton W-kernel
        # backend 里实现。cuda backend 有自己的 fp32 累加路径，并且会忽略这个
        # flag；这里显式拒绝这种组合，让配置错误时直接报错，而不是静默跑 fp32。
        if self.use_tensor_core and self.impl_backend != "triton":
            raise ValueError(
                "vortex_use_tensor_core is only supported with "
                f"vortex_impl_backend='triton'; got '{self.impl_backend}'."
            )
        self._created = True
        return self


# 模块级单例，是公开 package API 的一部分。
ctx: Final[Context] = Context()


def get_ctx() -> Context:
    return ctx


__all__ = ["Context", "MetaData", "ctx", "get_ctx"]
