    import torch
from typing import Tuple
from .context import Context
from .planner_sglang import get_sglang_plan_decode_v2_module
from .prefill_sglang import get_sglang_prefill_module


def get_decode_planner(policy: str = None):
    """构造 flashinfer/CSR decode planner 闭包。

    外层函数先加载/编译 SGLang decode planner 模块，并把模块引用绑定到内部
    ``plan_decode`` 闭包里。后续每次 decode forward 只调用闭包，不需要重复查找模块。
    """

    module = get_sglang_plan_decode_v2_module(
        policy_body=policy,
        verbose=True,
        fallback_to_default=True,
    )
    def plan_decode(
        cached_seq_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        ctx: Context
    ):
        """把当前 decode batch 的 page/block 信息写入 ``ctx.metadata``。

        flashinfer 路径使用 CSR 风格的 ``dense/sparse_kv_indptr`` 和
        ``dense/sparse_kv_indices``；同时写入 workload scheduler 的
        ``winfo_*`` 输出。
        """
        md = ctx.metadata
        module.sglang_plan_decode_v2(
            cached_seq_lens,
            md.dense_kv_indptr,
            md.dense_kv_indices,
            md.sparse_kv_indptr,
            md.sparse_kv_indices,
            md.kv_last_page_len,
            req_to_token,
            req_indices,
            md.winfo_q_indices,
            md.winfo_is_first_workload_per_batch,
            md.winfo_kv_offsets,
            md.winfo_kv_lens,
            md.winfo_num_workloads,
            md.winfo_chunk_size,
            ctx.page_size,
            ctx.block_size,
            ctx.num_kv_heads,
            ctx.topk_val,
            ctx.topk_ratio,
            ctx.block_reserved_bos,
            ctx.block_reserved_eos,
            ctx.workload_chunk_size
        )

        md.set_batch_size(cached_seq_lens.shape[0])

    return plan_decode


def get_decode_planner_trtllm(policy: str = None):
    """构造直接输出 trtllm-ready metadata 的 decode planner。

    trtllm planner 是 **indptr-free** 的：它不会读写
    ``dense_kv_indptr`` / ``sparse_kv_indptr`` / ``dense_kv_indices`` /
    ``sparse_kv_indices``。底层 CUDA kernel 会填充：

      * ``ctx.metadata.dense_block_tables``：dense 路径的全部已选 page；
      * ``ctx.metadata.sparse_block_tables``：只填 BOS+EOS 槽位，中间部分稍后
        由 topk kernel 填；
      * ``ctx.metadata.dense_seqlens`` / ``ctx.metadata.sparse_seqlens``：
        int32 token 数，会被 ``trtllm_batch_decode_with_kv_cache``、trtllm topk
        kernel 和 Schedule.S Triton kernel 使用；Schedule.S kernel 会通过
        ``ceil(tokens / block_size)`` 推导每行 block 数；
      * ``ctx.metadata.kv_last_page_len``：语义和 flashinfer 路径相同；
      * ``ctx.metadata.winfo_*``：workload scheduler 输出；
        ``winfo_kv_offsets[j]`` 携带 ``row * max_blocks_per_seq + col``，
        这样 Schedule.W kernel preamble 在使用
        ``indices = dense_block_tables.view(-1)`` 时能正确解析 page id。
    """
    module = get_sglang_plan_decode_v2_module(
        policy_body=policy,
        verbose=True,
        fallback_to_default=True,
    )

    def plan_decode_trtllm(
        cached_seq_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        ctx: Context,
    ):
        """把当前 decode batch 的 trtllm block-table metadata 写入 ``ctx.metadata``。"""
        md = ctx.metadata
        module.sglang_plan_decode_v2_trtllm(
            cached_seq_lens,
            md.kv_last_page_len,
            md.dense_block_tables,
            md.sparse_block_tables,
            md.dense_seqlens,
            md.sparse_seqlens,
            req_to_token,
            req_indices,
            md.winfo_q_indices,
            md.winfo_is_first_workload_per_batch,
            md.winfo_kv_offsets,
            md.winfo_kv_lens,
            md.winfo_num_workloads,
            md.winfo_chunk_size,
            ctx.page_size,
            ctx.block_size,
            ctx.num_kv_heads,
            ctx.topk_val,
            ctx.topk_ratio,
            ctx.block_reserved_bos,
            ctx.block_reserved_eos,
            ctx.workload_chunk_size,
        )
        md.set_batch_size(cached_seq_lens.shape[0])

    return plan_decode_trtllm


def get_prefill_planner():
    """prefill 路径对应的 planner 工厂，结构类似 :func:`get_decode_planner`。

    第一次调用时触发 prefill 模块的一次性 JIT compile，然后返回一个闭包。
    这个闭包会直接调用 ``sglang_plan_prefill``，模块引用已经绑定好，不需要每次查找。
    """
    module = get_sglang_prefill_module()

    def plan_prefill(
        cached_seq_lens: torch.Tensor,
        dense_kv_indptr: torch.Tensor,
        dense_kv_indices: torch.Tensor,
        input_seq_lens: torch.Tensor,
        qo_indptr_ragged: torch.Tensor,
        qo_indptr_paged: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        batch_table: torch.Tensor,
        page_size: int,
        num_kv_heads: int,
    ):
        """调用 SGLang prefill planner，填充 prefill 路径需要的 indptr/indices。"""
        module.sglang_plan_prefill(
            cached_seq_lens,
            dense_kv_indptr,
            dense_kv_indices,
            input_seq_lens,
            qo_indptr_ragged,
            qo_indptr_paged,
            kv_last_page_len,
            req_to_token,
            req_indices,
            batch_table,
            page_size,
            num_kv_heads,
        )

    return plan_prefill


def get_chunkwise_nh2hn_transpose():
    """``Chunkwise_NH2HN_Transpose`` kernel 的工厂函数。

    它也采用 :func:`get_decode_planner` 的闭包模式，让模块引用在 backend
    初始化时只绑定一次。
    """
    module = get_sglang_prefill_module()

    def chunkwise_nh2hn_transpose(
        x: torch.Tensor,
        indptr: torch.Tensor,
        batch_table: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        return module.Chunkwise_NH2HN_Transpose(
            x, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim,
        )

    return chunkwise_nh2hn_transpose


def get_chunkwise_hn2nh_transpose():
    """``Chunkwise_HN2NH_Transpose`` kernel 的工厂函数。

    它也采用 :func:`get_decode_planner` 的闭包模式。
    """
    module = get_sglang_prefill_module()

    def chunkwise_hn2nh_transpose(
        x: torch.Tensor,
        y: torch.Tensor,
        indptr: torch.Tensor,
        batch_table: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return module.Chunkwise_HN2NH_Transpose(
            x, y, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim,
        )

    return chunkwise_hn2nh_transpose
