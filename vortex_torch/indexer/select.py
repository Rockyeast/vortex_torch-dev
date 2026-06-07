"""选择类 indexer 算子。

本文件里的算子会 *产生中间* block-table / seqlens tensor。它们不会写入框架
传进来的 ``o``，后者是现有 :class:`vortex_torch.indexer.topK` / "TopKOut"
负责的路径。这里产生的中间结果需要后续算子（待补）拷贝到最终的
``o`` / ``ctx.metadata.sparse_seqlens`` buffer，供
``trtllm_batch_decode_with_kv_cache`` 消费。
"""
from __future__ import annotations

from typing import FrozenSet

import torch

from ..abs import FORMAT, vOp, vTensor
from ..utils import Schedule
from .context import Context


class TopK(vOp):
    r"""
    显式 ``k`` 的 block-table top-k（trtllm backend，两个输出）。

    :Math:
        对请求行 :math:`i`，给定 dense blocks 上的 score :math:`s`、
        ``bos`` / ``eos`` 预留 block 数，以及
        :math:`L_i = \lceil \text{seqlen}_i / \text{block\_size}\rceil`：

        .. math::

            \mathcal{B}_i = \begin{cases}
                \{0,\dots,L_i-1\}, & L_i \le \text{bos}+k+\text{eos}, \\[2pt]
                [0,\text{bos}) \;\cup\; \operatorname*{top\text{-}k}_{\,\text{bos}\le j<L_i-\text{eos}} s_j
                \;\cup\; [L_i-\text{eos}, L_i), & \text{otherwise}.
            \end{cases}

        如果某一行本来就能放进预算，就直接复制 dense blocks；如果行太长，则保留
        开头 ``bos`` 和结尾 ``eos`` blocks，中间按 score 选 top-``k``。
    :__init__: ``TopK(k)``；要 *选择* 的 block 数，不包含预留的 BOS/EOS。
        每行 sparse block 数是 ``bos + k + eos``。
    :__call__: ``block_tables, seqlens = op(score, ctx=ctx)``；``score`` 是
        RAGGED ``[S, 1, 1]``，输出 ``block_tables``（RAGGED int32）和
        ``seqlens``（BATCHED int32），两者都会自动创建。送进
        ``trtllm_batch_decode_with_kv_cache`` 前，需要把这对结果交给
        :class:`Union`。
    :Note: **仅支持 trtllm**。flashinfer 下会 assert；flashinfer / CSR 布局请使用
        :func:`topK`。
    """

    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self, k: int):
        super().__init__()
        try:
            k_int = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"TopK: k must be an integer, got {k!r}") from e
        if k_int < 1:
            raise ValueError(f"TopK: k must be >= 1, got {k_int}")
        self.k = k_int
        self.schedule = Schedule.S
        self.block_tables_buffer: vTensor = None  # 在 profile() 里填充
        self.seqlens_buffer: vTensor = None

    def profile(self, x: vTensor, ctx: Context):
        prefix = self._prefix()

        # ---- 输入校验（基本对齐 topK）----
        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected x to be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )
        assert x._format in self._supported_formats, (
            f"{prefix}no implementation for x._format={x._format}. "
            f"Supported: {sorted(self._supported_formats, key=lambda f: f.value)}"
        )

        # ---- 仅支持 trtllm ----
        backend = (
            getattr(ctx, "vortex_attention_backend", None) or "flashinfer"
        ).lower()
        assert backend == "trtllm", (
            f"{prefix}TopK(k) is only supported under the trtllm attention "
            f"backend; got vortex_attention_backend={backend!r}. Use the "
            f"regular ``topK()`` op for flashinfer / CSR layouts."
        )

        # ---- 创建两个中间输出 ----
        # block_tables：RAGGED int32。memory_init 会给 RAGGED 中间量选择
        # ``leading = ctx.max_num_blocks``，也就是
        # ``eff_bs * max_blocks_per_seq``。这正好等于 CUDA kernel 按
        # ``[eff_bs, max_blocks_per_seq]`` 连续内存寻址时需要的字节规模。
        self.block_tables_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=torch.int32,
            device=x.device,
            _format=FORMAT.RAGGED,
            tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.block_tables_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))

        # seqlens：BATCHED int32。memory_init 会给 BATCHED 中间量选择
        # ``leading = ctx.max_bs * ctx.num_kv_heads``。
        self.seqlens_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=torch.int32,
            device=x.device,
            _format=FORMAT.BATCHED,
            tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.seqlens_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))

        # 双输出算子。``compiler/graph.py`` 迁移后，Graph 已经支持 multi-output。
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([
            self.block_tables_buffer.tensor_id,
            self.seqlens_buffer.tensor_id,
        ])

        return self.block_tables_buffer, self.seqlens_buffer


__all__ = ["TopK"]
