import torch
from typing import FrozenSet
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import Schedule

class topK(vOp):
    r"""
    每个请求各自执行的 top-k page 选择器，并支持强制保留 BOS/EOS page。

    这是 ``forward_indexer`` 的终点算子：它把每个 page 的分数转换成
    每个请求最终要看的稀疏 page 集合。

    :Math:
        对某个请求的每页分数 :math:`X_p`（:math:`p=0,\dots,S-1`），
        如果开头强制保留区域是
        :math:`\mathcal{B}=\{0,\dots,n_{\mathrm{bos}}-1\}`，
        末尾强制保留区域是
        :math:`\mathcal{E}=\{S-n_{\mathrm{eos}},\dots,S-1\}`，
        那最终选中的 page 集合是：

        .. math::

            \mathcal{S} = \mathcal{B}\,\cup\,\mathcal{E}\,\cup\,
            \operatorname*{top\text{-}k}_{\,p\,\notin\,\mathcal{B}\cup\mathcal{E}} X_p,
            \qquad k = \texttt{topk\_val}.
    :__init__: ``topK()`` 不需要参数；``topk_val`` 预算以及 BOS/EOS
        强制保留数量会在运行时从 :class:`Context` 中读取。
    :__call__: ``op(score, o, ctx=ctx)``；``score`` 是 ``[S, 1, 1]``，
        表示每个 page 一个标量分数，并且必须是 ``RAGGED`` 格式。
        选中的 page id 会 **原地写入** ``o``。函数不返回值。
    :Note: 每个 flow 的 ``forward_indexer`` 都必须以这个算子
        或 :class:`approxTopK` 结束。
    """

    # 支持的输入格式；目前只支持 RAGGED。
    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self):
        super().__init__()
        self.schedule = Schedule.S

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> None:
        r"""trace 阶段：校验 ``x`` ``[S, 1, 1]`` 和 ``o``。
        两者都必须是同一设备上的 ``RAGGED`` ``vTensor``。随后注册这个算子。
        这里不分配内存，也不返回值；``o`` 会在 execute 阶段被原地填充。"""
        prefix = self._prefix()

        # ---- 类型检查 ----
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"

        # ---- 维度数量检查 ----
        assert x.dim() == 3, (
            f"{prefix}expected x to be 3D, "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert o.dim() == 3, (
            f"{prefix}expected o to be 3D, "
            f"got ndim={o.dim()} shape={tuple(o.shape)}"
        )

        # ---- x 的形状检查 ----
        # x 应该在第 1、2 维上都是 1，即每个 page 只携带一个标量分数。
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )

        # ---- 检查当前格式是否有实现 ----
        x_fmt = x._format
        assert x_fmt in self._supported_formats, (
            f"{prefix}no implementation for x._format={x_fmt}. "
            f"Supported: {sorted(self._supported_formats, key=lambda f: f.value)}"
        )

        # ---- 对 `o` 做轻量合理性检查 ----
        # 这里只检查设备一致性；精确的 (S_pack, D0, D1) 形状由上游约定和具体实现保证。
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # 在上下文中记录图结构
        ctx.output_tensor_to_op_list[o.tensor_id] = len(ctx.op_list)   # 记录这个输出张量由当前算子产生
        ctx.op_list.append(self)  # 把当前算子加入上下文的算子列表
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # 记录当前算子的输入张量
        ctx.op_to_output_tensor_list.append([o.tensor_id])  # 记录当前算子的输出张量


class approxTopK(topK):
    r"""
    近似版 :class:`topK`，使用更快的自适应 8-bit radix 选择。

    :Math:
        最终选中集合 :math:`\mathcal{S}` 与 :class:`topK` 的语义一致。
        区别是：对非强制保留 page 做 top-:math:`k` 搜索时，如果某一轮
        radix 选择中阈值桶还需要补的数量 :math:`r` 满足：

        .. math::

            r \;\le\; \texttt{tolerate\_ratio}\cdot k,

        就提前停止；剩下的位置按到达顺序填充。
    :__init__: ``approxTopK(tolerate_ratio=0.0)``；近似预算范围是
        ``[0, 1]``。``0.0`` 表示精确模式，会跑完所有 radix 轮次；
        数值越高越省计算，但结果越松，常见吞吐甜点大约是 ``0.05–0.15``。
    :__call__: ``op(score, o, ctx=ctx)``；签名与 :class:`topK` 相同。
        ``score`` 是 ``[S, 1, 1]``、``RAGGED``，并把 page id 写入 ``o``。
    :Note: 每个请求内部的索引 **不是排序好的**；下游不能假设输出有序。
    """

    def __init__(self, tolerate_ratio: float = 0.0):
        super().__init__()
        try:
            tol = float(tolerate_ratio)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"approxTopK: tolerate_ratio must be a number in [0.0, 1.0], "
                f"got {tolerate_ratio!r}"
            ) from e
        if not (0.0 <= tol <= 1.0):
            raise ValueError(
                f"approxTopK: tolerate_ratio={tol} out of range [0.0, 1.0]. "
                f"0.0 = exact (4 rounds); 1.0 = cheapest single-round; "
                f"typical sweet spot 0.05 - 0.15."
            )
        self.tolerate_ratio = tol


class Union(vOp):
    r"""
    对两组 ``(block_table, seqlens)`` 做逐行并集，供 trtllm 输出函数使用。

    :Math:
        对请求行 :math:`i`，假设已有两组选中的 block-id 集合
        :math:`\mathcal{B}_i^{0}` 和 :math:`\mathcal{B}_i^{1}`，
        它们来自两次 :class:`vortex_torch.indexer.TopK` 调用。
        最终稀疏集合是去重后的并集，并且把 dense 末尾 block
        :math:`\ell_i` 固定放在最后：

        .. math::

            \mathcal{U}_i = \big(\mathcal{B}_i^{0}\cup\mathcal{B}_i^{1}\setminus\{\ell_i\}\big)
            \;\Vert\; \{\ell_i\},

        并且
        :math:`\text{seqlens}_i = u_i\cdot\text{block\_size} + \text{last\_block\_len}_i`，
        其中 :math:`u_i = |\mathcal{U}_i| - 1`。把 :math:`\ell_i`
        固定放在最后，可以让 trtllm 从正确槽位读取最后一个不满 block
        的 token 数量。
    :__init__: ``Union()`` 不需要参数。
    :__call__: ``op((bt_0, sl_0), (bt_1, sl_1), o, ctx=ctx)``；
        输入两组 RAGGED ``(block_table, seqlens)`` 元组，把并集写入
        ``o``（即 ``ctx.metadata.sparse_block_tables``），并且作为副作用
        更新 ``ctx.metadata.sparse_seqlens``。
    :Note: **只支持 trtllm**；在 flashinfer 下会 assert。它是 trtllm
        ``forward_indexer`` 的终点算子，可以作为 :func:`topK` 的替代。
    """

    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self):
        super().__init__()
        self.schedule = Schedule.S

    def profile(self, i_0, i_1, o: vTensor, ctx: Context):
        prefix = self._prefix()

        assert isinstance(i_0, tuple) and len(i_0) == 2, (
            f"{prefix}i_0 must be a (block_table, seqlens) tuple; got {type(i_0).__name__}"
        )
        assert isinstance(i_1, tuple) and len(i_1) == 2, (
            f"{prefix}i_1 must be a (block_table, seqlens) tuple; got {type(i_1).__name__}"
        )
        bt_0, sl_0 = i_0
        bt_1, sl_1 = i_1
        for name, t in (("bt_0", bt_0), ("sl_0", sl_0),
                        ("bt_1", bt_1), ("sl_1", sl_1),
                        ("o",    o)):
            assert isinstance(t, vTensor), (
                f"{prefix}{name} must be a vTensor; got {type(t).__name__}"
            )
        assert bt_0._format in self._supported_formats, (
            f"{prefix}bt_0._format={bt_0._format} not supported"
        )
        assert bt_1._format in self._supported_formats, (
            f"{prefix}bt_1._format={bt_1._format} not supported"
        )
        assert o._format in self._supported_formats, (
            f"{prefix}o._format={o._format} not supported"
        )

        backend = (
            getattr(ctx, "vortex_attention_backend", None) or "flashinfer"
        ).lower()
        assert backend == "trtllm", (
            f"{prefix}Union() is only supported under the trtllm attention "
            f"backend; got vortex_attention_backend={backend!r}."
        )

        # 声明 ``o`` 是这个算子的唯一输出。
        ctx.output_tensor_to_op_list[o.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([
            bt_0.tensor_id, sl_0.tensor_id,
            bt_1.tensor_id, sl_1.tensor_id,
        ])
        ctx.op_to_output_tensor_list.append([o.tensor_id])

