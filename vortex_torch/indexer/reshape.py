import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class Reshape(vOp):
    r"""
    indexer 侧的 same-numel reshape：只重排内部两个轴。

    :Math:
        .. math::

            X\in\mathbb{R}^{S\times x_1\times y_1} \;\longrightarrow\;
            Y\in\mathbb{R}^{S\times x_2\times y_2},\qquad x_2\,y_2 = x_1\,y_1,

        对每个前导索引 :math:`s` 独立处理：先把 :math:`x_1 y_1` 个内部元素
        按 row-major 顺序拉平，再重新解释成 :math:`(x_2, y_2)` 布局。
        这对应 Triton tile 内的 :func:`tl.reshape`，除了既有 load/store 外不做
        额外数据搬运。
    :__init__: ``Reshape(-1, x2, y2)``；前导维必须是 ``-1``，表示保留
        ``S`` 轴；``x2*y2`` 必须等于输入的 ``x1*y1``，会在 trace 阶段检查。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` ``[S, x_1, y_1]`` ->
        ``[S, x_2, y_2]``。输入是 ``BATCHED`` 时输出才是 ``BATCHED``，
        否则输出是 ``RAGGED``。
    """

    def __init__(self, batch_dim: int, x2: int, y2: int):
        super().__init__()
        cls = self.__class__.__name__
        assert batch_dim == -1, (
            f"{cls}.__init__: leading dim must be -1 (auto-infer; the "
            f"leading S axis is preserved), got {batch_dim}"
        )
        assert isinstance(x2, int) and x2 >= 1, (
            f"{cls}.__init__: x2 must be a positive int, got {x2!r}"
        )
        assert isinstance(y2, int) and y2 >= 1, (
            f"{cls}.__init__: y2 must be a positive int, got {y2!r}"
        )
        self.x2 = x2
        self.y2 = y2

        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # 融合进 per-workload kernel。
        self.schedule = Schedule.W

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[S, x1, y1]``，并检查 same-numel 约束
        ``x1*y1 == x2*y2``。随后确定输出格式、注册算子，并返回
        ``[S, x2, y2]`` 输出的 ``vTensor`` 视图。"""
        prefix = self._prefix()

        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, x1, y1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        x1, y1 = x.shape[1], x.shape[2]
        in_numel = x1 * y1
        out_numel = self.x2 * self.y2
        assert in_numel == out_numel, (
            f"{prefix}same-numel reshape required: "
            f"x.shape[1]*x.shape[2] = {x1}*{y1} = {in_numel}  vs  "
            f"target x2*y2 = {self.x2}*{self.y2} = {out_numel}"
        )

        # 输入是 BATCHED 时输出才保持 BATCHED；否则输出是 RAGGED。
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        # 纯元数据 vTensor。前导维 0 只是占位符；实际 ``S`` 由运行时 pipeline 知道。
        self.output_buffer = vTensor(
            shape=(0, self.x2, self.y2),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # fused ``tl.reshape``（Schedule.W）要求内部维度是 2 的幂，并且 tile
        # 不能包含 padding。否则 reshape 时 padding lane 可能被混进真实位置。
        # 如果输入或输出的真实内部维度需要 padding，就退回独立的 Schedule.S
        # torch reshape，只处理真实区域 ``[:, :x1, :y1] -> [:, :x2, :y2]``。
        # 这种路径对格式和维度更宽容；如果两边都不需要 padding，就保留快速
        # fused path。
        self.schedule = (
            Schedule.S
            if (x.needs_padding() or self.output_buffer.needs_padding())
            else Schedule.W
        )

        # 在 indexer graph 里登记当前算子和输入/输出关系。
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
