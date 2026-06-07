import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class MaskSlice(vOp):
    r"""
    在某个内部轴上生成“按位置决定”的切片 mask。

    :Math:
        对目标轴 ``dim``（``1`` -> :math:`D_0`，``2`` -> :math:`D_1`）上的
        索引 :math:`i`，其他轴保持 broadcast：

        .. math::

            Y_{\dots,i,\dots} = \begin{cases} \alpha, & \text{start} \le i < \text{end}, \\ \beta, & \text{otherwise}. \end{cases}
    :__init__: ``MaskSlice(start, end, dim, alpha=1.0, beta=0.0)``；在
        ``dim`` 轴（只能是 1 或 2）的 ``[start, end)`` 区间写
        :math:`\alpha`，其他位置写 :math:`\beta`。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` 是 ``[S, D_0, D_1]``，输出同形状。
        这是纯 **位置** 写入器，``x`` 的数值不会被读取；只有 ``x`` 是
        ``BATCHED`` 时输出才是 ``BATCHED``。
    :Note: 只支持 ``dim in {1, 2}``，因为 packed ``S`` 轴是结构轴，不能在这里
        做切片 mask。
    """

    def __init__(
        self,
        start: int,
        end: int,
        dim: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.start = int(start)
        self.end = int(end)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        self.schedule = Schedule.W

        prefix = self._prefix()
        assert self.dim in (1, 2), (
            f"{prefix}__init__: dim must be 1 or 2, got dim={self.dim}"
        )
        assert self.start <= self.end, (
            f"{prefix}__init__: require start <= end, got "
            f"start={self.start}, end={self.end}"
        )

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        prefix = self._prefix()
        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, D0, D1], got shape={tuple(x.shape)}"
        )

        # 输入是 BATCHED 时输出才保持 BATCHED；否则输出是 RAGGED。
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        dim_size = x.shape[self.dim]
        assert 0 <= self.start <= self.end <= dim_size, (
            f"{prefix}[start, end) = [{self.start}, {self.end}) out of "
            f"bounds for dim={self.dim} (size={dim_size})"
        )

        # 纯元数据 vTensor，不读取 x 的值，只借用 x 的形状、device 和布局。
        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )
        # 在上下文里记录图结构
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
