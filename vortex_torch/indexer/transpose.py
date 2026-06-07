import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Transpose(vOp):
    r"""
    对前导轴上每个 slice，交换内部两个轴。

    :Math:
        .. math::

            Y_{s,d_1,d_0} = X_{s,d_0,d_1},

        对每个前导索引 :math:`s` 独立执行。
    :__init__: ``Transpose()``；不需要参数。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` ``[S, D_0, D_1]`` ->
        ``[S, D_1, D_0]``。输入是 ``BATCHED`` 时输出才是 ``BATCHED``，
        否则输出是 ``RAGGED``。
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[S, D_0, D_1]``，确定输出格式，并返回
        ``[S, D_1, D_0]`` 转置结果的 ``vTensor`` 视图。"""
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # 输入是 BATCHED 时输出才保持 BATCHED；否则输出是 RAGGED。
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        # 创建输出 buffer 的元数据：[S, D1, D0]。
        # S 来自运行时上下文，表示 pipeline 中的 page/token 数量。
        S = ctx.max_num_pages
        D0, D1 = x.shape[1], x.shape[2]
        # 纯元数据 vTensor，不做真实内存分配。编译后的代码会提供存储；
        # 这里仅需要 shape/dtype/device 供 codegen 使用。
        self.output_buffer = vTensor(
            shape=(S, D1, D0),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=-1,  # 如有需要，由调用方或图登记逻辑设置
        )

        for t in [x]:
            if t._format == FORMAT.PAGED:
                ctx.add_aux_flops(
                    t.shape[1] * t.shape[2]
                )

        return self.output_buffer
