import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class GeMV(vOp):
    r"""
    每个请求各自执行的批量矩阵-向量乘，:math:`O = Y X^{\top}`。

    :Math:
        批量 query :math:`X\in\mathbb{R}^{B\times 1\times D}`，
        打包后的 page :math:`Y\in\mathbb{R}^{S\times 1\times D}`。
        对属于请求 :math:`i(s)` 的 page :math:`s`：

        .. math::

            O_{s,0,0} = \sum_{d=0}^{D-1} Y_{s,0,d}\,X_{i(s),0,d}
                      = \langle Y_s,\, X_{i(s)} \rangle,
            \qquad O\in\mathbb{R}^{S\times 1\times 1}.
    :__init__: ``GeMV()`` 不需要参数。
    :__call__: ``o = op(x, y, ctx=ctx)``；``x`` 是 ``[B, 1, D]``，
        ``y`` 是 ``[S, 1, D]``，两者 ``D`` 必须一致；返回 ``o``，
        形状是 ``[S, 1, 1]``。只有当两个输入都是 ``BATCHED`` 时，
        输出才是 ``BATCHED``，否则输出是 ``RAGGED``。
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[B, 1, D]`` / ``y`` ``[S, 1, D]``，
        注册这个算子，并返回一个描述 ``[S, 1, 1]`` 输出的 ``vTensor``
        视图。详见类 docstring。"""
        prefix = self._prefix()

        # 类型检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # 维度数量 / 形状检查
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        assert x.shape[1] == 1, f"{prefix}expected x.shape[1] == 1, got {tuple(x.shape)}"
        assert y.shape[1] == 1, f"{prefix}expected y.shape[1] == 1, got {tuple(y.shape)}"
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # 只有当两个输入都是 BATCHED 时，输出才是 BATCHED；否则输出是 RAGGED。
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )
        # 纯元数据 vTensor，不需要 torch.empty 分配真实内存。
        self.output_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # 在上下文里记录图结构
        ctx.tensor_list.append(self.output_buffer)  # 记录输出 buffer
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # 记录输出张量由当前算子产生
        ctx.op_list.append(self)  # 记录当前算子
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])  # 记录当前算子的输入张量
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # 记录当前算子的输出张量

        return self.output_buffer



# ------------------------------ GeMM ------------------------------ #
class GeMM(vOp):
    r"""
    每个 page 各自执行的矩阵-矩阵乘，:math:`O[s] = Y[s]\,X[s]^{\top}`。

    :Math:
        :math:`Y\in\mathbb{R}^{S\times N_y\times K}`,
        :math:`X\in\mathbb{R}^{(B\text{ or }S)\times N_x\times K}`。
        对每个 page :math:`s`，计算 :math:`O_s = Y_s X_s^{\top}`。
        也就是说 ``GeMM(x, y) = y xᵀ``：

        .. math::

            O_{s,a,b} = \sum_{k=0}^{K-1} Y_{s,a,k}\,X_{s,b,k},
            \qquad O\in\mathbb{R}^{S\times N_y\times N_x}.
    :__init__: ``GeMM()`` 不需要参数。
    :__call__: ``o = op(x, y, ctx=ctx)``；``x`` 是 ``[B|S, N_x, K]``，
        ``y`` 是 ``[S, N_y, K]``，两者 ``K`` 必须一致；返回 ``o``，
        形状是 ``[S, N_y, N_x]``。只有当两个输入都是 ``BATCHED`` 时，
        输出才是 ``BATCHED``，否则输出是 ``RAGGED``。
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[B|S, N_x, K]`` / ``y`` ``[S, N_y, K]``
        且 ``K`` 一致，注册这个算子，并返回一个描述 ``[S, N_y, N_x]``
        输出的 ``vTensor`` 视图。详见类 docstring。"""
        prefix = self._prefix()

        # 类型检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # 维度数量 / 形状检查
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        # K 维必须一致
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # 只有当两个输入都是 BATCHED 时，输出才是 BATCHED；否则输出是 RAGGED。
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )

        # 输出的逻辑内部形状：Ny x Nx
        Ny, Nx = y.shape[1], x.shape[1]

        # 纯元数据 vTensor，不需要 torch.empty 分配真实内存。
        self.output_buffer = vTensor(
            shape=(0, Ny, Nx),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # 在上下文里记录图结构
        ctx.tensor_list.append(self.output_buffer)  # 记录输出 buffer
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # 记录输出张量由当前算子产生
        ctx.op_list.append(self)  # 记录当前算子
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])  # 记录当前算子的输入张量
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # 记录当前算子的输出张量

        return self.output_buffer
