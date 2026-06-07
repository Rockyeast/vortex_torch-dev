import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ReduceType, Schedule

class Reduce(vOp):
    r"""
    对 rank-3 逻辑张量的某一个轴做通用 1-D reduction。

    :Math:
        对输入 :math:`X\in\mathbb{R}^{N\times D_0\times D_1}`，
        以及沿某个轴做的 reduction :math:`\rho`（mean / max / min /
        L2-norm / sum，由子类决定）：

        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{n,0,d} = \rho_{\,0 \le i < D_0}\, X_{n,i,d}, \\
            (\text{dim}=2):\quad & Y_{n,d,0} = \rho_{\,0 \le j < D_1}\, X_{n,d,j}.
            \end{aligned}

        ``dim=0`` 会把打包后的前导轴压缩成每个 ``(batch, kv\_head)``
        一行摘要。
    :__init__:
        ``Reduce(dim=1)``；要 reduction 的逻辑轴，只能是 ``0`` / ``1`` / ``2``。
    :__call__:
        ``y = op(x, ctx=ctx)``；``x`` 是 ``[N, D_0, D_1]``。被 reduction
        的轴会保留，但大小变成 1。对于 ``dim ∈ {1, 2}``，只有输入是
        ``BATCHED`` 时输出才是 ``BATCHED``；``dim=0`` 要求输入是
        ``RAGGED``，输出是 ``BATCHED``。
    :Note:
        请使用具体子类：:class:`Max`、:class:`Min`、:class:`Mean`、
        :class:`L2Norm`、:class:`Sum`。
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        # dim==0 会跨 packed 前导轴做 reduction；结果是每个 (batch, kv_head)
        # 一个摘要，所以它不能融合进 per-block workload kernel，需要单独调度。
        self.schedule = Schedule.S if dim == 0 else Schedule.W
        prefix = self._prefix()
        assert self.dim in (0, 1, 2), (
            f"{prefix}__init__: dim must be 0, 1, or 2, got dim={self.dim}"
        )

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x``（``[N, D_0, D_1]``），确定输出格式，
        注册这个算子，并返回一个描述 reduction 输出的 ``vTensor`` 视图。
        输出形状详见类 docstring。"""
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [N, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        D0, D1 = x.shape[1], x.shape[2]

        if self.dim == 0:
            # 跨行 reduction：把 packed 前导轴压缩成每个 (batch, kv_head)
            # 一个摘要。输入必须是 RAGGED（per-page 或 per-token）；compiler
            # 会分配一个 BATCHED buffer，前导维是
            # ``ctx.max_bs * ctx.num_kv_heads``（见 indexer interface）。
            assert x._format == FORMAT.RAGGED, (
                f"{prefix}dim=0 reduce requires RAGGED input, got {x._format}"
            )
            self.output_format = FORMAT.BATCHED
            out_D0, out_D1 = D0, D1
        else:
            # 只有输入是 BATCHED 时，输出才是 BATCHED；否则输出是 RAGGED。
            self.output_format = (
                FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
            )
            out_D0 = 1 if self.dim == 1 else D0
            out_D1 = 1 if self.dim == 2 else D1

        # 纯元数据 vTensor，不需要 torch.empty 分配真实内存。
        self.output_buffer = vTensor(
            shape=(0, out_D0, out_D1),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # 在上下文里记录图结构
        ctx.tensor_list.append(self.output_buffer)  # 记录输出 buffer
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # 记录输出张量由当前算子产生
        ctx.op_list.append(self)  # 记录当前算子
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # 记录当前算子的输入张量
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # 记录当前算子的输出张量
        
        # 返回带有最终输出格式的 vTensor 视图
        return self.output_buffer


class Max(Reduce):
    r"""
    沿某个逻辑轴做 Max reduction，是 :class:`Reduce` 的具体子类。

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{n,0,d} = \max_{0 \le i < D_0} X_{n,i,d}, \\
            (\text{dim}=2):\quad & Y_{n,d,0} = \max_{0 \le j < D_1} X_{n,d,j}.
            \end{aligned}
    :__init__: ``Max(dim=1)``；要 reduction 的轴，``1`` 表示 :math:`D_0`，
        ``2`` 表示 :math:`D_1`。
    :__call__: ``y = op(x, ctx=ctx)``；``[N, D_0, D_1]`` 会变成
        ``[N, 1, D_1]``（``dim=1``）或 ``[N, D_0, 1]``（``dim=2``）。
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max


class Min(Reduce):
    r"""
    沿某个逻辑轴做 Min reduction，是 :class:`Reduce` 的具体子类。

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{n,0,d} = \min_{0 \le i < D_0} X_{n,i,d}, \\
            (\text{dim}=2):\quad & Y_{n,d,0} = \min_{0 \le j < D_1} X_{n,d,j}.
            \end{aligned}
    :__init__: ``Min(dim=1)``；要 reduction 的轴，``1`` 表示 :math:`D_0`，
        ``2`` 表示 :math:`D_1`。
    :__call__: ``y = op(x, ctx=ctx)``；``[N, D_0, D_1]`` 会变成
        ``[N, 1, D_1]``（``dim=1``）或 ``[N, D_0, 1]``（``dim=2``）。
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min


class Mean(Reduce):
    r"""
    沿某个逻辑轴做 Mean reduction，是 :class:`Reduce` 的具体子类。

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{n,0,d} = \frac{1}{D_0}\sum_{i=0}^{D_0-1} X_{n,i,d}, \\
            (\text{dim}=2):\quad & Y_{n,d,0} = \frac{1}{D_1}\sum_{j=0}^{D_1-1} X_{n,d,j}.
            \end{aligned}
    :__init__: ``Mean(dim=1)``；要 reduction 的轴，``1`` 表示 :math:`D_0`，
        ``2`` 表示 :math:`D_1`。
    :__call__: ``y = op(x, ctx=ctx)``；``[N, D_0, D_1]`` 会变成
        ``[N, 1, D_1]``（``dim=1``）或 ``[N, D_0, 1]``（``dim=2``）。
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean


class L2Norm(Reduce):
    r"""
    沿某个逻辑轴做 L2-norm reduction，是 :class:`Reduce` 的具体子类。

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{n,0,d} = \Big(\sum_{i=0}^{D_0-1} X_{n,i,d}^2\Big)^{1/2}, \\
            (\text{dim}=2):\quad & Y_{n,d,0} = \Big(\sum_{j=0}^{D_1-1} X_{n,d,j}^2\Big)^{1/2}.
            \end{aligned}
    :__init__: ``L2Norm(dim=1)``；要 reduction 的轴，``1`` 表示 :math:`D_0`，
        ``2`` 表示 :math:`D_1`。
    :__call__: ``y = op(x, ctx=ctx)``；``[N, D_0, D_1]`` 会变成
        ``[N, 1, D_1]``（``dim=1``）或 ``[N, D_0, 1]``（``dim=2``）。
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm


class Sum(Reduce):
    r"""
    沿某个逻辑轴做 Sum reduction，是 :class:`Reduce` 的具体子类。

    :Math:
        .. math::

            \begin{aligned}
            (\text{dim}=1):\quad & Y_{n,0,d} = \sum_{i=0}^{D_0-1} X_{n,i,d}, \\
            (\text{dim}=2):\quad & Y_{n,d,0} = \sum_{j=0}^{D_1-1} X_{n,d,j}.
            \end{aligned}
    :__init__: ``Sum(dim=1)``；要 reduction 的轴，``1`` 表示 :math:`D_0`，
        ``2`` 表示 :math:`D_1`。
    :__call__: ``y = op(x, ctx=ctx)``；``[N, D_0, D_1]`` 会变成
        ``[N, 1, D_1]``（``dim=1``）或 ``[N, D_0, 1]``（``dim=2``）。
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Sum
