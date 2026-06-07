import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ElementwiseBinaryOpType, Schedule

class Elementwise_Binary(vOp):
    r"""
    二元逐元素算子：把两个 tensor 按元素组合，支持 broadcasting。

    :Math:
        .. math::

            Z_{s,c,d} = g(X_{s,c,d},\, Y_{s,c,d};\, \alpha, \beta),

        其中 :math:`g` 由具体子类决定，可以是 max / min / affine-sum /
        product / comparison。内部 ``(C, D)`` 轴支持 broadcasting。
    :__init__: ``Elementwise_Binary(alpha=1.0, beta=1.0)``；某些算子会用到
        这些标量参数，例如 affine sum :math:`\alpha x + \beta y`。
    :__call__: ``z = op(x, y, ctx=ctx)``；``x`` / ``y`` 是 ``[S, C, D]``，
        在 ``C, D`` 上做 broadcasting；输出是
        ``[S, max(C_x,C_y), max(D_x,D_y)]``。只有两个输入都是 ``BATCHED`` 时，
        输出才是 ``BATCHED``，否则输出是 ``RAGGED``。
    :Note: 请使用具体子类：:class:`Maximum`、:class:`Minimum`、
        :class:`Add`、:class:`Multiply`，或比较 mask
        （:class:`WhereGreater`、:class:`WhereEqual` 等）。
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.op_type: Optional[ElementwiseBinaryOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` / ``y``，它们都必须是 rank-3，并且在
        ``C`` / ``D`` 上可以 broadcast；随后注册这个算子，并返回一个描述
        broadcast 输出的 ``vTensor`` 视图。形状规则见类 docstring。"""
        prefix = self._prefix()

        # 类型检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # 维度数量和基础形状检查
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs [S, C, D]; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )

        # 检查 C/D 两个内部维度是否可以 broadcast
        assert (x.shape[1] == y.shape[1] or x.shape[1] == 1 or y.shape[1] == 1), (
            f"{prefix}dim-1 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )
        assert (x.shape[2] == y.shape[2] or x.shape[2] == 1 or y.shape[2] == 1), (
            f"{prefix}dim-2 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )

        # 只有两个输入都是 BATCHED 时，输出才保持 BATCHED；否则输出是 RAGGED。
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )

        # 设备一致性检查
        assert x.device == y.device, (
            f"{prefix}x and y must be on the same device "
            f"(x.device={x.device}, y.device={y.device})"
        )

        # broadcast 后的输出内部形状 (C, D)
        C_out = max(x.shape[1], y.shape[1])
        D_out = max(x.shape[2], y.shape[2])

        # 纯元数据 vTensor，不需要 torch.empty 分配真实内存。
        self.output_buffer = vTensor(
            shape=(0, C_out, D_out),
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


class Maximum(Elementwise_Binary):
    r"""
    逐元素最大值，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \max(X_{s,c,d},\, Y_{s,c,d}).
    :__init__: ``Maximum(alpha=1.0, beta=1.0)``；``alpha`` / ``beta`` 不使用。
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Maximum


class Minimum(Elementwise_Binary):
    r"""
    逐元素最小值，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \min(X_{s,c,d},\, Y_{s,c,d}).
    :__init__: ``Minimum(alpha=1.0, beta=1.0)``；``alpha`` / ``beta`` 不使用。
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Minimum
        

class Add(Elementwise_Binary):
    r"""
    affine 组合 :math:`\alpha x + \beta y`，是 :class:`Elementwise_Binary`
    的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \alpha\,X_{s,c,d} + \beta\,Y_{s,c,d}.
    :__init__: ``Add(alpha=1.0, beta=1.0)``；:math:`x` 和 :math:`y`
        的乘子，默认得到 :math:`x+y`。
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Add
        

class Multiply(Elementwise_Binary):
    r"""
    逐元素乘法，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = X_{s,c,d}\cdot Y_{s,c,d}.
    :__init__: ``Multiply(alpha=1.0, beta=1.0)``；``alpha`` / ``beta`` 不使用。
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Mul


class WhereEqual(Elementwise_Binary):
    r"""
    比较 mask：:math:`x = y`，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} = Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereEqual()`` 不需要参数。
    :Note: 这是加性 mask，用于在 :class:`vortex_torch.indexer.Softmax` /
        :func:`vortex_torch.indexer.topK` 之前门控 score tensor。
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereEqual


class WhereNotEqual(Elementwise_Binary):
    r"""
    比较 mask：:math:`x \ne y`，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} \ne Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereNotEqual()`` 不需要参数。
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereNotEqual


class WhereGreater(Elementwise_Binary):
    r"""
    比较 mask：:math:`x > y`，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} > Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereGreater()`` 不需要参数。
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreater


class WhereGreaterEqual(Elementwise_Binary):
    r"""
    比较 mask：:math:`x \ge y`，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} \ge Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereGreaterEqual()`` 不需要参数。
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreaterEqual


class WhereLess(Elementwise_Binary):
    r"""
    比较 mask：:math:`x < y`，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} < Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereLess()`` 不需要参数。
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLess


class WhereLessEqual(Elementwise_Binary):
    r"""
    比较 mask：:math:`x \le y`，是 :class:`Elementwise_Binary` 的具体子类。

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} \le Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereLessEqual()`` 不需要参数。
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLessEqual
