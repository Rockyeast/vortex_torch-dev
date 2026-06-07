import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ElementwiseOpType, Schedule

class Elementwise(vOp):
    r"""
    一元逐元素算子：对每个元素独立套一个标量函数。

    :Math:
        .. math::

            Y_{s,c,d} = f(X_{s,c,d};\, \alpha, \beta),

        其中 :math:`f` 由具体子类决定，例如 ReLU / SiLU / Sigmoid /
        affine / abs / log / exp。
    :__init__: ``Elementwise(alpha=1.0, beta=1.0)``；标量参数
        :math:`\alpha`、:math:`\beta` 会被 :math:`f` 使用。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` 是 ``[S, C, D]``，输出形状相同。
        只有输入是 ``BATCHED`` 时输出才是 ``BATCHED``，否则输出是 ``RAGGED``。
    :Note: 请使用具体子类：:class:`Relu`、:class:`Silu`、
        :class:`Sigmoid`、:class:`Add_Mul`、:class:`Abs`、:class:`Log`、
        :class:`Exp`。
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.op_type: Optional[ElementwiseOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[S, C, D]``，注册这个算子，并返回一个
        同形状的 ``vTensor`` 视图。只有 ``x`` 是 ``BATCHED`` 时输出才是
        ``BATCHED``，否则输出是 ``RAGGED``。"""
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, C, D]. Got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # 只有输入是 BATCHED 时，输出才保持 BATCHED；否则输出是 RAGGED。
        # BATCHED 可以理解成 S 轴已经是规整布局；RAGGED 表示 S 轴是 page-packed。
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        C, D = x.shape[1], x.shape[2]

        # 纯元数据 vTensor，不需要 torch.empty 分配真实内存。
        self.output_buffer = vTensor(
            shape=(0, C, D),
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

        return self.output_buffer


class Relu(Elementwise):
    r"""
    带阈值和兜底值的 ReLU 类激活，是 :class:`Elementwise` 的具体子类。

    :Math:
        .. math::

            f(x;\alpha,\beta) = \begin{cases} x, & x \ge \alpha, \\ \beta, & x < \alpha. \end{cases}
    :__init__: ``Relu(alpha=0.0, beta=0.0)``；阈值 :math:`\alpha`，
        兜底值 :math:`\beta` 会在 :math:`x<\alpha` 时使用。
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu



class Silu(Elementwise):
    r"""
    带 affine 预变换的 SiLU 类激活，是 :class:`Elementwise` 的具体子类。

    :Math:
        .. math::

            f(x;\alpha,\beta) = \frac{x}{1 + \exp(\beta x + \alpha)}.
    :__init__: ``Silu(alpha=0.0, beta=0.0)``；指数内部的 bias
        :math:`\alpha` 和 slope :math:`\beta`。
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu
        

class Sigmoid(Elementwise):
    r"""
    带 affine 参数的 Sigmoid 激活，是 :class:`Elementwise` 的具体子类。

    :Math:
        .. math::

            f(x;\alpha,\beta) = \frac{1}{1 + \exp(\beta x + \alpha)}.
    :__init__: ``Sigmoid(alpha=0.0, beta=0.0)``；指数内部的 bias
        :math:`\alpha` 和 slope :math:`\beta`。
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid
        

class Add_Mul(Elementwise):
    r"""
    affine 变换 :math:`\beta x + \alpha`，是 :class:`Elementwise` 的具体子类。

    :Math:
        .. math::

            f(x;\alpha,\beta) = \beta x + \alpha.
    :__init__: ``Add_Mul(alpha=0.0, beta=1.0)``；加法项 :math:`\alpha`，
        乘法项 :math:`\beta`，默认值相当于恒等变换。
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    r"""
    affine 变换后取绝对值，是 :class:`Elementwise` 的具体子类。

    :Math:
        .. math::

            f(x;\alpha,\beta) = \lvert \beta x + \alpha \rvert.
    :__init__: ``Abs(alpha=0.0, beta=1.0)``；绝对值内部的加法项
        :math:`\alpha` 和乘法项 :math:`\beta`。
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs


class Log(Elementwise):
    r"""
    affine 变换后取自然对数，是 :class:`Elementwise` 的具体子类。

    :Math:
        .. math::

            f(x;\alpha,\beta) = \log(\beta x + \alpha).
    :__init__: ``Log(alpha=0.0, beta=1.0)``；对数内部的加法项
        :math:`\alpha` 和乘法项 :math:`\beta`。
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Log


class Exp(Elementwise):
    r"""
    affine 变换后取指数，是 :class:`Elementwise` 的具体子类。

    :Math:
        .. math::

            f(x;\alpha,\beta) = \exp(\beta x + \alpha).
    :__init__: ``Exp(alpha=0.0, beta=1.0)``；指数内部的加法项
        :math:`\alpha` 和乘法项 :math:`\beta`。
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Exp
