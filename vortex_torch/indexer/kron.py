import torch
from typing import Tuple, Optional, Union, Iterable
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class Kron(vOp):
    r"""
    在可配置的内部轴上做 Kronecker 展开。

    被选中的内部轴（``dim``）会做 Kronecker 展开；没有被选中的内部轴会做
    逐元素乘法，并支持 broadcasting。最前面的 :math:`S` 轴始终逐元素对齐，
    不做展开。

    :Math:
        对 :math:`X\in\mathbb{R}^{S\times x_1\times x_2}` 和
        :math:`Y\in\mathbb{R}^{S\times y_1\times y_2}`:

        .. math::

            \begin{aligned}
            \text{dim}=(1,2):\quad & O_{s,\,i\,y_1+j,\,k\,y_2+l} = X_{s,i,k}\,Y_{s,j,l}, \\
            \text{dim}=(1,):\quad  & O_{s,\,i\,y_1+j,\,d} = X_{s,i,d}\,Y_{s,j,d}, \\
            \text{dim}=(2,):\quad  & O_{s,\,c,\,k\,y_2+l} = X_{s,c,k}\,Y_{s,c,l}.
            \end{aligned}
    :__init__: ``Kron(dim=(1, 2))``；指定要展开的内部轴，每个轴只能是
        ``1`` 或 ``2``。没有列出的轴必须相等，或者可以 broadcast。
    :__call__: ``o = op(x, y, ctx=ctx)``；``x`` 是 ``[S, x_1, x_2]``，
        ``y`` 是 ``[S, y_1, y_2]``。被展开的轴输出大小是
        ``x.shape[a] * y.shape[a]``；broadcast 轴输出大小是
        ``max(x.shape[a], y.shape[a])``。只有两个输入都是 ``BATCHED`` 时，
        输出才是 ``BATCHED``。
    """

    def __init__(self, dim: Union[int, Iterable[int]] = (1, 2)):
        super().__init__()
        # 把 ``dim`` 标准化成排好序、没有重复元素、只包含 1/2 的 tuple。
        if isinstance(dim, int):
            dim_tuple: Tuple[int, ...] = (dim,)
        else:
            dim_tuple = tuple(dim)
        cls = self.__class__.__name__
        assert len(set(dim_tuple)) == len(dim_tuple), (
            f"{cls}.__init__: duplicate axes in dim={dim_tuple!r}"
        )
        for a in dim_tuple:
            assert isinstance(a, int) and a in (1, 2), (
                f"{cls}.__init__: dim entries must be 1 or 2, got {a!r}"
            )
        assert len(dim_tuple) >= 1, (
            f"{cls}.__init__: dim must list at least one axis, got empty"
        )
        self.dim: Tuple[int, ...] = tuple(sorted(dim_tuple))

        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # 融合进 per-workload kernel：每个 block tile 已经按 3D ``(W, C, D)``
        # 加载，Kronecker 展开也只发生在这个 tile 内部。
        self.schedule = Schedule.W

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""
        校验输入，创建输出 buffer，并返回一个 :class:`vTensor` 视图。

        对每个内部轴 ``a``（只能是 ``1`` 或 ``2``）：

        - 如果 ``a`` 在 :attr:`dim` 里，输出大小是
          ``x.shape[a] * y.shape[a]``，也就是 Kronecker 展开；
        - 否则 ``x.shape[a]`` 和 ``y.shape[a]`` 必须相等，或者其中一个是
          ``1`` 以便 broadcast；输出大小是 ``max(x.shape[a], y.shape[a])``。

        Parameters
        ----------
        x : vTensor
            左输入，逻辑形状是 ``[S, x1, x2]``。
        y : vTensor
            右输入，逻辑形状是 ``[S, y1, y2]``。
        ctx : Context
            执行上下文，用来记录图结构和辅助内存。

        Returns
        -------
        vTensor
            包装输出 buffer 的 :class:`vTensor` 视图。

        Raises
        ------
        AssertionError
            如果输入不是 :class:`vTensor`，rank 不是 3，非 Kron 轴不相等也
            不能 broadcast，或者 ``x`` 和 ``y`` 不在同一个 device 上。
        """
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs [S, C, D]; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )

        # 只有两个输入都是 BATCHED 时，输出才是 BATCHED；否则输出是 RAGGED。
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

        # 每个轴分别决定输出大小：在 ``dim`` 里的轴做 Kron 展开；
        # 不在 ``dim`` 里的轴做逐元素/broadcast。
        out_inner: Tuple[int, ...] = tuple(
            self._resolve_axis(x.shape[a], y.shape[a], a) for a in (1, 2)
        )
        C_out, D_out = out_inner

        # 纯元数据 vTensor，不需要 torch.empty 分配真实内存。
        self.output_buffer = vTensor(
            shape=(0, C_out, D_out),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # 在上下文里记录图结构，和 Elementwise_Binary 的模式一致。
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer

    # ---------------- 辅助函数 ----------------
    def _resolve_axis(self, sx: int, sy: int, axis: int) -> int:
        """计算某一个内部轴的输出大小。"""
        if axis in self.dim:
            return sx * sy
        # 逐元素轴：两个大小必须相等，或者其中一个是 1 以便 broadcast。
        assert (sx == sy) or (sx == 1) or (sy == 1), (
            f"{self._prefix()}dim={self.dim}: axis {axis} not in dim and not "
            f"broadcastable (x.shape[{axis}]={sx}, y.shape[{axis}]={sy})"
        )
        return max(sx, sy)
