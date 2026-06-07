import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Softmax(vOp):
    r"""
    沿 packed sequence 轴做分段 scaled softmax。

    :Math:
        前导轴是把 :math:`B` 个请求各自的片段
        :math:`\mathcal{I}_b` 打包拼接后的结果，总长度是
        :math:`S = \sum_b S_b`。对每个请求片段和每个通道
        :math:`(d_0,d_1)`，softmax 只在该片段 **内部** 计算：

        .. math::

            Y_{s,d_0,d_1} = \frac{\exp(\text{scale}\cdot X_{s,d_0,d_1})}
                 {\sum_{s'\in\mathcal{I}_b}\exp(\text{scale}\cdot X_{s',d_0,d_1})},
            \qquad s\in\mathcal{I}_b.
    :__init__: ``Softmax(dim=0, scale=1.0)``；``dim`` 必须是 ``0``，
        也就是 packed S 轴；``scale`` 会在指数运算前乘到 logits 上。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` ``[S, D_0, D_1]`` → 输出同形状。
    :Note: 只支持 ``RAGGED``。
    """

    # 根据 x_format 分发表，得到最终输出格式。
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
        # 如果后续添加更多 kernel，可以在这里扩展其他格式：
        # FORMAT.PAGED: FORMAT.PAGED,
    }

    def __init__(self, dim: int = 0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.S
        # 构造时校验 dim
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[S, D_0, D_1]``，根据 ``x._format``
        分发实现，注册这个算子，并返回输出 ``vTensor`` 视图。
        这里表示沿 ``dim=0`` 做分段 softmax。"""
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # 根据输入格式分发
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
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


class Normalize(vOp):
    r"""
    沿 packed sequence 轴做分段 :math:`L_2` 归一化。

    :Math:
        前导轴是把 :math:`B` 个请求各自的片段
        :math:`\mathcal{I}_b` 打包拼接后的结果。对每个请求片段和每个通道
        :math:`(d_0,d_1)`，每个值都会除以该片段内部的 :math:`L_2` 范数：

        .. math::

            Y_{s,d_0,d_1} = \frac{X_{s,d_0,d_1}}
                 {\sqrt{\sum_{s'\in\mathcal{I}_b} X_{s',d_0,d_1}^2}},
            \qquad s\in\mathcal{I}_b.
    :__init__: ``Normalize(dim=0)``；``dim`` 必须是 ``0``，也就是 packed S 轴。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` ``[S, D_0, D_1]`` → 输出同形状。
    :Note: 只支持 ``RAGGED``。
    """

    # 根据 x_format 分发表，得到最终输出格式。
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
        # 如果后续添加更多 kernel，可以在这里扩展其他格式：
        # FORMAT.PAGED: FORMAT.PAGED,
    }

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.S

        # 构造时校验 dim
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[S, D_0, D_1]``，根据 ``x._format``
        分发实现，注册这个算子，并返回输出 ``vTensor`` 视图。
        这里表示沿 ``dim=0`` 做分段 :math:`L_2` 归一化。"""
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # 根据输入格式分发
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
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


class Conv1d(vOp):
    r"""
    沿 packed sequence 轴做分段 depth-wise causal 1-D convolution。

    :Math:
        前导轴是把 :math:`B` 个请求各自的片段打包拼接后的结果。在每个片段内，
        每个通道 :math:`(d_0,d_1)` 都会使用自己的 :math:`K`-tap causal filter
        :math:`W\in\mathbb{R}^{K\times D_0\times D_1}` 做卷积：

        .. math::

            Y_{s,d_0,d_1} = \sum_{k=0}^{K-1} W_{k,d_0,d_1}\, X_{s-k,\,d_0,d_1},

        如果 :math:`s-k` 已经越过当前片段开头，则令 :math:`X_{s-k}=0`。
        该算子只在中间范围
        :math:`[b_{\text{bos}},\,S_b-b_{\text{eos}})` 上运行，即避开
        ``ctx.block_reserved_bos`` / ``block_reserved_eos``；保留的 BOS/EOS 行
        既不读也不写。
    :__init__: ``Conv1d(weight, dim=0, dtype=torch.bfloat16, device=None)``；
        ``weight`` 是形状为 ``[K, D_0, D_1]`` 的 Python 嵌套 list，
        kernel size :math:`K` = ``len(weight)``；``dim`` 必须是 ``0``。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` ``[S, D_0, D_1]`` → 输出同形状；
        ``weight`` 的内部维度必须匹配 ``(D_0, D_1)``。
    :Note: 只支持 ``RAGGED``。
    """

    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
    }

    def __init__(
        self,
        weight: list,
        dim: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert isinstance(weight, list), (
            f"Conv1d: weight must be a Python list, got {type(weight)}"
        )
        weight_tensor = torch.tensor(weight, dtype=dtype, device=device)
        assert weight_tensor.dim() == 3, (
            f"Conv1d: weight must be a 3D nested list [K, D0, D1], "
            f"got shape {tuple(weight_tensor.shape)}"
        )
        self.dim = dim
        self.weight = weight_tensor
        self.kernel_size = weight_tensor.shape[0]
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.S

        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[S, D_0, D_1]``，其中 weight 的内部维度
        必须匹配；把 ``weight`` 迁移到 ``x`` 的设备上；注册这个算子；
        并返回输出 ``vTensor`` 视图。"""
        prefix = self._prefix()

        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert self.weight.shape[1] == x.shape[1] and self.weight.shape[2] == x.shape[2], (
            f"{prefix}weight inner dims {tuple(self.weight.shape[1:])} "
            f"must match input inner dims {tuple(x.shape[1:])}"
        )

        # ``__init__`` 会把 ``self.weight`` 放到用户传入的设备上；如果用户没传，
        # 默认在 CPU 上。但生成的 Triton kernel 会从 GPU 读取 ``weight``。
        # 因此这里迁移设备：``profile`` 是第一次知道推理设备的地方。
        # 如果已经在正确设备上，``.to`` 是 no-op，所以每次 re-profile 都调用也安全。
        if self.weight.device != x.device:
            self.weight = self.weight.to(device=x.device)
        if not self.weight.is_contiguous():
            self.weight = self.weight.contiguous()

        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
