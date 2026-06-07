import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Save(vOp):
    r"""
    把每个 page 的 indexer 状态持久保存到预先分配好的 cache 字段里，
    这样跨 decode step 也能继续使用。它通常和 :class:`Load` 配套使用。

    :Math:
        .. math::

            O \leftarrow X

        这是格式/布局拷贝：``RAGGED`` -> ``PAGED``，不做数学运算。
    :__init__: ``Save()``；不需要参数。
    :__call__: ``op(x, o, ctx=ctx)``；``x`` 是 ``[S, D_0, D_1]``（``RAGGED``），
        会被 **原地写入** 预先分配的 ``o``（``PAGED``，内部 ``D_0`` / ``D_1``
        必须匹配）。没有返回值。
    :Note: 这是持久状态模式里的写入侧；使用 ``Save`` 的 flow 要求 engine 设置
        ``disable_radix_cache=True``。
    """

    # 按 x_format 分发到对应的输出格式。
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.PAGED,
        # 如果以后支持其他格式，在这里继续加。
    }

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.W

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` / ``o``，两者都必须是 rank-3，内部
        ``D_0`` / ``D_1`` 匹配，格式兼容。随后注册这个算子，并把 ``o``
        作为输出视图返回；这里不会分配新的 buffer。"""
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"
        assert o.dim() == 3, f"{prefix}expected 3D o [S, D0, D1], got {tuple(o.shape)}"

        # 形状检查：D0/D1 必须匹配。S 轴可能因为布局不同而不同，具体实现会处理。
        assert x.shape[1] == o.shape[1], (
            f"{prefix}expected matching D0: x.shape[1]={x.shape[1]} vs o.shape[1]={o.shape[1]}"
        )
        assert x.shape[2] == o.shape[2], (
            f"{prefix}expected matching D1: x.shape[2]={x.shape[2]} vs o.shape[2]={o.shape[2]}"
        )

        # 根据 x 的格式分发
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        # 输出格式必须和分发表推导出的格式一致
        assert o._format == self.output_format, (
            f"{prefix}output format mismatch. Expected {self.output_format}, got {o._format}"
        )

        # 设备一致性检查
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Save 是一个“带副作用的写入算子”：它会写回调用方提供的 cache 字段。
        # 这里故意不把 ``o.tensor_id`` 登记成由 Save 产生，也就是不改
        # ``output_tensor_to_op_list``。原因是：如果图里别处有 Load 读取同一个
        # cache 字段，它应该读到上一个 step 的值，而不是本次 Save 写入后的值。
        # 如果把 producer 覆盖成 Save，就可能在 DAG 里制造 Load -> Save 的环。
        #
        # 所以这里改为把当前 op id 记录到 ``side_effect_op_ids``。compiler 会从
        # 这个集合开始做 op DFS，确保 Save 不会被 DCE 当成无用节点删掉；同时目标
        # tensor 会被提升成最终输出，让子图为它生成 ``tl.store``。
        save_op_id = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([o.tensor_id])
        ctx.side_effect_op_ids.append(save_op_id)

        return o


class Load(vOp):
    r"""
    读回由 :class:`Save` 持久保存的 per-page 值。它是跨 decode step
    持久状态模式里的读取侧。

    :Math:
        .. math::

            Y \leftarrow X

        这是格式/布局拷贝：``PAGED`` -> ``RAGGED``，不做数学运算。
    :__init__: ``Load()``；不需要参数。
    :__call__: ``y = op(x, ctx=ctx)``；``x`` 是 ``[S, D_0, D_1]``（``PAGED``），
        返回一个内部形状相同、但格式为 ``RAGGED`` 的新视图。
    :Note: 持久状态模式里的读取侧，见 :class:`Save`。
    """

    # 按 x_format 分发到对应的输出格式。
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.PAGED: FORMAT.RAGGED,
        # 如果以后支持其他格式，在这里继续加。
    }

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[S, D_0, D_1]``，注册这个算子，并返回
        一个新分配的输出 ``vTensor`` 视图。输出内部形状相同，格式为
        ``RAGGED``。"""
        prefix = self._prefix()

        # 类型和维度数量检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"

        # 根据 x 的格式分发
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        # 纯元数据 vTensor，带一个新的 tensor_id。这里和 Softmax 的模式一致，
        # 方便 compiler graph 跟踪这个 buffer。
        D0, D1 = x.shape[1], x.shape[2]
        self.output_buffer = vTensor(
            shape=(0, D0, D1),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # 在上下文里记录图结构，和 Softmax / Conv1d 的约定一致。
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
