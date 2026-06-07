"""vortex 计算图和编译器里使用的纯元数据 tensor 类型。

:class:`vTensor` **不携带真实存储**。它只是一个很小的描述对象，
记录系统在建图和 codegen 阶段理解一个 tensor 所需的信息：

  * ``shape``        — int 元组，表示真实逻辑形状。
  * ``padded_shape`` — int 元组，每个相关维度向上补到最近的 2 的幂。
                       只有 ``shape[1]`` / ``shape[2]`` 可能和 ``shape``
                       不同；``shape[0]`` 是前导轴，不受 Triton block-shape
                       必须为 2 的幂这个约束影响。
  * ``dtype``        — :class:`torch.dtype`
  * ``device``       — :class:`torch.device` / str / ``None``
  * ``_format``      — :class:`FORMAT`，即 BATCHED / RAGGED / PAGED
  * ``tensor_id``    — int，编译器使用的计算图级别身份标识

为什么需要 ``padded_shape``：Triton 的 block-shape constexpr
（例如 ``tl.zeros``、``tl.arange``、``tl.reshape``、
``tl.make_block_ptr.block_shape`` 等）通常要求每个维度是 2 的幂。
真实模型里会出现非 2 的幂维度，例如 Qwen3-14B 中
``num_attention_heads // num_key_value_heads == 5``。编译器会根据
``padded_shape`` 生成 tile 大小，但内存寻址数学（stride、每行 offset、
``Mean`` 中的除数等）仍然基于真实 ``shape``。当
``padded_shape != shape`` 时，codegen 会额外生成 load/store mask，
只读写真实形状对应的 lane。如果 ``shape`` 已经是 2 的幂，
则 ``padded_shape == shape``，不会生成额外 mask，2 的幂模型没有额外开销。

它还提供 ``dim()``，与 ``torch.Tensor`` 保持接口一致，这样已有的
profile 阶段校验代码（例如 ``assert x.dim() == 3``）可以继续工作。

这里刻意 **不支持** torch op，没有 ``__torch_function__`` 覆盖，也不继承
``torch.Tensor``。``vTensor`` 只是元数据。运行时/execute 路径真正参与计算的
真实张量仍然是普通 ``torch.Tensor`` 实例。
"""

from __future__ import annotations
import torch
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union


class FORMAT(Enum):
    """tensor 的存储/布局格式。

    属性:
        BATCHED: 标准 dense batch 张量，例如 ``[B, N, D]``。
        RAGGED: ragged 张量，每个 batch 的序列长度或元素数量可以不同。
        PAGED: paged 张量，用于被切成 page/chunk 的大数据或流式数据。
        PARAMETER: 跨 batch 共享的学习常量，例如
            :class:`~vortex_torch.indexer.Parameter`。它没有 request/page 轴，
            值会烘焙进编译后的函数。收到 PARAMETER 操作数的算子
            （例如 ``GeMM``）会走独立的 ``Schedule.S`` ``torch.matmul``，
            而不是融合进 per-workload kernel；这样大权重不会进入 tiled kernel。
    """

    BATCHED = 0
    RAGGED = 1
    PAGED = 2
    PARAMETER = 3


def _next_pow2(n: int) -> int:
    """把 ``n`` 向上取整到最近的 2 的幂。

    ``_next_pow2(1) == 1``。如果 ``n <= 0``，返回 ``1``。这是防御性处理；
    正常情况下编译器不会查询非正维度。正整数且本来就是 2 的幂时，
    返回输入本身。
    """
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _compute_padded_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    """返回补齐后的 ``shape``，只把 ``shape[1]`` 和 ``shape[2]`` 向上补到
    最近的 2 的幂。其他轴保持不变。

    原因：Triton 对每个 workload tile（``[chunk, D_0, D_1]``）的第 1、
    第 2 维 block-shape constexpr 通常要求是 2 的幂。前导轴
    ``shape[0]`` 是 ragged/paged buffer 数量，由 ``workload_chunk_size`` /
    ``num_blocks_per_page`` 决定，配置上已经是 2 的幂。只 padding 内部两轴，
    改动最小，也避免影响依赖 ``shape[0]`` 的寻址数学。
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        return shape
    padded = list(shape)
    for i in (1, 2):
        if i < len(padded):
            padded[i] = _next_pow2(padded[i])
    return tuple(padded)


class vTensor:
    """纯元数据虚拟 tensor。

    携带 graph builder、compiler 和 codegen 层需要的描述字段。它不拥有任何
    GPU / CPU 内存，并且刻意不能参与 torch op。真正想对数据做计算的代码，
    应该单独持有 ``torch.Tensor``；``vTensor`` 只用于计算图记账。
    """

    __slots__ = ("shape", "padded_shape", "dtype", "device", "_format", "tensor_id")

    shape: tuple
    padded_shape: tuple
    dtype: torch.dtype
    device: Optional[Union[torch.device, str]]
    _format: FORMAT
    tensor_id: int

    def __init__(
        self,
        shape: Sequence[int] = (),
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[torch.device, str]] = None,
        _format: FORMAT = FORMAT.BATCHED,
        tensor_id: int = -1,
        padded_shape: Optional[Sequence[int]] = None,
    ) -> None:
        if not isinstance(tensor_id, int):
            raise TypeError(f"tensor_id must be int, got {type(tensor_id).__name__}")
        if not isinstance(_format, FORMAT):
            raise TypeError(f"_format must be a FORMAT enum, got {type(_format).__name__}")

        # 标准化 ``shape``，让 ``shape[i]``、``len(shape)`` 和 ``tuple(shape)``
        # 的行为都类似 ``torch.Tensor.shape``。
        self.shape = tuple(int(s) for s in shape)
        # 默认根据 ``shape`` 推导 ``padded_shape``。只有在手动从已有 padded 视图
        # 构造 tensor 时，调用方才需要覆盖它；这种情况很少，主要用于 pickle/copy 路径。
        if padded_shape is None:
            self.padded_shape = _compute_padded_shape(self.shape)
        else:
            self.padded_shape = tuple(int(s) for s in padded_shape)
        self.dtype = dtype
        self.device = device
        self._format = _format
        self.tensor_id = tensor_id

    # -------- 形状辅助方法 --------
    def dim(self) -> int:
        """维度数量；对齐 :meth:`torch.Tensor.dim`。"""
        return len(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    def size(self, dim: Optional[int] = None):
        """对齐 :meth:`torch.Tensor.size`。"""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def needs_padding(self) -> bool:
        """当且仅当 ``padded_shape != shape`` 时返回 True。

        这表示至少一个内部维度不是 2 的幂，codegen 必须生成 load/store mask。
        """
        return self.padded_shape != self.shape

    # -------- 字符串表示 --------
    def __repr__(self) -> str:
        pad = "" if self.padded_shape == self.shape else f", padded={self.padded_shape}"
        return (
            f"vTensor(shape={self.shape}{pad}, dtype={self.dtype}, "
            f"device={self.device}, _format={self._format}, "
            f"tensor_id={self.tensor_id})"
        )

    # -------- pickle / copy --------
    def __reduce__(self):
        return (
            _rebuild_vtensor,
            (self.shape, self.dtype, self.device, self._format, self.tensor_id, self.padded_shape),
        )


def _rebuild_vtensor(shape, dtype, device, _format, tensor_id, padded_shape=None):
    return vTensor(
        shape=shape, dtype=dtype, device=device, _format=_format,
        tensor_id=tensor_id, padded_shape=padded_shape,
    )


# -------- 便捷构造函数 --------
def as_vtensor(
    x: Any = None,
    _format: FORMAT = FORMAT.BATCHED,
    tensor_id: int = -1,
    *,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> vTensor:
    """构造一个 :class:`vTensor`。

    支持三种调用方式。它们都会返回新的 ``vTensor``；如果输入已经是
    ``vTensor``，则会在原对象上重新打标签并返回同一个对象。

    1. **给已有 vTensor 重新打标签**：``as_vtensor(vt, fmt, tid)``
       会原地覆盖 ``vt._format`` 和 ``vt.tensor_id``，并返回 ``vt``。
       当调用方想把已有 tensor 描述符用新的 id 加入计算图时很有用。
       ``padded_shape`` 会被保留。

    2. **从 torch.Tensor 提取元数据**：``as_vtensor(real, fmt, tid)``
       会从 ``real`` 读取 ``shape``、``dtype``、``device``，并返回一个全新的
       ``vTensor``。``padded_shape`` 会由 ``shape`` 推导。原始 tensor
       **不会** 被保留；vTensor 是纯元数据。

    3. **通过 kwargs 直接构造**：
       ``as_vtensor(_format=fmt, tensor_id=tid, shape=..., dtype=..., device=...)``.
       当没有真实 torch tensor 可用时使用这种方式；当编译路径完全虚拟化后，
       这是常见情况。
    """
    if isinstance(x, vTensor):
        x._format = _format
        x.tensor_id = tensor_id
        return x

    if isinstance(x, torch.Tensor):
        return vTensor(
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            _format=_format,
            tensor_id=tensor_id,
        )

    if x is None:
        return vTensor(
            shape=shape if shape is not None else (),
            dtype=dtype if dtype is not None else torch.bfloat16,
            device=device,
            _format=_format,
            tensor_id=tensor_id,
        )

    raise TypeError(
        f"as_vtensor: cannot convert {type(x).__name__} to vTensor; "
        "pass a torch.Tensor, an existing vTensor, or shape/dtype/device kwargs."
    )


if __name__ == "__main__":
    # 直接构造
    a = vTensor(shape=(2, 3, 4), dtype=torch.bfloat16, device="cuda:0",
                _format=FORMAT.RAGGED, tensor_id=0)
    print("a:", a, "padded:", a.padded_shape, "needs_padding:", a.needs_padding())

    # Pow2 stays unchanged
    p = vTensor(shape=(8, 4, 128), dtype=torch.bfloat16,
                _format=FORMAT.BATCHED, tensor_id=1)
    print("p:", p, "needs_padding:", p.needs_padding())

    # Non-pow2 inner dim rounds up
    q = vTensor(shape=(8, 5, 128), dtype=torch.bfloat16,
                _format=FORMAT.BATCHED, tensor_id=2)
    print("q:", q, "padded:", q.padded_shape, "needs_padding:", q.needs_padding())
    assert q.padded_shape == (8, 8, 128)
