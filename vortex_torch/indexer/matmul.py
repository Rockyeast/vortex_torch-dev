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
        self._param = None        # set when the y operand is a Vortex.Parameter

    # ---------------- profile 阶段 ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""trace 阶段：校验 ``x`` ``[B|S, N_x, K]`` / ``y`` ``[S, N_y, K]``
        且 ``K`` 一致，注册这个算子，并返回一个描述 ``[S, N_y, N_x]``
        输出的 ``vTensor`` 视图。详见类 docstring。

        如果 ``y`` 是 :class:`~vortex_torch.indexer.Parameter`
        （``FORMAT.PARAMETER``，跨 batch 共享的学习常量），这个算子不会融合进
        per-workload kernel，而是对当前 layer 的权重切片走独立的
        ``Schedule.S`` ``torch.matmul`` 路径。详见 :meth:`_profile_param`。"""
        prefix = self._prefix()

        # 类型检查
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # 维度数量 / 形状检查
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        # 跨 batch 共享的学习常量操作数 -> 走 Schedule.S torch.matmul 路径。
        # 这里必须放在普通 K 维检查之前：Parameter 收缩的是展平后的 activation
        # 维度（H*d），不一定等于普通 fused path 里的 x 最后一维。
        if y._format == FORMAT.PARAMETER:
            return self._profile_param(x, y, ctx)

        # 普通 fused path：K 维必须一致。
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

    # ------------------------------------------------------------------ #
    # Parameter（跨 batch 共享常量）路径：Schedule.S + torch.matmul
    # ------------------------------------------------------------------ #
    def _profile_param(self, x: vTensor, y, ctx: Context) -> vTensor:
        r"""``y`` 是 :class:`Parameter`，也就是烘焙进算子的常量权重。它收缩
        最后一维 ``K``（要求 ``x.shape[2] == K``），运行时由
        :meth:`compute_param` 计算。因为它走 ``Schedule.S`` ``torch.matmul``，
        大权重不会进入 tiled workload kernel。

        支持两种权重形状，输出都保持 3D BATCHED：

        * **plain**：权重值 ``[L, N_y, K]``，
          ``O[b,a,nx] = Σ_k W[a,k] x[b,nx,k]`` -> ``[B, N_y, N_x]``。
          这是标准 GeMM 收缩。
        * **per-head / batched**：权重值 ``[L, H, N_y, K]``。此时 ``x`` 把 head
          折进 ``N_x`` 轴（``x.shape[1] = H*N_x``）；算子内部做 batched matmul
          ``O[b,h,a,c] = Σ_k W[h,a,k] x[b,h,c,k]``，再把 head 折回 row 轴，
          输出 3D ``[B, H*N_y, N_x]``。4D 形态只在 launcher 内部临时存在，
          计算图 tensor 仍然保持 3D。"""
        prefix = self._prefix()
        assert x._format == FORMAT.BATCHED, (
            f"{prefix}a Parameter operand requires a BATCHED activation (per-request); "
            f"got x._format={x._format}"
        )
        assert int(x.shape[2]) == int(y.shape[2]), (
            f"{prefix}K mismatch: x.shape[2]={x.shape[2]} vs Parameter K={y.shape[2]} "
            f"(Reshape the activation so its last dim matches the Parameter's K)"
        )
        self.schedule = Schedule.S
        self._param = y
        # 在这里把权重一次性放到目标 device 并转成 bf16。这个阶段发生在编译期 /
        # cuda graph capture 之前，同时快照 host 侧 layer 查找表。
        y.materialize(device=x.device, dtype=torch.bfloat16)

        Ny = int(y.shape[1])
        if y.value.dim() == 4:                       # per-head [L, H, N_y, K]
            H = int(y.value.shape[1])
            assert int(x.shape[1]) % H == 0, (
                f"{prefix}per-head Parameter (H={H}) needs x.shape[1] divisible by H, "
                f"got x.shape[1]={x.shape[1]}"
            )
            Nx = int(x.shape[1]) // H
            out_N = H * Ny
        else:                                        # plain [L, N_y, K]
            Nx = int(x.shape[1])
            out_N = Ny

        self.output_format = FORMAT.BATCHED
        self.output_buffer = vTensor(
            shape=(0, out_N, Nx), dtype=ctx.vortex_dtype, device=x.device,
            _format=FORMAT.BATCHED, tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        # The Parameter is NOT a graph input (it is baked on the op); only x is.
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
        return self.output_buffer

    @torch.no_grad()
    def compute_param(self, x: torch.Tensor, cur_layer: int) -> torch.Tensor:
        r"""运行时路径（``Schedule.S`` launcher）。先取当前 layer 的权重：
        ``W = self._param.gather(cur_layer)``，权重已经是 device 上的 bf16。

        plain 权重 ``[N_y, K]``：计算
        ``O = Σ_k W[a,k] x[b,nx,k]`` -> ``[B, N_y, N_x]``。

        per-head 权重 ``[H, N_y, K]``：把 ``x`` 从 ``[B, H*N_x, K]`` reshape 成
        ``[B, H, N_x, K]``，做 batched 乘法
        ``O[b,h,a,c]=Σ_k W[h,a,k] x[b,h,c,k]``，最后折回 3D
        ``[B, H*N_y, N_x]``。

        注意这里没有 ``.to(device)`` / ``.item()``，因此对 CUDA graph capture
        是安全的；4D 形态只在 launcher 内部临时存在。"""
        assert x.dtype == torch.bfloat16, (
            f"{self._prefix()}compute_param expects a bf16 activation, got {x.dtype}"
        )
        W = self._param.gather(cur_layer)                      # [Ny,K] or [H,Ny,K] bf16
        if W.dim() == 2:                                        # plain
            return torch.einsum("nk,bxk->bnx", W, x)          # [B, Ny, Nx]
        # per-head batched: x [B, H*Nx, K] -> [B, H, Nx, K]; fold head back into rows.
        H, Ny, _ = W.shape
        B, NxH, K = x.shape
        x4 = x.reshape(B, H, NxH // H, K)                       # [B, H, Nx, K]
        O = torch.einsum("hak,bhck->bhac", W, x4)             # [B, H, Ny, Nx]
        return O.reshape(B, H * Ny, NxH // H)                  # [B, H*Ny, Nx]
