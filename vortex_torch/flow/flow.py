from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import torch

from ..abs import ContextBase
from ..utils import resolve_dtype

class vFlow(ABC):
    r"""
    flow 风格稀疏注意力模块的基类。

    这个抽象在概念上类似 :class:`torch.nn.Module`，但它专门用于
    **稀疏注意力 flow**。一个具体 flow 需要负责：

    - 维护结构化的 key/value cache；
    - 定义如何从稀疏 page 中 **挑选** 需要看的 page，通常是 top-k 路由；
    - 定义新 token/page 到来后，如何 **更新** / **汇总** cache 摘要。

    Query 张量
    ------------
    传给 :meth:`forward_indexer` 的 query 张量 ``q`` 逻辑形状是：

    .. math::

        q \in \mathbb{R}^{B \times H_q \times D},

    其中：

    - :math:`B` 是类似 batch 的轴，常见情况是 ``batch_size * num_heads``；
    - :math:`H_q` 是每个 batch/head 下的 query 位置数量；
    - :math:`D` 是每个 head 的向量维度。

    实际运行中，``q`` 通常用 :class:`torch.bfloat16` 存储。
    如果把它当成普通三维数组看，可以理解成下面这样::

        q = [
            B0: [
                query_head_0 = [d0, d1, d2, d3, ...],
                query_head_1 = [d0, d1, d2, d3, ...],
                query_head_2 = [d0, d1, d2, d3, ...],
            ],
            B1: [
                query_head_0 = [d0, d1, d2, d3, ...],
                query_head_1 = [d0, d1, d2, d3, ...],
                query_head_2 = [d0, d1, d2, d3, ...],
            ],
        ]

    也就是一共有 :math:`B` 组；每组有 :math:`H_q` 个 query head；
    每个 query head 是一个长度为 :math:`D` 的向量。

    稀疏索引张量
    -------------------
    :meth:`forward_indexer` 生成的稀疏索引张量 ``o`` 逻辑形状是：

    .. math::

        o \in \mathbb{R}^{S_{\text{sparse}} \times 1 \times 1},

    它存的是整数 page 索引。打包后的稀疏长度为：

    .. math::

        S_{\text{sparse}}
        = \sum_{i=0}^{B-1} S_{\text{sparse}, i},

    对每个请求 :math:`i`，如果它有 :math:`S_i` 个候选 page，则：

    .. math::

        S_{\text{sparse}, i}
        = \min\Bigl(
            S_i,\;
            \text{topk\_val}
            + \text{page\_reserved\_bos}
            + \text{page\_reserved\_eos}
        \Bigr).

    这里：

    - ``topk_val`` 是 indexer 选出的 page 数量；
    - ``page_reserved_bos`` 是开头区域固定保留的 page 数，通常对应 BOS 区域；
    - ``page_reserved_eos`` 是末尾区域固定保留的 page 数，通常对应 EOS 区域；

    这些值通常由运行时上下文提供。

    Cache 张量：两种逻辑视图
    --------------------------------
    每个 cache 条目 ``cache[key]`` 都是 rank-3 张量。这里既包括标准的
    ``"k"`` / ``"v"``，也包括 :meth:`create_cache` 声明的额外条目。
    同一个张量会被 **按两种不同逻辑布局理解**：

    1. **Indexer 视图（page-packed）**，用于 :meth:`forward_indexer`：

       .. math::

           \text{cache[key]} \sim
           \mathbb{R}^{S \times r \times c},

       

       :math:`(r, c)` 是每个 key 的内部形状。它要么由
       :meth:`create_cache` 声明，要么对 ``"k"`` / ``"v"`` 隐式给出。

       这里 :math:`S` 是最前面的 page 轴。内部实现里它是打包轴，常记为
       :math:`S_{\mathrm{pack}}`，由所有请求的 page 拼接得到。作为使用者，
       可以先把 :math:`S` 理解成“这个请求的 page 数”；vFlow kernel 和
       :class:`ContextBase` 会自动处理每个请求的 page 数与打包布局之间的映射。
    
    2. **Cache 更新视图（batch-major）**，用于 :meth:`forward_cache`：

       .. math::

           \text{cache[key]} \sim
           \mathbb{R}^{B \times r \times c}.

       最前面的轴是请求/batch 索引 :math:`B`，内部形状 :math:`(r, c)`
       与 indexer 视图相同。

    运行时会通过 :class:`ContextBase`，借助 indptr 数组和布局元数据，
    负责在这两种视图之间完成映射。

    Cache 元信息
    --------------
    子类只需要通过 :meth:`create_cache` 声明 **额外** cache 张量，例如：

        {
            "centroids": (1, head_dim),
            "my_aux_tensor": (block_size, head_dim),
            ...
        }

    辅助方法 :meth:`get_cache_meta_info` 会再自动加入标准条目：

    .. math::

        \text{k} &: (\text{block\_size}, \text{head\_dim}), \\
        \text{v} &: (\text{block\_size}, \text{head\_dim}),

    因此子类不能自己添加 ``"k"`` 或 ``"v"``。

    Token ratio
    -----------
    :meth:`get_token_ratio` 会计算一个简单比例，用来估计相对于一个
    ``k`` / ``v`` page，每个 head 额外用了多少 cache 存储：

    .. math::

        \text{token\_ratio}
        = \sum_{\text{key}}
          \frac{r_{\text{key}} \cdot c_{\text{key}}}
               {\text{block\_size} \cdot \text{head\_dim}}.

    这个比例会忽略最前面的维度，不管它是 :math:`B` 还是 :math:`S`，
    只比较内部形状和基准 ``(block_size, head_dim)`` 的大小。

    子类职责
    -------------------------
    具体 flow 必须实现：

    - :meth:`forward_indexer(q, o, cache, ctx)`:
      使用 :math:`S` 视图下的 cache，根据 query 计算稀疏 page 索引
      或路由分数。

    - :meth:`forward_cache(cache, loc, ctx)`:
      使用 :math:`B`-major 视图和位置信息更新 cache 张量。

    - :meth:`create_cache(block_size, head_dim)`:
      为所有额外 cache 张量声明内部形状 :math:`(r, c)`，不包括
      ``"k"`` 和 ``"v"``。
    """

    def __init__(self):
        super().__init__()

        self.block_size = None
        self.head_dim = None
        self.kv_cache_dtype = None
        self.q_data_type = None
        self.intermediate_dtype = None
        self.cache_meta_info = None
        self.token_ratio = None

    # ------------------------------------------------------------------ #
    # 需要由具体 flow 实现的抽象 API
    # ------------------------------------------------------------------ #
    @abstractmethod
    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: "ContextBase",
    ):
        r"""
        根据 query 和 cache 计算稀疏 page 索引，或等价的路由信息。

        标准形状
        ----------------
        - ``q``（queries）：

          .. math::

              q \in \mathbb{R}^{B \times H_q \times D},

          通常用 :class:`torch.bfloat16` 存储。
          直观地说，就是 ``B`` 组，每组 ``H_q`` 个 query head，
          每个 query head 里有 ``D`` 个数字。

        - ``o``（稀疏索引）：

          .. math::

              o \in \mathbb{R}^{S_{\text{sparse}} \times 1 \times 1},

          整数 dtype，例如 :class:`torch.int32` 或 :class:`torch.int64`。
          打包长度 :math:`S_{\text{sparse}}` 在类 docstring 中定义。

        - ``cache[key]``（indexer 视图）：

          .. math::

              \text{cache[key]}
              \sim \mathbb{R}^{S \times r \times c},

          :math:`(r, c)` 是每个 key 的内部维度，来自
          :meth:`get_cache_meta_info`。

        - ``ctx``:

          :class:`ContextBase` 的实例，携带 page 布局、indptr 数组，
          以及 ``topk_val``、``page_reserved_bos``、
          ``page_reserved_eos`` 等配置。

        接口约定
        --------
        具体实现应该：

        - 按 :math:`S` 视图理解 ``cache``；
        - 使用 ``q`` 和相关 cache 张量给 page 打分或选择 page；
        - 遵守从 ``ctx`` 得到的每个请求边界；
        - 将结果稀疏索引或路由表示原地写入 ``o``。

        ``o`` 里整数的精确定义，例如绝对 page 索引还是偏移量，
        由运行时约定决定，并且必须和后续 kernel 保持一致。
        """
        pass

    @abstractmethod
    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: "ContextBase",
    ):
        r"""
        在 batch-major 视图下更新或重新计算 cache 张量。

        标准形状
        ----------------
        - ``cache[key]``（cache 更新视图）：

          .. math::

              \text{cache[key]}
              \sim \mathbb{R}^{B \times r \times c},

          其中 :math:`B` 是请求数量，:math:`(r, c)` 与 indexer 视图中的
          内部维度相同。

        - ``loc``:

          位置/布局元数据，例如 page 索引或 token 位置。它用于决定在生成
          每个请求的摘要时，应该对哪些 page 或 token 做聚合。

        - ``ctx``:

          执行上下文，与 :meth:`forward_indexer` 中使用的是同类实例。
          它携带运行时参数和布局信息。

        接口约定
        --------
        常见操作包括重新计算每个请求的摘要，例如：

        - 将 ``cache["k"]`` 平均或池化到形状为 ``[B, r, c]`` 的
          ``cache["centroids"]``；
        - 维护 indexer 阶段需要的辅助统计量。

        具体实现可以原地更新 ``cache`` 中的任意条目，只要遵守
        :meth:`get_cache_meta_info` 声明的形状即可。
        """
        pass

    @abstractmethod
    def create_cache(
        self,
        block_size: int,
        head_dim: int,
    ) -> Dict[str, Tuple[Tuple[int, int]]]:
        r"""
        声明非 ``"k"`` / 非 ``"v"`` cache 张量的内部形状。

        这个方法 **不会分配** 张量。它只声明每个 key 的内部维度
        :math:`(r, c)`；运行时会根据 cache 是用于 :meth:`forward_cache`
        还是 :meth:`forward_indexer`，自动加上合适的前导轴
        :math:`B` 或 :math:`S`。

        参数
        ----------
        block_size : int
            每个 block 的 token 数，也就是 ``"k"`` / ``"v"`` cache slot 的
            内部长度。对标准 ``"k"`` 和 ``"v"`` 条目来说，这是第一个内部维度。

        head_dim : int
            head 维度。对标准 ``"k"`` 和 ``"v"`` 条目来说，这是第二个维度。

        返回
        -------
        Dict[str, Tuple[int, int]]
            从 cache 张量名到内部形状 ``(r, c)`` 的映射，不包括 ``"k"`` 和
            ``"v"``。例如::

                {
                    "centroids": (1, head_dim),
                }

        注意
        -----
        ``"k"`` 和 ``"v"`` 是保留 key，**不能** 出现在返回字典里。
        它们会由 :meth:`get_cache_meta_info` 自动加入，内部形状是
        ``(block_size, head_dim)``。
        """
        pass

    # ------------------------------------------------------------------ #
    # 运行时用于分配 cache 和统计 cache 占用的辅助 API
    # ------------------------------------------------------------------ #
    def get_cache_meta_info(
        self
    ) -> Dict[str, Tuple[Tuple[int, int], torch.dtype]]:
        
        return self.cache_meta_info

    def get_token_ratio(
        self, 
        ) -> float:
        
        return self.token_ratio

    def initialize(self,
        block_size: int,
        head_dim: int,
        kv_cache_dtype: Union[torch.dtype, str],
        q_data_type: Union[torch.dtype, str],
        intermediate_dtype: Union[torch.dtype, str] = torch.bfloat16,
        ):
        r"""
        可选初始化方法。运行时分配完 cache 张量后会调用它。

        flow 可以在这里设置自己需要的内部状态或不变量。默认实现不是空操作，
        它会记录基础形状、dtype、cache 元信息和 token_ratio；具体 flow 如有
        额外需求，可以覆盖这个方法。

        参数
        ----------
        block_size : int
            每个 block 的 token 数。
        head_dim : int
            head 维度。
        kv_cache_dtype : torch.dtype or str
            key/value cache 的数据类型。可以传 :class:`torch.dtype`，
            也可以传 :data:`vortex_torch.utils.DTYPE_STR_TO_TORCH`
            中的标准字符串，例如 ``"bfloat16"``、``"fp8_e5m2"``。
        q_data_type : torch.dtype or str
            query 张量的数据类型。字符串约定与 ``kv_cache_dtype`` 相同。
        intermediate_dtype : torch.dtype or str
            中间张量的数据类型，默认是 ``torch.bfloat16``。字符串约定与
            ``kv_cache_dtype`` 相同。
        """

        self.block_size = block_size
        self.head_dim = head_dim
        self.kv_cache_dtype = resolve_dtype(kv_cache_dtype)
        self.q_data_type = resolve_dtype(q_data_type)
        self.intermediate_dtype = resolve_dtype(intermediate_dtype)
        self.token_ratio = 0.0
        raw_cache_meta_info = self.create_cache(block_size, head_dim)
        assert "k" not in raw_cache_meta_info, "create_cache must not declare 'k' key"
        assert "v" not in raw_cache_meta_info, "create_cache must not declare 'v' key"
        
        raw_cache_meta_info["k"] = (block_size, head_dim)
        raw_cache_meta_info["v"] = (block_size, head_dim)

        total_bytes = 0
        # 转成 key -> ((r, c), dtype) 的格式，方便 indexer 和 cache 更新阶段访问
        self.cache_meta_info = {}
        for key, (r, c) in raw_cache_meta_info.items():
            if key in ["k", "v"]:
                dtype = self.kv_cache_dtype
            else:
                dtype = self.intermediate_dtype  # 辅助张量默认用中间 dtype；如有需要可进一步定制
            total_bytes += r * c * torch._utils._element_size(dtype)
            self.cache_meta_info[key] = ((r, c), dtype)
        
        self.token_ratio = total_bytes / (block_size * head_dim * torch._utils._element_size(self.kv_cache_dtype))
