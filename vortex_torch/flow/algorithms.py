import torch
from typing import Dict

from .flow import vFlow
from ..indexer import topK, approxTopK, GeMV, Softmax, Max, Sum, GeMM, Maximum, Multiply, Add, L2Norm, Save, Load, Mean, MaskSlice, Kron
from ..cache import Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm, Fill as CFill, MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave, MeanInterleave as CMeanInterleave
from ..abs import ContextBase
from .registry import register

# 注意：算子对象不能复用，即使它们语义一样。每个算子内部会初始化自己的中间缓冲区。
# 例如 QUEST 里需要两个 Multiply，就必须定义两个 Multiply 对象。

# 在 forward_indexer 里，q 逻辑上可看成 [1, H_q, D] 或 [B, H_q, D]（通常 B=1），
# cache["xxx"] 逻辑上可看成 [S, r, c]，其中 r/c 来自 create_cache。
# 在 forward_cache 里，cache["xxx"] 逻辑上可看成 [B, r, c]。
# 在 forward_cache 里，如果某个 page_id 出现在 loc 中，这个 page 只会被计算一次；
# 整个计算过程中，同一个 page_id 也只会在 loc 中出现一次。
# 因此这里的张量都保持 3 维；Mean、Max、Min 等 Reduce 算子也会保留维度。
# 提示 1：GeMM(x, y) = yx^t，和常见矩阵乘定义可能不一样。
# 提示 2：forward_cache 里除了 cache["k"]，也可以用 cache["v"] 来收集信息。

@register("block_sparse_attention")
class BlockSparseAttention(vFlow):
    r"""
    用 **key centroid** 相似度做 block-sparse 路由。

    每个 page/block 存一个 centroid，也就是这块里所有 key 的平均值。
    生成时用 query 和每块 centroid 做相似度，选最相关的块。
    这对应 Kinetics [sadhukhan2025kinetics]_ 里的 block-top-:math:`k` 思路
    (arXiv:2506.05333)。

    **缓存。** :meth:`forward_cache` 用 :class:`CMean` 给每个 page 存一个 centroid：

    .. math::

        c_p \;=\; \frac{1}{|p|} \sum_{k \in p} k \;\in\; \mathbb{R}^{D}.

    **路由。** 先把 query 在 head 维度上求平均，
    :math:`\bar q = \tfrac{1}{H_q}\sum_{h} q_h`。
    :meth:`forward_indexer` 用 query 平均值和每块 centroid 的点积打分，
    最后用 :class:`topK` 保留分数最高的块：

    .. math::

        \operatorname{score}(p) \;=\; \langle \bar q,\; c_p \rangle.

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["centroids"]`` 在 indexer
    视角是 ``[S, 1, D]``（page-packed，:math:`S` 是压平后的 page 轴），
    在 cache 更新视角是 ``[B, 1, D]``（batch-major）。

    References
    ----------
    .. [sadhukhan2025kinetics]
       Ranajoy Sadhukhan, Zhuoming Chen, Haizhong Zheng, Yang Zhou,
       Emma Strubell, Beidi Chen.
       *Kinetics: Rethinking Test-Time Scaling Laws*. arXiv:2506.05333, 2025.
    """

    def __init__(self):
        super().__init__()
        # indexer 侧算子：负责给历史块打分并 topK
        self.gemm = GeMM()
        self.mean = Mean(dim=1)
        self.output_func = topK()

        # cache 侧算子：负责更新历史摘要
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""用 ``q`` 和 cache 摘要给每个 page 打分，把选中的 page 写入 ``o``。
        最后一步是 :class:`topK`。具体公式见类文档。"""
        q_mean = self.mean(q, ctx=ctx)
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""根据刚写入的 key/value 刷新每个 page 的 cache 摘要。公式见类文档。"""
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""声明这个策略额外需要的 per-page cache 张量；``"k"`` 和 ``"v"``
        会由框架自动添加。``block_size`` 是每块 token 数，``head_dim`` 是每个 head 的维度。"""
        return {
            "centroids": (1, head_dim),
        }


@register("gqa_block_sparse_attention")
class GQABlockSparseAttention(vFlow):
    r"""
    GQA 场景下的 block-sparse 路由：对 page 分数做 softmax。

    每个 page 存一个 key centroid。每个 grouped-query head 分别和这些
    centroid 打分；每个 head 的 page 分数会先变成 page 维度上的 softmax 分布；
    一个 page 的最终分数取所有 head 概率里的最大值
    （参考 arXiv:2502.11089 里的 GQA sparse-attention 形式）。

    **缓存。** 用 :class:`CMean` 存每个 page 的 centroid：
    :math:`c_p = \frac{1}{|p|}\sum_{k\in p} k`。

    **路由。** 对 grouped-query head :math:`q_h` 和 page :math:`p`，
    温度系数 :math:`\tau = 0.09 \approx 1/\sqrt{D}`：

    .. math::

        a_{p,h} = \operatorname{softmax}_{p}\!\big(\tau\,\langle q_h, c_p\rangle\big),
        \qquad
        \operatorname{score}(p) = \max_{h} a_{p,h},

    然后 :class:`topK` 保留分数最高的 page。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["centroids"]`` 是
    ``[S, 1, D]``（indexer 视角）/ ``[B, 1, D]``（cache 视角）。
    """

    def __init__(self):
        super().__init__()
        # indexer 侧算子
        self.gemm = GeMM()
        self.softmax = Softmax(dim=0, scale=0.09)
        self.max_op = Max(dim=2)
        self.output_func = topK()

        # cache 侧算子
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""用 ``q`` 和 cache 摘要给每个 page 打分，把选中的 page 写入 ``o``。
        最后一步是 :class:`topK`。具体公式见类文档。"""
        score = self.gemm(q, cache["centroids"], ctx=ctx)
        normalized_score = self.softmax(score, ctx=ctx)
        aggr_score = self.max_op(normalized_score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""根据刚写入的 key/value 刷新每个 page 的 cache 摘要。公式见类文档。"""
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""声明这个策略额外需要的 per-page cache 张量；``"k"`` 和 ``"v"``
        会由框架自动添加。``block_size`` 是每块 token 数，``head_dim`` 是每个 head 的维度。"""
        return {
            "centroids": (1, head_dim),
        }



@register("gqa_quest_sparse_attention")
class GQAQuestSparseAttention(vFlow):
    r"""
    GQA 风格的 QUEST sparse attention 策略。

    这个策略使用类似 QUEST sparse attention 的 **query-envelope matching**
    （见 https://arxiv.org/abs/2406.10774）。它给每个 page 维护 key 的
    **最大值** 和 **最小值** 包络，然后用这些包络估计 query-key 相似度的保守上界。

    **缓存。** :meth:`forward_cache` 通过 :class:`CMax` / :class:`CMin`
    给每个 page :math:`p` 存按坐标维度的 max/min 包络：

    .. math::

        M_p = \max_{k\in p} k, \qquad m_p = \min_{k\in p} k \;\in\; \mathbb{R}^{D}.

    **路由。** 对每个 grouped-query head :math:`q_h`，QUEST bound 会在每个
    特征维度上取 ``q*max`` 和 ``q*min`` 两个带符号乘积里的较大值，
    然后在特征维度求和，最后在 head 维度取最大：

    .. math::

        \operatorname{score}(p)
        = \max_{h} \sum_{d=1}^{D}
          \max\!\big(q_{h,d}\,M_{p,d},\; q_{h,d}\,m_{p,d}\big),

    然后 :class:`topK` 保留分数最高的 page。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["max"]`` 和 ``cache["min"]``
    在 indexer 视角是 ``[S, 1, D]``（page-packed），在 cache 视角是
    ``[B, 1, D]``（batch-major）。
    """

    def __init__(self):
        super().__init__()

        # indexer 侧算子
        self.mul_max = Multiply()      # q * max
        self.mul_min = Multiply()      # q * min
        self.maximum_op = Maximum()    # 逐元素 max(q*max, q*min)
        self.sum = Sum(dim=2)          # 在特征维 D 上求和
        self.max_op = Max(dim=1)       # 在 grouped-query head 轴上取最大
        self.output_func = topK()      # 产生 sparse indices

        # cache 侧算子
        self.reduction_max = CMax(dim=1)  # 每个 page 上对 k 取最大包络
        self.reduction_min = CMin(dim=1)  # 每个 page 上对 k 取最小包络

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""用 ``q`` 和 cache 摘要给每个 page 打分，把选中的 page 写入 ``o``。
        最后一步是 :class:`topK`。具体公式见类文档。"""
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.maximum_op(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""根据刚写入的 key/value 刷新每个 page 的 cache 摘要。公式见类文档。"""
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""声明这个策略额外需要的 per-page cache 张量；``"k"`` 和 ``"v"``
        会由框架自动添加。``block_size`` 是每块 token 数，``head_dim`` 是每个 head 的维度。"""
        return {
            "max": (1, head_dim),
            "min": (1, head_dim),
        }



@register("lserve_sparse_attention")
class LServeSparseAttention(vFlow):
    r"""
    LSERVE：在 **sub-block** 粒度上做 QUEST 包络。

    每个 page 会被切成连续的 sub-block，每个 sub-block 有
    :attr:`LSERVE_BLOCK_SIZE` 个 token；每个 sub-block 都存一份 key 的
    max/min 包络。这样可以便宜地估计某个 sub-block 内 query-key 点积的上界。
    一个 page 的分数由它最匹配的 (head, sub-block) 决定，所以只要 page 里有一个
    局部区域很相关，这个 page 就可能被选中。

    **缓存。** :meth:`forward_cache` 会给每个 page :math:`p` 的每个 sub-block
    :math:`b` 存按坐标维度的 max/min 包络；sub-block 数量为
    :math:`n_b = \text{block\_size} / \text{LSERVE\_BLOCK\_SIZE}` sub-blocks
    ，由 :class:`CMaxInterleave` / :class:`CMinInterleave` 完成：

    .. math::

        M_{p,b} = \max_{k\in b} k, \qquad m_{p,b} = \min_{k\in b} k.

    **路由。** query head 通过 :class:`Kron` 和 sub-block 包络组合，
    然后 QUEST bound 会同时在 head 和 sub-block 维度上取最大：

    .. math::

        \operatorname{score}(p)
        = \max_{h,\,b} \sum_{d=1}^{D}
          \max\!\big(q_{h,d}\,M_{p,b,d},\; q_{h,d}\,m_{p,b,d}\big),

    然后 :class:`topK` 保留分数最高的 page。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["max"]`` / ``cache["min"]``
    是 ``[S, n_b, D]``（indexer）/ ``[B, n_b, D]``（cache）。
    如果 ``block_size == LSERVE_BLOCK_SIZE``，每个 page 只有一个 sub-block
    (:math:`n_b = 1`)，等价于对整个 page 存一个包络。
    """
    LSERVE_BLOCK_SIZE = 16
    def __init__(self):
        super().__init__()

        # indexer 侧算子
        self.mul_max = Kron(dim=1)     # q * max
        self.mul_min = Kron(dim=1)     # q * min
        self.maximum_op = Maximum()    # 逐元素 max(q*max, q*min)
        self.sum = Sum(dim=2)          # 在特征维 D 上求和
        self.max_op = Max(dim=1)       # 在 grouped-query head 轴上取最大
        self.output_func = topK()      # 产生 sparse indices

        # cache 侧算子
        self.reduction_max = CMaxInterleave(dim=1, k=self.LSERVE_BLOCK_SIZE)  # 每个 sub-block 上对 k 取最大包络
        self.reduction_min = CMinInterleave(dim=1, k=self.LSERVE_BLOCK_SIZE)  # 每个 sub-block 上对 k 取最小包络

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""用 ``q`` 和 cache 摘要给每个 page 打分，把选中的 page 写入 ``o``。
        最后一步是 :class:`topK`。具体公式见类文档。"""
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.maximum_op(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""根据刚写入的 key/value 刷新每个 page 的 cache 摘要。公式见类文档。"""
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""声明这个策略额外需要的 per-page cache 张量；``"k"`` 和 ``"v"``
        会由框架自动添加。``block_size`` 是每块 token 数，``head_dim`` 是每个 head 的维度。"""
        return {
            "max": (block_size // self.LSERVE_BLOCK_SIZE, head_dim),
            "min": (block_size // self.LSERVE_BLOCK_SIZE, head_dim),
        }


@register("lserve_centroid_sparse_attention")
class LServeCentroidSparseAttention(vFlow):
    r"""
    在 sub-block 粒度上做 centroid 路由。

    每个 page 会被切成连续的 sub-block，每个 sub-block 有
    :attr:`SUB_BLOCK_SIZE` 个 token；每个 sub-block 存一个 centroid
    （也就是 key 的平均值）。一个 page 的分数取 query 和它所有 sub-block
    centroid 中最匹配的那个，所以只要有一个局部区域相关，这个 page 就可能被选中。

    **缓存。** :meth:`forward_cache` 给每个 page :math:`p` 的每个 sub-block
    :math:`b` 存一个 centroid；sub-block 数量为
    :math:`n_b = \text{block\_size} / \text{SUB\_BLOCK\_SIZE}` sub-blocks
    ，由 :class:`CMeanInterleave` 完成：

    .. math::

        c_{p,b} = \frac{1}{|b|} \sum_{k\in b} k.

    **路由。** 先在 head 维度上得到 query summary：
    :math:`\bar q = \frac{1}{H_q}\sum_h q_h`，

    .. math::

        \operatorname{score}(p) = \max_{b}\, \langle \bar q,\; c_{p,b} \rangle,

    然后 :class:`topK` 保留分数最高的 page。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["centroids"]`` 是
    ``[S, n_b, D]``（indexer）/ ``[B, n_b, D]``（cache）。
    如果 ``block_size == SUB_BLOCK_SIZE``，每个 page 只有一个 sub-block
    (:math:`n_b = 1`)，等价于整个 page 只有一个 centroid。
    """
    SUB_BLOCK_SIZE = 16

    def __init__(self):
        super().__init__()

        # indexer 侧算子
        self.mean = Mean(dim=1)        # 在 grouped-query head 维度上平均 query
        self.gemm = GeMM()             # q_summary · sub-block centroids
        self.max_sub = Max(dim=1)      # 在 sub-block centroid 维度上取最大
        self.output_func = topK()      # 产生 sparse indices

        # cache 侧算子：对每个 sub-block 内的 key 求平均
        self.reduction = CMeanInterleave(dim=1, k=self.SUB_BLOCK_SIZE)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""用 ``q`` 和 cache 摘要给每个 page 打分，把选中的 page 写入 ``o``。
        最后一步是 :class:`topK`。具体公式见类文档。"""
        q_summary = self.mean(q, ctx=ctx)
        score = self.gemm(q_summary, cache["centroids"], ctx=ctx)
        page_score = self.max_sub(score, ctx=ctx)
        self.output_func(page_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""根据刚写入的 key/value 刷新每个 page 的 cache 摘要。公式见类文档。"""
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""声明这个策略额外需要的 per-page cache 张量；``"k"`` 和 ``"v"``
        会由框架自动添加。``block_size`` 是每块 token 数，``head_dim`` 是每个 head 的维度。"""
        return {
            "centroids": (block_size // self.SUB_BLOCK_SIZE, head_dim),
        }


@register("masked_quest_sparse_attention")
class MaskedQuestSparseAttention(vFlow):
    r"""
    带 feature 维 mask 的 QUEST 路由，用来丢掉低信号通道。

    每个 page 存 key 的按坐标 max/min 包络；它们组合后可以给 page 内最大的
    query-key 点积提供上界。在特征维度求和之前，:class:`MaskSlice` 会把前
    ``MASK_END`` 个特征维置零。这是一种便宜的、只依赖位置的屏蔽方式，用来排除
    低信号通道（例如幅度很大的 "sink" 维度）。这个 mask 只按位置生成，不需要通过
    ``ctx`` 传额外状态。

    **缓存。** 通过 :class:`CMax` / :class:`CMin` 存每个 page 的 key 包络：
    :math:`M_p = \max_{k\in p} k` 和
    :math:`m_p = \min_{k\in p} k`。

    **路由。** mask 定义为：当 :math:`d < \text{MASK\_END}` 时
    :math:`w_d = 0`，否则 :math:`w_d = 1`：

    .. math::

        \operatorname{score}(p)
        = \max_{h} \sum_{d=1}^{D} w_d \,
          \max\!\big(q_{h,d}\,M_{p,d},\; q_{h,d}\,m_{p,d}\big),

    然后 :class:`topK` 保留分数最高的 page。mask 作用在 ``dim=2``，
    也就是特征维 :math:`D`，所以 ``MASK_END``（默认 8）必须满足
    :math:`\le D`；对验证中的 :math:`D\in\{32,64,128\}` 是安全的。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["max"]`` / ``cache["min"]``
    是 ``[S, 1, D]``（indexer）/ ``[B, 1, D]``（cache）。
    """

    MASK_END = 8  # 屏蔽 [0, MASK_END) 这些特征；对 D in {32, 64, 128} 安全

    def __init__(self):
        super().__init__()

        # indexer 侧算子
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        # 特征轴上的纯位置 mask：[0, MASK_END) 上 α=0，其余位置 β=1。
        self.feature_mask = MaskSlice(
            start=0, end=self.MASK_END, dim=2, alpha=0.0, beta=1.0
        )
        self.mul_mask = Multiply()
        self.sum = Sum(dim=2)
        self.max_op = Max(dim=1)
        self.output_func = topK()

        # cache 侧算子
        self.reduction_max = CMax(dim=1)
        self.reduction_min = CMin(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        s_max = self.mul_max(q, cache["max"], ctx=ctx)      # [S, H_q, D]
        s_min = self.mul_min(q, cache["min"], ctx=ctx)      # [S, H_q, D]
        s = self.maximum_op(s_max, s_min, ctx=ctx)          # [S, H_q, D]
        mask = self.feature_mask(s, ctx=ctx)                # [S, H_q, D]
        masked_s = self.mul_mask(s, mask, ctx=ctx)          # [S, H_q, D]
        score = self.sum(masked_s, ctx=ctx)                 # [S, H_q, 1]
        aggr_score = self.max_op(score, ctx=ctx)            # [S, 1, 1]
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "max": (1, head_dim),
            "min": (1, head_dim),
        }


@register("centered_block_sparse_attention")
class CenteredBlockSparseAttention(vFlow):
    r"""
    带 per-request **均值中心化** 的 centroid block-sparse 路由。

    每个 page 存一个 centroid（key 的平均值）；page 分数来自 query-centroid
    相似度。在选择之前，会减掉同一个 request 内所有 page 的平均分。
    所以 page 比的是它比平均水平高多少，而不是原始相似度有多大。

    **缓存。** 通过 :class:`CMean` 存每个 page 的 centroid：
    :math:`c_p = \frac{1}{|p|}\sum_{k\in p} k`。

    **路由。** 先把每个 head 的点积在 head 维度上平均：
    :math:`s_p = \frac{1}{H_q}\sum_h \langle q_h, c_p\rangle`。
    再计算同一个 request 内所有 page 的平均分：
    :math:`\bar s = \frac{1}{S}\sum_{p} s_p`
    （这是一个 ``dim=0`` 的跨 page :class:`Mean`）：

    .. math::

        \operatorname{score}(p) = s_p - \bar s,

    然后 :class:`topK` 保留分数最高的 page。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["centroids"]`` 是
    ``[S, 1, D]``（indexer）/ ``[B, 1, D]``（cache）。
    """

    def __init__(self):
        super().__init__()
        # indexer 侧算子
        self.mul = Multiply()
        self.sum_d = Sum(dim=2)
        self.mean_h = Mean(dim=1)
        self.mean_seq = Mean(dim=0)            # Schedule.S，RAGGED → BATCHED
        self.center = Add(alpha=1.0, beta=-1.0)  # score - mean_seq
        self.output_func = topK()

        # cache 侧算子
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        s = self.mul(q, cache["centroids"], ctx=ctx)        # RAGGED
        score_d = self.sum_d(s, ctx=ctx)                    # RAGGED
        score = self.mean_h(score_d, ctx=ctx)               # RAGGED [S, 1, 1]
        mean_seq = self.mean_seq(score, ctx=ctx)            # BATCHED [B*H_kv, 1, 1]
        centered = self.center(score, mean_seq, ctx=ctx)    # RAGGED via (R, B) dispatch
        self.output_func(centered, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
        }


@register("running_avg_block_sparse")
class RunningAvgBlockSparse(vFlow):
    r"""
    带 per-page **running score** 的 centroid block-sparse 路由
    （也是 :class:`Save` / :class:`Load` 的示例）。

    每个 page 存一个 centroid（key 的平均值）。这个策略不是只看当前 step 的分数，
    而是把每个 page 的 query-centroid 分数累积到一个带指数衰减的 running score 里：
    持续相关的 page 会积累分数，不再相关的 page 会逐渐衰减。

    **缓存。** 通过 :class:`CMean` 存每个 page 的 centroid :math:`c_p`。
    持久的 ``running_score`` 会在 page 第一次填充时用 :class:`CFill` 初始化为 0；
    之后它由 :meth:`forward_indexer` 维护。

    **路由。** 设 :math:`\bar q_t = \frac{1}{H_q}\sum_h q_{h,t}`，
    衰减系数 :math:`\alpha`（即 ``ALPHA`` = 0.5），旧值通过 :class:`Load` 读取：

    .. math::

        r_t(p) = \alpha\, r_{t-1}(p) + \langle \bar q_t,\; c_p \rangle,

    新的 :math:`r_t(p)` 通过 :class:`Save` 写回持久 cache，并交给
    :class:`topK` 选块。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["centroids"]`` 是
    ``[S, 1, D]`` / ``[B, 1, D]``；``cache["running_score"]`` 是
    ``[S, 1, 1]`` / ``[B, 1, 1]``。

    .. note::
       因为这个策略会用 ``Save`` 保存每步状态，使用它的 engine 必须设置
       ``disable_radix_cache=True``。
    """
    ALPHA = 0.5

    def __init__(self):
        super().__init__()
        # indexer 侧算子
        self.mean        = Mean(dim=1)
        self.gemm        = GeMM()
        self.load_score  = Load()
        self.fuse        = Add(alpha=self.ALPHA, beta=1.0)
        self.save_score  = Save()
        self.output_func = topK()

        # cache 侧算子
        self.reduction = CMean(dim=1)
        # 每个新 block 完成时，把持久的 per-block 标量初始化为 0。
        # 如果不初始化，block 分配后的第一次 Load 可能读到这个内存槽里之前残留的值，
        # 通常是上一条 sequence 的旧值。
        self.init_running_score = CFill(alpha=0.0)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean       = self.mean(q, ctx=ctx)                               # [1, 1, D]
        current      = self.gemm(q_mean, cache["centroids"], ctx=ctx)      # [S, 1, 1]
        last_running = self.load_score(cache["running_score"], ctx=ctx)    # [S, 1, 1]
        running      = self.fuse(last_running, current, ctx=ctx)           # α*last + current
        self.save_score(running, cache["running_score"], ctx=ctx)          # 写回持久状态
        self.output_func(running, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_running_score(cache["running_score"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids":     (1, head_dim),  # 由 forward_cache 维护
            "running_score": (1, 1),         # 由 forward_indexer 通过 Save 维护
        }


@register("venergy_gated_centroid")
class VEnergyGatedCentroid(vFlow):
    r"""
    用 **value-block energy** 做 gate 的 centroid 路由。

    每个 page 存一个 key centroid（key 的平均值）。page 分数等于
    query-centroid 点积乘以这个 page 的平均 value 幅度（也就是 "energy"）。
    这样即使某个 page 的 key centroid 和 query 很像，只要它的 value 能量很小，
    分数也会被压低。

    **缓存。** :meth:`forward_cache` 存每个 page 的 key centroid
    :math:`c_p`（:class:`CMean`）和 value energy。value energy 是这个 page 里
    value token 的 :math:`L_2` norm 的平均值：先用 :class:`CL2Norm` 在
    :math:`D` 维上求每个 value token 的长度，再用 :class:`CMean` 在 token 维上平均。

    .. math::

        e_p = \frac{1}{|p|} \sum_{k\in p} \lVert v_k \rVert_2.

    **路由。** 设 :math:`\bar q = \frac{1}{H_q}\sum_h q_h`：

    .. math::

        \operatorname{score}(p) = \langle \bar q,\; c_p \rangle \cdot e_p,

    然后 :class:`topK` 保留分数最高的 page。

    **形状。** ``q`` 是 ``[B, H_q, D]``；``cache["centroids"]`` 是
    ``[S, 1, D]`` / ``[B, 1, D]``；``cache["v_energy"]`` 是
    ``[S, 1, 1]`` / ``[B, 1, 1]``。
    """

    def __init__(self):
        super().__init__()
        # indexer 侧算子
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.gate = Multiply()
        self.output_func = topK()

        # cache 侧算子
        self.k_mean = CMean(dim=1)
        self.v_tok_norm = CL2Norm(dim=2)   # [1, block_size, D] → [1, block_size, 1]
        self.v_energy = CMean(dim=1)        # [1, block_size, 1] → [1, 1, 1]

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "v_energy":  (1, 1),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        v_tok = self.v_tok_norm(cache["v"], None, loc=loc, ctx=ctx)   # [1, block_size, 1]
        self.v_energy(v_tok, cache["v_energy"], loc=loc, ctx=ctx)      # [1, 1, 1]

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                         # [1, 1, D]
        dot = self.gemm(q_mean, cache["centroids"], ctx=ctx)     # [S, 1, 1]
        score = self.gate(dot, cache["v_energy"], ctx=ctx)       # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)

# 给写策略/agent 的提示：
# 算子对象不能复用，即使它们语义一样。每个算子内部会初始化自己的中间缓冲区。
# 例如 QUEST attention 里需要两个 Multiply，就必须定义两个 Multiply 对象。
# flow 里不能直接使用原生 torch ops。所有 tensor（q、o、cache[...]、中间结果）
# 在 forward_indexer 里必须走 vortex_torch.indexer 的算子，
# 在 forward_cache 里必须走 vortex_torch.cache 的算子。

# 在 forward_indexer 里，q 逻辑上可看成 [1, H_q, D] 或 [B, H_q, D]（通常 B=1），
# cache["xxx"] 逻辑上可看成 [S, r, c]，其中 r/c 来自 create_cache。
# 在 forward_cache 里，cache["xxx"] 逻辑上可看成 [B, r, c]。
# 在 forward_cache 里，如果某个 page_id 出现在 loc 中，这个 page 只会被计算一次；
# 整个计算过程中，同一个 page_id 也只会在 loc 中出现一次。
# 所以用户不能指望在 forward_cache 里对同一个 page 多次累加。
# 因此这里的张量都保持 3 维；Mean、Max、Min 等 Reduce 算子也会保留维度。

# 提示 1：GeMM(x, y) = yx^t，和常见矩阵乘定义可能不一样。
# 提示 2：forward_cache 里除了 cache["k"]，也可以用 cache["v"] 来收集信息。
