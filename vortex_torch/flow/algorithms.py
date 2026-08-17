import torch
from typing import Dict

from .flow import vFlow
from ..indexer import topK, approxTopK, GeMV, Softmax, Max, Sum, GeMM, Maximum, Multiply, Add, L2Norm, Save, Load, Mean, MaskSlice, Kron
from ..cache import Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm, Fill as CFill, MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave, MeanInterleave as CMeanInterleave
from ..abs import ContextBase
from .registry import register

# The ops are not reusable, even if they have the same semantic meaning. Internally, they will initialize different memory buffer.
# For example, in Quest attention, we need to define two multiply operators.

# In forward indexer, q can be viewed as [1, H_q, D] or [B, H_q, D] (B=1) and cache["xxx"] can be viewed as [S, r, c] (r, c defined in create_cache) logically.
# In forward cache, the cache["xxx"] is viewed as [B, r, c]  (r, c defined in create_cache) logically.
# In forward cache, each page is computed only once if page_id appears in loc. During the entire computation, each page id will appear in loc only once.
# Thus, all the tensors have 3 dimensions. Reduce operators (Mean, Max, Min, etc) will always keep the dims.
# Tips 1: GeMM(x, y) = yx^t, which might be different from typical definitions.
# Tips 2: Except cache["k"], cache["v"] can also be used in forward_cache to collect information.

@register("block_sparse_attention")
class BlockSparseAttention(vFlow):
    r"""
    Block-sparse routing by **key-centroid** similarity.

    Keep one centroid per page (the mean of its keys) and select the pages
    whose centroid best aligns with the query — the block-top-:math:`k` idea
    from Kinetics [sadhukhan2025kinetics]_ (arXiv:2506.05333).

    **Cache.** :meth:`forward_cache` stores one centroid per page with
    :class:`CMean`:

    .. math::

        c_p \;=\; \frac{1}{|p|} \sum_{k \in p} k \;\in\; \mathbb{R}^{D}.

    **Routing.** With the query summary
    :math:`\bar q = \tfrac{1}{H_q}\sum_{h} q_h`, :meth:`forward_indexer`
    scores each page by the centroid dot product and keeps the highest with
    :class:`topK`:

    .. math::

        \operatorname{score}(p) \;=\; \langle \bar q,\; c_p \rangle.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["centroids"]`` is
    ``[S, 1, D]`` in the indexer (page-packed, :math:`S` = packed page axis)
    and ``[B, 1, D]`` in the cache (batch-major).

    References
    ----------
    .. [sadhukhan2025kinetics]
       Ranajoy Sadhukhan, Zhuoming Chen, Haizhong Zheng, Yang Zhou,
       Emma Strubell, Beidi Chen.
       *Kinetics: Rethinking Test-Time Scaling Laws*. arXiv:2506.05333, 2025.
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemm = GeMM()
        self.mean = Mean(dim=1)
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""Score each page from ``q`` and the cached state, then write the selected
        page indices into ``o`` (ends in :class:`topK`). See the class docstring
        for the scoring math."""
        q_mean = self.mean(q, ctx=ctx)
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""Refresh this flow's per-page cache state from the freshly written keys /
        values. See the class docstring for the formulas."""
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""Declare this flow's per-page cache tensors; ``"k"`` and ``"v"`` are added
        automatically. ``block_size`` is the per-block token count, ``head_dim``
        the head dimension. See the class docstring."""
        return {
            "centroids": (1, head_dim),
        }


@register("gqa_block_sparse_attention")
class GQABlockSparseAttention(vFlow):
    r"""
    Grouped-query block-sparse routing with a **softmax over pages**.

    Each page keeps a centroid (the mean of its keys). Every grouped-query
    head is scored against the centroids separately; each head's per-page
    scores are turned into a softmax distribution over pages, and a page's
    final score is the **max** of that probability across heads
    (cf. the GQA sparse-attention formulation in arXiv:2502.11089).

    **Cache.** Per-page centroid :math:`c_p = \frac{1}{|p|}\sum_{k\in p} k`
    via :class:`CMean`.

    **Routing.** For grouped-query head :math:`q_h` and page :math:`p`, with
    temperature :math:`\tau = 0.09 \approx 1/\sqrt{D}`,

    .. math::

        a_{p,h} = \operatorname{softmax}_{p}\!\big(\tau\,\langle q_h, c_p\rangle\big),
        \qquad
        \operatorname{score}(p) = \max_{h} a_{p,h},

    then :class:`topK` keeps the highest-scoring pages.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["centroids"]`` is
    ``[S, 1, D]`` (indexer) / ``[B, 1, D]`` (cache).
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemm = GeMM()
        self.softmax = Softmax(dim=0, scale=0.09)
        self.max_op = Max(dim=2)
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""Score each page from ``q`` and the cached state, then write the selected
        page indices into ``o`` (ends in :class:`topK`). See the class docstring
        for the scoring math."""
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
        r"""Refresh this flow's per-page cache state from the freshly written keys /
        values. See the class docstring for the formulas."""
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""Declare this flow's per-page cache tensors; ``"k"`` and ``"v"`` are added
        automatically. ``block_size`` is the per-block token count, ``head_dim``
        the head dimension. See the class docstring."""
        return {
            "centroids": (1, head_dim),
        }


@register("frozen_dynamic_top_p")
class FrozenDynamicTopP(vFlow):
    """Marker flow dispatched by the dedicated Dynamic Top-P runtime."""

    is_frozen_dynamic_top_p = True

    def forward_indexer(self, q, o, cache, ctx):
        raise RuntimeError(
            "frozen_dynamic_top_p must be dispatched by VortexFlashInferBackend"
        )

    def forward_cache(self, cache, loc, ctx):
        raise RuntimeError(
            "frozen_dynamic_top_p cache updates must be dispatched by VortexCachePool"
        )

    def create_cache(self, block_size: int, head_dim: int):
        if block_size != 16:
            raise ValueError("frozen_dynamic_top_p requires block_size=16")
        return {
            "centroids": ((4, head_dim), torch.bfloat16),
            "centroid_counts": ((4, 1), torch.uint8),
        }



@register("gqa_quest_sparse_attention")
class GQAQuestSparseAttention(vFlow):
    r"""
    GQA-style QUEST sparse attention flow.

    This flow uses **query–envelope matching** similar to QUEST sparse
    attention (see https://arxiv.org/abs/2406.10774). For each request,
    it maintains per-page **max** and **min** envelopes of keys and uses
    them to compute a conservative upper bound on query–key similarity.

    **Cache.** :meth:`forward_cache` stores, per page :math:`p`, the
    coordinate-wise envelopes via :class:`CMax` / :class:`CMin`:

    .. math::

        M_p = \max_{k\in p} k, \qquad m_p = \min_{k\in p} k \;\in\; \mathbb{R}^{D}.

    **Routing.** For each grouped-query head :math:`q_h` the QUEST bound takes,
    per coordinate, the larger of the two signed products, sums over features,
    and finally maxes over heads:

    .. math::

        \operatorname{score}(p)
        = \max_{h} \sum_{d=1}^{D}
          \max\!\big(q_{h,d}\,M_{p,d},\; q_{h,d}\,m_{p,d}\big),

    then :class:`topK` keeps the highest-scoring pages.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["max"]`` and ``cache["min"]``
    are ``[S, 1, D]`` in the indexer (page-packed) and ``[B, 1, D]`` in the
    cache (batch-major).
    """

    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mul_max = Multiply()      # q * max
        self.mul_min = Multiply()      # q * min
        self.maximum_op = Maximum()    # elementwise max(q*max, q*min)
        self.sum = Sum(dim=2)          # sum over feature dim D
        self.max_op = Max(dim=1)       # max over grouped-query axis
        self.output_func = topK()      # produce sparse indices

        # Cache-side ops
        self.reduction_max = CMax(dim=1)  # page-wise max envelope over k
        self.reduction_min = CMin(dim=1)  # page-wise min envelope over k

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""Score each page from ``q`` and the cached state, then write the selected
        page indices into ``o`` (ends in :class:`topK`). See the class docstring
        for the scoring math."""
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
        r"""Refresh this flow's per-page cache state from the freshly written keys /
        values. See the class docstring for the formulas."""
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""Declare this flow's per-page cache tensors; ``"k"`` and ``"v"`` are added
        automatically. ``block_size`` is the per-block token count, ``head_dim``
        the head dimension. See the class docstring."""
        return {
            "max": (1, head_dim),
            "min": (1, head_dim),
        }



@register("lserve_sparse_attention")
class LServeSparseAttention(vFlow):
    r"""
    LSERVE: QUEST envelopes at **sub-block** granularity.

    Each page is split into consecutive sub-blocks of :attr:`LSERVE_BLOCK_SIZE`
    tokens, and a coordinate-wise max/min key envelope is kept **per
    sub-block**. The envelopes give a cheap upper bound on the query–key dot
    product within a sub-block; a page is ranked by its single best-matching
    (head, sub-block) pair, so one relevant sub-region is enough to select the
    page.

    **Cache.** :meth:`forward_cache` stores, for each of the
    :math:`n_b = \text{block\_size} / \text{LSERVE\_BLOCK\_SIZE}` sub-blocks
    :math:`b` of page :math:`p`, coordinate-wise envelopes via
    :class:`CMaxInterleave` / :class:`CMinInterleave`:

    .. math::

        M_{p,b} = \max_{k\in b} k, \qquad m_{p,b} = \min_{k\in b} k.

    **Routing.** Query heads are combined with the sub-block envelopes via
    :class:`Kron`, and the QUEST bound is maximized over both heads and
    sub-blocks:

    .. math::

        \operatorname{score}(p)
        = \max_{h,\,b} \sum_{d=1}^{D}
          \max\!\big(q_{h,d}\,M_{p,b,d},\; q_{h,d}\,m_{p,b,d}\big),

    then :class:`topK` keeps the highest pages.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["max"]`` / ``cache["min"]``
    are ``[S, n_b, D]`` (indexer) / ``[B, n_b, D]`` (cache). With
    ``block_size == LSERVE_BLOCK_SIZE`` there is one sub-block per page
    (:math:`n_b = 1`), i.e. a single envelope over the whole page.
    """
    LSERVE_BLOCK_SIZE = 16
    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mul_max = Kron(dim=1)     # q * max
        self.mul_min = Kron(dim=1)     # q * min
        self.maximum_op = Maximum()    # elementwise max(q*max, q*min)
        self.sum = Sum(dim=2)          # sum over feature dim D
        self.max_op = Max(dim=1)       # max over grouped-query axis
        self.output_func = topK()      # produce sparse indices

        # Cache-side ops
        self.reduction_max = CMaxInterleave(dim=1, k=self.LSERVE_BLOCK_SIZE)  # page-wise max envelope over k
        self.reduction_min = CMinInterleave(dim=1, k=self.LSERVE_BLOCK_SIZE)  # page-wise min envelope over k

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""Score each page from ``q`` and the cached state, then write the selected
        page indices into ``o`` (ends in :class:`topK`). See the class docstring
        for the scoring math."""
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
        r"""Refresh this flow's per-page cache state from the freshly written keys /
        values. See the class docstring for the formulas."""
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""Declare this flow's per-page cache tensors; ``"k"`` and ``"v"`` are added
        automatically. ``block_size`` is the per-block token count, ``head_dim``
        the head dimension. See the class docstring."""
        return {
            "max": (block_size // self.LSERVE_BLOCK_SIZE, head_dim),
            "min": (block_size // self.LSERVE_BLOCK_SIZE, head_dim),
        }


@register("lserve_centroid_sparse_attention")
class LServeCentroidSparseAttention(vFlow):
    r"""
    Centroid routing at sub-block granularity.

    Each page is split into consecutive sub-blocks of :attr:`SUB_BLOCK_SIZE`
    tokens, and a centroid (the mean of its keys) is kept **per sub-block**. A
    page is ranked by the query's best match against any of its sub-block
    centroids, so one relevant sub-region is enough to select the page.

    **Cache.** :meth:`forward_cache` stores, for each of the
    :math:`n_b = \text{block\_size} / \text{SUB\_BLOCK\_SIZE}` sub-blocks
    :math:`b` of page :math:`p`, a centroid via :class:`CMeanInterleave`:

    .. math::

        c_{p,b} = \frac{1}{|b|} \sum_{k\in b} k.

    **Routing.** With the query summary
    :math:`\bar q = \frac{1}{H_q}\sum_h q_h`,

    .. math::

        \operatorname{score}(p) = \max_{b}\, \langle \bar q,\; c_{p,b} \rangle,

    then :class:`topK` keeps the highest pages.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["centroids"]`` is
    ``[S, n_b, D]`` (indexer) / ``[B, n_b, D]`` (cache). With
    ``block_size == SUB_BLOCK_SIZE`` there is one sub-block per page
    (:math:`n_b = 1`), i.e. a single centroid over the whole page.
    """
    SUB_BLOCK_SIZE = 16

    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mean = Mean(dim=1)        # average query over grouped-query heads
        self.gemm = GeMM()             # q_summary · sub-block centroids
        self.max_sub = Max(dim=1)      # max over sub-block centroids
        self.output_func = topK()      # produce sparse indices

        # Cache-side op: interleaved per-sub-block mean over keys
        self.reduction = CMeanInterleave(dim=1, k=self.SUB_BLOCK_SIZE)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""Score each page from ``q`` and the cached state, then write the selected
        page indices into ``o`` (ends in :class:`topK`). See the class docstring
        for the scoring math."""
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
        r"""Refresh this flow's per-page cache state from the freshly written keys /
        values. See the class docstring for the formulas."""
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""Declare this flow's per-page cache tensors; ``"k"`` and ``"v"`` are added
        automatically. ``block_size`` is the per-block token count, ``head_dim``
        the head dimension. See the class docstring."""
        return {
            "centroids": (block_size // self.SUB_BLOCK_SIZE, head_dim),
        }


@register("masked_quest_sparse_attention")
class MaskedQuestSparseAttention(vFlow):
    r"""
    QUEST routing with a feature-axis mask that drops low-signal channels.

    Each page keeps a coordinate-wise max/min key envelope; their combination
    upper-bounds the largest query–key dot product in the page. Before summing
    over features, a :class:`MaskSlice` zeroes the leading ``MASK_END`` feature
    coordinates of that bound — a cheap, position-only way to exclude
    low-signal channels (e.g. large-magnitude "sink" dimensions). The mask is a
    pure position writer, so no extra state is threaded through ``ctx``.

    **Cache.** Per-page key envelopes :math:`M_p = \max_{k\in p} k` and
    :math:`m_p = \min_{k\in p} k` via :class:`CMax` / :class:`CMin`.

    **Routing.** With the mask :math:`w_d = 0` for :math:`d < \text{MASK\_END}`
    and :math:`w_d = 1` otherwise,

    .. math::

        \operatorname{score}(p)
        = \max_{h} \sum_{d=1}^{D} w_d \,
          \max\!\big(q_{h,d}\,M_{p,d},\; q_{h,d}\,m_{p,d}\big),

    then :class:`topK` keeps the highest pages. The mask is applied on
    ``dim=2`` (the feature dim :math:`D`), so ``MASK_END`` (default 8) must be
    :math:`\le D` — safe for the verification sweep :math:`D\in\{32,64,128\}`.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["max"]`` / ``cache["min"]``
    are ``[S, 1, D]`` (indexer) / ``[B, 1, D]`` (cache).
    """

    MASK_END = 8  # mask [0, MASK_END) features; safe for D in {32, 64, 128}

    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        # Position-only mask on the feature axis: α=0 on [0, MASK_END), β=1 elsewhere.
        self.feature_mask = MaskSlice(
            start=0, end=self.MASK_END, dim=2, alpha=0.0, beta=1.0
        )
        self.mul_mask = Multiply()
        self.sum = Sum(dim=2)
        self.max_op = Max(dim=1)
        self.output_func = topK()

        # Cache-side ops
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
    Centroid block-sparse routing with per-request **mean-centering**.

    Each page keeps a centroid (the mean of its keys); pages are scored by
    query–centroid similarity, and the per-request mean score across pages is
    subtracted before selection — so a page competes by how far *above
    average* it is, not by raw similarity.

    **Cache.** Per-page centroid :math:`c_p = \frac{1}{|p|}\sum_{k\in p} k`
    via :class:`CMean`.

    **Routing.** With the per-head dot averaged over heads,
    :math:`s_p = \frac{1}{H_q}\sum_h \langle q_h, c_p\rangle`, and its
    per-request mean over pages :math:`\bar s = \frac{1}{S}\sum_{p} s_p`
    (a cross-page :class:`Mean` with ``dim=0``),

    .. math::

        \operatorname{score}(p) = s_p - \bar s,

    then :class:`topK` keeps the highest pages.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["centroids"]`` is
    ``[S, 1, D]`` (indexer) / ``[B, 1, D]`` (cache).
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.mul = Multiply()
        self.sum_d = Sum(dim=2)
        self.mean_h = Mean(dim=1)
        self.mean_seq = Mean(dim=0)            # Schedule.S, RAGGED → BATCHED
        self.center = Add(alpha=1.0, beta=-1.0)  # score - mean_seq
        self.output_func = topK()

        # Cache-side ops
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
    Centroid block-sparse routing with a per-page **running score**
    (a :class:`Save` / :class:`Load` demo).

    Each page keeps a centroid (the mean of its keys). Instead of scoring on
    the current step alone, the per-page query–centroid score is accumulated
    into an exponentially-decayed running score across decode steps: pages that
    stay relevant accumulate, pages that fade decay.

    **Cache.** Per-page centroid :math:`c_p` via :class:`CMean`; the persistent
    ``running_score`` is zero-initialised with :class:`CFill` when a page is
    first filled (thereafter it is owned by :meth:`forward_indexer`).

    **Routing.** With :math:`\bar q_t = \frac{1}{H_q}\sum_h q_{h,t}`, decay
    :math:`\alpha` (= ``ALPHA`` = 0.5), and the previous value read via
    :class:`Load`,

    .. math::

        r_t(p) = \alpha\, r_{t-1}(p) + \langle \bar q_t,\; c_p \rangle,

    the new :math:`r_t(p)` is persisted via :class:`Save` and fed to
    :class:`topK`.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["centroids"]`` is
    ``[S, 1, D]`` / ``[B, 1, D]``; ``cache["running_score"]`` is ``[S, 1, 1]``
    / ``[B, 1, 1]``.

    .. note::
       Because it ``Save``\ s per-step state, an engine using this flow must
       set ``disable_radix_cache=True``.
    """
    ALPHA = 0.5

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.mean        = Mean(dim=1)
        self.gemm        = GeMM()
        self.load_score  = Load()
        self.fuse        = Add(alpha=self.ALPHA, beta=1.0)
        self.save_score  = Save()
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)
        # Zero-initialise the persistent per-block scalar when each new
        # block completes. Without this, the first ``Load`` after a
        # block is allocated reads whatever was in that memory slot
        # before — typically stale values from a prior sequence.
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
        self.save_score(running, cache["running_score"], ctx=ctx)          # persist
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
            "centroids":     (1, head_dim),  # maintained by forward_cache
            "running_score": (1, 1),         # maintained by forward_indexer (Save)
        }


@register("venergy_gated_centroid")
class VEnergyGatedCentroid(vFlow):
    r"""
    Centroid routing **gated by value-block energy**.

    Each page keeps a key centroid (the mean of its keys); a page is scored by
    the query–centroid dot product multiplied by the page's mean value
    magnitude (its "energy"), so pages whose values carry little energy are
    muted even when the key centroid aligns with the query.

    **Cache.** :meth:`forward_cache` stores a per-page key centroid
    :math:`c_p` (:class:`CMean`) and the value energy — the mean :math:`L_2`
    norm of its value tokens (:class:`CL2Norm` over :math:`D`, then
    :class:`CMean` over tokens):

    .. math::

        e_p = \frac{1}{|p|} \sum_{k\in p} \lVert v_k \rVert_2.

    **Routing.** With :math:`\bar q = \frac{1}{H_q}\sum_h q_h`,

    .. math::

        \operatorname{score}(p) = \langle \bar q,\; c_p \rangle \cdot e_p,

    then :class:`topK` keeps the highest pages.

    **Shapes.** ``q`` is ``[B, H_q, D]``; ``cache["centroids"]`` is
    ``[S, 1, D]`` / ``[B, 1, D]``; ``cache["v_energy"]`` is ``[S, 1, 1]`` /
    ``[B, 1, 1]``.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.gate = Multiply()
        self.output_func = topK()

        # cache-side
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

# For agent developers!
# The ops are not reusable, even if they have the same semantic meaning. Internally, they will initialize different memory buffer.
# For example, in Quest attention, we need to define two multiply operators.
# Native torch ops are not allowed anywhere in the flow. Every tensor (q, o, cache[...], intermediates) must go through vortex_torch.indexer ops in forward_indexer and vortex_torch.cache ops in forward_cache.

# In forward indexer, q can be viewed as [1, H_q, D] or [B, H_q, D] (B=1) and cache["xxx"] can be viewed as [S, r, c] (r, c defined in create_cache) logically.
# In forward cache, the cache["xxx"] is viewed as [B, r, c]  (r, c defined in create_cache) logically.
# In forward cache, each page is computed only once if page_id appears in loc. During the entire computation, each page id will appear in loc only once. Thus, users cannot accumulate tensors through forward_cache.
# Thus, all the tensors have 3 dimensions. Reduce operators (Mean, Max, Min, etc) will always keep the dims.

# Tips 1: GeMM(x, y) = yx^t, which might be different from typical definitions.
# Tips 2: Except cache["k"], cache["v"] can also be used in forward_cache to collect information.
