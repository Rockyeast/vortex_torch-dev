"""
Custom **block-table** Triton MLA decode kernel for vortex sparse attention.

Adapted from sglang's grouped flash-decode stage-1 kernel
(`triton_ops/decode_attention.py::_fwd_grouped_kernel_stage1`), with two
changes:

  1. **KV gather is driven by vortex's 2D ``block_table`` + ``seqlens``** (the
     metadata the indexer + ``get_decode_planner_trtllm`` produce) instead of a
     flat token-level ``kv_indices`` CSR. For a gathered position ``n`` the
     physical latent slot is ``block_table[b, n // block_size] * block_size +
     (n % block_size)`` — exactly the block-sparse selection the trtllm MLA
     decode consumes, but in Triton so it is **not geometry-locked** to
     DeepSeek's `qk_nope=128 / v_head=128` (works for GLM-4.7-Flash's
     `192 / 256`, i.e. the fused 576-d latent with any head count).

  2. **Single pass** (no kv-split / stage-2 reduction): for decode with a
     sparse KV budget the per-(batch, head-group) parallelism is ample, so the
     kernel does the online softmax over all selected blocks and writes the
     final output directly. This avoids the split scratch + cuda-graph buffer
     management.

The fused MLA latent is the single ``K_Buffer`` (= ``V_Buffer``): the score
uses the full 576 dims (``BLOCK_DMODEL=512`` latent ``+ BLOCK_DPE=64`` rope,
matching the absorbed query ``[q_nope_out | q_pe]``), and the value reduction
uses the first ``Lv = kv_lora_rank = 512`` dims.
"""
# 中文读法：Triton MLA kernel 集合。这里是底层数值 kernel，主要关注输入 block_table、seq_lens、latent cache 如何算出当前 token attention。
import torch
import triton
import triton.language as tl


_SM_COUNT_CACHE = {}


def _get_sm_count(device) -> int:
    """SM count, cached per device (host-side query, not a GPU sync)."""
    idx = device.index if hasattr(device, "index") else torch.cuda.current_device()
    if idx not in _SM_COUNT_CACHE:
        _SM_COUNT_CACHE[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT_CACHE[idx]


@triton.jit
# Triton kernel：基础 block-table MLA decode，一次处理 selected blocks 并做 attention。
def _fwd_blocktable_mla_kernel(
    Q,
    K_Buffer,          # fused latent [num_slots, Lk]  (Lk = 576)
    V_Buffer,          # same buffer; value uses first Lv (= 512) dims
    sm_scale,
    Seqlens,           # [bs] sparse token count per request
    Block_Table,       # [bs, max_blocks] selected page ids (int32)
    O,                 # [bs, q_head_num, Lv]
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_obs,
    stride_oh,
    stride_bt_b,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,   # 512 (kv_lora_rank)
    BLOCK_DPE: tl.constexpr,      # 64  (qk_rope_head_dim)
    BLOCK_DV: tl.constexpr,       # 512 (value width = kv_lora_rank)
    BLOCK_N: tl.constexpr,        # token tile (= BLOCK_SIZE)
    BLOCK_H: tl.constexpr,        # query-head tile (16)
    Lk: tl.constexpr,             # 576
    Lv: tl.constexpr,             # 512
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = 0  # single shared latent head

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    seqlen = tl.load(Seqlens + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    mask_dpe = offs_dpe < Lk
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_n in range(0, seqlen, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid = offs_n < seqlen
        # block-table gather: position -> physical latent slot
        page = tl.load(
            Block_Table + cur_batch * stride_bt_b + (offs_n // BLOCK_SIZE),
            mask=valid,
            other=0,
        )
        kv_loc = page * BLOCK_SIZE + (offs_n % BLOCK_SIZE)

        offs_buf_k = (
            kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head
            + offs_d[:, None]
        )
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(valid[None, :]) & (mask_d[:, None]),
            other=0.0,
        )
        qk = tl.dot(q, k.to(q.dtype))
        offs_buf_kpe = (
            kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head
            + offs_dpe[:, None]
        )
        kpe = tl.load(
            K_Buffer + offs_buf_kpe,
            mask=(valid[None, :]) & (mask_dpe[:, None]),
            other=0.0,
        )
        qk += tl.dot(qpe, kpe.to(qpe.dtype))
        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & valid[None, :], qk, float("-inf"))

        offs_buf_v = (
            kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v,
            mask=(valid[:, None]) & (mask_dv[None, :]),
            other=0.0,
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_dv[None, :]
    tl.store(
        O + offs_o,
        acc / e_sum[:, None],
        mask=(mask_h[:, None]) & (mask_dv[None, :]),
    )


# Python wrapper：准备参数后调用基础 Triton MLA decode kernel。
def decode_blocktable_mla(
    q: torch.Tensor,            # [bs, H, Lk]   fused [q_nope_out | q_pe], Lk=576
    latent: torch.Tensor,       # [num_slots, Lk]  fused [kv_c | k_pe]
    block_table: torch.Tensor,  # [bs, max_blocks] selected page ids
    seqlens: torch.Tensor,      # [bs] sparse token count
    sm_scale: float,
    block_size: int,
    kv_lora_rank: int,          # Lv = 512
    o: torch.Tensor = None,     # [bs, H, Lv] (allocated if None)
    block_n: int = None,
    num_warps: int = 8,         # tuned: 8 > 4 at high bs (~33% vs 30% peak)
    num_stages: int = 2,        # tuned: >2 OOMs/regresses (shmem-bound 512-wide k-tile)
) -> torch.Tensor:
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    assert Lk == 576 and Lv == 512, f"MLA latent must be 512+64=576 (got Lk={Lk}, Lv={Lv})"
    if o is None:
        o = q.new_empty((bs, H, Lv))

    kv_group_num = H  # single latent kv head
    BLOCK_H = 16
    BLOCK_DMODEL = 512
    BLOCK_DPE = 64
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = block_n if block_n is not None else block_size

    grid = (bs, triton.cdiv(H, min(BLOCK_H, kv_group_num)))

    _fwd_blocktable_mla_kernel[grid](
        q,
        latent,
        latent,
        sm_scale,
        seqlens,
        block_table,
        o,
        q.stride(0),
        q.stride(1),
        latent.stride(0),
        latent.stride(0),
        o.stride(0),
        o.stride(1),
        block_table.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=H,
        BLOCK_SIZE=block_size,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        Lk=Lk,
        Lv=Lv,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


# ====================================================================== #
# kv-split (flash-decode) variant: split the selected-KV loop across more
# programs for parallelism (the single-pass kernel serializes the whole KV
# loop per (batch, head-block), starving the GPU at low bs / long KV).
# Stage 1 writes per-split partial (out, lse); stage 2 reduces across splits.
# ====================================================================== #
_MIN_BLOCK_KV = 64


@triton.jit
# Split-kernel stage1：长序列/多 block 时先分段计算局部结果。
def _fwd_blocktable_split_stage1(
    Q, K_Buffer, V_Buffer, sm_scale,
    Seqlens, Block_Table, num_kv_splits,
    Mid_O,        # [bs, H, MAX_SPLITS, Lv]
    Mid_Lse,      # [bs, H, MAX_SPLITS]
    stride_qbs, stride_qh,
    stride_buf_kbs, stride_buf_vbs,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_lse_b, stride_lse_h,
    stride_bt_b,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr, Lk: tl.constexpr, Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    mask_dpe = offs_dpe < Lk

    seqlen = tl.load(Seqlens + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = tl.cdiv(tl.cdiv(seqlen, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, seqlen)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
        q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        qpe = tl.load(Q + off_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < split_kv_end
            page = tl.load(Block_Table + cur_batch * stride_bt_b + (offs_n // BLOCK_SIZE),
                           mask=valid, other=0)
            kv_loc = page * BLOCK_SIZE + (offs_n % BLOCK_SIZE)

            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + offs_d[:, None]
            k = tl.load(K_Buffer + offs_buf_k, mask=valid[None, :] & mask_d[:, None], other=0.0)
            qk = tl.dot(q, k.to(q.dtype))
            offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + offs_dpe[:, None]
            kpe = tl.load(K_Buffer + offs_buf_kpe, mask=valid[None, :] & mask_dpe[:, None], other=0.0)
            qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale
            qk = tl.where(mask_h[:, None] & valid[None, :], qk, float("-inf"))

            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :]
            v = tl.load(V_Buffer + offs_buf_v, mask=valid[:, None] & mask_dv[None, :], other=0.0)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
                    + split_kv_id * stride_mid_os + offs_dv[None, :])
        tl.store(Mid_O + offs_mid, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])
        offs_lse = cur_batch * stride_lse_b + cur_head * stride_lse_h + split_kv_id
        tl.store(Mid_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
# Split-kernel stage2：把 stage1 的局部结果归并成最终 attention 输出。
def _fwd_blocktable_split_stage2(
    Mid_O, Mid_Lse, O, Seqlens, num_kv_splits,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_lse_b, stride_lse_h,
    stride_obs, stride_oh,
    MIN_BLOCK_KV: tl.constexpr, MAX_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr, Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    seqlen = tl.load(Seqlens + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = tl.cdiv(tl.cdiv(seqlen, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    base_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    base_l = cur_batch * stride_lse_b + cur_head * stride_lse_h

    for s in range(0, MAX_SPLITS):
        if s < kv_splits and (kv_len_per_split * s) < seqlen:
            tv = tl.load(Mid_O + base_o + s * stride_mid_os + offs_d, mask=mask_d, other=0.0)
            tlse = tl.load(Mid_Lse + base_l + s)
            n_e_max = tl.maximum(tlse, e_max)
            old = tl.exp(e_max - n_e_max)
            p = tl.exp(tlse - n_e_max)
            acc = acc * old + p * tv
            e_sum = e_sum * old + p
            e_max = n_e_max

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
             acc / e_sum, mask=mask_d)


_MAX_KV_SPLITS = 16


def decode_blocktable_mla_split(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
):
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    if o is None:
        o = q.new_empty((bs, H, Lv))
    kv_group_num = H
    BLOCK_H = 16
    BLOCK_DMODEL, BLOCK_DPE = 512, 64
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = block_size
    n_head_blocks = triton.cdiv(H, min(BLOCK_H, kv_group_num))

    # num_kv_splits: PURE host arithmetic — no seqlens read (must stay
    # cuda-graph-capture safe; seqlens are dummy at capture). Target ~4 waves
    # over the SMs. Short seqlens just yield empty splits, which stage1/stage2
    # treat as no-ops, so no seqlen-based cap is needed.
    import torch
    sm_count = _get_sm_count(q.device)
    nsplit = max(1, min(_MAX_KV_SPLITS, (4 * sm_count) // max(1, bs * n_head_blocks)))
    num_kv_splits = torch.full((bs,), nsplit, dtype=torch.int32, device=q.device)

    mid_o = q.new_empty((bs, H, _MAX_KV_SPLITS, Lv), dtype=torch.float32)
    mid_lse = q.new_empty((bs, H, _MAX_KV_SPLITS), dtype=torch.float32)

    _fwd_blocktable_split_stage1[(bs, n_head_blocks, nsplit)](
        q, latent, latent, sm_scale, seqlens, block_table, num_kv_splits,
        mid_o, mid_lse,
        q.stride(0), q.stride(1), latent.stride(0), latent.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        block_table.stride(0),
        kv_group_num=kv_group_num, q_head_num=H, BLOCK_SIZE=block_size,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, MIN_BLOCK_KV=_MIN_BLOCK_KV, Lk=Lk, Lv=Lv,
        num_warps=4, num_stages=2,
    )
    _fwd_blocktable_split_stage2[(bs, H)](
        mid_o, mid_lse, o, seqlens, num_kv_splits,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        o.stride(0), o.stride(1),
        MIN_BLOCK_KV=_MIN_BLOCK_KV, MAX_SPLITS=_MAX_KV_SPLITS, BLOCK_DV=BLOCK_DV, Lv=Lv,
        num_warps=4, num_stages=2,
    )
    return o


@triton.jit
def _fwd_blocktable_tma_kernel(
    Q, Latent, num_slots, latent_stride0, sm_scale, Seqlens, Block_Table, O,
    stride_qbs, stride_qh, stride_obs, stride_oh, stride_bt_b,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr, BLOCK_H: tl.constexpr, Lk: tl.constexpr, Lv: tl.constexpr,
):
    # TMA block dims must be pow2; the 576 latent isn't, so use two descriptors
    # over the nope (512) and rope (64) column ranges (both pow2).
    desc_nope = tl.make_tensor_descriptor(
        Latent, shape=[num_slots, BLOCK_DMODEL], strides=[latent_stride0, 1],
        block_shape=[BLOCK_SIZE, BLOCK_DMODEL],
    )
    desc_rope = tl.make_tensor_descriptor(
        Latent + BLOCK_DMODEL, shape=[num_slots, BLOCK_DPE], strides=[latent_stride0, 1],
        block_shape=[BLOCK_SIZE, BLOCK_DPE],
    )
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    mask_d = offs_d < Lk
    mask_dpe = offs_dpe < Lk
    mask_dv = offs_dv < Lv

    seqlen = tl.load(Seqlens + cur_batch)
    n_blocks = tl.cdiv(seqlen, BLOCK_SIZE)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(Q + off_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for blk in range(0, n_blocks):
        page = tl.load(Block_Table + cur_batch * stride_bt_b + blk)
        # TMA bulk loads of the page's contiguous [BLOCK_SIZE, *] tiles.
        k_nope = tl.load_tensor_descriptor(desc_nope, [page * BLOCK_SIZE, 0])  # [BS, 512]
        k_pe = tl.load_tensor_descriptor(desc_rope, [page * BLOCK_SIZE, 0])    # [BS, 64]
        offs_n = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs_n < seqlen

        qk = tl.dot(q, tl.trans(k_nope).to(q.dtype))
        qk += tl.dot(qpe, tl.trans(k_pe).to(qpe.dtype))
        qk *= sm_scale
        qk = tl.where(mask_h[:, None] & valid[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(k_nope.dtype), k_nope)              # reuse tile for V (no reload)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_dv[None, :]
    tl.store(O + offs_o, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])


_tma_alloc_set = False


def decode_blocktable_mla_tma(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
):
    import torch
    global _tma_alloc_set
    if not _tma_alloc_set:
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
        )
        _tma_alloc_set = True
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    if o is None:
        o = q.new_empty((bs, H, Lv))
    BLOCK_H = 16
    grid = (bs, triton.cdiv(H, min(BLOCK_H, H)))
    _fwd_blocktable_tma_kernel[grid](
        q, latent, latent.shape[0], latent.stride(0), sm_scale, seqlens, block_table, o,
        q.stride(0), q.stride(1), o.stride(0), o.stride(1), block_table.stride(0),
        kv_group_num=H, q_head_num=H, BLOCK_SIZE=block_size,
        BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_H=BLOCK_H, Lk=Lk, Lv=Lv, num_warps=8, num_stages=2,
    )
    return o


@triton.jit
def _fwd_tma_split_stage1(
    Q, Latent, num_slots, latent_stride0, sm_scale,
    Seqlens, Block_Table, num_kv_splits, Mid_O, Mid_Lse,
    stride_qbs, stride_qh,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_lse_b, stride_lse_h, stride_bt_b,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr, BLOCK_H: tl.constexpr, Lk: tl.constexpr, Lv: tl.constexpr,
):
    desc_nope = tl.make_tensor_descriptor(
        Latent, shape=[num_slots, BLOCK_DMODEL], strides=[latent_stride0, 1],
        block_shape=[BLOCK_SIZE, BLOCK_DMODEL])
    desc_rope = tl.make_tensor_descriptor(
        Latent + BLOCK_DMODEL, shape=[num_slots, BLOCK_DPE], strides=[latent_stride0, 1],
        block_shape=[BLOCK_SIZE, BLOCK_DPE])

    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    mask_d = offs_d < Lk
    mask_dpe = offs_dpe < Lk
    mask_dv = offs_dv < Lv

    seqlen = tl.load(Seqlens + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)
    n_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    bps = tl.cdiv(n_blocks, kv_splits)        # blocks per split
    blk_start = split_id * bps
    blk_end = tl.minimum(blk_start + bps, n_blocks)

    offs_lse = cur_batch * stride_lse_b + cur_head * stride_lse_h + split_id

    if (split_id < kv_splits) and (blk_start < n_blocks):
        offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
        q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        qpe = tl.load(Q + off_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

        e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

        for blk in range(blk_start, blk_end):
            page = tl.load(Block_Table + cur_batch * stride_bt_b + blk)
            k_nope = tl.load_tensor_descriptor(desc_nope, [page * BLOCK_SIZE, 0])
            k_pe = tl.load_tensor_descriptor(desc_rope, [page * BLOCK_SIZE, 0])
            offs_n = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            valid = offs_n < seqlen
            qk = tl.dot(q, tl.trans(k_nope).to(q.dtype))
            qk += tl.dot(qpe, tl.trans(k_pe).to(qpe.dtype))
            qk *= sm_scale
            qk = tl.where(mask_h[:, None] & valid[None, :], qk, float("-inf"))
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(k_nope.dtype), k_nope)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
                    + split_id * stride_mid_os + offs_dv[None, :])
        tl.store(Mid_O + offs_mid, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])
        tl.store(Mid_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_h)
    else:
        # empty split — mark inactive so stage2 skips it.
        tl.store(Mid_Lse + offs_lse, tl.full([BLOCK_H], -float("inf"), tl.float32), mask=mask_h)


@triton.jit
def _fwd_tma_split_stage2(
    Mid_O, Mid_Lse, O,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_lse_b, stride_lse_h,
    stride_obs, stride_oh,
    MAX_SPLITS: tl.constexpr, BLOCK_DV: tl.constexpr, Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    base_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    base_l = cur_batch * stride_lse_b + cur_head * stride_lse_h
    for s in range(0, MAX_SPLITS):
        tlse = tl.load(Mid_Lse + base_l + s)
        if tlse > -float("inf"):   # skip inactive splits
            tv = tl.load(Mid_O + base_o + s * stride_mid_os + offs_d, mask=mask_d, other=0.0)
            n_e_max = tl.maximum(tlse, e_max)
            old = tl.exp(e_max - n_e_max)
            p = tl.exp(tlse - n_e_max)
            acc = acc * old + p * tv
            e_sum = e_sum * old + p
            e_max = n_e_max
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / e_sum, mask=mask_d)


def decode_blocktable_mla_tma_split(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
):
    """TMA + kv-splits combined: split the selected pages across programs (low-bs
    parallelism) and TMA-load each page's contiguous tile once, reusing it for
    q.k and p.v (high-bs bandwidth + no redundant V load)."""
    import torch
    global _tma_alloc_set
    if not _tma_alloc_set:
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        _tma_alloc_set = True
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    if o is None:
        o = q.new_empty((bs, H, Lv))
    kv_group_num = H
    BLOCK_H = 16
    BLOCK_DMODEL, BLOCK_DPE = 512, 64
    BLOCK_DV = triton.next_power_of_2(Lv)
    n_head_blocks = triton.cdiv(H, min(BLOCK_H, kv_group_num))

    # grid split count: PURE host arithmetic (no seqlens read — cuda-graph-capture
    # safe). The per-batch split count below is a GPU op on the live seqlens
    # (capturable, recomputed each replay, no host sync); trailing empty splits
    # are no-ops in stage1/stage2.
    sm_count = _get_sm_count(q.device)
    nsplit = max(1, min(_MAX_KV_SPLITS, (4 * sm_count) // max(1, bs * n_head_blocks)))
    n_blocks_each = torch.div(seqlens + block_size - 1, block_size, rounding_mode="floor").to(torch.int32)
    num_kv_splits = torch.clamp(n_blocks_each, max=nsplit).to(torch.int32)

    mid_o = q.new_empty((bs, H, nsplit, Lv), dtype=torch.float32)
    mid_lse = q.new_empty((bs, H, nsplit), dtype=torch.float32)

    _fwd_tma_split_stage1[(bs, n_head_blocks, nsplit)](
        q, latent, latent.shape[0], latent.stride(0), sm_scale,
        seqlens, block_table, num_kv_splits, mid_o, mid_lse,
        q.stride(0), q.stride(1),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1), block_table.stride(0),
        kv_group_num=kv_group_num, q_head_num=H, BLOCK_SIZE=block_size,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV,
        BLOCK_H=BLOCK_H, Lk=Lk, Lv=Lv, num_warps=8, num_stages=2,
    )
    _fwd_tma_split_stage2[(bs, H)](
        mid_o, mid_lse, o,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1), o.stride(0), o.stride(1),
        MAX_SPLITS=nsplit, BLOCK_DV=BLOCK_DV, Lv=Lv, num_warps=4, num_stages=2,
    )
    return o


# 自动选择入口：根据 shape/硬件在普通、split、TMA 等实现之间选一个。
def decode_blocktable_mla_auto(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
):
    """Adaptive: single-pass when there is enough (batch x head-block)
    parallelism to fill the GPU; kv-split when parallelism-starved (low bs /
    long KV), where the split's extra mid-buffer traffic pays off. Avoids the
    split's fp32 mid-buffer round-trip at high bs (where it would dominate the
    KV traffic) and the single-pass GPU under-utilization at low bs."""
    bs, H, _ = q.shape
    BLOCK_H = 16
    n_head_blocks = triton.cdiv(H, min(BLOCK_H, H))
    # Decision uses only host-known bs (NO seqlens.max().item() sync — that would
    # serialize + break cuda-graph capture). Single-pass fills the GPU once the
    # program count reaches ~half the SMs (B200=148); below that the split
    # recovers idle SMs. Dense branch uses TMA (>= plain single-pass at block>=32).
    _SM_HALF = 74
    if bs * n_head_blocks < _SM_HALF:
        return decode_blocktable_mla_split(q, latent, block_table, seqlens, sm_scale,
                                           block_size, kv_lora_rank, o)
    return decode_blocktable_mla_tma(q, latent, block_table, seqlens, sm_scale,
                                     block_size, kv_lora_rank, o)


# ====================================================================== #
# plan(): per-decode-step kernel dispatch via a PRE-SET RULE.
#
# MUST be a pure function of (bs, block_size, num_query_heads) — NO profiling,
# NO seqlens, NO GPU reads/sync. The first call happens during cuda-graph
# CAPTURE, where (a) seqlens are dummy placeholders (timing them is meaningless)
# and (b) any sync/timing inside capture is illegal. So the choice is decided
# from host-known shape ints only, and is therefore stable across capture/replay
# (a fixed kernel is captured for each (bs, block, H)).
#
# Calibrated from the B200 (bs, block, H) bandwidth sweep at H=16/20, sel=2048
# (marks/mla/plan_sweep2.jsonl). The candidate pool is the OPTIMIZED kernels
# (kv_split_opt / single_pass_opt); the old single_pass / kv_split / tma are
# dominated and no longer dispatched. Driver = optimized program count
# prog = bs * ceil(H / BLOCK_H_opt(H)):
#
#   * kv_split_opt is the default. At these low head counts (16/20) a single-
#     pass program count of prog<=128 leaves the 148-SM B200 under-filled, so
#     the split's extra split-dimension wins almost everywhere; its BLOCK_H=32
#     also removes the redundant 2nd head-block at H=20 (vs the old kv_split:
#     +18..+72% at high bs).
#   * single_pass_opt overtakes it only in the fully-saturated, bandwidth-bound
#     corner — large pages (block>=64) AND enough programs to fill the GPU
#     (prog>=128). There the split's fp32 mid-buffer round-trip costs more than
#     its parallelism buys (H=16 bs=128 blk=64: 2688 vs 2273 GB/s, +18%).
#
# Validated against the measured oracle (marks/mla/plan_sweep2.jsonl): matches
# the best kernel (within ~2%, i.e. measurement noise) in every swept cell.
# ====================================================================== #
_PLAN_SATURATE_PROG = 128   # prog at/above which single-pass fills the SMs (B200)


# 计划选择：用 batch/head/block 信息决定 decode 走哪种 kernel 配置。
def plan_select(bs, block_size, num_query_heads):
    """Pure pre-set dispatch rule → kernel name. No GPU, no seqlens, no profiling
    (cuda-graph-capture safe — decided from host-known shape ints only).
    Calibrated from the B200 (bs, block, H) sweep at H=16/20."""
    bh = _opt_block_h(num_query_heads)
    prog = bs * ((num_query_heads + bh - 1) // bh)
    # single-pass only in the saturated + bandwidth-bound corner; else the split.
    if block_size >= 64 and prog >= _PLAN_SATURATE_PROG:
        return "single_pass_opt"
    return "kv_split_opt"


# 计划化 decode：外部已经选好 plan 时从这里执行。
def decode_blocktable_mla_plan(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
):
    bs, H, _ = q.shape
    if o is None:
        o = q.new_empty((bs, H, kv_lora_rank))
    name = plan_select(bs, block_size, H)
    return KERNELS[name](q, latent, block_table, seqlens, sm_scale,
                         block_size, kv_lora_rank, o)


# ====================================================================== #
# OPTIMIZED single-pass + TMA kernels.
#
# Two changes over the baselines above, both grounded in the decode geometry:
#
#  (1) **Last-block-only masking (no masked LOADS, no separate tail).** The
#      baseline applies the boundary mask ``valid = offs_n < seqlen`` on EVERY
#      KV tile — masked K/V loads (``other=0``) plus a ``tl.where(..., -inf)``.
#      Only the *last* tile can straddle ``seqlen``; for every full tile
#      ``valid`` is all-true, so that work is wasted. We drop the LOAD masks
#      entirely — each KV page is allocated full (``BLOCK_SIZE`` rows), so a
#      whole-tile read is always in-bounds even when the request ends mid-page —
#      and keep only a score-side ``tl.where(offs_n < seqlen, qk, -inf)`` that
#      is a no-op except on the last tile and simply drops the over-read tokens
#      out of the softmax. Crucially this stays a SINGLE loop: a peeled,
#      duplicated masked *tail block* inflates the register/code footprint and
#      collapses occupancy at small block_size (measured -14..-27%), wiping out
#      the gain. The dim masks (``mask_d/dpe/dv``) are dropped too (``offs_d <
#      576`` / ``offs_dv < 512`` are provably all-true), and invalid query-head
#      rows need no per-tile ``where``: they load q=0 (finite softmax) and are
#      discarded by the masked final store.
#
#  (2) **Larger query-head tile (``BLOCK_H``).** Each ``(batch, head_block)``
#      program streams the request's ENTIRE selected KV. With ``H`` query heads
#      that is ``ceil(H / BLOCK_H)`` independent re-reads of the same KV. The
#      baseline's ``BLOCK_H=16`` therefore re-reads the KV 8x at the real MLA
#      head count (H=128); raising ``BLOCK_H`` cuts the redundant traffic
#      (H=128: 16->8 blocks, 32->4, 64->2) and feeds the tensor cores wider
#      ``[BLOCK_H, *]`` MMA tiles. ``_opt_block_h`` picks it from H.
# ====================================================================== #
def _opt_block_h(H: int) -> int:
    """Query-head tile. Bigger => fewer redundant full-KV re-reads across
    head-blocks (H=128: 16->8 blocks, 32->4) and wider MMA. Capped at 32:
    BLOCK_H=64 is non-viable here — acc=[64,512]fp32 + double-buffered K/V tiles
    overflow B200 shared memory at block_size=64 (~352KB > 227KB) and trip a
    Triton vectorized-gather misalignment at block_size=32. 32 is the measured
    sweet spot (see report)."""
    return 16 if H <= 16 else 32


@triton.jit
def _fwd_blocktable_mla_kernel_opt(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Seqlens,
    Block_Table,
    O,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_obs,
    stride_oh,
    stride_bt_b,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)

    seqlen = tl.load(Seqlens + cur_batch)

    # query is loaded once; invalid head rows get 0 (=> finite softmax, dropped
    # by the masked store). No dim mask: offs_d<576 / offs_dpe<576 are all-true.
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    bt_base = Block_Table + cur_batch * stride_bt_b
    n_blocks = tl.cdiv(seqlen, BLOCK_N)

    # SINGLE loop over ceil(seqlen / BLOCK_N) tiles. Loads are UNMASKED: every
    # KV page is allocated full (BLOCK_SIZE rows), so reading a whole tile is
    # always in-bounds even when the request ends mid-page. Only the SCORE is
    # masked (``offs_n < seqlen``) — and that predicate is all-true for every
    # tile EXCEPT the last (the "only the last block needs a mask" property),
    # so it is a no-op in the steady state and just drops the partial tail's
    # over-read tokens out of the softmax. There is deliberately NO separate
    # masked tail block: duplicating the load/dot/softmax body inflates the
    # register + code footprint and collapses occupancy at small block_size
    # (measured -14..-27%); the in-loop predicate keeps the steady state intact.
    for blk in range(0, n_blocks):
        offs_n = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        page = tl.load(bt_base + (offs_n // BLOCK_SIZE))
        kv_loc = page * BLOCK_SIZE + (offs_n % BLOCK_SIZE)

        k = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_d[:, None])
        qk = tl.dot(q, k.to(q.dtype))
        kpe = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_dpe[:, None])
        qk += tl.dot(qpe, kpe.to(qpe.dtype))
        qk *= sm_scale
        qk = tl.where(offs_n[None, :] < seqlen, qk, float("-inf"))

        v = tl.load(V_Buffer + kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :])
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_dv[None, :]
    tl.store(O + offs_o, acc / e_sum[:, None], mask=mask_h[:, None])


def decode_blocktable_mla_opt(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
    block_n=None, block_h=None, num_warps=8, num_stages=2,
) -> torch.Tensor:
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    assert Lk == 576 and Lv == 512, f"MLA latent must be 512+64=576 (got Lk={Lk}, Lv={Lv})"
    if o is None:
        o = q.new_empty((bs, H, Lv))
    kv_group_num = H
    BLOCK_H = block_h if block_h is not None else _opt_block_h(H)
    BLOCK_DMODEL, BLOCK_DPE = 512, 64
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = block_n if block_n is not None else block_size
    grid = (bs, triton.cdiv(H, min(BLOCK_H, kv_group_num)))

    _fwd_blocktable_mla_kernel_opt[grid](
        q, latent, latent, sm_scale, seqlens, block_table, o,
        q.stride(0), q.stride(1), latent.stride(0), latent.stride(0),
        o.stride(0), o.stride(1), block_table.stride(0),
        kv_group_num=kv_group_num, q_head_num=H, BLOCK_SIZE=block_size,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, Lk=Lk, Lv=Lv,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


@triton.jit
def _fwd_blocktable_tma_kernel_opt(
    Q, Latent, num_slots, latent_stride0, sm_scale, Seqlens, Block_Table, O,
    stride_qbs, stride_qh, stride_obs, stride_oh, stride_bt_b,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr, BLOCK_H: tl.constexpr, Lk: tl.constexpr, Lv: tl.constexpr,
):
    desc_nope = tl.make_tensor_descriptor(
        Latent, shape=[num_slots, BLOCK_DMODEL], strides=[latent_stride0, 1],
        block_shape=[BLOCK_SIZE, BLOCK_DMODEL])
    desc_rope = tl.make_tensor_descriptor(
        Latent + BLOCK_DMODEL, shape=[num_slots, BLOCK_DPE], strides=[latent_stride0, 1],
        block_shape=[BLOCK_SIZE, BLOCK_DPE])
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)

    seqlen = tl.load(Seqlens + cur_batch)
    n_blocks = tl.cdiv(seqlen, BLOCK_SIZE)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    # SINGLE loop over all pages; TMA descriptor loads the full page (always
    # in-bounds). Only the score is masked, and (offs_n < seqlen) is all-true
    # except on the last page — no separate tail block (see single-pass note).
    for blk in range(0, n_blocks):
        page = tl.load(Block_Table + cur_batch * stride_bt_b + blk)
        k_nope = tl.load_tensor_descriptor(desc_nope, [page * BLOCK_SIZE, 0])  # [BS, 512]
        k_pe = tl.load_tensor_descriptor(desc_rope, [page * BLOCK_SIZE, 0])    # [BS, 64]
        offs_n = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        qk = tl.dot(q, tl.trans(k_nope).to(q.dtype))
        qk += tl.dot(qpe, tl.trans(k_pe).to(qpe.dtype))
        qk *= sm_scale
        qk = tl.where(offs_n[None, :] < seqlen, qk, float("-inf"))
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])   # p=0 for over-read cols => no contribution
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(k_nope.dtype), k_nope)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_dv[None, :]
    tl.store(O + offs_o, acc / e_sum[:, None], mask=mask_h[:, None])


def decode_blocktable_mla_tma_opt(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
    block_h=None, num_warps=8, num_stages=2,
):
    import torch
    global _tma_alloc_set
    if not _tma_alloc_set:
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        _tma_alloc_set = True
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    if o is None:
        o = q.new_empty((bs, H, Lv))
    BLOCK_H = block_h if block_h is not None else _opt_block_h(H)
    grid = (bs, triton.cdiv(H, min(BLOCK_H, H)))
    _fwd_blocktable_tma_kernel_opt[grid](
        q, latent, latent.shape[0], latent.stride(0), sm_scale, seqlens, block_table, o,
        q.stride(0), q.stride(1), o.stride(0), o.stride(1), block_table.stride(0),
        kv_group_num=H, q_head_num=H, BLOCK_SIZE=block_size,
        BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_H=BLOCK_H, Lk=Lk, Lv=Lv, num_warps=num_warps, num_stages=num_stages,
    )
    return o


@triton.jit
def _fwd_blocktable_split_stage1_opt(
    Q, K_Buffer, V_Buffer, sm_scale,
    Seqlens, Block_Table, num_kv_splits,
    Mid_O, Mid_Lse,
    stride_qbs, stride_qh,
    stride_buf_kbs, stride_buf_vbs,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_lse_b, stride_lse_h,
    stride_bt_b,
    kv_group_num: tl.constexpr, q_head_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr, Lk: tl.constexpr, Lv: tl.constexpr,
):
    # kv-split stage-1 with the opt kernels' two changes: adaptive BLOCK_H
    # (merges the wasteful 2nd head-block at H in (16,32]) and last-block-only
    # masking (unmasked full-page loads + a score-side `where` that only bites
    # the split's final tile). Stage-2 reduction is shared (unchanged).
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)

    seqlen = tl.load(Seqlens + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = tl.cdiv(tl.cdiv(seqlen, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, seqlen)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
        q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            page = tl.load(Block_Table + cur_batch * stride_bt_b + (offs_n // BLOCK_SIZE))
            kv_loc = page * BLOCK_SIZE + (offs_n % BLOCK_SIZE)

            k = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_d[:, None])
            qk = tl.dot(q, k.to(q.dtype))
            kpe = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_dpe[:, None])
            qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale
            qk = tl.where(offs_n[None, :] < split_kv_end, qk, float("-inf"))

            v = tl.load(V_Buffer + kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :])
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
                    + split_kv_id * stride_mid_os + offs_dv[None, :])
        tl.store(Mid_O + offs_mid, acc / e_sum[:, None], mask=mask_h[:, None])
        offs_lse = cur_batch * stride_lse_b + cur_head * stride_lse_h + split_kv_id
        tl.store(Mid_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_h)


def decode_blocktable_mla_split_opt(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
    block_h=None, num_warps=4,
):
    """kv-split (flash-decode) with the opt-kernel improvements. Wins across most
    of the (bs, block) plane at H=16/20: the split's extra parallelism fills the
    B200's 148 SMs (single-pass starves there), and BLOCK_H=32 removes the
    redundant 2nd-head-block KV pass at H=20."""
    import torch
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    if o is None:
        o = q.new_empty((bs, H, Lv))
    kv_group_num = H
    BLOCK_H = block_h if block_h is not None else _opt_block_h(H)
    BLOCK_DMODEL, BLOCK_DPE = 512, 64
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_N = block_size
    n_head_blocks = triton.cdiv(H, min(BLOCK_H, kv_group_num))

    sm_count = _get_sm_count(q.device)
    nsplit = max(1, min(_MAX_KV_SPLITS, (4 * sm_count) // max(1, bs * n_head_blocks)))
    num_kv_splits = torch.full((bs,), nsplit, dtype=torch.int32, device=q.device)

    mid_o = q.new_empty((bs, H, _MAX_KV_SPLITS, Lv), dtype=torch.float32)
    mid_lse = q.new_empty((bs, H, _MAX_KV_SPLITS), dtype=torch.float32)

    _fwd_blocktable_split_stage1_opt[(bs, n_head_blocks, nsplit)](
        q, latent, latent, sm_scale, seqlens, block_table, num_kv_splits,
        mid_o, mid_lse,
        q.stride(0), q.stride(1), latent.stride(0), latent.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        block_table.stride(0),
        kv_group_num=kv_group_num, q_head_num=H, BLOCK_SIZE=block_size,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, MIN_BLOCK_KV=_MIN_BLOCK_KV, Lk=Lk, Lv=Lv,
        num_warps=num_warps, num_stages=2,
    )
    _fwd_blocktable_split_stage2[(bs, H)](
        mid_o, mid_lse, o, seqlens, num_kv_splits,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        o.stride(0), o.stride(1),
        MIN_BLOCK_KV=_MIN_BLOCK_KV, MAX_SPLITS=_MAX_KV_SPLITS, BLOCK_DV=BLOCK_DV, Lv=Lv,
        num_warps=4, num_stages=2,
    )
    return o


# registry for the microbenchmark (marks/mla/bench_triton_mla.py)
KERNELS = {
    "single_pass": decode_blocktable_mla,
    "kv_split": decode_blocktable_mla_split,
    "adaptive": decode_blocktable_mla_auto,
    "tma": decode_blocktable_mla_tma,
    "tma_split": decode_blocktable_mla_tma_split,
    "plan": decode_blocktable_mla_plan,
    "single_pass_opt": decode_blocktable_mla_opt,
    "tma_opt": decode_blocktable_mla_tma_opt,
    "kv_split_opt": decode_blocktable_mla_split_opt,
}
