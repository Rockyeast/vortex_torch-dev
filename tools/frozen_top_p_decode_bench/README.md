# Frozen Top-P decode speed on H200 (sm_90)

Work from 2026-11-05 and 2026-11-09/10 on the decode path of `frozen_dynamic_top_p`.
Model: DeepSeek-R1-Distill-Qwen-1.5B (28 layers, 12 Q heads, 2 KV heads, head_dim 128), one H200, SGLang 0.5.15.post1.
Selection rule, block counts, tail handling and the online softmax are unchanged; only how the work is executed changed.

## What changed in `vortex_torch/`

| File | Change |
|---|---|
| `frozen_dynamic_top_p/isolated_paged_attention_triton.py` | New `_paged_gqa_decode_split_tc_kernel` and `paged_gqa_decode_attention_with_tail_splitk_tc`: one program per KV head and split, the whole Q group together, both products on tensor cores, several selected blocks per iteration. The earlier split-K kernel ran one program per Q head, so each K/V block was read once per Q head (6 times here). Probabilities go through two FP16 terms so the value product keeps FP32-level accuracy. |
| `frozen_dynamic_top_p/centroid_scoring_triton.py` | The scoring grid is sized by the workspace capacity (2048 blocks). Programs wholly past the sequence's real length now return after marking their tile empty. Tile width 32 blocks at batch >= 32. |
| `frozen_dynamic_top_p/fps_centroid_triton.py` | `fps_r4_lloyd1_scatter_from_locations`: the centroid update takes token locations and derives the slot and the "block just completed" test inside the kernel. Four small torch kernels per layer and step are gone. |
| `frozen_dynamic_top_p/top_p_selection_triton.py` | Bucketed selector (sort width from the row's real length, 11-05). 16 warps when the workspace width is 2048. |
| `frozen_dynamic_top_p/training_selection.py` | Kernel widths rounded to 64 blocks so microbatches share compiled variants (11-05). |
| `engine/sgl/frozen_dynamic_top_p.py` | Decode uses the tensor-core attention when Q/K/V are BF16 and head_dim is a power of two. Launch by batch: up to 4 sequences 32 splits x 4 blocks, up to 16 8 x 4, above 8 x 2. |

Switches (all read at import or launch): `VORTEX_FAST_MIN_MAJOR` (9 enables the fast path on H200, default 10),
`VORTEX_TC_ATTENTION=0` and `VORTEX_FOLD_FINALIZE=0` restore the previous attention and centroid-update paths,
`VORTEX_TEST_TOP_P` / `VORTEX_TEST_FIXED_K` are test overrides for the selector.

## Results

Milliseconds per decode step, greedy, prompts are the student's own AIME responses cut to the given length
(`spd.py`: wall time of 576 new tokens minus wall time of 64, divided by 512; two repeats averaged).

| Context | Concurrency | Dense (FlashInfer) | Sparse before | Sparse after | vs dense |
|---|---|---|---|---|---|
| 15K | 1 | 2.41 | 4.97 | 2.94 | 22% slower |
| 15K | 16 | 4.08 | 6.55 | 4.07 | equal |
| 15K | 64 | 9.00 | 10.63 | 6.52 | 1.38x |
| 15K | 128 | 15.53 | 16.19 | 9.62 | 1.61x |
| 14K | 128 | 14.20 | 16.12 | 9.47 | 1.50x |
| 14K | 256 | 27.73 | 28.36 | 16.44 | 1.69x |
| 8K | 128 | 10.23 | 13.95 | 8.31 | 1.23x |
| 8K | 256 | 17.53 | 24.42 | 13.13 | 1.34x |
| 4K | 128 | 6.46 | 11.15 | 6.10 | 1.06x |

"Sparse before" is this branch's parent plus the 11-05 changes. Concurrency 64 and 128 were measured on two cards
in both orders; concurrency 256 is one run. Sparse is slower than dense below roughly 250K history tokens in flight
(concurrency x context length), because scoring and selection are a fixed cost that dense attention does not pay.

AIME24+25, 60 problems, 32 samples each, 128 concurrent: dense 25.7%, sparse after 22.7%, sparse before 20.5%
(untrained student). After minus before: +2.1 points, 95% interval [-1.2, +5.6]. Wall time for 960 generations:
dense 1021 s, sparse after 822 s, sparse before 1486 s.

## Checks

- `parity_score_attention.py`: scoring output bitwise equal to the previous kernel; tensor-core attention within 7e-5
  relative error of the per-Q-head kernel.
- `parity_attention_fp64.py`: against an FP64 reference of the same definition, both attention kernels have the same
  error (1.7e-3, the BF16 output rounding floor), including short contexts and short tails.
- `parity_finalize_and_splits.py`: centroid update from token locations gives bitwise equal centroids and counts;
  32 splits versus 8 within 1e-6.
- Greedy generation is not token-identical to the previous path: outputs fork at near ties, with a mean log-prob
  difference of 0.003-0.007 before the fork (dense versus sparse on the same span: 0.014-0.030).

## Known limits

- A row past 1024 completed blocks (16384 tokens) sorts at width 2048: selection goes from about 16 to about 31 us per layer.
- More than 256 running requests falls outside the captured CUDA graphs and the frozen path fails (seen 2026-11-04).
- The tensor-core attention needs BF16 Q/K/V; FP8 KV caches take the previous kernel.

## Scripts

`spd.sh` / `spd.py` end-to-end step time, `prof.sh` / `prof.py` per-kernel profile of real decode steps,
`stage_times.py` and `select_times.py` per-stage timing in CUDA graphs, `aime.sh` the AIME check.
The scripts assume the layout they were run in: the working directory mounted at `/w` inside the container,
the model at `/w/model`, AIME responses at `/w/aime/dense.jsonl`, and code copies selected through `PYTHONPATH`.
