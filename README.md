<p align="center">
  <img
    alt="Vortex"
    src="assets/vortex_logo_flat.png"
    width="55%"
  />
</p>

<h3 align="center">
Vortex: A Flexible and Efficient Sparse Attention Framework
</h3>

<p align="center">
  <a href="https://infini-ai-lab.github.io/vortex_torch/"><b>Documentation</b></a>
</p>


Vortex is a lightweight, modular framework for building **custom sparse attention algorithms** for LLM inference.  
It exists to make it easy for researchers and engineers to **prototype**, **extend**, and **deploy** advanced sparsity patterns on modern inference backends such as SGLang—without modifying core model code.

Vortex allows you to express novel sparse attention concisely while relying on an optimized execution engine.

<figure>
  <img src="assets/demo.gif" alt="Demo" />
  <figcaption align="center"><em>OpenHands generate a sparse attention algorithm (up to 2.7X speedup in this example).</em></figcaption>
</figure>


---

## ✨ Key Features

- **Easy Programming**  
  Program sparse attention with a PyTorch-like frontend. No worrying about batching, caching & paged attention.

- **High Performance**  
  Built to work with FlashInfer & CUDA Graph & Radix Attention for efficient LLM inference.

---

## 🚀 Installation

```bash
git clone -b v1 --recursive https://github.com/Infini-AI-Lab/vortex_torch.git

# Install SGLang dependency (support 0.4.9)
cd third_party/sglang
bash install.sh
cd ../../

# Install Vortex
cd vortex_torch
pip install -e .
```

---

## ☁️ Run On Modal

This repo now includes [`modal_app.py`](modal_app.py), which builds a CUDA-capable image, installs the pinned `third_party/sglang` dependency, mounts a persistent Hugging Face cache volume, and runs the benchmark remotely on Modal.

Important: [`setup.py`](setup.py) only compiles kernels for `sm_89` and `sm_90`, so you should use `L40S`, `H100`, or `H200` class GPUs on Modal. `A10`, `L4`, and `A100` are not good defaults for this repo as-is.

1. Make sure the local checkout has the SGLang submodule metadata available:

```bash
git submodule update --init --recursive
```

2. Make sure Modal is installed and authenticated locally:

```bash
python3 -m ensurepip --upgrade
python3 -m pip install --user modal
python3 -m modal setup
python3 -m modal token info
python3 -m modal profile current
```

3. Run a remote smoke test on a compatible GPU:

```bash
VORTEX_MODAL_GPU=L40S python3 -m modal run modal_app.py
```

4. Run the verification benchmark remotely:

```bash
VORTEX_MODAL_GPU=L40S python3 -m modal run modal_app.py \
  --command verify \
  --vortex-module-name gqa_block_sparse_attention \
  --model-name Qwen/Qwen3-1.7B \
  --trials 2 \
  --topk-val 30 \
  --mem 0.8
```

5. Switch to Hopper if you want a faster card:

```bash
VORTEX_MODAL_GPU=H100 python3 -m modal run modal_app.py --command verify
```

If the model weights are already downloaded once, later runs will reuse the Modal volume named `vortex-hf-cache`.

---

## 🤖 AI-Generated Sparse Attention

Vortex is designed not only for hand-crafted sparsity patterns but also for AI-generated sparse attention.

Our demo shows how to use SOTA agents OpenHands (https://openhands.dev/) to generate sparse attention algorithms.

```bash
export LLM_API_KEY=YOUR_API_KEY
python openhands_gen.py

```

The usage and installation guide of OpenHands can be found in https://docs.openhands.dev/sdk. 

Note: Some operators are not yet fused or fully optimized, which may lead to increased memory usage. Tune down the `mem_fraction_static` if CUDA OOM. This can also impact generation speed during inference. 

---

## 🧩 Quick Example: Custom Sparse Attention

```python
@register("custom_sparse_attention")
class CustomSparseAttention(vFlow):

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemv = GeMV()
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,                 # viewed as [1, H_q, D]
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],  # viewed as [S, r, c] depending on create_cache()
        ctx: ContextBase,
    ):
        q_mean = q.mean(dim=1, keepdim=True)
        score = self.gemv(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],  # viewed as [B, r, c] depending on create_cache()
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        # triggered only when a page is finished
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
        }
```

---

## 🏃 Using Your Sparse Attention with SGLang

```python
llm = sgl.Engine(
    model_path="Qwen/Qwen3-0.6B",
    disable_cuda_graph=False,
    page_size=16,
    vortex_topk_val=30,
    disable_overlap_schedule=True,    # Mandatory
    attention_backend="flashinfer",   # Mandatory
    enable_vortex_sparsity=True,      # Otherwise full attention is used
    vortex_page_reserved_bos=1,
    vortex_page_reserved_eos=1,
    vortex_layers_skip=list(range(1)),  # Full attention for layer 0
    vortex_module_path="path/to/custom_sparse_attention.py",
    vortex_module_name="custom_sparse_attention", # the registered name for your algorithm
    vortex_max_seq_lens=8192,
    mem_fraction_static=0.85,
)
```

If `vortex_module_path` is not provided, Vortex will automatically search in `vortex_torch.flow.algorithms`.

---


## 📘 API Reference

👉 https://infini-ai-lab.github.io/vortex_torch/

