# Runtime validation matrix

The install contract and the GPU runtime contract are tested separately. A
successful wheel/import test does not prove CUDA Graph, overlap scheduling, or
multi-GPU execution.

Run one isolated GPU case on Modal:

```bash
modal run tests/gpu/modal_validation.py --case baseline-eager
modal run tests/gpu/modal_validation.py --case cuda-graph
modal run tests/gpu/modal_validation.py --case overlap
modal run tests/gpu/modal_validation.py --case cuda-graph-overlap
modal run tests/gpu/modal_validation.py --case dp2
modal run tests/gpu/modal_validation.py --case parameter-cuda-graph
```

Run the complete matrix with:

```bash
modal run tests/gpu/modal_validation.py --case all
```

The cases have deliberately different acceptance checks:

| Case | What must be observed |
| --- | --- |
| `baseline-eager` | Vortex server starts and generates with graph and overlap disabled. |
| `cuda-graph` | `/server_info` keeps graph enabled, SGLang logs a successful capture, and generation succeeds. |
| `overlap` | `/server_info` keeps overlap enabled and concurrent generation succeeds with graph disabled. |
| `cuda-graph-overlap` | Both settings remain enabled and concurrent generation succeeds after graph capture. |
| `dp2` | SGLang reports two DP states and explicitly routed requests succeed on DP ranks 0 and 1. |
| `parameter-cuda-graph` | A checkpoint-backed, per-layer `Parameter` flow is compiled, graph-captured, and used for generation. |

`parameter-cuda-graph` validates the runtime mechanism, including checkpoint
loading, `materialize`, `layer_lookup`, and `cur_layer`. It does not validate the
quality of a trained compressor. That requires a real compressor checkpoint and
the corresponding model/accuracy dataset as a separate, heavier experiment.

## Verified snapshot (2026-07-12)

The complete matrix above passed on Modal with the following environment:

| Component | Verified value |
| --- | --- |
| GPU | NVIDIA L40S (SM 89) |
| CUDA image | CUDA 13.0.2, Ubuntu 24.04 |
| Python | 3.12 |
| PyTorch | 2.11.0 |
| SGLang | 0.5.12.post1 |
| Transformers | 5.6.0 |
| kernels | 0.12.3 |
| Model | Qwen/Qwen3-0.6B |

The AstraFlow RaaS integration smoke also passed on the same environment. It
exercised the real launch path rather than importing the packages in isolation:

```text
AstraFlow manager
  -> astraflow.raas.entrypoint --vortex-config ...
  -> Vortex SGLang plugin and compiled sparse-attention code
  -> SGLang server ready
  -> /generate returns output tokens
```

This snapshot establishes ordinary extra installation, one-node DP=2,
CUDA Graph, overlap scheduling, their combined path, and the checkpoint-backed
`Parameter` mechanism on L40S. It does not establish multi-node execution,
other GPU architectures, trained-compressor accuracy, or production latency and
throughput targets; those require separate validation cases and acceptance
thresholds.
