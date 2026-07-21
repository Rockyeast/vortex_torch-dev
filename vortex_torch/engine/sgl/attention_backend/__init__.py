"""Vortex attention backends for sglang.

Each backend lives in its own sibling module here (``flashinfer.py``,
future ``triton.py``, ``flash_attention.py``, …) and exposes one
``AttentionBackend`` subclass. The package re-exports their public
names so existing call sites such as::

    from vortex_torch.engine.sgl.attention_backend import VortexFlashInferBackend

continue to work unchanged after the split.

Adding a new backend:
  1. Drop ``my_backend.py`` next to this file. Subclass
     ``sglang.srt.layers.attention.base_attn_backend.AttentionBackend``;
     implement ``init_forward_metadata``, ``forward_extend``,
     ``forward_decode`` etc.
  2. Re-export the class below — keep the registry deterministic so
     ``model_runner.py`` can pick it by name.
"""
# 中文读法：attention backend 导出表。integration.py 注册 backend 时会从这里拿到具体实现。
from vortex_torch.engine.sgl.attention_backend.flashinfer import (
    VortexFlashInferBackend,
)

from vortex_torch.engine.sgl.attention_backend.trtllm import (
    VortexTRTLLMBackend,
)

from vortex_torch.engine.sgl.attention_backend.trtllm_mla import (
    VortexTRTLLMMLABackend,
)

from vortex_torch.engine.sgl.attention_backend.triton_mla import (
    VortexTritonMLABackend,
)

from vortex_torch.engine.sgl.attention_backend.cuda_mla import (
    VortexCudaMLABackend,
)

__all__ = [
    "VortexFlashInferBackend",
    "VortexTRTLLMBackend",
    "VortexTRTLLMMLABackend",
    "VortexTritonMLABackend",
    "VortexCudaMLABackend",
]