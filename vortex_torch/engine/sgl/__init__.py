"""sglang adapter for vortex_torch.

Re-exports the engine entry points from ``api`` so existing callers using
``from vortex_torch.engine.sgl import get_engine`` continue to work. The
re-export is **lazy** (PEP 562): importing this package — e.g. to reach the
light ``config`` submodule — does not pull in ``api`` (and thus sglang's
engine) until one of the names below is actually accessed.
"""
# 中文读法：SGL 集成层的包入口。这里只做轻量导出和延迟加载，避免 import vortex_torch 时立刻把 SGLang/FlashInfer/Triton 全部拉进来。

__all__ = [
    "DEFAULT_SCHEDULE_POLICY",
    "EngineConfigError",
    "MODEL_PATH",
    "check_engine_config",
    "get_engine",
    "get_engine_from_json",
]


def __getattr__(name):
    if name in __all__:
        from vortex_torch.engine.sgl import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
