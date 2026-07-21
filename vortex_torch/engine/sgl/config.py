"""Independent vortex configuration object.

中文读法：这个文件只做一件事，把外部传进来的 Vortex 启动参数统一整理成
``VortexConfig``，再让 SGLang 的 ``ServerArgs`` 能保存这个对象。它不启动
SGLang，也不执行 sparse attention；真正运行时接入在 ``integration.py`` 和
各 attention backend 里。

All vortex hyper-parameters live here, in one dataclass owned by vortex_torch,
instead of as ~18 scattered ``vortex_*`` fields on sglang's ``ServerArgs``.
``ServerArgs`` keeps a single ``vortex: Optional[VortexConfig]`` field (the
spawn-safe channel: sglang pickles ``ServerArgs`` to its worker), plus a small
backward-compatible ``__getattr__`` shim so the many existing
``server_args.vortex_*`` / ``server_args.enable_vortex_sparsity`` read sites keep
working unchanged.

Two entry points populate it:
  * Python: ``sgl.Engine(vortex_topk_val=..., enable_vortex_sparsity=True, ...)``
    still works — the official SGLang plugin folds those flat kwargs into a
    ``VortexConfig`` at the ``ServerArgs`` boundary.
  * Explicit: ``sgl.Engine(vortex=VortexConfig(topk_val=..., ...))``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple


# Vortex 与 SGLang plugin 约定的接口版本。两边版本不一致时直接拒绝启动。
VORTEX_SGLANG_ABI = 1
# 按“能力”列出 Vortex 正常运行必须具备的接入点，用于给错误信息分组。
_REQUIRED_SGLANG_HOOKS = frozenset(
    {
        "server_args.vortex_config",
        "model_runner.build_sparse_flow",
        "kv_cache.kv_cell_size",
        "kv_cache.make_kv_pool",
        "model_utils.fused_kv_opt_out",
        "disaggregation.rebuild_aux",
    }
)
# 按完整 Python 路径列出 plugin.py 必须注册的 8 个 SGLang 目标函数。
VORTEX_SGLANG_PLUGIN_TARGETS = frozenset(
    {
        "sglang.srt.server_args.ServerArgs.__init__",
        "sglang.srt.server_args.ServerArgs.__post_init__",
        "sglang.srt.server_args.ServerArgs.add_cli_args",
        "sglang.srt.model_executor.model_runner.ModelRunner.configure_kv_cache_dtype",
        "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_cell_size",
        "sglang.srt.model_executor.model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._init_pools",
        "sglang.srt.models.utils.enable_fused_set_kv_buffer",
        "sglang.srt.disaggregation.decode.DecodeTransferQueue.pop_transferred",
    }
)


def validate_sglang_runtime_contract() -> None:
    """确认当前 SGLang 已加载匹配的 Vortex plugin。

    这里不做 attention 计算，只在启动阶段检查 ABI、能力声明、ServerArgs.vortex
    字段和 8 个 hook。缺少任意一项都立即报错，避免服务启动后静默走普通
    SGLang，或者运行到 KV pool/decode 阶段才失败。
    """

    try:
        from sglang.srt.plugins import load_plugins

        # 先让 SGLang 发现并执行 pyproject.toml 声明的 plugin.py::register()。
        load_plugins()
        from sglang.srt.server_args import ServerArgs
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Vortex's SGLang plugin is not active. Install vortex_torch "
            "with its 'sglang' extra (sglang==0.5.12.post1)."
        ) from exc

    # plugin.register() 会把实际安装的 ABI 和能力声明挂到 ServerArgs 上。
    actual_abi = getattr(ServerArgs, "_vortex_sglang_abi", None)
    actual_hooks = frozenset(getattr(ServerArgs, "_vortex_sglang_hooks", ()))

    missing_hooks = sorted(_REQUIRED_SGLANG_HOOKS - actual_hooks)
    server_arg_fields = {field.name for field in fields(ServerArgs)}

    errors = []
    if actual_abi != VORTEX_SGLANG_ABI:
        errors.append(
            f"ABI {actual_abi!r} is installed; ABI {VORTEX_SGLANG_ABI} is required"
        )
    if missing_hooks:
        errors.append(f"missing hooks: {', '.join(missing_hooks)}")
    if "vortex" not in server_arg_fields:
        errors.append("ServerArgs has no 'vortex' field")
    from sglang.srt.plugins.hook_registry import HookRegistry

    # “已声明”只说明 plugin 想注册；HookRegistry._patched 才说明 hook 真正生效。
    declared_targets = frozenset(
        getattr(ServerArgs, "_vortex_sglang_targets", ())
    )
    missing_declarations = sorted(
        VORTEX_SGLANG_PLUGIN_TARGETS - declared_targets
    )
    missing_applied = sorted(declared_targets - HookRegistry._patched)
    if missing_declarations:
        errors.append(
            "plugin did not declare targets: "
            + ", ".join(missing_declarations)
        )
    if missing_applied:
        errors.append(
            "plugin hooks were not applied: " + ", ".join(missing_applied)
        )

    if errors:
        raise RuntimeError(
            "Incompatible official SGLang plugin runtime for Vortex ("
            + "; ".join(errors)
            + "). Reinstall matching vortex_torch and SGLang packages."
        )


# Vortex 的标准配置对象。外部可以用 JSON string、旧式 vortex_* kwargs、
# 或直接传 VortexConfig，最后都会统一落到这个 dataclass 上。
@dataclass
class VortexConfig:
    """All vortex sparse-attention hyper-parameters (defaults mirror the former
    ``ServerArgs.vortex_*`` defaults exactly, so behaviour is unchanged)."""

    # 稀疏 attention 每次保留的 page/block 数量；越小越稀疏。
    topk_val: int = 30
    # topk 的上限，给动态 topk / ratio 场景兜底。
    max_topk_val: Optional[int] = None
    # 哪些 transformer layer 不走 Vortex sparse attention。
    layers_skip: Optional[List[int]] = None
    # BOS/EOS 这类特殊位置通常要保留，避免稀疏选择把边界信息丢掉。
    block_reserved_bos: int = 1
    block_reserved_eos: int = 1
    # 最大序列长度；-1 表示不在这里额外限制。
    max_seq_lens: int = -1
    # 编译/profile 时按多少 request/chunk 分块处理。
    workload_chunk_size: int = 32
    # Vortex cache / kernel 使用的数据精度。
    dtype: str = "bfloat16"
    # 用户自定义 vFlow 策略来源：可以给文件路径，也可以给模块名。
    module_path: Optional[str] = None
    module_name: Optional[str] = None
    # 一个 KV page/block 里包含多少 token，影响索引和 KV pool 布局。
    block_size: int = 16
    # 用比例控制 topk 的可选方式；0 表示主要用 topk_val。
    topk_ratio: float = 0.0
    # 编译产物缓存目录，避免重复编译。
    compilation_cache_dir: Optional[str] = None
    # 后端调度策略，留给 runtime/backend 使用。
    schedule_policy: Optional[str] = None
    # Vortex 最终挂到哪个 attention 后端路径：flashinfer / trtllm / cuda_mla 等。
    attention_backend: str = "flashinfer"
    # Vortex 自己生成/调用 kernel 时的实现后端。
    impl_backend: str = "triton"
    # 是否启用 tensor core 相关实现。
    use_tensor_core: bool = False

    @classmethod
    def from_flat(cls, flat: Dict[str, Any]) -> "VortexConfig":
        """Build from a dict of ``vortex_<name>`` keys (prefix stripped).

        中文读法：把 ``{"vortex_topk_val": 30}`` 或 ``{"topk_val": 30}``
        这类平铺 dict 清洗成 ``VortexConfig(topk_val=30)``。只保留
        VortexConfig dataclass 里真实存在的字段。
        """
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in flat.items():
            key = k[len("vortex_"):] if k.startswith("vortex_") else k
            if key in names:
                kw[key] = v
        return cls(**kw)


def normalize_vortex_config(value: Any) -> Optional[VortexConfig]:
    """把 CLI/Python 的多种输入统一成 ``VortexConfig``。

    CLI 的 ``--vortex-config`` 通常传入 JSON 字符串，Python 调用可能传 dict
    或已经构造好的 VortexConfig。后续 plugin/integration 只需要处理统一类型。
    """
    if value is None or isinstance(value, VortexConfig):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--vortex-config must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(
            "vortex must be a VortexConfig, JSON object, dict, or None; "
            f"got {type(value).__name__}"
        )
    return VortexConfig.from_flat(value)


# 生成旧式 vortex_* 参数的默认值表。plugin.py 的兼容 __getattr__ 会使用它，
# 让历史代码读取 server_args.vortex_topk_val 时仍得到和 VortexConfig 一致的默认值。
def legacy_defaults() -> Dict[str, Any]:
    return {f.name: f.default for f in fields(VortexConfig)}


# 兼容旧式 Python 写法：sgl.Engine(enable_vortex_sparsity=True, vortex_topk_val=...)。
def split_flat_kwargs(kwargs: Dict[str, Any]) -> Tuple[Optional[VortexConfig], Dict[str, Any]]:
    """Pop ``enable_vortex_sparsity`` + ``vortex_*`` from ``kwargs``.

    Returns ``(config_or_None, remaining_kwargs)``. The config is built iff
    ``enable_vortex_sparsity`` is truthy; otherwise the vortex_* keys are simply
    dropped (vortex stays off).
    """
    # enable_vortex_sparsity 是总开关；不开时即使有 vortex_* 字段也丢掉。
    enabled = bool(kwargs.pop("enable_vortex_sparsity", False))
    # 从 kwargs 里拿走所有旧式 vortex_* 字段，避免原生 SGLang ServerArgs 看到不认识的参数。
    flat = {k: kwargs.pop(k) for k in list(kwargs) if k.startswith("vortex_") and k != "vortex"}
    cfg = VortexConfig.from_flat(flat) if enabled else None
    return cfg, kwargs


# 小工具：后续 integration/backend 可以从 ServerArgs 或 ModelRunner 里统一取 VortexConfig。
def cfg(model_runner_or_server_args) -> Optional[VortexConfig]:
    """从 ModelRunner 或 ServerArgs 中统一取出当前 ``VortexConfig``。"""
    sa = getattr(model_runner_or_server_args, "server_args", model_runner_or_server_args)
    return getattr(sa, "vortex", None)
