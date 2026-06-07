from __future__ import annotations
import os
import uuid
import types
import inspect
import importlib.util
from typing import Any, Dict, Optional
from .flow import vFlow
from .registry import RegistryError, get as reg_get, list_keys

class FlowLoadError(Exception):
    """加载/执行用户模块或解析 flow 类时的错误。"""
    ...

class FlowInitError(Exception):
    """校验构造参数或创建 flow 实例时的错误。"""
    ...

def _load_module_from_file(file_path: str) -> types.ModuleType:
    """
    执行用户文件，让文件内部的 @register(...) 调用生效。

    这个函数不返回 flow 类；它只确保用户文件被执行，从而产生“注册类”
    这个副作用。
    """
    if not os.path.isfile(file_path):
        raise FlowLoadError(f"File does not exist: {file_path}")

    # 使用随机模块名，避免和已有模块重名，也避免 import 缓存带来的问题。
    mod_name = f"user_flow_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise FlowLoadError(f"Failed to create import spec for: {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        # 执行模块；用户文件顶层代码会在这里运行。
        # 文件里的任何 @register(...) 装饰器都会把类写入全局注册表。
        spec.loader.exec_module(module)
    except Exception as e:
        raise FlowLoadError(f"Failed to import user module: {e}") from e

    return module

def _validate_kwargs(cls: type, init_kwargs: Dict[str, Any]) -> None:
    """
    在真正实例化前，根据类的 __init__ 签名校验构造参数。

    这样参数传错时，能给出更清楚的错误信息。
    """
    sig = inspect.signature(cls.__init__)
    try:
        # 第一个位置参数是 self；这里用一个占位值先绑定。
        sig.bind_partial(None, **(init_kwargs or {}))
    except TypeError as e:
        raise FlowInitError(f"Constructor arguments mismatch: {e}") from e

def build_vflow(
    selected: str,
    init_kwargs: Optional[Dict[str, Any]] = None,
    user_file: Optional[str] = None
) -> vFlow:
    """
    根据已经注册过的类创建一个 vFlow 实例。

    参数:
        selected: 注册名，用来决定要实例化哪个子类。
        init_kwargs: 传给子类构造函数的关键字参数字典。
        user_file: 可选的用户文件绝对路径。若提供，会先执行这个文件，
                   让文件内部的新注册生效。

    行为:
        - 如果提供了 user_file，先执行/导入该文件，触发里面的
          @register(...) 调用。
        - 然后根据 selected 从全局注册表里取出对应类。
        - 校验构造函数参数，并实例化该类。

    抛出:
        FlowLoadError: 用户文件导入失败，或 selected 对应的注册名不存在。
        FlowInitError: 构造参数校验失败，或实例化失败。
    """
    if user_file:
        _load_module_from_file(user_file)

    try:
        FlowCls = reg_get(selected)
    except RegistryError as e:
        # 把当前可用注册名也放进错误信息里，方便用户排查。
        raise FlowLoadError(f"{e} (available: {list_keys()})")

    _validate_kwargs(FlowCls, init_kwargs or {})
    try:
        return FlowCls(**(init_kwargs or {}))
    except Exception as e:
        raise FlowInitError(f"Failed to instantiate: {e}") from e
