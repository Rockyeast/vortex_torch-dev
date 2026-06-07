# registry.py
from typing import Dict, Type, Union
from .flow import vFlow
from .flow_mla import vFlowMLA

# 全局注册表：注册名 -> vFlow 子类（普通 MHA）或 vFlowMLA 子类（latent attention）
_FlowBase = (vFlow, vFlowMLA)
_REGISTRY: Dict[str, Type[Union[vFlow, vFlowMLA]]] = {}

class RegistryError(Exception):
    """注册类或查找注册名时的错误。"""
    ...

def register(name: str):
    """
    用户用这个装饰器注册自己的 vFlow / vFlowMLA 子类。

    示例:
        @register("cls_a")
        class MyFlow(vFlow): ...
    """
    def deco(cls):
        if not issubclass(cls, _FlowBase):
            raise RegistryError(f"{cls.__name__} must inherit from vFlow or vFlowMLA")
        if name in _REGISTRY:
            raise RegistryError(f"Registration name '{name}' already exists")
        _REGISTRY[name] = cls
        return cls
    return deco

def get(name: str) -> Type[vFlow]:
    """根据注册名返回对应类；如果找不到则抛出异常。"""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise RegistryError(f"Registration name '{name}' not found")

def has(name: str) -> bool:
    """检查某个名字是否已经注册。"""
    return name in _REGISTRY

def list_keys():
    """列出所有已注册的名字。"""
    return list(_REGISTRY.keys())
