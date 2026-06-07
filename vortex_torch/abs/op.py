from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from .context_base import ContextBase
from ..utils import Mode, Schedule


class vOp(ABC):
    """支持 profile 和 execute 两种模式的虚拟算子基类。

    这个抽象基类给所有虚拟算子提供统一接口。算子主要有两个阶段：

    - **Profile 阶段**：用于预计算形状、分配缓冲区，或收集统计信息。
    - **Execute 阶段**：真正执行算子计算。

    子类 **必须** 实现 :meth:`profile`。:meth:`execute` 是可选的：
    如果某个算子只通过编译/codegen 路径运行，可以不实现它；对这种算子调用
    :meth:`execute` 会抛出 :class:`NotImplementedError`。
    :meth:`__call__` 会根据传入的 context 自动分发到不同模式。
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.schedule = Schedule.S  #: 调度类型，例如 W 或 S；具体实现可能会用到。

    @abstractmethod
    def profile(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        """抽象方法：profile。

        在 profile 或准备阶段调用。

        常见用途：
          - 分配持久输出缓冲区。
          - 计算静态形状。
          - 收集性能统计信息。

        子类必须实现这个方法。

        参数:
            *args: 位置参数。
            ctx (ContextBase, optional): 执行上下文。
            **kwargs: 额外关键字参数。

        返回:
            Any: profile 操作的结果。

        抛出:
            NotImplementedError: 子类没有实现该方法时抛出。
        """

        raise NotImplementedError

    
    def execute(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        """可选方法：在运行时执行算子。

        如果子类参与运行时 eager 路径，应该覆盖这个方法。如果子类只通过
        编译后的 codegen 路径运行，可以不实现它；这种情况下调用
        :meth:`execute` 会抛出 :class:`NotImplementedError`，并提示该算子
        只支持 codegen 路径。

        参数:
            *args: 传给算子的输入位置参数。
            ctx (ContextBase, optional): 执行上下文。
            **kwargs: 额外关键字参数。

        返回:
            Any: 算子执行结果。如果没有实现，则不会返回。

        抛出:
            NotImplementedError: 子类没有覆盖该方法时抛出。
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not implement execute(); "
            f"this operator runs only through the compiled codegen path."
        )


    def __call__(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        """可调用入口。

        根据 ``ctx`` 中指定的执行模式，把调用分发给 :meth:`profile`
        或 :meth:`execute`。

        行为：
          - 如果 ``ctx.mode == "profile"``，调用 :meth:`self.profile(*args, **kwargs)`
          - 如果 ``ctx.mode == "execute"`` 或 ``ctx.mode is None``，调用 :meth:`self.execute(*args, **kwargs)`
          - 其他模式会抛出 :class:`ValueError`

        参数:
            *args: 传给底层方法的位置参数。
            ctx (ContextBase, optional): 包含 mode 的执行上下文。
            **kwargs: 额外关键字参数。

        返回:
            Any: :meth:`profile` 或 :meth:`execute` 的返回结果。

        抛出:
            ValueError: 如果 ``ctx.mode`` 不是 ``"profile"`` 或 ``"execute"``。
        """
        
        if ctx.mode is None or ctx.mode == Mode.execute:
            return self.execute(*args, ctx=ctx, **kwargs)
        if ctx.mode == Mode.profile:
            return self.profile(*args, ctx=ctx, **kwargs)
        raise ValueError(f"Unknown mode: {ctx.mode!r}, expected 'profile' or 'execute'")
    
    # ------------------------------ 辅助方法 ------------------------------ #
    def _prefix(self) -> str:
        """给 assert/log 信息加上类名前缀。"""
        return f"{self.__class__.__name__}: "
