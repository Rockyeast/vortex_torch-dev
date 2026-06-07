import torch
from typing import List, Dict, Set, DefaultDict, Tuple, Optional, Callable
from collections import defaultdict, deque
from ..context import Context
from ...abs import vTensor, vOp
from ...utils import Schedule

# =====================================================================
# 数据结构
# =====================================================================

class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class OpDAG:
    """
    轻量级 op DAG。节点是 op，边来自 tensor 的 producer/consumer 关系。
    """
    def __init__(self):
        self.nodes: List[int] = []                                       # topo 顺序中的 op id
        self.successors: DefaultDict[int, Set[int]] = defaultdict(set)   # op -> 消费它的 op
        self.predecessors: DefaultDict[int, Set[int]] = defaultdict(set) # op -> 生产它输入的 op

    def add_node(self, op_id: int):
        self.nodes.append(op_id)

    def add_edge(self, producer_op_id: int, consumer_op_id: int):
        if consumer_op_id not in self.successors[producer_op_id]:
            self.successors[producer_op_id].add(consumer_op_id)
            self.predecessors[consumer_op_id].add(producer_op_id)


class SubgraphDAG:
    """
    以 subgraph 粒度观察 op-level DAG。
    fusion 阶段用它检查一次 merge 是否会制造环。
    """
    def __init__(self, op_dag: OpDAG, uf: UnionFind):
        self._op_dag = op_dag
        self._uf = uf

    def _build_sg_successors(self) -> DefaultDict[int, Set[int]]:
        """根据当前 UnionFind 状态重建 subgraph-level successor 映射。"""
        succ: DefaultDict[int, Set[int]] = defaultdict(set)
        for op_id in self._op_dag.nodes:
            sg = self._uf.find(op_id)
            for s in self._op_dag.successors[op_id]:
                s_sg = self._uf.find(s)
                if sg != s_sg:
                    succ[sg].add(s_sg)
        return succ

    def can_merge(self, sg_a: int, sg_b: int) -> bool:
        """对 merge ``sg_a`` 和 ``sg_b`` 做单方向环检查。

        具体检查：是否存在一条从 ``sg_b`` 到 ``sg_a`` 的路径，并且这条路径不使用
        直接的 ``sg_a -> sg_b`` 边？如果存在这种反向路径，merge 后会在
        ``sg_a`` 一侧形成环。

        如果需要 **双向** 安全检查（例如融合两个没有直接边、但共享输入的 W op，
        也就是 sibling ops），请改用 :meth:`can_merge_bidir`。
        """
        succ = self._build_sg_successors()
        # 检查时移除直接边
        succ[sg_a].discard(sg_b)

        visited: Set[int] = {sg_b}
        queue = deque([sg_b])
        while queue:
            node = queue.popleft()
            for nxt in succ[node]:
                if nxt == sg_a:
                    return False  # 存在反向路径 -> 会形成环
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return True  # 安全

    def can_merge_bidir(self, sg_a: int, sg_b: int) -> bool:
        """在不假设方向的情况下，``sg_a`` 和 ``sg_b`` 是否可以融合？

        当且仅当二者之间任意方向存在 **非平凡** 路径时，merge 会形成环。
        非平凡路径指长度 >= 2，并且经过至少一个其他 subgraph 的路径。单独的直接边
        （``a -> b`` 或 ``b -> a``）是安全的：融合后它会变成 subgraph 内部依赖，
        由 op 的拓扑顺序处理。

        注意：这不等价于 ``can_merge(a, b) and can_merge(b, a)``。
        :meth:`can_merge` 只会移除 ``sg_a -> sg_b`` 边，这意味着两个调用中的
        一个仍然会“看到”直接边，从而拒绝相邻的 W-pair。
        """
        succ = self._build_sg_successors()

        def _reaches_via_other(src: int, dst: int) -> bool:
            # 先走到 ``src`` 的非 ``dst`` successor（过滤掉直接 ``src -> dst`` 边），
            # 然后 BFS。
            visited: Set[int] = {src}
            queue: deque = deque()
            for nxt in succ[src]:
                if nxt != dst and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
            while queue:
                node = queue.popleft()
                for nxt in succ[node]:
                    if nxt == dst:
                        return True
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            return False

        return not (
            _reaches_via_other(sg_a, sg_b) or _reaches_via_other(sg_b, sg_a)
        )


class Graph:
    def __init__(
        self,
        tensor_list: List[vTensor],
        op_list: List[vOp],
        output_tensor_to_op_list: List[Optional[int]],
        op_to_input_tensor_list: List[List[int]],
        op_to_output_tensor_list: List[List[int]],
        input_tensor_ids: List[int],
        output_tensor_ids: List[int],
        global_input_tensor_ids: List[int],
        global_output_tensor_ids: List[int],
    ):
        self.tensor_list: List[vTensor] = tensor_list
        self.op_list: List[vOp] = op_list
        self.output_tensor_to_op_list: List[Optional[int]] = output_tensor_to_op_list
        self.op_to_input_tensor_list: List[List[int]] = op_to_input_tensor_list
        # 每个 op 对应一组输出 tensor id。单输出 op 只有一个元素；多输出 op
        # （例如 ``TopK`` 返回 ``(block_table, seqlens)``）会为每个输出记录一项。
        # 如果某个 codegen 只需要第一个输出，应使用 ``[op_id][0]``。
        self.op_to_output_tensor_list: List[List[int]] = op_to_output_tensor_list
        self.input_tensor_ids: List[int] = input_tensor_ids
        self.output_tensor_ids: List[int] = output_tensor_ids
        self.global_input_tensor_ids: List[int] = global_input_tensor_ids
        self.global_output_tensor_ids: List[int] = global_output_tensor_ids
        self.schedule = op_list[0].schedule if (op_list and op_list[0] is not None) else None
        self.forward: Callable = None

    def __repr__(self) -> str:
        return (
            f"Graph("
            f"num_tensors={len(self.tensor_list)}, "
            f"num_ops={len(self.op_list)}, "
            f"input_tensor_ids={self.input_tensor_ids}, "
            f"output_tensor_ids={self.output_tensor_ids}, "
            f"global_input_tensor_ids={self.global_input_tensor_ids}, "
            f"global_output_tensor_ids={self.global_output_tensor_ids})"
        )


# =====================================================================
# Helpers
# =====================================================================

def _as_tensor_id_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _build_local_graph(
    global_tensor_list: List[vTensor],
    global_op_list: List[vOp],
    global_output_tensor_to_op_list,
    global_op_to_input_tensor_list: List[List[int]],
    global_op_to_output_tensor_list,
    selected_global_op_ids: List[int],
    global_input_tensor_ids: List[int],
    global_output_tensor_ids: List[int],
) -> Graph:
    """
    根据一组全局 op id 构造一个自包含的局部 Graph。

    返回的 Graph 内部所有结构 id 都会重新映射成本地图里的 id。
    """
    # 第 1 步：统一输出表示。
    # 有的 op 是单输出，有的 op 是多输出；这里全部转成 List[int]，
    # 后面就可以统一按“输出 tensor 列表”处理。
    global_op_to_output_tensor_ids: List[List[int]] = [
        _as_tensor_id_list(x) for x in global_op_to_output_tensor_list
    ]

    # 第 2 步：记录当前要切出来的 op，并准备收集这些 op 涉及的所有 tensor。
    selected_global_op_set: Set[int] = set(selected_global_op_ids)
    selected_global_tensor_ids: Set[int] = set()

    # 第 3 步：收集局部图需要的 tensor。
    # 一个局部 graph 里只要某个 op 用到了某个输入/输出 tensor，
    # 这个 tensor 就必须出现在 local_tensor_list 里。
    for global_op_id in selected_global_op_ids:
        for tid in global_op_to_input_tensor_list[global_op_id]:
            selected_global_tensor_ids.add(tid)
        for tid in global_op_to_output_tensor_ids[global_op_id]:
            selected_global_tensor_ids.add(tid)

    # 第 4 步：排序，让 global tensor id -> local tensor id 的映射稳定。
    sorted_global_tensor_ids: List[int] = sorted(selected_global_tensor_ids)

    # 第 5 步：建立全局 id 到局部 id 的映射。
    # 例如 selected_global_op_ids = [2, 4, 5] 时：
    #   global op2 -> local op0
    #   global op4 -> local op1
    #   global op5 -> local op2
    global_op_id_to_local: Dict[int, int] = {
        gid: lid for lid, gid in enumerate(selected_global_op_ids)
    }
    # tensor 也一样重编号；局部图内部只使用连续的 local tensor id。
    global_tensor_id_to_local: Dict[int, int] = {
        gid: lid for lid, gid in enumerate(sorted_global_tensor_ids)
    }

    # 第 6 步：根据刚才收集到的 id，从全局列表里取出真正属于局部图的对象。
    local_tensor_list = [global_tensor_list[gid] for gid in sorted_global_tensor_ids]
    local_op_list = [global_op_list[gid] for gid in selected_global_op_ids]

    # 第 7 步：构造局部版 tensor -> producer op 反查表。
    # 如果某个 tensor 的 producer 不在当前 selected op 里，
    # 那它对这个局部图来说就是外部输入，producer 记为 None。
    local_output_tensor_to_op_list: List[Optional[int]] = []
    for gtid in sorted_global_tensor_ids:
        producer = global_output_tensor_to_op_list[gtid]
        if producer is None or producer not in selected_global_op_set:
            local_output_tensor_to_op_list.append(None)
        else:
            local_output_tensor_to_op_list.append(global_op_id_to_local[producer])

    # 第 8 步：构造局部版 op -> input tensor 表。
    # 每个输入 tensor id 都从 global id 改成 local id。
    local_op_to_input_tensor_list: List[List[int]] = [
        [global_tensor_id_to_local[tid] for tid in global_op_to_input_tensor_list[gid]]
        for gid in selected_global_op_ids
    ]

    # 第 9 步：构造局部版 op -> output tensor 表。
    # 每个输出 tensor id 同样从 global id 改成 local id。
    # 支持多输出 op：每个条目都是该 op 的输出 tensor_id 列表（至少 1 个）。
    # 只消费第一个输出的 codegen 可以用 ``[op_id][0]``。
    local_op_to_output_tensor_list: List[List[int]] = []
    for gid in selected_global_op_ids:
        outs = global_op_to_output_tensor_ids[gid]
        if len(outs) < 1:
            raise RuntimeError(
                f"Graph.op_to_output_tensor_list: op {gid} has no outputs"
            )
        local_op_to_output_tensor_list.append(
            [global_tensor_id_to_local[t] for t in outs]
        )

    # 第 10 步：把局部 tensor/op 表和输入输出边界打包成 Graph。
    # input_tensor_ids / output_tensor_ids 使用 local id；
    # global_input_tensor_ids / global_output_tensor_ids 保留全局 id，
    # 方便外层知道这个 subgraph 对应原始 ctx 里的哪些 tensor。
    return Graph(
        tensor_list=local_tensor_list,
        op_list=local_op_list,
        output_tensor_to_op_list=local_output_tensor_to_op_list,
        op_to_input_tensor_list=local_op_to_input_tensor_list,
        op_to_output_tensor_list=local_op_to_output_tensor_list,
        input_tensor_ids=[global_tensor_id_to_local[t] for t in global_input_tensor_ids],
        output_tensor_ids=[global_tensor_id_to_local[t] for t in global_output_tensor_ids],
        global_input_tensor_ids=list(global_input_tensor_ids),
        global_output_tensor_ids=list(global_output_tensor_ids),
    )


# =====================================================================
# 阶段 1：把基于 tensor 的 context 记录转换成 op 级别 DAG
# =====================================================================

def _build_op_dag(
    op_list: List[vOp],
    output_tensor_to_op_list,
    op_to_input_tensor_list: List[List[int]],
    op_to_output_tensor_list,
    final_output_tensor_ids: List[int],
    side_effect_op_ids: List[int] = (),
) -> Tuple[OpDAG, Set[int]]:
    """
    构造只包含“能从最终输出 tensor 反向追溯到”的 op 的 OpDAG。
    节点按拓扑顺序保存，也就是依赖先于使用者。

    一个 pipeline 有三类反向搜索起点：
      * attention 输出 ``tensor_id == 1``；
      * 有生产者、但图内没有消费者的 tensor，也就是孤立 sink；
      * 带副作用的 op，例如 ``Save``。这类 op 会故意不把目标 tensor
        标成自己的输出生产者，因此需要通过 ``side_effect_op_ids`` 额外传入。

    这三类都会作为反向 DFS 的起点，保证它们依赖的 op 不会被误删。
    """
    visited: Set[int] = set()
    topo_order: List[int] = []

    def dfs(op_id: int):
        if op_id in visited:
            return
        visited.add(op_id)
        for tid in op_to_input_tensor_list[op_id]:
            producer = output_tensor_to_op_list[tid]
            if producer is not None:
                dfs(producer)
        topo_order.append(op_id)

    for tid in final_output_tensor_ids:
        producer = output_tensor_to_op_list[tid]
        if producer is None:
            # 这是正常情况：``Save`` op 的 cache-field 目标在设计上不会登记到
            # ``output_tensor_to_op_list`` 里；Save op 会通过
            # ``side_effect_op_ids`` 单独作为反向 DFS 起点。
            continue
        dfs(producer)

    for op_id in side_effect_op_ids:
        dfs(op_id)

    reachable: Set[int] = set(topo_order)

    dag = OpDAG()
    for op_id in topo_order:
        dag.add_node(op_id)

    for consumer_op_id in topo_order:
        for tid in op_to_input_tensor_list[consumer_op_id]:
            producer_op_id = output_tensor_to_op_list[tid]
            if producer_op_id is not None and producer_op_id in reachable:
                dag.add_edge(producer_op_id, consumer_op_id)

    return dag, reachable


# =====================================================================
# 阶段 2：在 op DAG 上融合 W 调度的 op，且保证不引入环
# =====================================================================

def _fuse_w_ops(op_dag: OpDAG, op_list: List[vOp]) -> UnionFind:
    """
    融合任意一对 W-scheduled op，前提是融合后不会让 subgraph DAG 出现环。

    这里不是只融合有直接 W→W 生产/消费边的 op。两个共享输入的 sibling op
    （例如对同一个 KV 字段做两个 reduction）也可以被融合到同一个 kernel，
    只要它们没有同时位于另一条跨 subgraph 路径的两端。

    返回值是 UnionFind，表示最终每个 op 属于哪个融合后的 subgraph。
    """
    uf = UnionFind()
    for op_id in op_dag.nodes:
        uf.add(op_id)

    sg_dag = SubgraphDAG(op_dag, uf)

    def _w_subgraph_reps() -> List[int]:
        """当前唯一的 W-subgraph 代表元，按 op 拓扑顺序排列。"""
        seen_set: Set[int] = set()
        reps: List[int] = []
        for op_id in op_dag.nodes:
            if op_list[op_id].schedule != Schedule.W:
                continue
            rep = uf.find(op_id)
            if rep not in seen_set:
                seen_set.add(rep)
                reps.append(rep)
        return reps

    changed = True
    while changed:
        changed = False
        reps = _w_subgraph_reps()
        # 每轮做 O(N^2) 的配对扫描；每次成功融合后 N 至少减少 1，
        # 所以外层循环最多跑 N 轮。
        for i in range(len(reps)):
            sg_a = uf.find(reps[i])
            for j in range(i + 1, len(reps)):
                sg_b = uf.find(reps[j])
                if sg_a == sg_b:
                    continue
                if sg_dag.can_merge_bidir(sg_a, sg_b):
                    uf.union(sg_a, sg_b)
                    changed = True
                    # ``reps`` 已经过期，重新开始外层扫描。
                    break
            if changed:
                break

    return uf


# =====================================================================
# 阶段 3：把融合后的 op 组重新转换成 Graph 对象
# =====================================================================

def _build_all_graphs(
    op_dag: OpDAG,
    uf: UnionFind,
    reachable_ops: Set[int],
    tensor_list: List[vTensor],
    op_list: List[vOp],
    output_tensor_to_op_list,
    op_to_input_tensor_list: List[List[int]],
    op_to_output_tensor_list,
    tensor_to_consumers: DefaultDict[int, List[int]],
    final_output_tensor_ids: List[int],
) -> Tuple[Graph, List[Graph]]:
    """把融合后的 subgraph 分组转换回 Graph 对象。"""

    op_to_output_tensor_ids: List[List[int]] = [
        _as_tensor_id_list(x) for x in op_to_output_tensor_list
    ]
    topo_op_ids = op_dag.nodes
    final_output_set: Set[int] = set(final_output_tensor_ids)

    # --- 完整图 ---
    full_input_set: Set[int] = set()
    for op_id in topo_op_ids:
        for tid in op_to_input_tensor_list[op_id]:
            producer = output_tensor_to_op_list[tid]
            if producer is None or producer not in reachable_ops:
                full_input_set.add(tid)

    full_graph = _build_local_graph(
        global_tensor_list=tensor_list,
        global_op_list=op_list,
        global_output_tensor_to_op_list=output_tensor_to_op_list,
        global_op_to_input_tensor_list=op_to_input_tensor_list,
        global_op_to_output_tensor_list=op_to_output_tensor_list,
        selected_global_op_ids=topo_op_ids,
        global_input_tensor_ids=sorted(full_input_set),
        global_output_tensor_ids=sorted(final_output_set),
    )

    # --- 收集 subgraph，并做 subgraph 级别拓扑排序 ---
    sg_key_to_op_ids: DefaultDict[int, List[int]] = defaultdict(list)
    sg_key_order: List[int] = []
    seen: Set[int] = set()
    for op_id in topo_op_ids:
        sg_key = uf.find(op_id)
        if sg_key not in seen:
            seen.add(sg_key)
            sg_key_order.append(sg_key)
        sg_key_to_op_ids[sg_key].append(op_id)

    sg_key_to_id: Dict[int, int] = {k: i for i, k in enumerate(sg_key_order)}
    sg_successors: DefaultDict[int, Set[int]] = defaultdict(set)
    sg_indegree: Dict[int, int] = {i: 0 for i in range(len(sg_key_order))}

    for op_id in topo_op_ids:
        c_sg = sg_key_to_id[uf.find(op_id)]
        for pred in op_dag.predecessors[op_id]:
            p_sg = sg_key_to_id[uf.find(pred)]
            if p_sg != c_sg and c_sg not in sg_successors[p_sg]:
                sg_successors[p_sg].add(c_sg)
                sg_indegree[c_sg] += 1

    queue = deque([s for s, d in sg_indegree.items() if d == 0])
    topo_sg_ids: List[int] = []
    while queue:
        s = queue.popleft()
        topo_sg_ids.append(s)
        for nxt in sg_successors[s]:
            sg_indegree[nxt] -= 1
            if sg_indegree[nxt] == 0:
                queue.append(nxt)

    if len(topo_sg_ids) != len(sg_key_order):
        raise RuntimeError("Cycle detected in subgraph DAG")

    # 把每个 op 映射到最终拓扑排序后的 subgraph id。
    op_to_subgraph_id: Dict[int, int] = {}
    for new_id, old_id in enumerate(topo_sg_ids):
        for op_id in sg_key_to_op_ids[sg_key_order[old_id]]:
            op_to_subgraph_id[op_id] = new_id

    # --- 构造每个 subgraph ---
    subgraphs: List[Graph] = []
    for new_id, old_id in enumerate(topo_sg_ids):
        sg_op_ids = sg_key_to_op_ids[sg_key_order[old_id]]

        input_set: Set[int] = set()
        output_set: Set[int] = set()

        for op_id in sg_op_ids:
            for tid in op_to_input_tensor_list[op_id]:
                producer = output_tensor_to_op_list[tid]
                if producer is None:
                    input_set.add(tid)
                elif producer in op_to_subgraph_id and op_to_subgraph_id[producer] != new_id:
                    input_set.add(tid)

        for op_id in sg_op_ids:
            for out_tid in op_to_output_tensor_ids[op_id]:
                if out_tid in final_output_set:
                    output_set.add(out_tid)
                    continue
                for consumer in tensor_to_consumers.get(out_tid, []):
                    if consumer in op_to_subgraph_id and op_to_subgraph_id[consumer] != new_id:
                        output_set.add(out_tid)
                        break

        subgraphs.append(_build_local_graph(
            global_tensor_list=tensor_list,
            global_op_list=op_list,
            global_output_tensor_to_op_list=output_tensor_to_op_list,
            global_op_to_input_tensor_list=op_to_input_tensor_list,
            global_op_to_output_tensor_list=op_to_output_tensor_list,
            selected_global_op_ids=sg_op_ids,
            global_input_tensor_ids=sorted(input_set),
            global_output_tensor_ids=sorted(output_set),
        ))

    return full_graph, subgraphs


# =====================================================================
# 主入口
# =====================================================================

def contruct_graph(ctx: Context) -> Tuple[Graph, List[Graph]]:

    tensor_list: List[vTensor] = ctx.tensor_list
    op_list: List[vOp] = ctx.op_list
    output_tensor_to_op_list = ctx.output_tensor_to_op_list
    op_to_input_tensor_list: List[List[int]] = ctx.op_to_input_tensor_list
    op_to_output_tensor_list = ctx.op_to_output_tensor_list

    # --- consumer 反查表 ---
    # 这个表有两个用途：
    # 1. 找出哪些 tensor 是没有消费者的 sink；
    # 2. 在阶段 3 判断哪些 tensor 会跨 subgraph 边界。
    tensor_to_consumers: DefaultDict[int, List[int]] = defaultdict(list)
    for consumer_op_id, input_tids in enumerate(op_to_input_tensor_list):
        for tid in input_tids:
            tensor_to_consumers[tid].append(consumer_op_id)

    # 必须在死代码消除后保留下来的终点 tensor：
    # 1. 永远包含 attention 输出 ``tensor_id == 1``；
    # 2. 包含所有“有生产者、但图内没有消费者”的孤立 sink；
    # 3. 包含每个 ``Save`` 的 cache-field 目标。这些目标不会表现成普通 orphan，
    #    因为它们的 producer 槽故意保持为 ``None``，避免产生 Load -> Save 环；
    #    细节见 ``indexer/save_load.py``。
    side_effect_op_ids: List[int] = list(getattr(ctx, "side_effect_op_ids", []))
    save_target_tids = {
        tid
        for op_id in side_effect_op_ids
        for tid in op_to_output_tensor_list[op_id]
    }
    final_output_tensor_ids: List[int] = sorted({
        1,
        *(
            tid for tid, producer in enumerate(output_tensor_to_op_list)
            if producer is not None and not tensor_to_consumers.get(tid)
        ),
        *save_target_tids,
    })

    # ---------------------------------------------------------------
    # 阶段 1：转换成 op 级别 DAG，节点是 op，边来自 tensor 依赖。
    # ---------------------------------------------------------------
    op_dag, reachable_ops = _build_op_dag(
        op_list=op_list,
        output_tensor_to_op_list=output_tensor_to_op_list,
        op_to_input_tensor_list=op_to_input_tensor_list,
        op_to_output_tensor_list=op_to_output_tensor_list,
        final_output_tensor_ids=final_output_tensor_ids,
        side_effect_op_ids=side_effect_op_ids,
    )

    # ---------------------------------------------------------------
    # 阶段 2：融合 W 调度 op，只允许不会造环的 merge。
    # ---------------------------------------------------------------
    uf = _fuse_w_ops(op_dag, op_list)

    # ---------------------------------------------------------------
    # 阶段 3：转换回 Graph 对象，包括完整图和各个 subgraph。
    # ---------------------------------------------------------------
    full_graph, subgraphs = _build_all_graphs(
        op_dag=op_dag,
        uf=uf,
        reachable_ops=reachable_ops,
        tensor_list=tensor_list,
        op_list=op_list,
        output_tensor_to_op_list=output_tensor_to_op_list,
        op_to_input_tensor_list=op_to_input_tensor_list,
        op_to_output_tensor_list=op_to_output_tensor_list,
        tensor_to_consumers=tensor_to_consumers,
        final_output_tensor_ids=final_output_tensor_ids,
    )

    return full_graph, subgraphs
