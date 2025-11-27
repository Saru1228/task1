import jax
import jax.numpy as jnp
from collections import defaultdict, deque


def get_activation_fn(name):
    if name == "relu":
        return jax.nn.relu
    elif name == "sigmoid":
        return jax.nn.sigmoid
    elif name == "tanh":
        return jax.nn.tanh
    elif name == "softmax":
        return jax.nn.softmax
    else:
        raise ValueError(f"Unsupported activation: {name}")


def topo_sort(nodes, connections):
    """拓扑排序，确保节点按依赖顺序计算"""
    indegree = defaultdict(int)
    graph = defaultdict(list)

    for conn in connections:
        if conn.enabled:
            graph[conn.in_node].append(conn.out_node)
            indegree[conn.out_node] += 1

    q = deque([n.id for n in nodes if indegree[n.id] == 0])
    order = []

    while q:
        nid = q.popleft()
        order.append(nid)
        for neighbor in graph[nid]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)

    return order


def build_forward(genome):
    """
    返回一个 JAX 可微前向传播函数 f(x, params)
    """

    incoming_edges = defaultdict(list)
    for conn in genome.connections:
        if conn.enabled:
            incoming_edges[conn.out_node].append(conn.in_node)

    node_by_id = {n.id: n for n in genome.nodes.values()}
    input_nodes = [n.id for n in genome.nodes.values() if n.type == "input"]
    output_nodes = [n.id for n in genome.nodes.values() if n.type == "output"]
    topo_order = topo_sort(genome.nodes.values(), genome.connections)

    def f(x, params):
        node_outputs = {}

        # 输入节点直接赋值
        for i, nid in enumerate(input_nodes):
            if i < len(x):
                node_outputs[nid] = x[i]
            else:
                node_outputs[nid] = 1.0

        # 拓扑顺序依次计算中间节点
        for nid in topo_order:
            if nid in node_outputs:
                continue

            inputs = incoming_edges.get(nid, [])
            total_input = 0.0
            for src in inputs:
                if src not in node_outputs:
                    print(f"[Warning] Node {src} not computed before {nid}, assigning 0.0")
                    node_outputs[src] = 0.0  # fallback 默认值
                w = params.get((src, nid), 0.0)
                total_input += node_outputs[src] * w

            activation = get_activation_fn(node_by_id[nid].activation)
            node_outputs[nid] = activation(total_input)

        # 输出节点值，若缺失则用0.0填充
        output_values = []
        for nid in output_nodes:
            if nid not in node_outputs:
                print(f"[Warning] Output node {nid} not computed, defaulting to 0.0")
            val = node_outputs.get(nid, 0.0)
            output_values.append(val)

        # 保证所有输出节点都在拓扑排序中（即可以被计算）
        for out_id in output_nodes:
            if out_id not in topo_order:
                raise ValueError(f"[Invalid Network] Output node {out_id} is not reachable.")
        return jnp.array(output_values)

    return f
