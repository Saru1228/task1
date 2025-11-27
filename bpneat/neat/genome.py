from bpneat.neat.gene import NodeGene, ConnectionGene
import networkx as nx
import os
import matplotlib.pyplot as plt
from collections import defaultdict

class Genome:
    def __init__(self, nodes: dict, connections: list):
        self.nodes = nodes  # dict[node_id] = NodeGene
        self.connections = connections  # list of ConnectionGene

    def add_node(self, node: NodeGene):
        self.nodes[node.id] = node

    def add_connection(self, conn: ConnectionGene):
        self.connections.append(conn)

    def summary(self):
        print("== Nodes ==")
        for node in self.nodes.values():
            print(node)
        print("== Connections ==")
        for conn in self.connections:
            print(conn)

    def draw(self, filename=None):
        G = nx.DiGraph()

        # 节点
        for node in self.nodes.values():
            label = f"{node.id}\n{node.type}"
            G.add_node(node.id, label=label, type=node.type)

        # 边（只显示启用的）
        for conn in self.connections:
            if conn.enabled:
                G.add_edge(conn.in_node, conn.out_node,
                           weight=conn.weight, label=f"{conn.weight:.2f}")

        # 节点层布局
        layer_nodes = defaultdict(list)
        for node in self.nodes.values():
            layer_nodes[node.type].append(node.id)

        layer_order = ["input", "hidden", "output"]
        pos = {}
        x_spacing, y_spacing = 3, 1.5

        for x, layer in enumerate(layer_order):
            for i, node_id in enumerate(layer_nodes[layer]):
                pos[node_id] = (x * x_spacing, -i * y_spacing)

        node_labels = {n: G.nodes[n]['label'] for n in G.nodes}
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        # 绘图
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=False, node_color='skyblue',
                node_size=1200, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

def make_initial_genome(input_dim=2, output_dim=1, use_bias=True):
    nodes = {}
    connections = []

    node_id = 0
    input_ids = []

    # 输入节点
    for _ in range(input_dim):
        nodes[node_id] = NodeGene(id=node_id, type_='input', activation='relu')
        input_ids.append(node_id)
        node_id += 1

    # bias 节点
    if use_bias:
        nodes[node_id] = NodeGene(id=node_id, type_='input', activation='linear')
        bias_id = node_id
        node_id += 1
    else:
        bias_id = None

    # 输出节点
    output_ids = []
    for _ in range(output_dim):
        nodes[node_id] = NodeGene(id=node_id, type_='output', activation='relu')
        output_ids.append(node_id)
        node_id += 1

    # input -> output
    innov = 1
    for i in input_ids:
        for o in output_ids:
            connections.append(ConnectionGene(i, o, 1.0, True, innov))
            innov += 1

    # bias -> output
    if bias_id is not None:
        for o in output_ids:
            connections.append(ConnectionGene(bias_id, o, -1.0, True, innov))
            innov += 1

    return Genome(nodes=nodes, connections=connections)
