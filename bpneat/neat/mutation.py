import random
import copy
from bpneat.neat.gene import ConnectionGene, NodeGene
from bpneat.neat.genome import Genome

class InnovationTracker:
    def __init__(self):
        self.counter = 1
        self.conn_history = {}  # {(src, tgt): innovation_number}

    def get_innovation(self, src, tgt):
        key = (src, tgt)
        if key in self.conn_history:
            return self.conn_history[key]
        else:
            self.counter += 1
            self.conn_history[key] = self.counter
            return self.counter
        
# === 添加连接突变 ===
def mutate_add_connection(genome, innovation_tracker):
    node_ids = list(genome.nodes.keys())
    tried = 0
    max_attempts = 20

    while tried < max_attempts:
        in_node = random.choice(node_ids)
        out_node = random.choice(node_ids)
        tried += 1

        # 禁止自连接，禁止输入 <- 输出
        if in_node == out_node:
            continue
        if genome.nodes[in_node].type == 'output':
            continue
        if genome.nodes[out_node].type == 'input':
            continue

        # 跳过重复连接
        if any(conn.in_node == in_node and conn.out_node == out_node for conn in genome.connections):
            continue

        weight = random.uniform(-1.0, 1.0)
        innov = innovation_tracker.get_innovation(in_node, out_node)
        genome.connections.append(ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=weight,
            enabled=True,
            innovation_number=innov
        ))
        return True

    return False


# === 添加节点突变 ===
def mutate_add_node(genome, innovation_tracker):
    enabled = [c for c in genome.connections if c.enabled]
    if not enabled:
        return False

    conn = random.choice(enabled)
    conn.enabled = False

    new_node_id = max(genome.nodes.keys()) + 1
    genome.add_node(NodeGene(
        id=new_node_id,
        type_='hidden',
        activation='relu'
    ))

    genome.add_connection(ConnectionGene(
        in_node=conn.in_node,
        out_node=new_node_id,
        weight=1.0,
        enabled=True,
        innovation_number=innovation_tracker.get_innovation(conn.in_node, new_node_id)
    ))

    genome.add_connection(ConnectionGene(
        in_node=new_node_id,
        out_node=conn.out_node,
        weight=conn.weight,
        enabled=True,
        innovation_number=innovation_tracker.get_innovation(new_node_id, conn.out_node)
    ))

    return True


# === 总突变接口 ===
def mutate(genome, innovation_tracker, rate_add_conn=0.5):
    if random.random() < rate_add_conn:
        return mutate_add_connection(genome, innovation_tracker)
    else:
        return mutate_add_node(genome, innovation_tracker)
# === 基因交叉 ===
def crossover(parent1, parent2):
    conn_map1 = {c.innovation_number: c for c in parent1.connections}
    conn_map2 = {c.innovation_number: c for c in parent2.connections}

    all_innovs = sorted(set(conn_map1.keys()).union(conn_map2.keys()))
    child_connections = []

    for innov in all_innovs:
        if innov in conn_map1 and innov in conn_map2:
            gene = random.choice([conn_map1[innov], conn_map2[innov]])
        elif innov in conn_map1:
            gene = conn_map1[innov]
        else:
            gene = conn_map2[innov]
        child_connections.append(copy.deepcopy(gene))

    used_nodes = set()
    for c in child_connections:
        used_nodes.add(c.in_node)
        used_nodes.add(c.out_node)

    child_nodes = {
        nid: copy.deepcopy(parent1.nodes.get(nid) or parent2.nodes[nid])
        for nid in used_nodes
    }

    return Genome(nodes=child_nodes, connections=child_connections)
