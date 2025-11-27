class NodeGene:
    def __init__(self, id, type_, activation="relu"):
        self.id = id
        self.type = type_  # 'input', 'hidden', 'output'
        self.activation = activation

    def __repr__(self):
        return f"NodeGene(id={self.id}, type='{self.type}', activation='{self.activation}')"


class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return (f"ConnectionGene({self.in_node} -> {self.out_node}, "
                f"w={self.weight:.3f}, {status}, innov={self.innovation_number})")
