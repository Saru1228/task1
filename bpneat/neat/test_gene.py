from gene import NodeGene, ConnectionGene

n1 = NodeGene(0, "input")
n2 = NodeGene(1, "hidden", "relu")
n3 = NodeGene(2, "output", "sigmoid")

c1 = ConnectionGene(0, 1, 0.5, True, 1)
c2 = ConnectionGene(1, 2, -1.2, True, 2)

print(n1)
print(n2)
print(n3)
print(c1)
print(c2)
