import os
from bpneat.neat.genome import make_initial_genome
import matplotlib as plt

def test_genome():
    genome = make_initial_genome()

    print("== Nodes ==")
    for n in genome.nodes:
        print(n)

    print("\n== Connections ==")
    for c in genome.connections:
        print(c)

    print("\n== Drawing ==")
    genome.draw("figs/xor_genome.png")
    print("\n == Over ==")

if __name__ == "__main__":
    test_genome()
